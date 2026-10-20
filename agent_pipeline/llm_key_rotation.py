"""Rotates through multiple API keys configured for a single backend.

A free Gemini account can generate several keys, each with its own
separate rate limit, so a student who sets up GOOGLE_API_KEY_1 through
GOOGLE_API_KEY_10 (see llm_backends.py) gets several times the effective
quota of any single key, without needing to fall through to a different
provider at all.

Rotation reacts to two different conditions, and treats them
differently on purpose:

- A rate limit means *that key's own quota* is used up. It's likely to
  stay that way for a while (minutes to a full day, depending on the
  provider), so a rate-limited key is removed from rotation for the rest
  of the session -- there's nothing to gain by trying it again soon.
  The one exception is a backend's *last* key: removing it would leave
  the student stuck until a kernel restart (which, in AgenticPipeline
  and Homework, erases the agent's memory of the run), even after a
  short per-minute limit has long cleared. So the last key is set
  aside instead, and gets one more try each time the student sends a
  message while no other key is active; if that works, it's back in
  rotation.
- The provider's servers being unreachable or overloaded
  (rate_limits.is_server_unavailable_error) says nothing about that
  key's own quota at all -- it's the provider's infrastructure being
  busy at that moment, for every key equally. Permanently dropping a key
  over a transient 503 would waste a perfectly good key for the rest of
  the session. Instead it goes to the back of the line: still part of
  the active rotation, just given a turn to let the moment pass.

Each Gemini key here comes from its own separate project (see the
README's setup instructions), so an overloaded response through one
key's project isn't necessarily true of another project's key at the
same moment.

Two ways this class gets used:

1. Standalone (multi-provider fallback off, or only one backend has a
   key configured): invoke() is fully self-contained, trying every
   currently-active key of *this* backend once per call (a round scaled
   to how many distinct keys exist -- one key means one try, ten keys
   means up to ten), then handing control back if none worked. See
   invoke()'s own docstring for the reasoning.
2. Wrapped inside FallbackChatModel (llm_fallback.py), when more than
   one backend has a key configured: FallbackChatModel calls
   try_current() directly instead of invoke(), taking exactly one
   attempt against this backend's current key and handing control
   straight back to the orchestrator either way (success or a specific
   exception to react to) -- so it can interleave with *another*
   backend's key on every single failure, alternating providers on a
   server-unavailable error rather than exhausting one backend's whole
   pool before trying the next. Rate-limit removal and server-error
   requeue bookkeeping (and their own print lines) happen here either
   way; only the round-shape and the cross-backend interleaving belong
   to the caller.

Sticky on success: once a key works, later calls try that same key
first again, rather than restarting from the front of the original
list -- no reason to re-test keys that were already skipped past.

"No added delay" is not the same as "fast": each try is still a real
network call, and each one can itself take up to the configured request
timeout (60s, see llm_backends.py) before it even fails, especially for
a hung-connection-style server-unavailable error rather than a fast
error response. Getting that "up to 60s per real call, not several
minutes" bound to actually hold depends on llm_backends.py capping each
SDK's own internal retry-on-failure behavior to a single attempt
(max_retries=1, both backends) -- without that, one nominal "try" here
could silently be the underlying SDK retrying 2-6 more times on its own
beneath this module, each up to the timeout, before this code ever sees
the failure. Checked live: ChatGroq defaults max_retries to 2,
ChatGoogleGenerativeAI to 6; both are pinned to 1 in llm_backends.py
specifically so this module's own attempt-and-try-count accounting
reflects what's actually happening over the network, not additional
hidden retries underneath it.

A rate limit never counts against the server-error budget or pauses
anything. A line is printed (plain print(), not a widget call -- this
module has no display dependency, and never touches the LangChain
message list, so nothing it prints reaches either the model's own
context or the graded session log; see chat_widget.py and
session_logger.py) for every single key change: every rate-limit
removal, every server-error requeue.
"""

from collections import deque

from .backoff import MESSAGE_BUDGET
from .rate_limits import is_rate_limit_error, is_server_unavailable_error

_TOTAL_SERVER_ERROR_BUDGET = 800
_PROGRESS_PRINT_EVERY = 5  # how often the quiet per-key server-error retry shows a "still working" line, not every single one -- same constant name/value as graph_core.py's, kept separate since this module doesn't import that one


class _Cursor:
    """Holds the queue of key indices still active (in try order), plus
    the counters status() reports. A plain object rather than fields on
    RotatingKeyChatModel itself, so bind_tools() can return a new
    wrapper around freshly-bound models while both the bound and unbound
    versions share the same queue and counts -- graph_core.py calls
    invoke() on both an unbound model and its own bind_tools() result
    within one session, and a change through either one needs to be
    visible to the other.

    last_error is tracked here too, not just as a local in invoke(),
    because once the queue is empty a later invoke() call never enters
    the retry loop at all -- without a remembered error to raise, that
    call would have nothing to raise."""

    def __init__(self, n: int):
        self.queue = deque(range(n))
        self.last_error = None
        self.switches = 0  # every failed attempt, either kind, counted once each
        self.removed = 0  # keys dropped for good (rate limit)
        self.requeued = 0  # cumulative count of server-unavailable failures for this backend alone (standalone use only -- FallbackChatModel keeps its own separate global count when interleaving across backends)
        self.exhausted = False  # standalone use only; True once this backend's own 800-try server-error budget has been used up
        # Per-key detail, indexed the same way the queue's entries are
        # (original position, not current queue position), for a
        # /status-style listing -- the aggregate counters above can say
        # "2 removed" but not *which* two, which isn't enough to show a
        # student an actual per-key list.
        self.key_state = ["active"] * n  # "active", "rate_limited", or "set_aside"
        self.key_server_error_count = [0] * n
        self.set_aside = None  # index of this backend's last key, rate-limited but kept for a later retry (see the module docstring); None otherwise


class RotatingKeyChatModel:
    """Drop-in replacement for a single LangChain chat model. Exposes the
    same bind_tools()/invoke() surface graph_core.py relies on, backed
    by one already-built model per key. See the module docstring for the
    two different ways this gets driven (invoke() standalone, or
    try_current() from inside FallbackChatModel).

    Both is_rate_limit_error and is_server_unavailable_error are used
    rather than a plain isinstance check, the same reason they're used
    everywhere else in this project: Groq's integration never actually
    raises langchain_core's ModelRateLimitError, only Gemini's does.
    Checking the type alone would mean a Groq key that gets rate-limited
    (observed live as a 413 with a rate_limit_exceeded body, not even a
    plain 429) never triggers rotation at all, silently defeating
    multi-key setup for Groq specifically."""

    def __init__(self, llms: list[object], cursor: _Cursor = None, name: str = None):
        if not llms:
            raise ValueError("RotatingKeyChatModel needs at least one model to rotate through.")
        self._llms = llms
        self._cursor = cursor if cursor is not None else _Cursor(len(llms))
        self._name = name  # e.g. "groq" -- only used to label this backend's own print lines when interleaved with another backend's identically-numbered keys

    def bind_tools(self, tools):
        return RotatingKeyChatModel(
            [llm.bind_tools(tools) for llm in self._llms], cursor=self._cursor, name=self._name
        )

    def has_keys(self) -> bool:
        """Whether this backend still has at least one active key --
        False once every key has been rate-limited away for good.
        FallbackChatModel checks this before bothering to interleave in
        a backend that has nothing left to offer."""
        return bool(self._cursor.queue)

    def has_set_aside(self) -> bool:
        """Whether this backend's last key is set aside after a rate
        limit (see the module docstring) -- not active, but still worth
        one try per message once nothing else is left."""
        return self._cursor.set_aside is not None

    def _label(self, idx: int) -> str:
        prefix = f"{self._name} " if self._name else ""
        return f"{prefix}key {idx + 1}"

    def try_current(self, messages):
        """Exactly one attempt against the current front-of-queue key.
        Returns the result on success. On a rate limit, removes that key
        for good, prints that, and re-raises. On a server-unavailable
        error, moves that key to the back of the queue (still active),
        prints that, and re-raises. Anything else re-raises unchanged.
        Does no looping and no round/budget decisions of its own -- see
        the module docstring for who owns those in each of the two ways
        this class gets used."""
        cursor = self._cursor
        if not cursor.queue:
            if cursor.set_aside is not None:
                return self._try_set_aside(messages)
            raise cursor.last_error
        idx = cursor.queue[0]
        try:
            return self._llms[idx].invoke(messages)
        except Exception as exc:
            if is_rate_limit_error(exc):
                cursor.queue.popleft()
                cursor.removed += 1
                cursor.last_error = exc
                cursor.switches += 1
                if cursor.queue:
                    cursor.key_state[idx] = "rate_limited"
                    print(
                        f"{self._label(idx)} rate-limited -- removed for the rest of this session "
                        f"({cursor.removed} of {len(self._llms)} keys now removed, "
                        f"{len(cursor.queue)} still active)."
                    )
                else:
                    cursor.key_state[idx] = "set_aside"
                    cursor.set_aside = idx
                    print(
                        f"{self._label(idx)} rate-limited -- it's the last key left, so it's set "
                        "aside instead of removed. Wait a minute, then send another message to "
                        "try it again."
                    )
                raise
            if not is_server_unavailable_error(exc):
                raise
            cursor.requeued += 1
            cursor.key_server_error_count[idx] += 1
            cursor.last_error = exc
            cursor.switches += 1
            cursor.queue.popleft()
            cursor.queue.append(idx)
            # Mostly silent, not a line per attempt (up to
            # _ROUND_SIZE/_TOTAL_SERVER_ERROR_BUDGET times) -- throttled
            # to the first one and every _PROGRESS_PRINT_EVERY'th after,
            # by cursor.requeued (cumulative for this backend, so the
            # exact 1st/5th/10th... doesn't reset at a turn boundary,
            # just keeps a steady occasional heartbeat across the
            # session), the same treatment graph_core.py's empty-retry
            # line got. The rate-limit removal print just above, and
            # the total-exhaustion prints below/in llm_fallback.py,
            # print every time, unaffected -- those are rarer and
            # actionable.
            if cursor.requeued == 1 or cursor.requeued % _PROGRESS_PRINT_EVERY == 0:
                print("Processing your request...")
            raise

    def _try_set_aside(self, messages):
        """One try against the set-aside last key. On success it's back
        in rotation (active again, no longer counted as removed). On a
        rate limit or server error it stays set aside for the next
        message; either way the exception is re-raised."""
        cursor = self._cursor
        idx = cursor.set_aside
        try:
            result = self._llms[idx].invoke(messages)
        except Exception as exc:
            if is_rate_limit_error(exc):
                cursor.last_error = exc
                cursor.switches += 1
                print(
                    f"{self._label(idx)} is still rate-limited. Wait a bit longer before the next "
                    "message; if this keeps happening, its daily quota is used up."
                )
            elif is_server_unavailable_error(exc):
                cursor.last_error = exc
                cursor.switches += 1
                cursor.requeued += 1
                cursor.key_server_error_count[idx] += 1
            raise
        cursor.set_aside = None
        cursor.queue.append(idx)
        cursor.removed -= 1
        cursor.key_state[idx] = "active"
        print(f"{self._label(idx)} is working again -- back in rotation.")
        return result

    def invoke(self, messages):
        """Self-contained retry loop for standalone use (see the module
        docstring): at most one round, scaled to how many distinct keys
        are currently active. A single-key backend gets exactly one try
        before handing control back -- retrying the *same* key again
        immediately, with no delay and nothing about the situation
        changed, doesn't improve the odds, it just burns through the try
        budget faster for no benefit. An N-key backend gets up to N tries
        in that round, since those can plausibly be up or down
        independently of each other. Once a round ends without success,
        raises rather than continuing to retry silently -- a message
        explains what happened and asks the student to wait a bit before
        sending another message (which starts the next round), and not
        to restart or stop the kernel, since that would lose the
        in-memory rotation state."""
        cursor = self._cursor
        if not cursor.queue and cursor.set_aside is not None and not cursor.exhausted:
            # Only the set-aside last key is left: one try per message.
            # LAB 2 DEVIATION: wait first. This branch is only reached because
            # that key was rate-limited, so trying it the instant the student
            # presses send mostly spends the one try on a limit that has not
            # cleared yet. A few seconds is not a full minute, but the student
            # has usually spent some of that minute typing, and a wait here
            # costs one short pause where failing costs the whole message.
            if MESSAGE_BUDGET.remaining() > 0:
                print("Waiting a moment for the rate limit to clear...")
                MESSAGE_BUDGET.wait(3, cursor.last_error)
            return self.try_current(messages)
        tried_this_call = set()  # original key indices already tried once this call
        while cursor.queue and not cursor.exhausted:
            idx = cursor.queue[0]
            if idx in tried_this_call:
                # Every currently-active key has had its one try for this
                # call already. Stop here instead of trying one a second
                # time in the same round.
                break
            try:
                return self.try_current(messages)
            except Exception as exc:
                if is_rate_limit_error(exc):
                    continue  # try_current() already advanced the queue past this key
                if not is_server_unavailable_error(exc):
                    raise
                tried_this_call.add(idx)
                # LAB 2 DEVIATION: a short jittered wait before the next key,
                # for the same reason as llm_fallback.py's. A server error is
                # the one failure that tends to fix itself.
                MESSAGE_BUDGET.wait(len(tried_this_call), exc)
                if cursor.requeued >= _TOTAL_SERVER_ERROR_BUDGET:
                    print(
                        f"{self._label(idx)} hit a server error on try {cursor.requeued} of "
                        f"{_TOTAL_SERVER_ERROR_BUDGET} -- that was the last one. "
                        "Not available, tries exhausted."
                    )
                    cursor.exhausted = True
                    cursor.queue.clear()
                    break
        if cursor.queue and not cursor.exhausted:
            print(
                "That was every active key for this backend, none worked this "
                "round. Please wait a few minutes, then send another message "
                "to try again -- don't restart or stop the kernel, that would "
                "lose this progress."
            )
        raise cursor.last_error

    def skip_current(self) -> None:
        """Moves the current front-of-queue key to the back, with none of
        try_current()'s bookkeeping: no removed/requeued counter change,
        no print, no exception involved at all. For a caller retrying
        after something that isn't a rate limit or a server error (an
        empty response with nothing usable in it, say) and wants a
        different key on the next attempt, without holding the empty
        response against the key it just used. A no-op on a one-key or
        empty queue -- there's nothing else to rotate to."""
        if len(self._cursor.queue) > 1:
            self._cursor.queue.rotate(-1)

    def status(self) -> dict:
        """Which numbered key is at the front of the queue (the one
        that's active now, or would be tried next), how many keys this
        backend has in total, how many are still active (including any
        currently at the back of the line after a server error), the two
        separate counts that make up the difference (keys removed for
        good vs. server-error requeue events), and whether this
        backend's own server-error retry budget has been fully used up
        (standalone use only -- when interleaved via FallbackChatModel,
        the budget that actually matters is its global one, not this)."""
        return {
            "key_number": (self._cursor.queue[0] + 1) if self._cursor.queue else None,
            "set_aside_key": (self._cursor.set_aside + 1) if self._cursor.set_aside is not None else None,
            "total_keys": len(self._llms),
            "active_keys": len(self._cursor.queue),
            "keys_removed": self._cursor.removed,
            "keys_requeued": self._cursor.requeued,
            "key_switches": self._cursor.switches,
            "exhausted": self._cursor.exhausted,
        }

    def key_list(self) -> list[dict]:
        """One entry per key this backend has, in original numbering
        order (key 1, key 2, ...), for a /status-style listing rather
        than just the aggregate counts status() returns. `current` marks
        whichever key is at the front of the queue right now (the one a
        call would use next), independent of `state`."""
        current_idx = self._cursor.queue[0] if self._cursor.queue else None
        return [
            {
                "key_number": idx + 1,
                "state": self._cursor.key_state[idx],
                "server_error_count": self._cursor.key_server_error_count[idx],
                "current": idx == current_idx,
            }
            for idx in range(len(self._llms))
        ]

"""Optional multi-provider fallback for the LLM backends.

Off by default. A notebook has to explicitly opt in (a "multi-provider
mode" toggle in the setup cell) before this is used at all; the plain
single-backend path (llm_backends.build_llm_for_backend, used directly)
is unaffected and remains what runs when fallback isn't enabled.

When enabled, FallbackChatModel interleaves across every backend with a
usable key configured, one attempt at a time, rather than exhausting
one backend's whole key pool before ever trying another. On a rate
limit, the failing key is removed for good from its own backend (see
llm_key_rotation.py) and the very next backend in the chain is tried
immediately -- rate limits never pause anything. On the provider's
servers being unreachable or overloaded
(rate_limits.is_server_unavailable_error: a connection failure, a
timeout with no response, or a 5xx -- confirmed against each SDK's own
exception hierarchy, not guessed at), the failing key is moved to the
back of its own backend's queue (still active, not removed), and the
*next backend* in the chain is tried next, not another key of the same
one -- switching provider is far more likely to actually help than
retrying a different key of a provider whose infrastructure is having a
moment, since that's an outage dimension separate from any one key's
own quota. Concretely: Groq hits a server error -> Gemini is tried
next; if Gemini also hits one -> back to Groq (a *different* Groq key,
if it has more than one; the same one again, if it only has one, since
there's nothing else on that side to offer) -> and so on, continuing to
alternate for up to 25 cumulative server-error tries across every
configured backend combined, before handing control back to the
student. Any other kind of failure (a bad model name, a malformed
request, an auth error) is not treated as a reason to switch backends,
since silently masking those could make a real bug look like normal
fallback behavior.

This interleaving is driven through each backend's try_current()/
has_keys() (llm_key_rotation.py), not its self-contained invoke() --
try_current() takes exactly one attempt and hands control straight back
here either way, which is what makes per-attempt interleaving possible
in the first place. build_llm_for_backend always returns a
RotatingKeyChatModel, even for a single key, specifically so every
chain entry here exposes those consistently -- no bare-model special
case to work around.

The cumulative server-error budget (25 tries per round, 800 total before
giving up on this session's fallback entirely) lives here, globally
across every backend in the chain, not per backend -- it's one ongoing
question of "has anything responded yet," not a separate budget per
provider. See llm_key_rotation.py's module docstring for why a round is
no longer a fixed number regardless of what's configured: the same
"don't hammer the same resource with no delay" reasoning applies here
too, it just spans providers instead of spanning one provider's own
keys.
"""

from .backoff import MESSAGE_BUDGET
from .llm_backends import build_llm_for_backend, has_api_key
from .rate_limits import is_rate_limit_error, is_server_unavailable_error

_FALLBACK_ORDER = ("groq", "gemini")
_ROUND_SIZE = 25
_TOTAL_SERVER_ERROR_BUDGET = 800


def available_backends(primary: str, env_path: str = "config/.env") -> list[str]:
    """Backend names with a usable key configured, primary first, then
    the rest of _FALLBACK_ORDER, each name included at most once."""
    ordered = [primary] + [b for b in _FALLBACK_ORDER if b != primary]
    return [b for b in ordered if has_api_key(b, env_path=env_path)]


class _FallbackCursor:
    """Same idea as llm_key_rotation._Cursor, for the same reason: a
    plain shared object rather than fields on FallbackChatModel itself,
    so bind_tools() can hand back a new wrapper around freshly-bound
    backends while the bound and unbound versions (agent_node uses
    llm_with_tools, discuss_node uses llm directly, per graph_core.py)
    both see the same active backend and the same global counts."""

    def __init__(self, first_backend: str):
        self.active_backend = first_backend
        self.backend_switches = 0  # times the backend that actually answered changed from the previous one
        self.requeued = 0  # cumulative server-unavailable failures across every backend combined
        self.switches = 0  # cumulative failed attempts of either kind, across every backend combined
        self.exhausted = False
        self.last_error = None
        self.skip_backend = None  # one-shot: set by skip_current(), consumed by the very next invoke() call


class FallbackChatModel:
    """Drop-in replacement for a single LangChain chat model. Exposes the
    same bind_tools()/invoke() surface graph_core.py relies on, and
    interleaves across every backend in `chain`, one attempt at a time --
    see the module docstring for the full reasoning."""

    def __init__(self, chain: list[tuple[str, object]], cursor: _FallbackCursor = None):
        if not chain:
            raise ValueError("FallbackChatModel needs at least one backend to try.")
        self._chain = chain  # [(backend_name, llm_instance), ...], in try-order
        self._cursor = cursor if cursor is not None else _FallbackCursor(chain[0][0])

    def bind_tools(self, tools):
        return FallbackChatModel(
            [(name, llm.bind_tools(tools)) for name, llm in self._chain], cursor=self._cursor
        )

    def invoke(self, messages):
        """One round, interleaved across every backend in `chain` that
        still has an active key: try each in turn (skipping any with none
        left), one attempt each, cycling back to the start of the chain
        until either something succeeds, the round's global budget of
        _ROUND_SIZE server-error tries runs out, the cumulative
        _TOTAL_SERVER_ERROR_BUDGET is used up for good, or every backend
        is completely out of keys. A rate limit removes that key from its
        own backend and moves on immediately, uncounted against either
        budget; only a server-unavailable error counts a try and can end
        a round. See the module docstring for the full reasoning."""
        cursor = self._cursor
        if cursor.exhausted:
            raise cursor.last_error
        # One-shot: skip_current() can ask the very next call to try a
        # different backend first, e.g. after an empty response that
        # wasn't a rate limit or a server error. Consumed here so it
        # only ever applies to this one call, and only for this call's
        # first pass through the chain -- never skips the only backend
        # that actually still has keys.
        skip = cursor.skip_backend
        cursor.skip_backend = None
        available = [name for name, backend in self._chain if backend.has_keys()]
        # Backends whose last key was already set aside (after a rate
        # limit) before this message -- only these get a retry below if
        # nothing active is left, so a key set aside during this same
        # call isn't immediately hit again.
        set_aside_before = [
            name for name, backend in self._chain
            if hasattr(backend, "has_set_aside") and backend.has_set_aside()
        ]
        skip_first_pass = skip in available and len(available) > 1
        first_pass = True
        round_start = cursor.requeued
        while cursor.requeued - round_start < _ROUND_SIZE:
            progressed = False
            for name, backend in self._chain:
                if not backend.has_keys():
                    continue
                if first_pass and skip_first_pass and name == skip:
                    continue
                progressed = True
                try:
                    result = backend.try_current(messages)
                except Exception as exc:
                    if is_rate_limit_error(exc):
                        # backend already removed the key and printed
                        # its own line; the very next backend in the
                        # chain is tried immediately, no pause.
                        cursor.last_error = exc
                        cursor.switches += 1
                        continue
                    if not is_server_unavailable_error(exc):
                        raise
                    cursor.requeued += 1
                    cursor.switches += 1
                    cursor.last_error = exc
                    if cursor.requeued >= _TOTAL_SERVER_ERROR_BUDGET:
                        print(
                            f"Still unavailable after {_TOTAL_SERVER_ERROR_BUDGET} tries across "
                            "every configured backend -- not available, tries exhausted."
                        )
                        cursor.exhausted = True
                        raise exc
                    if cursor.requeued - round_start >= _ROUND_SIZE:
                        break  # this round's budget is used; stop mid-pass rather than start another backend
                    # LAB 2 DEVIATION: wait before the next try. This loop
                    # allows 25 tries a round and 800 a session, and with no
                    # pause a provider having a bad minute was answered with 25
                    # full-size requests in a couple of seconds. The wait is
                    # exponential with full jitter, prefers the provider's own
                    # Retry-After when it sends one, and comes out of the
                    # shared per-message budget, so this can slow the loop down
                    # but can never make the notebook look frozen.
                    MESSAGE_BUDGET.wait(cursor.requeued - round_start, exc)
                    continue
                else:
                    if name != cursor.active_backend:
                        cursor.backend_switches += 1
                        cursor.active_backend = name
                    return result
            first_pass = False
            if not progressed:
                # Every configured backend is out of active keys (all
                # rate-limited away) -- nothing left to interleave. A
                # backend's set-aside last key still gets one try per
                # message, since its quota may have reset by now.
                return self._try_set_aside_keys(messages, set_aside_before)
        print(
            "That was this round's tries across every configured backend, none "
            "worked. Please wait a few minutes, then send another message to "
            "try again -- don't restart or stop the kernel, that would lose "
            "this progress."
        )
        raise cursor.last_error

    def _try_set_aside_keys(self, messages, names):
        """One try each against the set-aside last key of every backend
        in `names`, in chain order. Returns the first success; if none
        works, re-raises the last error, same as having nothing left."""
        cursor = self._cursor
        for name, backend in self._chain:
            if name not in names or not backend.has_set_aside():
                continue
            try:
                result = backend.try_current(messages)
            except Exception as exc:
                if not (is_rate_limit_error(exc) or is_server_unavailable_error(exc)):
                    raise
                cursor.last_error = exc
                cursor.switches += 1
                continue
            if name != cursor.active_backend:
                cursor.backend_switches += 1
                cursor.active_backend = name
            return result
        raise cursor.last_error

    def skip_current(self) -> None:
        """Asks the very next invoke() call to try a different backend
        first, skipping whichever one is currently active_backend (never
        skips it if it's the only backend with keys left; invoke() just
        ignores the request in that case). Also rotates that backend's
        own key queue, via its own skip_current(), so cycling back to it
        later on doesn't just land on the exact same key again. For a
        caller retrying after something that isn't a rate limit or a
        server error (an empty response, say) and doesn't want to hold
        it against either the key or the backend that produced it."""
        cursor = self._cursor
        cursor.skip_backend = cursor.active_backend
        for name, backend in self._chain:
            if name == cursor.active_backend and hasattr(backend, "skip_current"):
                backend.skip_current()
                break

    def status(self) -> dict:
        """Which backend most recently answered, how many times that's
        changed, and the global server-error counts this session (across
        every configured backend combined, not per backend) -- plus,
        merged in, that backend's own key_number/total_keys/keys_removed
        from RotatingKeyChatModel.status(), since "which key" is still
        useful to show even though the retry budget itself is global
        now."""
        info = {
            "backend": self._cursor.active_backend,
            "backend_switches": self._cursor.backend_switches,
            "key_switches": self._cursor.switches,
            "keys_requeued": self._cursor.requeued,
            "exhausted": self._cursor.exhausted,
        }
        active_llm = next((llm for name, llm in self._chain if name == self._cursor.active_backend), None)
        if active_llm is not None and hasattr(active_llm, "status"):
            backend_status = active_llm.status()
            info["key_number"] = backend_status.get("key_number")
            info["total_keys"] = backend_status.get("total_keys")
            info["keys_removed"] = backend_status.get("keys_removed")
        return info

    def key_list(self) -> list[dict]:
        """Every key across every configured backend, each entry tagged
        with which backend it belongs to, for a /status-style listing.
        A backend with no key_list() of its own (shouldn't happen given
        how this project builds backends, but not assumed) is skipped
        rather than raising.

        Each RotatingKeyChatModel marks `current` on whichever key sits
        at the front of *its own* queue, with no idea whether this
        backend is the one actually in use right now -- an idle backend
        still has a front-of-queue key, it's just never been touched.
        Left alone, that means every configured backend shows its own
        "in use now" key simultaneously. Only cursor.active_backend is
        really active, so every other backend's `current` is forced to
        False here."""
        rows = []
        for name, backend in self._chain:
            if not hasattr(backend, "key_list"):
                continue
            is_active_backend = name == self._cursor.active_backend
            for entry in backend.key_list():
                if not is_active_backend:
                    entry = {**entry, "current": False}
                rows.append({"backend": name, **entry})
        return rows


def build_fallback_llm(primary: str, temperature: float = 0, env_path: str = "config/.env"):
    """Builds a FallbackChatModel starting with `primary` and
    interleaving, on a rate limit or server-unavailable error, across
    every other backend with a configured key. Returns (llm,
    backend_names) so the caller can show which backends are actually in
    the chain. Raises the same error get_api_key would if even the
    primary has no usable key."""
    backend_names = available_backends(primary, env_path=env_path)
    chain = [(name, build_llm_for_backend(name, temperature=temperature, env_path=env_path)) for name in backend_names]
    return FallbackChatModel(chain), backend_names

"""LAB 2 DEVIATION (not in Lab 1): measuring the request instead of guessing.

graph_core.py used to cap the conversation at a fixed number of characters.
That works, but it is blind in two ways. Characters are not tokens, and the
ratio differs per model and per text. More importantly, the messages are not
the whole request: the system prompt and all fourteen tool descriptions ride
along on every agent call and are nowhere in the message list, so a ceiling
measured on messages alone cannot see most of what is actually sent. Measured
live, that invisible part was about 2,700 tokens, a third of a free Groq key's
8,000 per minute, before the student had said anything at all.

So this keeps a running picture of the real request, learned from what the
provider itself reports back (`usage_metadata.input_tokens` on every reply):

* `chars_per_token` converts our cheap character count into tokens.
* `overhead_tokens` is everything in a request that is *not* the messages,
  recovered as (reported input tokens) minus (our estimate of the messages).

From those two, `history_char_allowance()` answers the only question
_trim_history actually has: how much conversation fits under the target with
room left for the reply. Before the first reply arrives, both start at the
values measured on this lab's own notebook, so the first message behaves
exactly as the old fixed ceiling did; after it, the numbers are this student's
real ones, on their model, with their tools.

Why a target rather than the full per-minute limit: the limit counts the reply
too, and a retry after an empty reply pays the whole prompt a second time, so
aiming at the ceiling guarantees hitting it. There is no attempt to track the
remaining per-minute allowance across calls, which would need a clock the
provider does not share; staying under a per-request target is what a notebook
can honestly do, and it is what keeps a long session alive.
"""

# Measured on this lab's agent notebook, used until the provider's own numbers
# replace them: 2,668 tokens of system prompt plus fourteen tool schemas, and
# a target that reproduces the old 5,000-character ceiling on message one.
_DEFAULT_CHARS_PER_TOKEN = 4.0
_DEFAULT_OVERHEAD_TOKENS = 2668
_TARGET_INPUT_TOKENS = 4000
# Calibration is a measurement, so it can be noisy; these keep one odd reply
# from either collapsing the conversation to nothing or opening it wide.
_MIN_HISTORY_TOKENS = 800
_MAX_HISTORY_TOKENS = 4000
# A new observation never replaces the old picture outright, it moves it part
# of the way. One unusual call should not reset what many calls established.
_SMOOTHING = 0.3


class ContextBudget:
    def __init__(self):
        self.chars_per_token = _DEFAULT_CHARS_PER_TOKEN
        self.overhead_tokens = _DEFAULT_OVERHEAD_TOKENS
        self.observations = 0
        self.last_input_tokens = None

    def estimate_tokens(self, chars: int) -> float:
        return chars / self.chars_per_token

    def observe(self, history_chars: int, usage) -> None:
        """Learns from one real reply. `usage` is the reply's usage_metadata;
        anything without a usable input token count is ignored, since several
        backends omit it and a wrong number is worse than no number."""
        if not isinstance(usage, dict):
            return
        reported = usage.get("input_tokens")
        if not isinstance(reported, (int, float)) or reported <= 0:
            return
        self.last_input_tokens = int(reported)
        estimated_history = self.estimate_tokens(history_chars)
        overhead = reported - estimated_history
        # A request smaller than our estimate of its own messages means the
        # ratio is off, not that the overhead is negative. Fix the ratio in
        # that case and leave the overhead alone.
        if overhead < 0:
            if history_chars > 0 and reported > 0:
                observed_ratio = history_chars / reported
                self.chars_per_token += _SMOOTHING * (observed_ratio - self.chars_per_token)
        else:
            self.overhead_tokens += _SMOOTHING * (overhead - self.overhead_tokens)
        self.observations += 1

    def history_token_allowance(self) -> float:
        room = _TARGET_INPUT_TOKENS - self.overhead_tokens
        return max(_MIN_HISTORY_TOKENS, min(_MAX_HISTORY_TOKENS, room))

    def history_char_allowance(self) -> int:
        return int(self.history_token_allowance() * self.chars_per_token)

    def status_line(self) -> str:
        """What /status prints, so a student can see why the conversation is
        being shortened rather than guessing at it."""
        if self.observations == 0:
            return (
                f"Context: about {self.overhead_tokens:.0f} tokens of prompt and tool "
                f"descriptions before your conversation, about "
                f"{self.history_char_allowance():,} characters of conversation kept "
                "(estimated; not measured yet)."
            )
        return (
            f"Context: last request {self.last_input_tokens:,} tokens, of which about "
            f"{self.overhead_tokens:.0f} are the prompt and tool descriptions. Keeping "
            f"about {self.history_char_allowance():,} characters of conversation, "
            f"measured over {self.observations} replies."
        )


# One per process, like the rest of this pipeline's small shared state: the
# bound and unbound models and both graph nodes all describe the same request.
BUDGET = ContextBudget()

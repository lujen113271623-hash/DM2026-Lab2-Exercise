"""LAB 2 DEVIATION (not in Lab 1): waiting between retries.

Every retry loop in this pipeline retries *immediately*. On a rate limit that
is right, because the next attempt goes to a different key or a different
provider, and waiting would help nobody. On the provider's own servers being
unreachable or overloaded it is wrong: llm_fallback.py allows 25 server-error
tries per round and 800 across a session, and with no pause between them a
provider having a bad minute is answered with 25 full-size requests in a few
seconds. That burns a free tier's budget on failures, and it is the one case
where the thing we are waiting for actually does get better on its own.

So server-error retries wait here first, with three properties that matter:

* **Exponential**, so a short outage costs one short wait and a long one
  stops quickly rather than hammering at a fixed rate.
* **Full jitter** (`uniform(0, delay)`, not `delay`), because every student in
  a lab session hits the same provider at the same time. Without jitter a
  class retries in lockstep and rebuilds the spike it is backing off from.
* **Budgeted per message**, so the notebook can never appear to hang. When the
  budget is spent the caller gives up and hands control back to the student,
  which is what the chat already does today, just a few seconds later and
  after far fewer wasted calls.

Rate limits keep their existing no-pause behavior while another key or
provider is left to try. The single exception is the last key of the last
backend: there is nothing to rotate to, so a short wait is the only thing that
can help, and students with one key are exactly who runs into this.
"""

import random
import time

# One failure should cost a fraction of a second, not a visible stall; the cap
# keeps any single wait short enough that the chat still feels alive, and the
# per-message budget bounds the whole thing. Measured against the real retry
# counts: 25 server-error tries now spread over at most _TOTAL_SECONDS rather
# than firing in roughly a second.
_BASE_SECONDS = 0.5
_CAP_SECONDS = 8.0
_TOTAL_SECONDS = 20.0


def retry_after_seconds(exc) -> float | None:
    """The provider's own Retry-After header, when it sends one. Always
    preferred over our guess: it is the only number that reflects what the
    server actually knows. Defensive throughout, since this runs inside an
    except branch and must never raise an error of its own."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
        if raw is None:
            return None
        seconds = float(raw)
    except (AttributeError, TypeError, ValueError):
        return None
    return seconds if 0 <= seconds <= _TOTAL_SECONDS else None


class BackoffBudget:
    """How long this one message may spend waiting, in total. A fresh budget
    per message, so a long session never accumulates a longer and longer
    stall, and `waited` lets the caller report what it spent."""

    def __init__(self, total_seconds: float = _TOTAL_SECONDS):
        self.total = total_seconds
        self.spent = 0.0
        self.waits = 0

    def remaining(self) -> float:
        return max(0.0, self.total - self.spent)

    def wait(self, attempt: int, exc=None, sleep=None) -> bool:
        """Sleeps before retry number `attempt` (1 for the first retry).
        Returns False when the budget is gone, which the caller should treat
        as "stop retrying and hand control back", not as an error. `sleep` is
        injectable so the tests can assert the delays without taking them, and
        is resolved at call time rather than defaulted to time.sleep in the
        signature: a default argument binds once at import, which would make
        the real clock impossible to substitute for exactly the tests that
        need to."""
        remaining = self.remaining()
        if remaining <= 0:
            return False
        delay = retry_after_seconds(exc)
        if delay is None:
            ceiling = min(_CAP_SECONDS, _BASE_SECONDS * (2 ** max(0, attempt - 1)))
            delay = random.uniform(0, ceiling)
        delay = min(delay, remaining)
        if delay > 0:
            (sleep or time.sleep)(delay)
        self.spent += delay
        self.waits += 1
        return True

    def reset(self) -> None:
        self.spent = 0.0
        self.waits = 0


# One budget shared by every retry layer, reset once per student message by
# graph_core._invoke_retrying_empty. Shared deliberately: the empty-reply loop,
# the key rotation inside it and the provider fallback inside that are three
# nested loops, and what a student experiences is the total stall, not any one
# layer's share of it. Separate budgets would quietly add up.
MESSAGE_BUDGET = BackoffBudget()

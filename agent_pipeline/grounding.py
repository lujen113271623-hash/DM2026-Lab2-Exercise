"""Deterministic numeric-grounding check for the agent's discussion turns.

This is not another LLM call and not a check on reasoning quality. It
just asks whether every number the agent states in prose actually traces
back to a real value somewhere in the session's stored tool results. That
catches fabricated statistics, like a hallucinated correlation value or
an invented count, without adding a second layer of LLM judgment that
could itself be wrong.

False positives are expected and acceptable (for example "the top 3
terms" will flag "3" if nothing in the results happens to equal 3). This
is meant as an advisory flag for a TA to spot-check, not a hard gate that
blocks a response.
"""

import re

from .message_text import as_text

# Comma-grouped thousands ("2,257") are matched before plain decimals or
# integers, so a number like that reads as one value rather than splitting
# into "2" and "257".
#
# Two lookaround exclusions keep the regex from matching digits that
# aren't really standalone numbers:
# - (?<!\w) before the number skips a digit glued onto an identifier with
#   no real word boundary, such as the trailing "1" in "load_dataset_1"
#   (a result id, not a stated statistic).
# - (?!\d)(?!-[A-Za-z]) after the number skips a number embedded in a
#   hyphenated compound word, such as the "20" in "20-newsgroups" (part
#   of the dataset's name). The (?!\d) has to come first, otherwise the
#   regex engine can backtrack a longer digit run like "20" down to a
#   shorter one like "2" to satisfy the hyphen check against the wrong
#   substring, which lets a number like that leak through anyway.
_NUMBER_RE = re.compile(
    r"(?<!\w)-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)(?!-[A-Za-z])"
    r"|(?<!\w)-?\d+\.\d+(?!\d)(?!-[A-Za-z])"
    r"|(?<!\w)-?\d+(?!\d)(?!-[A-Za-z])"
)
_ROUND_DECIMALS = (0, 1, 2, 3, 4)


def _flatten_numeric(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_numeric(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flatten_numeric(v)


def extract_numbers(text) -> list[float]:
    numbers = []
    for m in _NUMBER_RE.findall(as_text(text)):
        try:
            numbers.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return numbers


def collect_grounded_values(results_store: dict) -> set[float]:
    grounded = set()
    for entry in results_store.values():
        for value in _flatten_numeric(entry["record"].summary):
            for d in _ROUND_DECIMALS:
                grounded.add(round(value, d))
    return grounded


def check_grounding(response_text, results_store: dict, tol: float = 0.01) -> list[float]:
    """Returns the list of numbers mentioned in `response_text` (str, or a
    list-of-content-blocks as some models return) that don't match (within
    `tol`) any real value stored in `results_store`."""
    grounded = collect_grounded_values(results_store)
    if not grounded:
        return []
    ungrounded = []
    for n in extract_numbers(response_text):
        if not any(abs(n - g) <= tol for g in grounded):
            ungrounded.append(n)
    return ungrounded

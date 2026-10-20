"""Detects a rate-limit error, or the provider's servers being
unreachable/overloaded, across every backend this project supports.
is_rate_limit_error is the older and more heavily-documented of the two
detectors below; is_server_unavailable_error was added later, for the
second condition -- see its own docstring for that one's reasoning.

langchain_core.exceptions.ModelRateLimitError is the type every backend
integration is supposed to raise a 429 as, but not all of them actually
do it. Confirmed by reading each integration's source: langchain_google_genai
wraps it correctly (GoogleRateLimitError subclasses ModelRateLimitError),
but langchain_groq doesn't wrap anything at all -- the raw provider SDK
exception (groq.RateLimitError) propagates untouched instead. That SDK
still sets a plain .status_code attribute on the exception
(groq._exceptions.APIStatusError sets self.status_code =
response.status_code directly, the same convention OpenAI-pattern
clients use), so that's the fallback check here rather than trusting
isinstance alone.

A single request that's simply too large for the per-minute token
budget is the same underlying condition, confirmed live: Groq returned
HTTP 413 ("Request too large... on tokens per minute (TPM)"), not 429,
for a conversation that had grown large enough (a sizeable system
prompt plus several turns of tool results) to exceed the free tier's
8000 TPM ceiling in one shot. The response body's own error code was
still "rate_limit_exceeded" despite the different HTTP status, so
status_code alone isn't a reliable signal either -- this also checks the
parsed body's error code when the SDK exposes one (groq.APIError sets
.body to the decoded JSON response).

Without this function -- if llm_fallback.py and chat_widget.py fell back
to a plain isinstance(exc, ModelRateLimitError) check instead of calling
this -- a Groq rate limit would never trigger multi-provider fallback,
and would never show the friendly "rate limited" message either; it
would show a raw traceback instead, on the backend most students
actually use.
"""

import httpx
from groq import APIConnectionError as GroqAPIConnectionError
from langchain_core.exceptions import ModelConnectionError, ModelRateLimitError, ModelTimeoutError


def _body_says_rate_limit(exc: BaseException) -> bool:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    error = body.get("error")
    return isinstance(error, dict) and error.get("code") == "rate_limit_exceeded"


def is_unbound_tool_call_error(exc: BaseException) -> bool:
    """True for Groq's gpt-oss-120b throwing a 400 because the model
    called a tool on a call where tools weren't bound (graph_core.py's
    discuss_node calls the plain, unbound `llm` -- see its own module
    docstring). Observed live, exact body:
    {"error": {"message": "Tool choice is none, but model called a
    tool", "type": "invalid_request_error", "code": "tool_use_failed",
    "failed_generation": "..."}}. This is model noise on that one call,
    not a sign the request itself is malformed or that anything is
    configured wrong -- a different key/backend on the very next
    attempt has a real chance of not repeating it, the same reasoning
    an empty response already gets retried for (see
    graph_core.py's _invoke_retrying_empty), rather than the "propagate
    immediately, don't mask a real bug" treatment every other
    unclassified error still gets."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    error = body.get("error")
    return isinstance(error, dict) and error.get("code") == "tool_use_failed"


def is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, ModelRateLimitError):
        return True
    # 429 (too many requests) is unambiguous by status code alone. Any
    # other status, including 413 (payload too large, which can also
    # mean a genuinely oversized or malformed request for reasons that
    # have nothing to do with a token budget), only counts here when the
    # response body itself confirms the code as rate_limit_exceeded, as
    # Groq's did, live, for a 413.
    if getattr(exc, "status_code", None) == 429:
        return True
    return _body_says_rate_limit(exc)


# 5xx and 529 ("overloaded") are the server-side codes both providers'
# SDKs actually surface for this. Read from each SDK's own source rather
# than assumed: Groq's InternalServerError (an APIStatusError subclass)
# sets .status_code from the real response; google-genai's ServerError
# family (GoogleAPIError, raised by langchain_google_genai for any 5xx)
# sets .code instead -- a different attribute name, same idea, so both
# need checking.
_SERVER_ERROR_STATUS_CODES = {500, 502, 503, 504, 529}


def is_server_unavailable_error(exc: BaseException) -> bool:
    """True for the provider's servers being unreachable or overloaded --
    a connection failure, a timeout with no response at all, or a 5xx --
    as opposed to a rate limit (is_rate_limit_error) or a real problem
    with the request itself (a bad model name, malformed input, auth).
    Confirmed against each SDK's actual exception hierarchy:
    - Groq (OpenAI-pattern client): APIConnectionError/APITimeoutError for
      a failed or timed-out connection (raised with no HTTP response, so
      no status_code at all); InternalServerError for an actual 5xx.
    - Gemini: langchain_google_genai wraps a raw connection failure as
      httpx.ConnectError/TimeoutException (google-genai's own retry logic
      already retries those internally before giving up), and wraps an
      actual 5xx as GoogleAPIError.
    - ModelConnectionError/ModelTimeoutError are langchain_core's own
      standardized types for this, checked in case either integration
      starts using them in a future version.
    """
    if isinstance(exc, (ModelConnectionError, ModelTimeoutError)):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, GroqAPIConnectionError):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    return status in _SERVER_ERROR_STATUS_CODES

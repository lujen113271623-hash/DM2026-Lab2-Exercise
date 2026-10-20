"""LLM backend construction for the agentic pipeline.

Two backends are supported, selected by name:
- "groq": gpt-oss-120b via Groq's API. Recommended, since Groq's free
  tier has much more headroom than Gemini's, whose free tier caps at 20
  requests per day for a given model, easy for a single student to
  exhaust mid-lab from normal use plus a couple of retries. This used to
  be Meta's Llama 3.3 70B, but Groq dropped that model from its catalog
  at some point after this was first built; confirmed live against
  Groq's /v1/models endpoint that Llama chat models are gone entirely
  and gpt-oss-120b is available and supports tool calling.
- "gemini": Google's Gemini API, same convention as Lab 2's setup
  (config/.env, GOOGLE_API_KEY).

A third backend, xAI's Grok, was supported earlier in this project but
has been dropped entirely; it's no longer an option anywhere in this
codebase.

Model names are centralized here so a future deprecation (this already
happened once, when gemini-2.0-flash was retired mid-project) only needs
a fix in one place.

Every backend is also built with a request timeout. Without one, a
provider that hangs (a dropped connection, a stuck generation) blocks the
call indefinitely with nothing to catch and display, since there's no
exception to catch until one actually happens. A hung student session
that never errors and never responds is a real failure mode this closes:
the call now surfaces as a normal, catchable timeout instead of leaving
the chat widget stuck on "Thinking..." forever.

A backend can also have more than one key configured: a free Gemini
account can generate several keys, each with its own separate rate
limit, so a student with GOOGLE_API_KEY_1 through GOOGLE_API_KEY_10 set
gets several times the effective quota of any single one. Numbered keys
(`{ENV_VAR}_1`, `{ENV_VAR}_2`, ...) take priority over the plain
singular variable if both are present; get_api_keys() returns all of
them, in order, and build_llm_for_backend() is what turns that list into
a single model that rotates between them (see llm_key_rotation.py).
build_llm() itself still only ever takes one key at a time -- it's the
low-level primitive both get_api_keys()-based construction and the tests
build on.
"""

import logging
import os
import warnings

# Both of these are benign, purely cosmetic messages from inside the
# Gemini libraries themselves, not anything about a student's own code
# or setup, confirmed by reading each library's source directly rather
# than guessed at: `google-genai`'s own logger warns once per process
# that its "automatic function calling" internals aren't the officially
# recommended call pattern (langchain_google_genai's own internal
# choice, not something students control), and `langchain_google_genai`
# warns that `gemini-3.6-flash` ignores the `temperature` this project
# always passes as 0, since that specific model line uses fixed
# sampling and can't be adjusted. Left visible, both look exactly like
# something broke; suppressed here, once, before either library is
# actually used, so every notebook that imports this module gets a
# clean run regardless of which cell first triggers a Gemini call.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*", category=UserWarning)

_ENV_VAR_BY_BACKEND = {"groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY"}
_MODEL_BY_BACKEND = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.6-flash",
}
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 60
_MAX_NUMBERED_KEYS = 10


def _is_placeholder(value: str) -> bool:
    """True for anything that looks like an unfilled .env.example value
    (e.g. "your-google-api-key", "your-second-google-api-key") rather
    than a real key. Pattern-based instead of an exact list, since the
    numbered slots (GOOGLE_API_KEY_2, _3, ...) each get their own
    placeholder text and a fixed list would need updating every time one
    gets added."""
    return value.lower().startswith("your-") and value.lower().endswith("-api-key")


def get_api_keys(backend: str, env_path: str = "config/.env") -> list[str]:
    """Returns every usable key configured for `backend`, in order. Looks
    for {ENV_VAR}_1 through {ENV_VAR}_10 first; if none of those are
    set, falls back to the plain {ENV_VAR}. Returns an empty list if
    nothing usable is configured -- callers decide whether that's an
    error (get_api_key does) or just a "not available yet" (has_api_key
    does)."""
    if backend not in _ENV_VAR_BY_BACKEND:
        raise ValueError(f"Unknown backend '{backend}'. Must be one of {list(_ENV_VAR_BY_BACKEND)}.")

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_path)
    env_var = _ENV_VAR_BY_BACKEND[backend]

    numbered = []
    for i in range(1, _MAX_NUMBERED_KEYS + 1):
        value = os.getenv(f"{env_var}_{i}")
        if value and not _is_placeholder(value):
            numbered.append(value)
    if numbered:
        return numbered

    single = os.getenv(env_var)
    if single and not _is_placeholder(single):
        return [single]
    return []


def get_api_key(backend: str, env_path: str = "config/.env") -> str:
    keys = get_api_keys(backend, env_path=env_path)
    if not keys:
        env_var = _ENV_VAR_BY_BACKEND[backend]
        raise RuntimeError(
            f"No usable key found for '{backend}'. Set {env_var} in {env_path} (or {env_var}_1, "
            f"{env_var}_2, ... if you have more than one key). Copy config/.env.example to {env_path} "
            f"first if you haven't."
        )
    return keys[0]


def has_api_key(backend: str, env_path: str = "config/.env") -> bool:
    return bool(get_api_keys(backend, env_path=env_path))


def build_llm_for_backend(
    backend: str, temperature: float = 0, env_path: str = "config/.env", timeout: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS
):
    """Builds whatever `backend` needs from its configured key(s), always
    wrapped in a RotatingKeyChatModel -- even for a single key. This is
    what notebooks should call instead of build_llm() directly, so a
    student with several Gemini keys gets the benefit of all of them
    without the notebook needing to know how many there are.

    Always wrapping, rather than returning a bare model for a single
    key, matters beyond just consistency: RotatingKeyChatModel is what
    actually reacts to a rate limit or server-unavailable error at all
    (print progress, a "please wait" handoff instead of a hard failure,
    the server-error retry budget) -- a bare model has none of that, it just fails
    immediately. A student with exactly one Groq key and multi-provider
    fallback off (a very common setup, since one Groq key is already
    enough -- see the README) would otherwise get zero retry benefit
    from any of this. It also means every chain entry FallbackChatModel
    (llm_fallback.py) interleaves across consistently exposes
    try_current()/has_keys(), with no bare-model special case to check
    for there."""
    keys = get_api_keys(backend, env_path=env_path)
    if not keys:
        get_api_key(backend, env_path=env_path)  # raises the standard, descriptive error
    models = [build_llm(backend, key, temperature=temperature, timeout=timeout) for key in keys]

    from .llm_key_rotation import RotatingKeyChatModel

    return RotatingKeyChatModel(models, name=backend)


def build_llm(backend: str, api_key: str, temperature: float = 0, timeout: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS):
    if backend == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=_MODEL_BY_BACKEND["groq"],
            groq_api_key=api_key,
            temperature=temperature,
            request_timeout=timeout,
            # Same fix as Gemini's max_retries=1 below, and the same
            # reason: ChatGroq defaults max_retries to 2, which was never
            # actually addressed here even though the Gemini side of this
            # exact problem already was. A single "attempt" as
            # RotatingKeyChatModel sees it could otherwise be this SDK
            # silently retrying 1-2 more times underneath, each up to
            # `timeout` seconds, before the failure ever reaches the
            # rotation/backoff logic that's supposed to be deciding that.
            # Retrying is already handled a layer up (llm_key_rotation.py,
            # llm_fallback.py), which reacts to the real rate-limit/
            # server-unavailable signal rather than blindly backing off.
            max_retries=1,
        )
    elif backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=_MODEL_BY_BACKEND["gemini"],
            google_api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            # langchain_google_genai defaults max_retries to 6, with fixed
            # exponential backoff between attempts (the library's own docs
            # flag this as a known issue -- it ignores a 429's suggested
            # retry_delay entirely). Combined with `timeout` per attempt,
            # a single flaky call can take several minutes to finally fail,
            # observed live: a tool-calling request hung far past the
            # configured 60s timeout before eventually erroring. max_retries=1
            # (not 0 -- the library treats 0 as "use the SDK's own default of
            # 5") makes exactly one attempt, so a stuck call fails within
            # `timeout` seconds instead. Retrying across rate limits is
            # already handled a layer up, by RotatingKeyChatModel and
            # FallbackChatModel (agent_pipeline/llm_key_rotation.py,
            # llm_fallback.py), which react to the real rate-limit signal
            # rather than blindly backing off.
            max_retries=1,
            # gemini-3.6-flash does internal "thinking" before answering,
            # variable-length and not something our own code controls or
            # sees. langchain_google_genai's own docs recommend a low
            # thinking_level specifically "for faster, lower-latency
            # responses" -- deciding on a tool call (this project's entire
            # workload) is exactly the kind of request that pushes thinking
            # time up. "minimal" is the lowest level Gemini 3+ models
            # support; this isn't a hard off switch (Gemini 3 has no
            # documented way to fully disable thinking the way Gemini 2.5's
            # thinking_budget=0 could), but it's the closest available and
            # meaningfully cut latency in testing.
            thinking_level="minimal",
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Must be one of {list(_ENV_VAR_BY_BACKEND)}.")

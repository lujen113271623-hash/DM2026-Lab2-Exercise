"""Shared LangGraph state-machine builder for every agent track in this
repo: the pipeline-runner in this package, and the agent-dev track in
agent_dev/. It enforces the same structural invariant for both:
at most one tool call per student turn, followed by a mandatory
discuss-only agent turn, then END. The graph never loops back into tool
selection within a single invocation. This is a structural guardrail
built into the graph topology, not just a system-prompt instruction, so
it holds even if a student tries to prompt the agent into chaining
multiple actions at once.

    agent --(no tool call)--> END
      |
      +-(tool call)--> tool_node --> discuss --> END

The discuss step never routes anywhere but END, so it can't restart tool
selection even if the model tries. That matters because "don't bind
tools to this call" turned out not to be enough on its own, on two
separate providers: Gemini has returned a populated tool_calls list from
a call made against a plain, never-bound model, apparently inferring it
from the tool-call/tool-result shape already sitting in the message
history rather than from what's registered on that specific request.
discuss_node clears any tool_calls it gets back before returning, so the
graph's guarantee doesn't depend on a provider behaving the way its API
is supposed to. Groq's gpt-oss-120b goes a step further: rather than
returning an unwanted tool call, the raw API call itself can fail with a
400 (observed: the model tried to invoke a tool named
"repo_browser.open_file" that was never bound to this call, and the
provider rejects the whole request instead of just ignoring the
attempt). discuss_node also wraps its call in a try/except for that
reason, and since the tool has already run by then, every failure there
(a rate limit and a server outage included) degrades to a fallback
message instead of aborting the turn -- see discuss_node. agent_node,
where nothing has run yet, still lets a rate limit or server outage
propagate to the chat widget's own message. Both conditions are checked with
rate_limits.is_rate_limit_error/is_server_unavailable_error rather than
an isinstance(exc, ModelRateLimitError) check, since that type turns out
not to be reliable either: Groq's integration never actually raises it,
only Gemini's does (see rate_limits.py).
"""

import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .grounding import check_grounding
from .message_text import as_text
from .backoff import MESSAGE_BUDGET
from .context_budget import BUDGET
from .rate_limits import is_rate_limit_error, is_server_unavailable_error, is_unbound_tool_call_error


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


# LAB 2 DEVIATION from Lab 1's agent_pipeline (see LAB1-SYNC-CHANGES.md).
# Upstream this is 25, i.e. a *floor* of 25 extra attempts even for a single
# key. Every attempt resends the whole prompt, which at Lab 2's prompt and tool
# sizes is about 2,700 tokens, so one empty reply could spend roughly 70,000
# tokens: several times over a free tier's whole 8,000-per-minute budget, for a
# retry that cannot help when there is no other key to rotate to. Measured live
# on 2026-10-04: this is what tripped the rate limit even at one message per
# minute. The retry's real purpose is to reach a *different* key, so the budget
# now follows the number of keys, with a small floor for the single-key case
# (where a second try does sometimes land) and a ceiling so a student with many
# keys cannot spend a whole minute's budget on one blank answer.
_MIN_EMPTY_RETRIES = 2
_MAX_EMPTY_RETRIES = 9
_PROGRESS_PRINT_EVERY = 5  # how often _invoke_retrying_empty's quiet retry loop shows a "still working" line, not every single attempt
_DISCUSS_PROVIDER_DOWN_TEXT = (
    "(The step above already ran and its result is saved, but the AI provider "
    "was busy or rate-limited while writing a comment on it. Wait a moment, "
    "then ask it to summarize what happened.)"
)


def _next_step_text(tool_call) -> str:
    """What discuss_node shows when the model answered only with a tool
    call it isn't allowed to make yet (one tool per turn): name the step
    it wants next so the student can approve it, rather than showing
    "no response"."""
    name = tool_call.get("name") or "another tool"
    args = tool_call.get("args") or {}
    arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    call = f"{name}({arg_text})" if arg_text else name
    return (
        f"(Done with that step. Next it wants to run {call}. "
        'Say "go ahead" to run it, or ask for something else.)'
    )


# LAB 2 DEVIATION from Lab 1's agent_pipeline (see LAB1-SYNC-CHANGES.md).
# The graph is stateless: every call resends the whole conversation, so a long
# session costs more per message than a short one, and past a point a single
# message exceeds a free tier's per-minute budget on its own, at which point
# waiting cannot help. Tool results are nearly all of that weight (a retrieval
# comparison is about 900 tokens, a results table about 650), and they keep
# being resent long after the agent has finished talking about them.
#
# So older tool results are replaced, on the way out only, by a one-line digest
# that keeps the result_id and the headline scalars. Nothing is dropped and the
# order never changes, so each assistant tool_call still sits directly beside
# its own tool result and providers stay happy. The full result is still in the
# notebook's own output and in session.results_store, so grounding still checks
# against real values and the student can ask to inspect any result_id again.
_KEEP_RECENT_TOOL_RESULTS = 3
_TOOL_RESULT_DIGEST_CHARS = 220
# Digesting old results slows the growth but does not stop it: the student's own
# messages, the agent's comments and the digests themselves still pile up, and
# measured over a twenty-message session that alone passes 8,000 tokens again.
# So the history also has a ceiling, chosen by measurement rather than by feel.
# Over a deliberately heavy twenty-five-message session (a retrieval comparison
# every fourth message), the worst single message costs:
#
#     no ceiling   23,267 tokens      6,000 chars    7,804 tokens
#     7,000 chars   8,613 tokens      5,000 chars    7,052 tokens
#
# 5,000 keeps roughly three recent turns in view and leaves about a thousand
# tokens of headroom under a free tier's 8,000 per minute, which matters because
# an empty reply is retried and the reply's own output tokens count too.
#
# That number is now the starting point rather than the rule. context_budget.py
# turns it into a token target and then corrects it from what the provider
# reports for each real request, which is the only way to see the system prompt
# and the fourteen tool schemas: they are sent on every agent call and appear
# nowhere in the message list a character count can measure.


def _digest_tool_result(text: str) -> str:
    """One short stand-in for an older tool result: enough for the model to
    remember that the step ran and which result_id to name, without paying
    for the whole table again on every later message."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        scalars = {k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))}
        head = dict(list(scalars.items())[:6])
        if "result_id" in data and "result_id" not in head:
            head = {"result_id": data["result_id"], **head}
        shortened = json.dumps(head)
    else:
        shortened = text[:_TOOL_RESULT_DIGEST_CHARS]
    return shortened + "  [earlier result, shortened; ask to inspect it by result_id for the full detail]"


def _message_chars(message) -> int:
    text = as_text(message.content)
    for call in getattr(message, "tool_calls", None) or []:
        text += call.get("name", "") + repr(call.get("args", {}))
    return len(text)


def _trim_history(messages):
    """What the model is actually shown, in two steps.

    First, every tool result except the most recent
    _KEEP_RECENT_TOOL_RESULTS is replaced by a digest. This keeps the list the
    same length and order, so each assistant tool_call still sits beside its
    own result and no provider sees a dangling call.

    Second, if the conversation is still longer than _MAX_HISTORY_CHARS, whole
    turns are dropped from the front. A turn starts at a student message, so
    cutting on those boundaries never separates a tool_call from its result. A
    leading system message is always kept. The agent then no longer sees the
    oldest exchanges, but nothing is lost: every result stays in
    session.results_store, so grounding still checks real values and
    inspect_result_tool still works on any result_id the student names."""
    positions = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    stale = set(positions[:-_KEEP_RECENT_TOOL_RESULTS] if _KEEP_RECENT_TOOL_RESULTS else positions)
    trimmed = []
    for i, message in enumerate(messages):
        text = as_text(message.content) if i in stale else None
        if text is not None and len(text) > _TOOL_RESULT_DIGEST_CHARS:
            message = ToolMessage(
                content=_digest_tool_result(text),
                tool_call_id=message.tool_call_id,
                name=message.name,
            )
        trimmed.append(message)

    ceiling = BUDGET.history_char_allowance()
    if sum(_message_chars(m) for m in trimmed) <= ceiling:
        return trimmed

    lead = trimmed[:1] if trimmed and isinstance(trimmed[0], SystemMessage) else []
    body = trimmed[len(lead):]
    starts = [i for i, m in enumerate(body) if isinstance(m, HumanMessage)]
    for start in starts:
        if sum(_message_chars(m) for m in body[start:]) <= ceiling:
            return lead + body[start:]
    # Even the newest turn on its own is over the ceiling (a very large tool
    # result): send just that turn rather than an empty conversation.
    return lead + body[starts[-1]:] if starts else trimmed


def _max_empty_retries_for(llm) -> int:
    """How many extra attempts an empty response gets before giving up.
    Scaled to how many keys are actually still active right now, across
    every configured backend (RotatingKeyChatModel/FallbackChatModel
    both expose key_list()) -- a student with one groq key and nine
    gemini keys gets up to nine extra tries, not a small fixed number
    that gives up long before every configured key has actually been
    tried once. The first attempt already used one active key, so the
    extra-retries budget is (active keys - 1); floored at
_MIN_EMPTY_RETRIES so that a single-key setup still gets a second and
    third try, and capped at _MAX_EMPTY_RETRIES so no setup can spend a
    whole rate-limit window on one blank answer. A plain model with no
    key_list() (single backend, single key, nothing to rotate to) gets
    the floor."""
    if not hasattr(llm, "key_list"):
        return _MIN_EMPTY_RETRIES
    active = sum(1 for k in llm.key_list() if k.get("state") == "active")
    return min(max(active - 1, _MIN_EMPTY_RETRIES), _MAX_EMPTY_RETRIES)


def _describe_current(llm) -> str:
    """Where a retry is about to land, for the progress print below --
    "gemini key 3" if the model exposes status() (FallbackChatModel,
    RotatingKeyChatModel), otherwise a generic fallback. Called AFTER
    skip_current() so it reports the key/backend the next attempt will
    actually use, not the one that just came back empty."""
    if not hasattr(llm, "status"):
        return "a different key/backend"
    s = llm.status()
    backend = s.get("backend")
    key_number = s.get("key_number")
    if backend and key_number:
        return f"{backend} key {key_number}"
    if key_number:
        return f"key {key_number}"
    return "a different key/backend"


def _invoke_once(llm, messages):
    """One llm.invoke(), with a narrow exception->None translation: only
    is_unbound_tool_call_error (Groq's gpt-oss-120b throwing a 400
    because the model called a tool on an unbound call -- rate_limits.py
    has the full story) gets caught here, since that's model noise on
    this one call worth retrying with a different key, the same
    reasoning an empty response already gets. Any other exception
    (a rate limit, a server error, anything unclassified) is left to
    propagate untouched -- this function never widens what already
    gets treated as retriable. Returns (response, tool_call_exc): a
    real response with tool_call_exc None, or (None, the exception)."""
    try:
        return llm.invoke(messages), None
    except Exception as exc:
        if is_unbound_tool_call_error(exc):
            return None, exc
        raise


def _invoke_retrying_empty(llm, messages, is_empty):
    """Calls llm.invoke(messages), and retries with a different key or
    backend each time (skip_current(), when the model exposes one --
    RotatingKeyChatModel, FallbackChatModel) for either of two
    conditions, until something usable comes back or every currently
    active key has had a turn (see _max_empty_retries_for):
    - the result is empty by is_empty's definition (no tool call, no
      usable text -- the call succeeded but came back with nothing in
      it), or
    - the call raised is_unbound_tool_call_error (see _invoke_once and
      rate_limits.py) -- model noise on that one call, not a sign
      anything is actually wrong with the request.
    A rate limit or a server error is never retried here; those already
    raise straight through _invoke_once and are handled by the caller.
    Neither retriable condition is held against the key or backend that
    produced it: no removal, no requeue -- neither one says anything
    about whether that key itself is fine. A generic "Working on your
    request... (n/max)" line is printed for each retry, deliberately
    without the technical reason or which key/backend it's trying next
    -- a routine, recovering condition shouldn't read as alarming on
    every one of up to 25 attempts. Returns the last response if
    the budget runs out on an empty result; re-raises the last
    exception if it runs out on a tool-call error, so the caller's own
    exception handling (agent_node/discuss_node) still applies to it."""
    MESSAGE_BUDGET.reset()
    history_chars = sum(_message_chars(m) for m in messages)
    response, tool_call_exc = _invoke_once(llm, messages)
    attempts = 0
    max_attempts = _max_empty_retries_for(llm)
    while (tool_call_exc is not None or is_empty(response)) and attempts < max_attempts:
        attempts += 1
        if hasattr(llm, "skip_current"):
            llm.skip_current()
        # Mostly silent, not a line per attempt: a technical reason
        # ("empty response", "model called an unbound tool") on every
        # one of up to 25 attempts reads as alarming for a routine,
        # already-recovering condition, and even the generic wording
        # got noisy at full frequency. Printed on the first attempt and
        # every _PROGRESS_PRINT_EVERY'th one after, so a long stretch
        # still shows occasional signs of life instead of going
        # completely silent and looking frozen. Rate-limit removals and
        # both total-exhaustion notices are unaffected by this
        # throttle -- those print every time, in llm_key_rotation.py /
        # llm_fallback.py, since they're rarer and actionable.
        if attempts == 1 or attempts % _PROGRESS_PRINT_EVERY == 0:
            print("Processing your request...")
        # LAB 2 DEVIATION: a short, jittered wait before trying again. With
        # several keys skip_current() has already moved to a different one and
        # this costs almost nothing; with a single key it is the difference
        # between asking the same key nine times in a second and giving it a
        # moment. The budget is shared with the layers below, so the total
        # stall for one message stays bounded either way.
        MESSAGE_BUDGET.wait(attempts)
        response, tool_call_exc = _invoke_once(llm, messages)
    if tool_call_exc is not None:
        raise tool_call_exc
    # LAB 2 DEVIATION: what the provider says this request actually cost, which
    # is what context_budget.py trims the next one against.
    BUDGET.observe(history_chars, getattr(response, "usage_metadata", None))
    return response


def build_agent_graph(tools, system_prompt, llm, session):
    """Builds and compiles the shared one-tool-call-per-turn graph for a
    given tool list, system prompt, LLM, and session (the session only
    needs a `results_store` dict-like attribute, used for grounding)."""
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def _gate_state_line():
        """One line of live state appended to the system prompt.

        Whether the student has already approved the gated tools is something
        the model would otherwise have to remember from a turn that
        _trim_history may well have dropped. Seen live: with the window on, the
        agent asked for access again many turns after it had been granted.
        The approval lives in session.granted_tools, so it is cheaper and more
        reliable to restate it every call than to hope the turn survived.
        Costs about fifteen tokens."""
        granted = sorted(getattr(session, "granted_tools", ()) or ())
        if not granted:
            return "\n\nCurrent state: the student has NOT yet approved the gated tools."
        return (
            "\n\nCurrent state: the student has already approved the gated tools "
            f"({', '.join(granted)}); they all run directly now, so never call "
            "request_tool_access_tool or grant_tool_access_tool again this session."
        )

    def _with_system_prompt(messages):
        # _trim_history shortens older tool results on the way to the model;
        # the graph's own state keeps the full messages (see _trim_history).
        messages = _trim_history(messages)
        if messages and isinstance(messages[0], SystemMessage):
            return messages
        return [SystemMessage(content=system_prompt + _gate_state_line())] + list(messages)

    def agent_node(state: GraphState) -> dict:
        try:
            response = _invoke_retrying_empty(
                llm_with_tools,
                _with_system_prompt(state["messages"]),
                lambda r: not r.tool_calls and not as_text(r.content).strip(),
            )
        except Exception as exc:
            if is_rate_limit_error(exc) or is_server_unavailable_error(exc):
                raise
            print(f"Call failed with {type(exc).__name__}: {exc}")
            response = AIMessage(content="")
        if not response.tool_calls:
            if not as_text(response.content).strip():
                # Observed live on more than one backend (first noticed on
                # Groq's gpt-oss-120b, but not exclusive to it): a call can
                # finish with finish_reason "stop", no tool call, and
                # nothing in the visible content -- the model's actual
                # answer ends up entirely inside its hidden reasoning
                # instead of ever reaching the final channel. Same root
                # cause discuss_node already guards against; without this,
                # the student sees a silent blank bubble with no indication
                # anything went wrong, and no way to tell it apart from the
                # agent simply having nothing to say.
                response.content = (
                    "(No response came back for that. Try asking again.)"
                )
            # Direct reply, no tool this turn (e.g. a discussion-only
            # follow-up) -- still ground-check it, since it can reference
            # numbers from earlier stored results just as easily.
            ungrounded = check_grounding(response.content, session.results_store)
            response.additional_kwargs["ungrounded_numbers"] = ungrounded
            # Only FallbackChatModel/RotatingKeyChatModel expose status()
            # (see llm_fallback.py, llm_key_rotation.py) -- a plain
            # single-backend, single-key model doesn't, so there's
            # nothing to attach for the common case, correctly.
            if hasattr(llm_with_tools, "status"):
                response.additional_kwargs["backend_status"] = llm_with_tools.status()
        return {"messages": [response]}

    def route_after_agent(state: GraphState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tool_node"
        return END

    def tool_node(state: GraphState) -> dict:
        last = state["messages"][-1]
        executed, *dropped = last.tool_calls

        tool = tools_by_name.get(executed["name"])
        if tool is None:
            # The model named a tool that isn't registered (a hallucinated
            # or misspelled name, e.g. "variance_filter" for
            # "variance_filter_tool"). A plain dict lookup here used to
            # raise KeyError, aborting the whole turn with a raw traceback.
            output = {
                "error": f"No tool named '{executed['name']}' exists.",
                "hint": f"Use one of the exact tool names available: {sorted(tools_by_name)}",
            }
        else:
            try:
                output = tool.invoke(executed["args"])
            except Exception as exc:  # a tool blowing up must not crash the whole
                # session -- surface it as a normal tool error instead.
                output = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "hint": "This tool call failed. Try different parameters rather "
                    "than repeating the same call.",
                }
        new_messages = [ToolMessage(content=str(output), tool_call_id=executed["id"], name=executed["name"])]

        for extra in dropped:
            note = (
                "Not executed: only one tool call is allowed per student turn. "
                "Ask for this step explicitly in your next message."
            )
            new_messages.append(ToolMessage(content=note, tool_call_id=extra["id"], name=extra["name"]))

        return {"messages": new_messages}

    def discuss_node(state: GraphState) -> dict:
        # `llm`, not `llm_with_tools` -- tools aren't bound here, so this
        # call has nothing to call a tool with. That's not sufficient by
        # itself (see the module docstring): any tool_calls that come back
        # anyway are cleared below rather than trusted, and the call
        # itself can fail outright on some providers (Groq's gpt-oss-120b
        # has thrown a 400 trying to invoke a tool that was never bound),
        # so it's wrapped here too. Unlike agent_node, a rate limit or
        # server outage is NOT re-raised here: by the time discuss runs,
        # the tool has already executed (a file written, access granted,
        # a result stored). Raising would abort the whole graph run, so
        # chat_widget.py would never save this turn to graph_state or the
        # session log, leaving the agent unaware of an action that really
        # happened and the graded log missing it. Instead the turn
        # completes with a plain note that the comment couldn't be
        # written; the key-rotation/exhaustion notices from
        # llm_key_rotation.py / llm_fallback.py still print as usual.
        try:
            response = _invoke_retrying_empty(
                llm,
                _with_system_prompt(state["messages"]),
                # A tool call counts as "not empty" here, so it is NOT
                # retried. Observed live: asked to use two spec files, the
                # model read one, then on this unbound call asked for a
                # tool to read the second. Same history at temperature 0
                # means every retry makes that same choice, so retrying it
                # burned ~25 calls (and rate-limited two keys) for nothing.
                # It's turned into a "next it wants to run X" note below
                # instead. Only a genuinely empty reply (no text, no tool
                # call) is still retried.
                lambda r: not r.tool_calls and not as_text(r.content).strip(),
            )
        except Exception as exc:
            if is_rate_limit_error(exc) or is_server_unavailable_error(exc):
                response = AIMessage(content=_DISCUSS_PROVIDER_DOWN_TEXT)
            else:
                print(f"Call failed with {type(exc).__name__}: {exc}")
                response = AIMessage(content="")
        wanted_next = response.tool_calls[0] if response.tool_calls else None
        if response.tool_calls:
            # Never executed: tools aren't bound here and the graph ends
            # after this node. Cleared from both places a provider may
            # keep them -- langchain_groq resends
            # additional_kwargs["tool_calls"] when .tool_calls is empty,
            # which would put an orphan tool call in the next request.
            response.tool_calls = []
            response.additional_kwargs.pop("function_call", None)
            response.additional_kwargs.pop("tool_calls", None)
        if not as_text(response.content).strip():
            if wanted_next is not None:
                response.content = _next_step_text(wanted_next)
            else:
                # Genuinely empty even after retries -- say that plainly
                # instead of leaving a blank turn.
                response.content = (
                    "(No response came back for that. Ask it to summarize what "
                    "happened, or go ahead with your next question.)"
                )
        ungrounded = check_grounding(response.content, session.results_store)
        response.additional_kwargs["ungrounded_numbers"] = ungrounded
        if hasattr(llm, "status"):
            response.additional_kwargs["backend_status"] = llm.status()
        return {"messages": [response]}

    graph = StateGraph(GraphState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("discuss", discuss_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tool_node": "tool_node", END: END})
    graph.add_edge("tool_node", "discuss")
    graph.add_edge("discuss", END)

    return graph.compile()

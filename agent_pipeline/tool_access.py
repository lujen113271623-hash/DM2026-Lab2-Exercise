"""Gates the student's own workspace tools behind explicit permission,
instead of handing every tool discovered in agent_dev_workspace/ straight
to the analysis agent the moment the notebook starts.

The agent can always see what's there and explain what a tool does --
that's the whole point of it being able to search and describe the
workspace -- but it can't actually run any of the student's own tools
until the student has agreed to it. Granting is a single, one-time
gate that covers the whole batch of the student's own tools at once,
not a separate request/approve round trip per tool: with up to eighteen
of them, gating each one individually turned this into most of a
session's turns before any real analysis could start. One
request_tool_access_tool proposal, one student approval,
one grant_tool_access_tool call -- after that, every one of the
student's own tools runs directly for the rest of the session, the
same as a provided tool already does. It's still the same two-turn
propose/approve shape as the write gate in the developer stage, and
the same structural rule still holds: the two calls can never land in
the same turn, since graph_core.py allows at most one tool call per
student turn.

Provided tools (premade_tools/ at the repo root) don't go through this
gate at all -- they're already-vetted infrastructure, not something the
student wrote and might still be buggy, so there's nothing to approve.
They're still visible to list_available_tools_tool and
describe_tool_tool alongside the student's own, so the agent can talk
about the whole pipeline, but only the student's tools ever go behind
the permission check, and that check now covers all of them together.

build_gated_tools() is the entry point: it takes the student's own
discovered tools and the already-usable provided tools, and returns the
full tool list that gets bound to the agent.
"""

from langchain_core.tools import StructuredTool, tool


def _gate_tool(real_tool, session):
    """Wraps one already-built tool so calling it before the student's
    own tools have been granted (as a batch) returns a plain, structured
    refusal instead of running the real function. Not an exception -- a
    denial here is an expected, routine outcome, not a bug in the tool,
    so it goes through the same return-a-dict path a tool uses to
    report any other outcome."""

    def gated_func(**kwargs):
        if real_tool.name not in session.granted_tools:
            return {
                "error": "not_granted",
                "hint": (
                    f"'{real_tool.name}' hasn't been granted access yet. Use "
                    "request_tool_access_tool to propose access to all of the "
                    "student's own tools at once, then wait for the student's "
                    "explicit approval before calling grant_tool_access_tool."
                ),
            }
        return real_tool.func(**kwargs)

    return StructuredTool.from_function(
        func=gated_func,
        name=real_tool.name,
        description=real_tool.description,
        args_schema=real_tool.args_schema,
    )


def _make_discovery_tools(session, all_by_name, premade_names, student_names):
    def _requires_approval(name):
        return name not in premade_names

    @tool
    def list_available_tools_tool() -> dict:
        """Lists every tool discovered, both the student's own and the
        already-provided ones: its name, description, whether it needs
        the student's approval before it can run, and whether it's
        currently granted (always true for provided tools, and true for
        every one of the student's own tools once the batch has been
        granted). Discovering a tool this way never grants it."""
        return {
            "tools": [
                {
                    "name": name,
                    "description": t.description,
                    "requires_approval": _requires_approval(name),
                    "granted": (not _requires_approval(name)) or name in session.granted_tools,
                }
                for name, t in sorted(all_by_name.items())
            ]
        }

    @tool
    def describe_tool_tool(tool_name: str) -> dict:
        """Returns the full description and parameters for one discovered
        tool by name, without granting access to it. Use this to explain
        what a tool actually does before proposing to use it."""
        real_tool = all_by_name.get(tool_name)
        if real_tool is None:
            return {"error": f"No tool named '{tool_name}' was found."}
        return {
            "name": real_tool.name,
            "description": real_tool.description,
            "parameters": real_tool.args,
            "requires_approval": _requires_approval(tool_name),
            "granted": (not _requires_approval(tool_name)) or tool_name in session.granted_tools,
        }

    @tool
    def request_tool_access_tool(reason: str) -> dict:
        """Proposes granting access to every one of the student's own
        tools at once (not one at a time), explaining why access is
        needed for what the student asked to do. This does not grant
        access by itself -- the student must approve in their next
        message before grant_tool_access_tool can be called. A no-op,
        reporting nothing left to grant, if every student tool is
        already granted."""
        ungranted = [name for name in student_names if name not in session.granted_tools]
        if not ungranted:
            return {"status": "already_granted", "tool_names": sorted(student_names)}
        session.pending_tool_request = {"reason": reason}
        return {"status": "pending_approval", "tool_names": sorted(ungranted), "reason": reason}

    @tool
    def grant_tool_access_tool() -> dict:
        """Grants access to every one of the student's own tools at
        once. Only call this after the student has explicitly approved
        the pending request in their own message -- never in the same
        turn as request_tool_access_tool, and never on an assumption
        that silence means yes."""
        pending = session.pending_tool_request
        if pending is None:
            return {"error": "No pending tool request to grant. Call request_tool_access_tool first."}
        session.granted_tools.update(student_names)
        session.pending_tool_request = None
        return {"status": "granted", "tool_names": sorted(student_names)}

    @tool
    def discard_tool_request_tool() -> dict:
        """Discards the pending batch tool request without granting it.
        Use this when the student declines, or asks for something else
        instead."""
        pending = session.pending_tool_request
        if pending is None:
            return {"error": "No pending tool request to discard."}
        session.pending_tool_request = None
        return {"status": "discarded"}

    return [
        list_available_tools_tool,
        describe_tool_tool,
        request_tool_access_tool,
        grant_tool_access_tool,
        discard_tool_request_tool,
    ]


def build_gated_tools(student_tools, session, premade_tools=()):
    """Takes the student's own discovered tools (from agent_dev_workspace/)
    and the already-usable provided tools (from premade_tools/, optional),
    and returns the full tool list that gets bound to the agent: the
    student's tools wrapped behind session.granted_tools (granted as one
    batch, not individually), the provided tools unwrapped since they
    need no permission, and search/describe/request/grant/discard tools
    that see all of them and let the agent negotiate access to the ones
    that need it."""
    premade_tools = list(premade_tools)
    premade_names = {t.name for t in premade_tools}
    # A student tool sharing a provided tool's name (e.g. their own
    # load_dataset_tool) would bind that name to the model twice, which
    # providers reject, failing every message. The provided, already-vetted
    # one wins; say so rather than dropping it silently.
    kept_student = []
    for t in student_tools:
        if t.name in premade_names:
            print(
                f"Skipped your tool '{t.name}': a provided tool already has that "
                "name, so the provided one is used instead."
            )
            continue
        kept_student.append(t)
    student_tools = kept_student
    all_by_name = {t.name: t for t in list(student_tools) + premade_tools}
    student_names = {t.name for t in student_tools}
    gated_student = [_gate_tool(t, session) for t in student_tools]
    return gated_student + premade_tools + _make_discovery_tools(session, all_by_name, premade_names, student_names)

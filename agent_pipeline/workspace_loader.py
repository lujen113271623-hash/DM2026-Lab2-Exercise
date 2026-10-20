"""Loads student-authored tool modules from an agent-dev workspace and
merges them into one list of LangChain tools, so a real analysis agent
can run on top of the student's own code instead of the hidden reference
implementation. This is what connects agent_dev/ (where the
tools get written) to DM2026-Lab1-AgenticPipeline.ipynb (where they
get run).

Each .py file in the workspace is expected to define a module-level
make_tools(session) function, matching the contract documented in
agent_dev/TOOLS_PIPELINE_BREAKDOWN.md. A file that doesn't define one,
or that fails to import or run, is skipped and reported rather than
treated as fatal, so a partially finished workspace still produces
whatever tools are ready.
"""

import importlib.util
import os


def load_workspace_tools(workspace_dir, session):
    """Returns (tools, problems).

    `tools` is the combined list of LangChain tool objects collected from
    every workspace file's make_tools(session). Each entry in `problems`
    is a (filename, message) pair for a file that couldn't contribute any
    tools, whether because it doesn't define make_tools, failed to
    import, or raised while building its tools.
    """
    tools = []
    problems = []
    # Tool name -> the file that first defined it. A second tool with the
    # same name (a backup copy like tools_data_old.py, or the same tool
    # pasted into two files) would otherwise be bound to the model twice,
    # which providers reject outright, failing every message in the
    # session. Files load in sorted order, so tools_data.py wins over
    # tools_data_old.py.
    seen = {}

    if not os.path.isdir(workspace_dir):
        return tools, [(workspace_dir, "workspace directory not found")]

    for fname in sorted(os.listdir(workspace_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        path = os.path.join(workspace_dir, fname)
        module_name = f"_student_workspace_{fname[:-3]}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:
            problems.append((fname, f"failed to import: {type(exc).__name__}: {exc}"))
            continue

        make_tools_fn = getattr(module, "make_tools", None)
        if make_tools_fn is None:
            problems.append((fname, "no make_tools(session) function found"))
            continue

        try:
            file_tools = make_tools_fn(session)
        except Exception as exc:
            problems.append((fname, f"make_tools() raised {type(exc).__name__}: {exc}"))
            continue

        for t in file_tools:
            name = getattr(t, "name", None)
            if name in seen:
                problems.append((fname, f"tool '{name}' skipped: same name already defined in {seen[name]}"))
                continue
            seen[name] = fname
            tools.append(t)

    return tools, problems

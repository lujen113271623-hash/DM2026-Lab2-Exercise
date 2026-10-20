"""Notebook-embedded chat widget wiring a LangGraph app, session state, and
JSONL logger together. It renders inline in the same Jupyter or Colab tab
the rest of the lab already runs in, with no separate server or URL.

It's generic by design: it takes an already-built graph `app` rather than
building one itself, so it can be reused by any agent variant (the
pipeline-runner in this package, and the separate agent-dev track) without
duplicating the widget code.
"""

import contextlib
import os
import sys
import traceback

import ipywidgets as widgets
import pandas as pd
from IPython.display import Javascript, Markdown, clear_output, display
from langchain_core.messages import AIMessage, HumanMessage

from .context_budget import BUDGET as CONTEXT_BUDGET
from .message_text import as_text
from .rate_limits import is_rate_limit_error, is_server_unavailable_error

# Line-level noise from deep inside the LLM SDKs that isn't ours to fix and
# isn't useful to a student -- observed live: "Dropping reasoning block
# from provider 'groq'; foreign reasoning is not replayed as a Gemini
# thought part." repeats once per historical turn that used Groq's
# reasoning-capable model, every single retry, once multi-provider
# fallback has switched off Groq -- 80+ near-identical lines burying the
# one progress line (a key rotation, a server error) a student actually
# needs to read. Couldn't be traced to an exact source line in either
# langchain_google_genai or the google-genai SDK (checked both, this
# exact wording isn't present verbatim in either as installed), so this
# filters by matched text at the point chat_widget.py already captures
# output, rather than patching a call site that couldn't be found.
_SUPPRESSED_OUTPUT_SUBSTRINGS = (
    "Dropping reasoning block from provider",
)


class _FilteredStream:
    """Wraps a real stream (Output's own captured stdout/stderr once
    `with output:` is already open) and drops only whole lines matching
    _SUPPRESSED_OUTPUT_SUBSTRINGS, forwarding everything else untouched
    and live, same as before this existed."""

    def __init__(self, target):
        self._target = target
        self._buffer = ""

    def write(self, s):
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not any(sub in line for sub in _SUPPRESSED_OUTPUT_SUBSTRINGS):
                self._target.write(line + "\n")

    def flush(self):
        if self._buffer and not any(sub in self._buffer for sub in _SUPPRESSED_OUTPUT_SUBSTRINGS):
            self._target.write(self._buffer)
        self._buffer = ""
        self._target.flush()
from .session_logger import SessionLogger

# ipywidgets' Output widget never auto-scrolls to the bottom when new
# content is appended, even with overflow_y: auto set -- the box is
# genuinely scrollable, it just stays pinned wherever it was, which
# looks exactly like "the chat is stuck" to anyone who doesn't realize
# they need to drag the scrollbar down themselves. There's no built-in
# ipywidgets option for this; this JS is the standard workaround, scoped
# to only this widget via a unique class name so it never touches any
# other Output widget elsewhere in the notebook.
_SCROLL_TO_BOTTOM_JS = """
(function() {
    var boxes = document.getElementsByClassName('chat-output-box');
    for (var i = 0; i < boxes.length; i++) {
        boxes[i].scrollTop = boxes[i].scrollHeight;
    }
})();
"""


def _safe_stem(name: str) -> str:
    """A student name turned into something safe to put in a filename."""
    if not name:
        return "unknown"
    stem = "".join(c if c.isalnum() else "_" for c in name).strip("_")
    return stem or "unknown"


def _display_tabular_fields(summary):
    """Render any list-of-dict fields in a tool's summary (e.g. the `rows`
    from inspect_data_tool, `top_10_by_variance` from variance_filter_tool)
    as an actual table -- pulled directly from the tool's structured
    result, not the agent's prose, so "show me the rows" always shows the
    real rows regardless of how the agent chooses to describe them."""
    for key, value in summary.items():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            display(Markdown(f"*{key}:*"))
            display(pd.DataFrame(value))


def _display_diff_fields(summary):
    """Render any string field named 'diff' or 'content_preview' in a
    tool's summary as a fenced code block -- same principle as
    _display_tabular_fields: a proposed code change is shown verbatim from
    the tool's actual result, not dependent on the agent's prose."""
    for key in ("diff", "content_preview"):
        value = summary.get(key)
        if isinstance(value, str) and value.strip():
            display(Markdown(f"*{key}:*\n```diff\n{value}\n```"))


def _status_changed(old: dict, new: dict) -> bool:
    """True only when something actually happened since the last turn we
    saw a status for -- not on every turn, which would mean printing the
    same "still on key 1 of 10" notice after every single message.
    Missing counts default to 0, so the very first turn (old is still
    {}) stays quiet unless a switch really happened on it, and a
    standalone RotatingKeyChatModel's status (no backend_switches at
    all, fallback off) compares cleanly too."""
    if not new:
        return False
    return (
        old.get("backend_switches", 0) != new.get("backend_switches", 0)
        or old.get("key_switches", 0) != new.get("key_switches", 0)
    )


def _format_status_notice(status: dict) -> str:
    """One line naming the currently active backend and key (if that
    backend rotates more than one), plus a breakdown of what's happened
    to its keys so far: how many were removed for good (rate limit --
    that key's own quota is used up, not coming back this session) versus
    how many times a key was sent to the back of the line instead of
    removed (a server error -- the provider's infrastructure being busy,
    not that key's quota, so it's still in the rotation and may well
    succeed on a later turn; see RotatingKeyChatModel for why these two
    are handled differently). key_number can be missing entirely if
    every key has been removed. "So far this session," not "today": this
    count is only ever reset by restarting the kernel, not by whatever
    time each provider's own daily quota actually resets at, so it can't
    honestly claim to track the calendar day.

    With multi-provider fallback off, the llm is a standalone
    RotatingKeyChatModel whose status() has no backend or
    backend_switches at all (there's only one backend), so that case
    gets its own key-only line instead of a KeyError on the first
    reply."""
    if "backend" not in status:
        if status.get("key_number") is not None:
            where = f"now on key {status['key_number']} of {status['total_keys']}"
        elif status.get("set_aside_key") is not None:
            where = f"key {status['set_aside_key']} set aside after a rate limit, tried again on your next message"
        else:
            where = "no active keys left"
        return (
            f"*(key rotation: {where} -- {status.get('keys_removed', 0)} of "
            f"{status.get('total_keys', 0)} keys rate-limited out for good, "
            f"{status.get('keys_requeued', 0)} more sent to the back of the line "
            "for a server error (still active), so far this session)*"
        )
    where = f"now on **{status['backend']}**"
    if status.get("key_number") is not None:
        where += f", key {status['key_number']} of {status['total_keys']}"
    parts = [f"backend switched {status['backend_switches']} time(s)"]
    if "keys_removed" in status:
        parts.append(
            f"{status['keys_removed']} of {status['total_keys']} {status['backend']} keys "
            f"rate-limited out for good, {status['keys_requeued']} more sent to the back of "
            "the line for a server error (still active), so far this session"
        )
    return f"*(multi-provider fallback: {where} -- {'; '.join(parts)})*"


def _format_key_list(llm) -> str:
    """Full per-key listing for the /status command: which keys are
    still active, which have been rate-limited out for good, how many
    server errors each has had, and which one's in use right now.
    Plain-text markdown, not `_format_status_notice`'s one-line
    summary -- this is the full picture on request, not a change
    notice. A setup with nothing to rotate (single backend, single
    key, no llm.key_list() at all) gets a short note instead of an
    empty listing."""
    if not hasattr(llm, "key_list"):
        return "*(no key rotation configured for this setup; a single key needs nothing to rotate)*"
    rows = llm.key_list()
    if not rows:
        return "*(no key rotation configured for this setup; a single key needs nothing to rotate)*"
    lines = ["**Key status, so far this session:**"]
    current_group = object()  # sentinel, never equals a real backend name
    for row in rows:
        group = row.get("backend", None)
        if group != current_group:
            current_group = group
            if group is not None:
                lines.append(f"\n*{group}:*")
        state = {
            "rate_limited": "rate-limited (removed for the rest of this session)",
            "set_aside": "rate-limited (last key left, set aside: tried again when you send another message)",
        }.get(row["state"], "active")
        extra = f", {row['server_error_count']} server error(s) so far" if row["server_error_count"] else ""
        marker = " ← in use now" if row["current"] else ""
        lines.append(f"- key {row['key_number']}: {state}{extra}{marker}")
    return "\n".join(lines)


def launch_chat(
    app,
    session,
    log_path,
    student_name,
    student_id,
    track=None,
    plots_dir=None,
    placeholder="Ask the agent...",
    llm=None,
):
    # Re-running this cell (a common thing to do -- after an error, or to
    # start over) would otherwise display() a brand-new widget appended
    # below whatever the cell already showed, since Jupyter doesn't clear
    # a cell's prior output on its own. That leaves an old, now-orphaned
    # widget instance (its own input box and Send button, bound to a
    # graph_state nothing still references) stacked above the fresh one,
    # which looks exactly like the input row sitting above the chat box
    # rather than below it. Clearing first means re-running this cell
    # always starts from a clean slate.
    clear_output(wait=True)
    logger = SessionLogger(log_path, student_name, student_id, track=track)
    graph_state = {"messages": []}
    last_status = {}  # backend_status from the most recent turn that had one; see _status_changed
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)

    # flex="0 0 auto" on both this box and the input row below is
    # deliberate, not decoration: a fixed-height Output next to a sibling
    # with no layout of its own is exactly the shape that lets a flexbox
    # engine compute the sibling's height as 0 and collapse it into this
    # box's own scrolling area instead of giving it its own row -- seen
    # live, reproducibly, in a real student's browser (the Send button
    # rendered on top of the conversation text, mid-scroll, instead of
    # staying below it). Explicit flex/width on both sides is the
    # standard fix for that failure mode.
    output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #ccc", height="420px", overflow_y="auto", flex="0 0 auto", width="100%"
        )
    )
    output.add_class("chat-output-box")
    text_input = widgets.Text(placeholder=placeholder, layout={"width": "80%"})
    send_button = widgets.Button(description="Send", button_style="primary")

    def on_send(_=None):
        prompt = text_input.value.strip()
        if not prompt:
            return
        text_input.value = ""
        with output:
            display(Markdown(f"**You:** {prompt}"))
            display(Javascript(_SCROLL_TO_BOTTOM_JS))

        if prompt.lower() in ("/status", "/keys"):
            # Handled entirely here, locally -- never touches graph_state,
            # never calls app.invoke(), never reaches the model, and isn't
            # logged to the session log. This is a harness-level question
            # about key rotation state, not a real turn of the
            # conversation the student is directing the agent through, so
            # it shouldn't count as one, cost an API call, or show up in
            # what's graded.
            with output:
                display(Markdown(_format_key_list(llm)))
                # LAB 2 DEVIATION: the other half of why a message fails. Keys
                # explain "which credential", the context budget explains "how
                # big the request got", and a student hitting a per-minute
                # limit cannot tell those apart without seeing both.
                display(Markdown("*" + CONTEXT_BUDGET.status_line() + "*"))
                display(Javascript(_SCROLL_TO_BOTTOM_JS))
            return

        text_input.disabled = True
        send_button.disabled = True
        send_button.description = "Thinking..."
        try:
            before = len(graph_state["messages"])
            before_result_ids = set(session.results_store.keys())
            # Build the pending turn as a local dict rather than mutating
            # graph_state directly. graph_state is shared, long-lived state;
            # if app.invoke() below raises (a Gemini 504, a rate limit,
            # anything), graph_state must come out of this turn exactly as
            # it went in. Assigning the human message onto graph_state
            # before invoke() used to leave an orphaned, never-answered
            # human turn permanently in history on any failure -- observed
            # live: a failed turn's question got silently answered on the
            # *next* successful turn instead of the new message, because
            # the model saw two unanswered human messages in a row and
            # responded to the substantive one.
            pending_state = {"messages": graph_state["messages"] + [HumanMessage(content=prompt)]}
            # Wrapped in `with output` specifically so that if
            # RotatingKeyChatModel is backing off on repeated server
            # errors (llm_key_rotation.py), its plain print() progress
            # notices land live in the chat box during the wait, rather
            # than in the notebook cell's own output area below it --
            # ipywidgets' Output redirects stdout for the whole duration
            # a `with` block is open, not just for display() calls made
            # directly inside it.
            #
            # app.invoke() itself is called inside a nested try/except,
            # not left to raise directly out of the `with output` block.
            # Confirmed live in ipywidgets 7.8.1: Output.__exit__ ends
            # with `return True if ip else None` -- it displays an
            # exception passing through it, then *swallows* it, rather
            # than letting it propagate, specifically because it's
            # designed for interactive IPython use where nothing else
            # would otherwise show a raised exception. That's exactly
            # wrong here: it silently ate the real error (a Gemini 503,
            # a rate limit) before the except block below ever saw it to
            # classify and show the friendly message for, and execution
            # fell through to the next line with result_state still
            # unset, raising an unrelated UnboundLocalError that
            # replaced the real error entirely. Catching it here first
            # and re-raising only after `with output` has cleanly exited
            # (no exception in flight at that point, so nothing left for
            # Output.__exit__ to intercept) gets both: live progress
            # prints during the call, and the real exception reaching
            # the classification below.
            caught_exc = None
            with output:
                try:
                    with contextlib.redirect_stdout(_FilteredStream(sys.stdout)), \
                         contextlib.redirect_stderr(_FilteredStream(sys.stderr)):
                        result_state = app.invoke(pending_state)
                except Exception as exc:
                    caught_exc = exc
            if caught_exc is not None:
                raise caught_exc
            graph_state["messages"] = result_state["messages"]
            new_messages = graph_state["messages"][before + 1 :]  # skip the human message itself
            new_result_ids = set(session.results_store.keys()) - before_result_ids

            ungrounded = []
            for m in new_messages:
                if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                    ungrounded = m.additional_kwargs.get("ungrounded_numbers", [])
                    with output:
                        display(Markdown(f"**Agent:** {as_text(m.content)}"))
                        if ungrounded:
                            display(Markdown(f"*(unverified numbers flagged: {ungrounded})*"))
                    # Only present at all with multi-provider fallback
                    # enabled (see graph_core.py); absent otherwise, so
                    # this stays quiet for the common single-backend case.
                    new_status = m.additional_kwargs.get("backend_status")
                    if new_status and _status_changed(last_status, new_status):
                        with output:
                            display(Markdown(_format_status_notice(new_status)))
                        last_status.clear()
                        last_status.update(new_status)

            for result_id in new_result_ids:
                summary = session.results_store[result_id]["record"].summary
                with output:
                    _display_tabular_fields(summary)
                    _display_diff_fields(summary)

            logger.log_turn(prompt, new_messages, ungrounded_numbers=ungrounded)

            if plots_dir and getattr(session, "pending_figure", None) is not None:
                # Named after the result_id the same turn's tool call
                # produced (e.g. plot_reduce_1.png), so it's traceable
                # back to the matching tool_result_summary in the log by
                # name alone. Falls back to a turn count if a figure ever
                # shows up with no new result this turn, which shouldn't
                # happen given how these tools are built, but a plot
                # silently not saving would be worse than an odd filename.
                plot_id = next(iter(new_result_ids), None) or f"turn{len(graph_state['messages'])}"
                # Student-name-prefixed, same reasoning session_logger.py
                # already applies to the encrypted log's own filename: if
                # a batch of plots/ folders ever gets flattened into one
                # place (an LMS bulk download, a manually zipped
                # submissions folder), two different students' plots for
                # the same tool (e.g. both producing a "reduce_1") would
                # otherwise silently overwrite each other. Track-prefixed
                # too, since agentic-pipeline and homework both save into
                # the same shared plots/ folder within one student's own
                # submission.
                filename = f"{_safe_stem(student_name)}_{track}_plot_{plot_id}.png" if track else f"{_safe_stem(student_name)}_plot_{plot_id}.png"
                plot_path = os.path.join(plots_dir, filename)
                # Metadata embedded directly in the PNG itself (a real
                # tEXt chunk, readable with PIL.Image.open(...).text or
                # any PNG metadata tool) -- survives a rename, a copy into
                # a flattened folder, or a student not following the
                # naming convention above, none of which the filename
                # alone would survive. machine_id is the exact same hash
                # already in this session's encrypted log header (see
                # session_logger.py's _machine_id()), so a plot can be
                # cross-checked against the same machine-sharing signal
                # without needing the log at all.
                session.pending_figure.savefig(
                    plot_path, bbox_inches="tight", dpi=100,
                    metadata={
                        "student_name": student_name or "",
                        "student_id": student_id or "",
                        "track": track or "",
                        "result_id": plot_id,
                        "machine_id": logger.machine_id,
                    },
                )
                with output:
                    display(Markdown(f"*(plot saved to `{plot_path}`)*"))

            if getattr(session, "pending_figure", None) is not None:
                with output:
                    display(session.pending_figure)
                session.pending_figure = None
        except Exception as exc:
            if is_rate_limit_error(exc):
                with output:
                    display(
                        Markdown(
                            "**Rate limited.** The LLM provider's request quota has been hit "
                            "(this is common on free-tier API keys, which allow only a small "
                            "number of requests per day/minute). Wait a minute and send another "
                            "message: per-minute limits clear quickly. If it keeps happening, "
                            "the daily quota is used up (it resets once a day); type `/status` "
                            "to see your keys. Don't restart the kernel in AgenticPipeline or "
                            "Homework: that erases the agent's memory of your run.\n\n"
                            f"*(details: {exc})*"
                        )
                    )
            elif is_server_unavailable_error(exc):
                with output:
                    display(
                        Markdown(
                            "**Provider unreachable.** A connection failure, a timeout, or a "
                            "5xx error -- the provider is overloaded or having an outage, not "
                            "that anything here is broken (see the notice above, if this backend "
                            "or multi-provider fallback already tried more than once before "
                            "giving up).\n\n"
                            f"*(details: {exc})*"
                        )
                    )
            else:
                # Never fail silently -- an uncaught exception in an ipywidgets
                # callback otherwise shows up nowhere in the notebook UI, which
                # looks exactly like "nothing happened" when you send a message.
                with output:
                    display(Markdown("**Error** (see traceback below):"))
                    print(traceback.format_exc())
        finally:
            text_input.disabled = False
            send_button.disabled = False
            send_button.description = "Send"
            with output:
                display(Javascript(_SCROLL_TO_BOTTOM_JS))

    send_button.on_click(on_send)
    text_input.on_submit(on_send)

    input_row = widgets.HBox(
        [text_input, send_button], layout=widgets.Layout(flex="0 0 auto", width="100%", align_items="center")
    )
    display(widgets.VBox([output, input_row], layout=widgets.Layout(width="100%")))
    return graph_state, logger

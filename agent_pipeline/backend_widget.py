"""Notebook-embedded backend picker.

This replaces an earlier version that used input() for this step. input()
depends on the notebook frontend having its stdin channel wired up to the
kernel, and not every frontend does that by default -- some just sit at
"kernel busy" forever with no visible box to type into and no error, which
looks exactly like the notebook has frozen. A clickable widget doesn't have
that dependency: it renders as normal cell output the same way the chat
widget below it already does.
"""

import ipywidgets as widgets
from IPython.display import display

from .llm_backends import get_api_keys


def select_backend(valid_backends=("groq", "gemini")):
    """Displays a name field, a student ID field, a backend dropdown, a
    fallback checkbox, and a Confirm button. Returns a plain dict, empty
    until the student fills in both identifiers, picks a backend, and
    clicks Confirm, at which point it's filled in with "student_name",
    "student_id", "backend", and "multi_provider". Run this cell, fill it
    in, click Confirm, then move on -- the next cell checks for those
    keys and tells you plainly if you skipped this step.

    This only checks that a usable key exists; it doesn't read or store
    one. The graph-launch cell builds the actual model itself, straight
    from config/.env, through build_llm_for_backend -- which is also
    where a backend with several keys configured (GOOGLE_API_KEY_1
    through GOOGLE_API_KEY_10, say) gets turned into a model that
    rotates between all of them. See llm_backends.py.

    Both identifiers go into the session log's header in plain text (see
    session_logger.py) so a batch of submissions can be matched back to
    students without depending on how each file happens to be named."""
    choice = {}

    name_input = widgets.Text(placeholder="Your name, as it should appear on the submission", description="Name:")
    id_input = widgets.Text(placeholder="Your student ID", description="Student ID:")
    backend_dropdown = widgets.Dropdown(options=valid_backends, description="Backend:")
    fallback_checkbox = widgets.Checkbox(value=False, description="Enable multi-provider fallback")
    confirm_button = widgets.Button(description="Confirm", button_style="primary")
    status_output = widgets.Output()

    def on_confirm(_=None):
        status_output.clear_output()
        student_name = name_input.value.strip()
        student_id = id_input.value.strip()
        backend = backend_dropdown.value
        with status_output:
            if not student_name or not student_id:
                print("Error: enter both your name and your student ID before confirming.")
                return
            keys = get_api_keys(backend)
            if not keys:
                print(
                    f"Error: no usable key found for '{backend}'. Copy config/.env.example to "
                    f"config/.env and paste in your key, then run this cell again."
                )
                return
            choice["student_name"] = student_name
            choice["student_id"] = student_id
            choice["backend"] = backend
            choice["multi_provider"] = fallback_checkbox.value
            print(f"Name: {student_name}")
            print(f"Student ID: {student_id}")
            key_word = "key" if len(keys) == 1 else "keys"
            print(f"Using backend: {backend}. {len(keys)} API {key_word} loaded.")
            print(f"Multi-provider fallback: {'enabled' if fallback_checkbox.value else 'disabled'}.")
            print("Set. Move on to the next cell.")

    confirm_button.on_click(on_confirm)
    display(widgets.VBox([name_input, id_input, backend_dropdown, fallback_checkbox, confirm_button, status_output]))
    return choice

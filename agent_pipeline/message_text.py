"""Normalizes a LangChain message's `.content` to plain text.

Different model and provider versions return `.content` differently.
Some return a plain string; a newer Gemini model instead returns a list
of content blocks, such as `[{"type": "text", "text": "...", "extras":
{...}}]`. Anything that reads message text, including the grounding
check and the chat display, needs to handle both forms, or it breaks the
moment the underlying model or library version changes.
"""


def as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""

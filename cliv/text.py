"""Text helpers shared by the agent and offline fallback."""

import re


def strip_markdown_fences(text: str) -> str:
    """Remove a surrounding markdown code fence from model output."""

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    code_block_match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    return text

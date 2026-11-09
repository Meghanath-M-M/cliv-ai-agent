"""Regex intent fallback for local models without tool calling.

This is deliberately kept as a last resort.  It is fast and predictable,
but brittle; the primary offline path is Ollama tool calling.  The
classifier is isolated here so it can be tested and replaced without
touching the agent loop.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

FILE_EXTENSIONS = r"(?:py|js|ts|html|css|md|txt|json|yaml|yml|sh|rs|go|java|cpp|c|h)"


def classify_intent(user_input: str) -> Tuple[str, Optional[str]]:
    """Return ``(category, target)`` for a natural-language request.

    Categories: conversation, project_analysis, file_create, file_read,
    file_fix, file_remove, file_rename, list_dir, unknown.
    """

    lowered = user_input.lower().strip()

    file_match = re.search(
        r"['\"]?([\w\-\./]+\.(?:" + FILE_EXTENSIONS + r"))['\"]?", lowered
    )
    extracted_file = file_match.group(1) if file_match else None

    social_patterns = [
        r"^\s*(hi+|hello|hey|greetings|howdy|hola|yo|sup)\b",
        r"^\s*(thanks?|thank\s+(?:you|u)|thx|ty)\b",
        r"^\s*(bye|goodbye|see\s+ya|cya|later|peace)\b",
        r"^\s*(ok|okay|cool|nice|great|awesome|good\s+job|well\s+done)\b",
    ]
    for pattern in social_patterns:
        if re.search(pattern, lowered):
            return ("conversation", None)

    if not extracted_file:
        conversation_patterns = [
            r"^\s*(who\s+(?:are|r)\s+(?:you|u)|what\s+(?:are|r)\s+(?:you|u)|what\s+is\s+(?:your|ur)\s+name)",
            r"^\s*(what\s+can\s+(?:you|u)\s+do|what\s+(?:are|r)\s+(?:your|ur)\s+capabilities|how\s+can\s+(?:you|u)\s+help)",
            r"^\s*(how\s+(?:are|r)\s+(?:you|u)|how's\s+it\s+going|whats\s+up|what's\s+up)",
            r"^\s*(can\s+(?:you|u)\s+help|could\s+(?:you|u)\s+help)\s*$",
        ]
        for pattern in conversation_patterns:
            if re.search(pattern, lowered):
                return ("conversation", None)

    project_analysis_patterns = [
        r"\b(?:analys|analyz)e\s+(?:this|the|my|our)\s+(?:project|repo|repository|codebase)\b",
        r"\b(?:project|repo|repository|codebase)[\s-]+(?:analysis|overview|dna|architecture|map|status)\b",
        r"\b(?:understand|explain|inspect|map)\s+(?:this|the|my|our)\s+(?:project|repo|repository|codebase)\b",
        r"\bwhat\s+(?:is|'s)\s+(?:this|the)\s+(?:project|repo|repository|codebase)\s+(?:about|architecture|structure)\b",
        r"\b(?:architecture|dependency)\s+(?:of|for)\s+(?:this|the|my|our)\s+(?:project|repo|repository|codebase)\b",
        r"\b(?:show|display|print)\s+(?:the\s+)?(?:dependency|import)\s+graph\b",
    ]
    if any(re.search(pattern, lowered) for pattern in project_analysis_patterns):
        return ("project_analysis", None)

    if (
        re.search(
            r"(?:list|show|ls|display)\s+(?:all\s+)?(?:the\s+)?(?:files?|contents?|items?)",
            lowered,
        )
        or re.search(
            r"what\s+(?:files?|.*)\s+(?:are|is)\s+(?:here|in\s+this|present)",
            lowered,
        )
        or re.search(
            r"what\s*(?:'s|s|is)\s+in\s+(?:the\s+)?(?:this\s+)?(?:current\s+)?(?:directory|folder|dir)",
            lowered,
        )
        or re.search(
            r"(?:show|list|display)\s+(?:me\s+)?(?:the\s+)?(?:files?|contents?|items?)\s+"
            r"(?:here|in\s+(?:this|the\s+current)\s+(?:dir|directory|folder))",
            lowered,
        )
    ):
        return ("list_dir", None)

    rename_match = re.search(
        r"(?:rename|mv|move)\s+(?:the\s+)?(?:file\s+)?['\"]?([\w\-\.]+\.\w+)['\"]?\s+(?:to|as|->)?\s*['\"]?([\w\-\.]+\.\w+)['\"]?",
        lowered,
    )
    if rename_match:
        return ("file_rename", f"{rename_match.group(1)}|{rename_match.group(2)}")

    create_patterns = [
        r"(?:create|make|write|generate|build)\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?(?:called|named)?\s*['\"]?([\w\-\.]+\.\w+)['\"]?",
        r"(?:create|make|write|generate|build)\s+['\"]?([\w\-\.]+\.\w+)['\"]?",
    ]
    for pattern in create_patterns:
        match = re.search(pattern, lowered)
        if match:
            filename = match.group(1)
            conversational = [
                "make sense",
                "make sure",
                "make it",
                "make this",
                "write down",
                "write back",
                "generate ideas",
                "build on",
            ]
            if not any(phrase in lowered for phrase in conversational):
                return ("file_create", filename)

    fix_patterns = [
        r"(?:fix|debug|correct|repair|patch)(?:\s+\w+){0,4}\s+(?:the\s+)?(?:file\s+)?['\"]?([\w\-\.]+\.\w+)['\"]?",
        r"(?:fix|debug|correct|repair)(?:\s+(?:the|any|all|some|this|that|a|an))?(?:\s+(?:error|errors|bug|bugs|issue|issues|problem|problems))?(?:\s+(?:in|with|on|of|for))\s+['\"]?([\w\-\.]+\.\w+)['\"]?",
    ]
    for pattern in fix_patterns:
        match = re.search(pattern, lowered)
        if match:
            return ("file_fix", match.group(1))

    read_patterns = [
        r"(?:show|read|display|open|view|cat|print|get|see)(?:\s+(?:me|us|the|this|that|my))?(?:\s+(?:content|contents|file|code|script))?(?:\s+of)?\s+['\"]?([\w\-\.]+\.\w+)['\"]?",
    ]
    for pattern in read_patterns:
        match = re.search(pattern, lowered)
        if match:
            return ("file_read", match.group(1))

    remove_patterns = [
        r"(?:remove|delete|rm|destroy)\s+(?:the\s+)?(?:file\s+)?['\"]?([\w\-\.]+\.\w+)['\"]?",
    ]
    for pattern in remove_patterns:
        match = re.search(pattern, lowered)
        if match:
            return ("file_remove", match.group(1))

    if extracted_file:
        if any(
            keyword in lowered
            for keyword in ["fix", "debug", "correct", "repair", "error", "bug"]
        ):
            return ("file_fix", extracted_file)
        if any(
            keyword in lowered
            for keyword in ["show", "read", "display", "view", "see", "open", "get"]
        ):
            return ("file_read", extracted_file)
        if any(keyword in lowered for keyword in ["remove", "delete", "rm", "destroy"]):
            return ("file_remove", extracted_file)
        if any(
            keyword in lowered
            for keyword in ["create", "make", "write", "generate", "build"]
        ):
            return ("file_create", extracted_file)
        return ("file_read", extracted_file)

    return ("unknown", None)


def extract_file(user_input: str) -> Optional[str]:
    """Extract the first filename-like token from a request."""

    _, target = classify_intent(user_input)
    return target

from __future__ import annotations

import re


RESERVED_TAG_PATTERN = re.compile(r"<\|im_(?:start|end)\|>")


def sanitize_reserved_tags(text: str) -> str:
    """Turn literal control tags in user text into harmless full-width text."""

    return RESERVED_TAG_PATTERN.sub(
        lambda match: match.group(0).replace("<", "＜").replace(">", "＞"),
        str(text),
    )


def format_prompt(
    source: str,
    source_label: str = "<|im_start|>",
    target_label: str = "<|im_end|>",
    *,
    sanitize: bool = True,
) -> str:
    """Render one source document with the text-simplification task protocol."""

    safe_source = sanitize_reserved_tags(source) if sanitize else str(source)
    return f"{source_label}{safe_source}{target_label}"

"""Private, privacy-sensitive helpers — PRIVACY-CRITICAL.

PRIVACY-CRITICAL
================
This module is the only place in the SDK that touches customer prompt
content directly. Code here must obey three rules:

  1. NEVER pass prompt content to a logger (`logger.*`) — not even in
     a formatted string, not even at DEBUG level. CI enforces this
     with `tests/unit/test_privacy_firewall.py`.
  2. NEVER store prompt content on a long-lived object — compute and
     discard within the current function call.
  3. NEVER include prompt content in exception arguments. If a
     computation fails, log `type(exc).__name__` only.

If you add a new helper here, add a corresponding enforcement test.
"""

from __future__ import annotations

from typing import Any


def estimate_content_length(kwargs: dict[str, Any]) -> int:
    """Return the total character length of prompt content in kwargs.

    Walks messages/system/contents and sums string lengths WITHOUT
    concatenating them into a joined string. The returned integer is
    safe to log — it is not reversible to prompt content.

    Args:
        kwargs: The LLM call kwargs dict. Handles OpenAI/Anthropic/Bedrock
            messages (Bedrock Converse content blocks are ``{"text": ...}``
            dicts, covered by the block walk), the Anthropic string system
            prompt, the Bedrock system block LIST, and Google contents.

    Returns:
        Total character count (0 if no recognizable content keys).
    """
    total = 0

    messages = kwargs.get("messages", [])
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if isinstance(text, str):
                        total += len(text)

    system = kwargs.get("system")
    if isinstance(system, str):
        total += len(system)
    elif isinstance(system, list):
        # Bedrock Converse: system is a list of SystemContentBlock dicts.
        # Non-text blocks (cachePoint/guardContent) carry no countable text.
        for block in system:
            if isinstance(block, dict):
                text = block.get("text", "")
                if isinstance(text, str):
                    total += len(text)

    contents = kwargs.get("contents")
    if isinstance(contents, str):
        total += len(contents)
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                total += len(item)
            elif isinstance(item, dict):
                text = item.get("text", "")
                if isinstance(text, str):
                    total += len(text)

    return total


def estimate_tokens_from_length(char_count: int, provider: str) -> int:
    """Convert a character count to a token estimate using per-provider ratios.

    These are heuristic ratios that match tiktoken's observed behavior.
    They are NOT tiktoken-exact — the exact path is intentionally removed
    because it required materializing the joined prompt text.

    Args:
        char_count: Number of characters in the prompt content.
        provider: One of "openai", "anthropic", "google", "bedrock".

    Returns:
        Estimated token count.
    """
    ratio_by_provider = {
        "openai": 4.0,
        "anthropic": 3.8,
        "google": 4.0,
        # Bedrock hosts many model families with different tokenizers; 4.0 is
        # the deliberate multi-family default (pre-flight estimate only — the
        # settled cost always comes from provider-reported usage).
        "bedrock": 4.0,
    }
    ratio = ratio_by_provider.get(provider, 4.0)
    return max(1, int(char_count / ratio))

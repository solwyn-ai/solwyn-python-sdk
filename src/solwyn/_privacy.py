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


def _chars_to_embedding_tokens(char_count: int, provider: str) -> int:
    """Chars -> tokens via the provider ratio, with 0 chars staying 0.

    ``estimate_tokens_from_length`` floors at 1 token for any positive char
    count; that floor is wrong for a purely pre-tokenized batch (0 text chars),
    which would otherwise gain a spurious 1-token estimate. Guard the zero case.
    """
    if char_count <= 0:
        return 0
    return estimate_tokens_from_length(char_count, provider)


def _estimate_embedding_list_tokens(items: list[Any], provider: str) -> int:
    """Token estimate for a list-valued embeddings ``input=`` (lengths only).

    Element type disambiguates the three list shapes without materializing any
    text — nothing is concatenated, stored, or returned beyond the integer:

    - ``str`` items: a text batch; character lengths summed then ratio-converted.
    - ``int`` items: token ids of ONE pre-tokenized sequence — each counts as a
      single token (``len()`` of the list IS the token count; running
      chars/ratio over token ids would be wrong).
    - ``list`` items: a batch of pre-tokenized sequences — each inner list
      contributes its own length.
    """
    char_total = 0
    token_total = 0
    for item in items:
        if isinstance(item, str):
            char_total += len(item)
        elif isinstance(item, bool):
            # bool is an int subclass but is never a token id — ignore it.
            continue
        elif isinstance(item, int):
            # list[int]: one pre-tokenized sequence; each int is one token id.
            token_total += 1
        elif isinstance(item, list):
            # list[list[int]]: this inner sequence's length is its token count.
            token_total += len(item)
    return token_total + _chars_to_embedding_tokens(char_total, provider)


def estimate_embedding_input_tokens(kwargs: dict[str, Any], provider: str) -> int:
    """Estimate input tokens for an embeddings request's ``input=`` (lengths only).

    The request-side fallback used when an embeddings response reports no usable
    usage (``measure_request`` in the embeddings ``MediaSurfaceSpec``). Recognizes
    the four ``input=`` shapes the OpenAI embeddings API accepts, measuring
    LENGTHS only — the input text/ids are never retained, logged, or concatenated:

    - ``str``: one text; character count converted to tokens via the provider ratio.
    - ``list[str]``: a batch of texts; summed character count ratio-converted.
    - ``list[int]``: ONE pre-tokenized sequence; ``len()`` IS the token count.
    - ``list[list[int]]``: a batch of pre-tokenized sequences; summed inner lengths.

    Pre-tokenized ids map to tokens 1:1 — dividing their count by the char/token
    ratio would undercount. Returns 0 when ``input=`` is absent or an unrecognized
    shape, so the caller keeps the billable quantity None rather than settling a
    zero-as-default price.
    """
    input_value = kwargs.get("input")
    if isinstance(input_value, str):
        return _chars_to_embedding_tokens(len(input_value), provider)
    if isinstance(input_value, list):
        return _estimate_embedding_list_tokens(input_value, provider)
    return 0


def _part_text_length(part: Any) -> int:
    """Sum string lengths of one delta/message part: content, reasoning text,
    and tool-call function arguments. Length-only; nothing is concatenated,
    stored, or returned beyond an integer."""
    total = 0
    content = getattr(part, "content", None)
    if isinstance(content, str):
        total += len(content)
    # DeepSeek-style reasoning text rides on a separate field.
    reasoning = getattr(part, "reasoning_content", None)
    if isinstance(reasoning, str):
        total += len(reasoning)
    # Tool-calling output lives entirely in function arguments — without this,
    # a tool-only response from a usage-less endpoint would estimate to zero.
    tool_calls = getattr(part, "tool_calls", None) or []
    for tool_call in tool_calls:
        arguments = getattr(getattr(tool_call, "function", None), "arguments", None)
        if isinstance(arguments, str):
            total += len(arguments)
    return total


def estimate_stream_chunk_content_length(chunk: Any) -> int:
    """Return the character length of an OpenAI-dialect stream chunk's delta text.

    Length-only measurement for the missing-streaming-usage fallback: when an
    OpenAI-compatible provider never emits a usage block, accumulated delta
    lengths (text + reasoning + tool-call arguments) estimate output tokens.
    Sums string lengths WITHOUT concatenating or storing the text. The
    returned integer is not reversible to content.

    Returns the total so far for any unexpected chunk shape. Never raises —
    this runs inside the live stream path on arbitrary endpoint output, and a
    malformed chunk must not convert a deliverable stream into a failure.
    """
    total = 0
    try:
        for choice in getattr(chunk, "choices", None) or []:
            total += _part_text_length(getattr(choice, "delta", None))
    except Exception:
        return total
    return total


def estimate_response_content_length(response: Any) -> int:
    """Return the character length of an OpenAI-dialect response's message text.

    Length-only measurement for the missing-usage fallback on NON-streaming
    responses from OpenAI-compatible providers. Counts message text, reasoning
    text, and tool-call arguments. Sums string lengths WITHOUT concatenating
    or storing the text. The returned integer is not reversible to content.

    Returns the total so far for any unexpected response shape. Never raises.
    """
    total = 0
    try:
        for choice in getattr(response, "choices", None) or []:
            total += _part_text_length(getattr(choice, "message", None))
    except Exception:
        return total
    return total

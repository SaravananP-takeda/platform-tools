"""
Context optimization utilities.

Strategies implemented
──────────────────────
1. History trimming   – keep only the most recent N turns instead of the full
                        conversation, to avoid ever-growing prompts.
2. Chunk deduplication – remove retrieved chunks that are near-duplicates of
                          each other (overlap > 50 % of tokens).
3. Context compression  – truncate individual chunks to fit within a token budget
                           while preserving the most informative content.
4. Budget allocation    – given a total token budget, allocate tokens fairly
                           across history, retrieved context, and system prompt.

The token counts are approximate (4 chars ≈ 1 token).  For production use,
replace `_rough_tokens` with tiktoken or the provider's tokeniser.
"""

from __future__ import annotations

from typing import List, Tuple

from copilot.providers import Message
from copilot.retrieval import Chunk
from copilot.cache import _tokenize


# ─── Token estimation ─────────────────────────────────────────────────────────

def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ─── History trimming ─────────────────────────────────────────────────────────

def trim_history(
    messages: List[Message],
    max_turns: int = 6,
) -> List[Message]:
    """
    Keep at most `max_turns` recent user+assistant exchanges.
    The system message (if present) is always preserved.
    """
    system = [m for m in messages if m.role == "system"]
    conversation = [m for m in messages if m.role != "system"]

    # Each turn is a (user, assistant) pair — 2 messages
    max_msgs = max_turns * 2
    if len(conversation) > max_msgs:
        conversation = conversation[-max_msgs:]

    return system + conversation


# ─── Chunk deduplication ──────────────────────────────────────────────────────

def deduplicate_chunks(chunks: List[Chunk], overlap_threshold: float = 0.5) -> List[Chunk]:
    """
    Remove chunks whose token overlap with a previously seen chunk exceeds the
    threshold.  Keeps the highest-scored chunk from each near-duplicate group.
    """
    kept: List[Chunk] = []
    kept_token_sets: List[set] = []

    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        chunk_tokens = set(_tokenize(chunk.text))
        if not chunk_tokens:
            continue

        duplicate = False
        for seen_tokens in kept_token_sets:
            if not seen_tokens:
                continue
            intersection = len(chunk_tokens & seen_tokens)
            union = len(chunk_tokens | seen_tokens)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard >= overlap_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(chunk)
            kept_token_sets.append(chunk_tokens)

    return kept


# ─── Context compression ──────────────────────────────────────────────────────

def compress_chunks(
    chunks: List[Chunk],
    budget_tokens: int = 1500,
) -> Tuple[List[str], int]:
    """
    Trim chunks to fit within `budget_tokens`.
    Returns (list_of_compressed_texts, total_tokens_used).
    Chunks are truncated at the sentence boundary when possible.
    """
    texts: List[str] = []
    used = 0

    for chunk in chunks:
        remaining = budget_tokens - used
        if remaining <= 0:
            break

        text = chunk.text
        tok_count = _rough_tokens(text)

        if tok_count <= remaining:
            texts.append(text)
            used += tok_count
        else:
            # Truncate to remaining budget (in characters ≈ tokens * 4)
            char_limit = remaining * 4
            truncated = text[:char_limit]
            # Try to cut at last sentence end
            last_period = max(
                truncated.rfind(". "),
                truncated.rfind(".\n"),
                truncated.rfind("! "),
                truncated.rfind("? "),
            )
            if last_period > char_limit // 2:
                truncated = truncated[: last_period + 1]
            texts.append(truncated + " [...]")
            used += _rough_tokens(truncated)
            break

    return texts, used


# ─── Full context builder ─────────────────────────────────────────────────────

def build_context(
    query: str,
    history: List[Message],
    chunks: List[Chunk],
    system_prompt: str,
    max_context_tokens: int = 3000,
    max_history_turns: int = 6,
) -> Tuple[List[Message], List[str]]:
    """
    Assemble the final message list for the LLM.

    Returns
    ───────
    (messages, context_texts)
      messages      – ready-to-send list including system, history, and user msg
      context_texts – the retrieved text passages that were included
    """
    # Reserve budget for system prompt + user query + some response slack
    system_tokens = _rough_tokens(system_prompt)
    query_tokens = _rough_tokens(query)
    slack = 300  # reserve for response preamble
    available = max(0, max_context_tokens - system_tokens - query_tokens - slack)

    # Allocate 60 % to retrieved context, 40 % to history
    context_budget = int(available * 0.60)
    history_budget = int(available * 0.40)

    # 1. Dedup and compress retrieved chunks
    deduped = deduplicate_chunks(chunks)
    context_texts, context_used = compress_chunks(deduped, context_budget)

    # 2. Trim history
    trimmed_history = trim_history(history, max_history_turns)
    # Further trim history if over budget
    history_tokens = sum(_rough_tokens(m.content) for m in trimmed_history)
    while history_tokens > history_budget and len(trimmed_history) > 2:
        removed = trimmed_history.pop(2)  # remove oldest non-system message
        history_tokens -= _rough_tokens(removed.content)

    # 3. Build retrieved-context block
    if context_texts:
        context_block = (
            "## Retrieved Context\n\n"
            + "\n\n---\n\n".join(context_texts)
            + "\n\n---"
        )
    else:
        context_block = ""

    # 4. Assemble messages
    system_msg = Message(
        role="system",
        content=system_prompt
        + (("\n\n" + context_block) if context_block else ""),
    )

    messages = [system_msg] + trimmed_history + [Message(role="user", content=query)]

    return messages, context_texts

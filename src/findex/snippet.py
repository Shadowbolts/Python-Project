import re
from collections.abc import Sequence
from pathlib import Path

from findex.tokenize import tokenize


def extract_snippet(
    text: str,
    query_terms: Sequence[str],
    window: int = 80,
    highlight_prefix: str = "**",
    highlight_suffix: str = "**",
) -> str:
    if not text:
        return ""

    normalized_terms: set[str] = set()
    for q in query_terms:
        for tok in tokenize(q):
            normalized_terms.add(tok)

    if not normalized_terms:
        snippet = text[: window * 2].strip()
        return snippet + ("..." if len(text) > len(snippet) else "")

    lower_text = text.casefold()
    best_pos = -1

    for term in normalized_terms:
        pos = lower_text.find(term)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    if best_pos == -1:
        snippet = text[: window * 2].strip()
        return snippet + ("..." if len(text) > len(snippet) else "")

    start = max(0, best_pos - window)
    end = min(len(text), best_pos + window)

    raw_snippet = text[start:end].strip()

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in normalized_terms) + r")\b",
        re.IGNORECASE,
    )
    highlighted = pattern.sub(rf"{highlight_prefix}\g<0>{highlight_suffix}", raw_snippet)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""

    return f"{prefix}{highlighted}{suffix}"


def snippet_for_doc(
    doc_path: str,
    query_terms: Sequence[str],
    window: int = 80,
) -> str:
    path = Path(doc_path)
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return extract_snippet(text, query_terms, window=window)
    except OSError:
        return ""

"""BM25 sparse retrieval for hybrid search."""
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Chinese-aware tokenizer using jieba, falls back to char-level."""
    try:
        import jieba
        tokens = jieba.lcut_for_search(text)
    except ImportError:
        # Fallback: simple character-level tokenization
        tokens = list(text)
    # Filter empty/whitespace-only tokens
    return [t.strip() for t in tokens if t.strip()]


def build_bm25_from_docs(documents: List[str]) -> "BM25Okapi":
    """Build a BM25Okapi index from a list of document texts.

    Args:
        documents: List of raw document texts.

    Returns:
        A rank_bm25 BM25Okapi instance, or None if documents is empty.
    """
    if not documents:
        return None
    from rank_bm25 import BM25Okapi
    tokenized = [_tokenize(doc) for doc in documents]
    return BM25Okapi(tokenized)


def bm25_search(
    bm25_index: "BM25Okapi",
    query: str,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """Search a BM25 index and return scored results.

    Args:
        bm25_index: A BM25Okapi instance.
        query: The search query.
        top_k: Maximum number of results.

    Returns:
        List of (document_index, score) tuples, sorted by score descending.
    """
    if bm25_index is None:
        return []
    tokenized_query = _tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)

    # Collect (index, score) pairs, filter zero scores
    scored = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

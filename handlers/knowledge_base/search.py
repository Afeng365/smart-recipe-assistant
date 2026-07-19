"""Knowledge base search orchestration and result formatting."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from settings.constant import KB_ROOT_PATH, KB_DEFAULT_TOP_K, KB_DEFAULT_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from the knowledge base."""
    content: str
    metadata: Dict[str, Any]
    score: float


def search_knowledge_base(
    query: str,
    kb_name: Optional[str] = None,
    top_k: int = KB_DEFAULT_TOP_K,
    score_threshold: float = KB_DEFAULT_SCORE_THRESHOLD,
) -> List[SearchResult]:
    """Search one or all knowledge bases for relevant recipe content.

    Args:
        query: The search query text.
        kb_name: Optional KB name to restrict search. If None, searches all KBs.
        top_k: Maximum number of results to return per KB.
        score_threshold: Minimum similarity score (0.0-1.0).

    Returns:
        List of SearchResult, sorted by score descending.
    """
    from handlers.knowledge_base.kb_manager import kb_manager
    from handlers.knowledge_base.embedding import EmbeddingModel

    # Get target KBs
    if kb_name:
        kb_info = kb_manager.get_kb(kb_name)
        if kb_info is None:
            logger.warning("KB not found: %s", kb_name)
            return []
        kb_names = [kb_name]
    else:
        all_kbs = kb_manager.list_kbs()
        kb_names = [kb["name"] for kb in all_kbs]

    if not kb_names:
        return []

    # Generate query embedding
    emb_model = EmbeddingModel()
    query_embedding = emb_model.embed_query(query)

    # Search each KB
    all_results: List[SearchResult] = []
    for name in kb_names:
        try:
            vs = kb_manager._get_vector_store(name)
            raw_results = vs.query(query_embedding, top_k=top_k, score_threshold=score_threshold)
            for r in raw_results:
                all_results.append(SearchResult(
                    content=r["content"],
                    metadata=r["metadata"],
                    score=r["score"],
                ))
        except Exception as e:
            logger.error("Error searching KB '%s': %s", name, e)
            continue

    # Sort by score descending
    all_results.sort(key=lambda x: x.score, reverse=True)
    return all_results


def format_search_results(results: List[SearchResult]) -> str:
    """Format search results into an LLM-readable context string.

    Args:
        results: List of search results.

    Returns:
        Formatted string for injection into LLM context.
    """
    if not results:
        return "【已知菜谱知识】\n（未找到相关内容）"

    lines = ["【已知菜谱知识】"]
    for r in results:
        source = r.metadata.get("source", "未知来源")
        lines.append(f"\n[来源：{source}，相关性：{r.score:.2f}]")
        lines.append(r.content)

    return "\n".join(lines)

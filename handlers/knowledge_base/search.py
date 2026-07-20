"""Knowledge base search — hybrid (dense + BM25) + optional Cross-Encoder rerank."""
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from settings.constant import (
    KB_ROOT_PATH,
    KB_DEFAULT_TOP_K,
    KB_DEFAULT_SCORE_THRESHOLD,
    KB_USE_BM25,
    KB_USE_RERANKER,
    KB_HYBRID_TOP_K_MULTIPLIER,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from the knowledge base."""
    content: str
    metadata: Dict[str, Any]
    score: float


# ── Reciprocal Rank Fusion ────────────────────────────────────────────

def _rrf_fusion(
    dense: List[SearchResult],
    sparse: List[Tuple[int, float]],
    all_docs: List[Dict[str, Any]],
    k: int = 60,
) -> List[SearchResult]:
    """Fuse dense (vector) and sparse (BM25) results with RRF.

    Args:
        dense: Results from vector search (already SearchResult objects).
        sparse: Results from BM25 — list of (doc_index, bm25_score).
        all_docs: Full document list from ChromaDB (same order as BM25 index).
        k: RRF constant (default 60).

    Returns:
        Fused list of SearchResult, sorted by RRF score descending.
    """
    rrf_scores: Dict[str, float] = {}

    # Dense contributions
    for rank, r in enumerate(dense):
        doc_id = f"{r.metadata.get('kb_name', '')}:{r.metadata.get('source', '')}:{r.metadata.get('chunk_index', rank)}"
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    # Sparse contributions
    for rank, (doc_idx, _) in enumerate(sparse):
        if doc_idx < len(all_docs):
            meta = all_docs[doc_idx].get("metadata", {})
            doc_id = f"{meta.get('kb_name', '')}:{meta.get('source', '')}:{meta.get('chunk_index', rank)}"
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    # Build fused candidate list: map back to original docs
    fused: List[Tuple[float, SearchResult]] = []

    # Add dense results with their RRF scores
    for r in dense:
        doc_id = f"{r.metadata.get('kb_name', '')}:{r.metadata.get('source', '')}:{r.metadata.get('chunk_index', 0)}"
        fused.append((rrf_scores.get(doc_id, 1.0 / (k + 100)), r))

    # Add sparse results not already in dense
    seen_contents = {r.content for _, r in fused}
    for doc_idx, _ in sparse:
        if doc_idx < len(all_docs):
            doc = all_docs[doc_idx]
            if doc["content"] not in seen_contents:
                r = SearchResult(
                    content=doc["content"],
                    metadata=doc["metadata"],
                    score=0.0,  # will be replaced by RRF score
                )
                doc_id = f"{doc['metadata'].get('kb_name', '')}:{doc['metadata'].get('source', '')}:{doc['metadata'].get('chunk_index', 0)}"
                fused.append((rrf_scores.get(doc_id, 1.0 / (k + 100)), r))
                seen_contents.add(doc["content"])

    # Sort by RRF score descending
    fused.sort(key=lambda x: x[0], reverse=True)
    result = []
    for rrf_score, sr in fused:
        sr.score = round(rrf_score, 4)
        result.append(sr)
    return result


# ── Main search entry point ───────────────────────────────────────────

def search_knowledge_base(
    query: str,
    kb_name: Optional[str] = None,
    top_k: int = KB_DEFAULT_TOP_K,
    score_threshold: float = KB_DEFAULT_SCORE_THRESHOLD,
) -> List[SearchResult]:
    """Search one or all KBs with optional BM25 hybrid + Cross-Encoder rerank.

    Pipeline:
      1. Dense (vector) retrieval — top_k * MULTIPLIER candidates
      2. [optional] BM25 sparse retrieval
      3. [optional] RRF fusion
      4. [optional] Cross-Encoder rerank → final top_k
    """
    from handlers.knowledge_base.kb_manager import kb_manager
    from handlers.knowledge_base.embedding import EmbeddingModel

    # Resolve KB names
    if kb_name:
        kb_info = kb_manager.get_kb(kb_name)
        if kb_info is None:
            logger.warning("KB not found: %s", kb_name)
            return []
        kb_names = [kb_name]
    else:
        kb_names = [kb["name"] for kb in kb_manager.list_kbs()]

    if not kb_names:
        return []

    # Generate query embedding
    emb_model = EmbeddingModel()
    query_embedding = emb_model.embed_query(query)

    # ── Per-KB search ──
    all_results: List[SearchResult] = []
    fetch_k = top_k * KB_HYBRID_TOP_K_MULTIPLIER if KB_USE_BM25 else top_k

    for name in kb_names:
        try:
            vs = kb_manager._get_vector_store(name)
            raw_results = vs.query(query_embedding, top_k=fetch_k, score_threshold=score_threshold)

            if not raw_results:
                continue

            # Convert to SearchResult
            dense_results = [
                SearchResult(
                    content=r["content"],
                    metadata=r["metadata"],
                    score=r["score"],
                )
                for r in raw_results
            ]

            # ── BM25 hybrid (per KB) ──
            if KB_USE_BM25:
                all_docs = vs.get_all_documents()
                doc_texts = [d["content"] for d in all_docs]
                if doc_texts:
                    from handlers.knowledge_base.bm25 import build_bm25_from_docs, bm25_search
                    bm25_index = build_bm25_from_docs(doc_texts)
                    sparse_results = bm25_search(bm25_index, query, top_k=top_k)
                    candidates = _rrf_fusion(dense_results, sparse_results, all_docs)
                else:
                    candidates = dense_results
            else:
                candidates = dense_results

            all_results.extend(candidates)
        except Exception as e:
            logger.error("Error searching KB '%s': %s", name, e)
            continue

    if not all_results:
        return []

    # Deduplicate by content
    seen = set()
    deduped = []
    for r in sorted(all_results, key=lambda x: x.score, reverse=True):
        if r.content not in seen:
            seen.add(r.content)
            deduped.append(r)

    # ── Cross-Encoder rerank ──
    if KB_USE_RERANKER and len(deduped) > top_k:
        from handlers.knowledge_base.reranker import RerankerModel
        reranker = RerankerModel()
        if reranker.available:
            docs_to_rerank = deduped[: min(len(deduped), top_k * 3)]
            reranked = reranker.rerank(query, [r.content for r in docs_to_rerank])
            # Keep best top_k after rerank
            reranked_results = []
            for idx, ce_score in reranked[:top_k]:
                docs_to_rerank[idx].score = round(ce_score, 4)
                reranked_results.append(docs_to_rerank[idx])
            return reranked_results

    return deduped[:top_k]


# ── Formatting ─────────────────────────────────────────────────────────

def format_search_results(results: List[SearchResult]) -> str:
    """Format search results into an LLM-readable context string."""
    if not results:
        return "【已知菜谱知识】\n（未找到相关内容）"

    lines = ["【已知菜谱知识】"]
    for r in results:
        source = r.metadata.get("source", "未知来源")
        lines.append(f"\n[来源：{source}，相关性：{r.score:.2f}]")
        lines.append(r.content)

    return "\n".join(lines)

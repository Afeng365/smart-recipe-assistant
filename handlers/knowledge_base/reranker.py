"""Cross-Encoder reranker for refining search results (singleton)."""
import logging
import os
from typing import List, Tuple

from settings.constant import KB_RERANKER_MODEL

logger = logging.getLogger(__name__)

# ── Fast-fail for HF download — avoid hanging ─────────────────────────
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")


class RerankerModel:
    """Cross-Encoder singleton for re-ranking search candidate pairs.

    Uses sentence-transformers CrossEncoder with a small,
    fast model (ms-marco-MiniLM-L-6-v2, ~80MB).

    Lazy-loads on first use.  Falls back gracefully if the model
    cannot be downloaded within the timeout.
    """
    _instance: "RerankerModel | None" = None
    _model = None
    _available: bool = False
    _checked: bool = False

    def __new__(cls, model_name: str = ""):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_name = model_name or KB_RERANKER_MODEL
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (useful for testing)."""
        cls._instance = None

    @property
    def available(self) -> bool:
        """Whether the re-ranker model was successfully loaded."""
        if not self._checked:
            self._try_load()
        return self._available

    def _try_load(self) -> None:
        """Attempt to load the CrossEncoder model (with 10s timeout)."""
        self._checked = True
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
            self._available = True
            logger.info("Reranker model loaded: %s", self._model_name)
        except Exception as e:
            self._available = False
            logger.info("Reranker model unavailable: %s — search continues without rerank", e)

    def rerank(
        self,
        query: str,
        documents: List[str],
    ) -> List[Tuple[int, float]]:
        """Re-rank documents by cross-encoding (query, doc) pairs.

        Returns (original_index, score) sorted score desc.
        Falls back to identity ranking if model unavailable.
        """
        if not documents:
            return []

        if not self.available:
            return [(i, 1.0) for i in range(len(documents))]

        pairs = [(query, doc) for doc in documents]
        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
            results = [(i, float(s)) for i, s in enumerate(scores)]
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            logger.error("Reranker prediction failed: %s", e)
            return [(i, 1.0) for i in range(len(documents))]

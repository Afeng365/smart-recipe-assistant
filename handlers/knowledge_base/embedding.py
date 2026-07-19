"""Embedding model wrapper using sentence-transformers (singleton)."""
import logging
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Singleton embedding model using BAAI/bge-small-zh-v1.5.

    Lazy-loads on first use to avoid startup delay and allow
    the application to start without the model being downloaded.
    """
    _instance: "EmbeddingModel | None" = None
    _model = None

    # BGE models benefit from a query instruction prefix
    _QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __new__(cls, model_name: str = "BAAI/bge-small-zh-v1.5"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_name = model_name
            cls._instance._loaded = False
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (useful for testing)."""
        cls._instance = None

    def load(self) -> None:
        """Load the model (idempotent — no-op if already loaded)."""
        if self._loaded:
            return
        logger.info("Loading embedding model: %s", self._model_name)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)
        self._loaded = True
        logger.info("Embedding model loaded, dimension=%d", self.dimension)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Batch-embed a list of texts. Returns list of float vectors."""
        self.load()
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text (with BGE query instruction)."""
        self.load()
        embedding = self._model.encode(
            self._QUERY_INSTRUCTION + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Embedding vector dimension (512 for bge-small-zh-v1.5)."""
        self.load()
        return self._model.get_sentence_embedding_dimension()

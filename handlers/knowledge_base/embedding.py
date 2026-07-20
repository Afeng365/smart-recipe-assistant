"""Embedding model wrapper using Ollama (local deployment, singleton)."""
import logging
from typing import List

import requests

from settings.constant import KB_OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

# Ollama embed API endpoint
OLLAMA_EMBED_URL = f"{KB_OLLAMA_BASE_URL}/api/embed"
OLLAMA_TAGS_URL = f"{KB_OLLAMA_BASE_URL}/api/tags"


class EmbeddingModel:
    """Singleton embedding model using Ollama qwen3-embedding:0.6b.

    Communicates with a local Ollama server via REST API.
    No local model download or GPU memory consumption — Ollama
    handles model lifecycle.

    The embedding dimension is auto-detected from the first API
    response, so switching Ollama models requires no code changes.
    """
    _instance: "EmbeddingModel | None" = None

    def __new__(cls, model_name: str = "qwen3-embedding:0.6b"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_name = model_name
            cls._instance._loaded = False
            cls._instance._dimension: int | None = None  # auto-detected
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (useful for testing)."""
        cls._instance = None

    def load(self) -> None:
        """Verify Ollama server is reachable (idempotent — no-op after first success)."""
        if self._loaded:
            return
        logger.info("Verifying Ollama connection at %s, model=%s",
                     KB_OLLAMA_BASE_URL, self._model_name)
        try:
            resp = requests.get(OLLAMA_TAGS_URL, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {KB_OLLAMA_BASE_URL}. "
                f"Make sure Ollama is running and the model '{self._model_name}' "
                f"is pulled: ollama pull {self._model_name}"
            ) from e
        self._loaded = True
        logger.info("Ollama connection verified")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Batch-embed a list of texts via Ollama API. Returns list of float vectors."""
        self.load()
        if not texts:
            return []
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": self._model_name, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        self._detect_dimension(embeddings)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text via Ollama API. Returns a single float vector."""
        self.load()
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": self._model_name, "input": [text]},
            timeout=60,
        )
        resp.raise_for_status()
        embedding = resp.json()["embeddings"][0]
        self._detect_dimension([embedding])
        return embedding

    def _detect_dimension(self, embeddings: List[List[float]]) -> None:
        """Auto-detect embedding dimension from the first API response."""
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])
            logger.info("Auto-detected embedding dimension: %d", self._dimension)

    @property
    def dimension(self) -> int:
        """Embedding vector dimension (auto-detected from the first API response)."""
        if self._dimension is None:
            # Trigger a quick embed to detect dimension
            self.embed(["dimension detection"])
        return self._dimension

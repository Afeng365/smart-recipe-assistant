"""ChromaDB vector store wrapper -- one collection per knowledge base."""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── 禁用 ChromaDB 遥测 ──────────────────────────────────────────────
# posthog 版本与 chromadb 0.5.23 不兼容，import 时即触发报错
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# 抑制 ChromaDB 遥测模块的 ERROR 日志（不影响功能）
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# ChromaDB client cache (one per persist_dir)
_clients: Dict[str, "chromadb.PersistentClient"] = {}


def _get_client(persist_dir: str) -> "chromadb.PersistentClient":
    """Get or create a ChromaDB PersistentClient (cached per persist_dir)."""
    if persist_dir not in _clients:
        import chromadb
        _clients[persist_dir] = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
    return _clients[persist_dir]


class VectorStore:
    """ChromaDB wrapper for a single knowledge base.

    Each KB maps to one ChromaDB collection named ``kb_{kb_name}``.
    Vectors are persisted to ``{persist_dir}/{kb_name}/chroma/``.
    """

    def __init__(self, kb_name: str, persist_dir: Path):
        self.kb_name = kb_name
        self.collection_name = f"kb_{kb_name}"
        persist_str = str(persist_dir)
        self.client = _get_client(persist_str)
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: List[List[float]],
    ) -> None:
        """Add documents to the vector store.

        Args:
            texts: Document chunk texts.
            metadatas: Metadata dicts (must include 'source', 'chunk_index', 'kb_name').
            ids: Unique IDs for each chunk.
            embeddings: Pre-computed embedding vectors matching each text.
        """
        if not texts:
            return
        self._collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )
        logger.info("Added %d chunks to collection '%s'", len(texts), self.collection_name)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Query the vector store for similar chunks.

        Args:
            query_embedding: The query embedding vector.
            top_k: Maximum number of results to return.
            score_threshold: Minimum similarity score (cosine distance).

        Returns:
            List of dicts with keys: content (str), metadata (dict), score (float).
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns cosine distance; convert to similarity: 1 - distance
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                similarity = 1.0 - distance
                if similarity >= score_threshold:
                    formatted.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": round(similarity, 4),
                    })
        return formatted

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete chunks by their IDs."""
        if not ids:
            return
        self._collection.delete(ids=ids)
        logger.info("Deleted %d chunks from '%s'", len(ids), self.collection_name)

    def delete_by_filter(self, filter_dict: Dict[str, Any]) -> None:
        """Delete chunks matching a metadata filter.

        Example:
            store.delete_by_filter({"source": "recipe.pdf"})
        """
        self._collection.delete(where=filter_dict)
        logger.info("Deleted chunks matching filter %s from '%s'",
                     filter_dict, self.collection_name)

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self._collection.count()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents in the collection (for BM25 index building).

        Returns:
            List of dicts with keys: id (str), content (str), metadata (dict).
        """
        if self._collection.count() == 0:
            return []
        results = self._collection.get(include=["documents", "metadatas"])
        docs = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                docs.append({
                    "id": results["ids"][i],
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        return docs

    def clear(self) -> None:
        """Delete the entire collection and re-create empty."""
        self.client.delete_collection(name=self.collection_name)
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared collection '%s'", self.collection_name)

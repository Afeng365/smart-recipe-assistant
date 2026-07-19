"""Knowledge Base Manager -- facade for KB lifecycle and document management."""
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from settings.constant import (
    KB_ROOT_PATH,
    KB_DB_PATH,
    KB_CONTENT_DIR_NAME,
    KB_CHROMA_DIR_NAME,
    KB_DEFAULT_EMBEDDING_MODEL,
    KB_DEFAULT_CHUNK_SIZE,
    KB_DEFAULT_CHUNK_OVERLAP,
    KB_MAX_FILE_SIZE_BYTES,
    KB_SUPPORTED_EXTENSIONS,
    KB_VALID_NAME_PATTERN,
)

logger = logging.getLogger(__name__)


def _validate_kb_name(name: str) -> None:
    """Validate KB name against allowed pattern."""
    if not name or not re.match(KB_VALID_NAME_PATTERN, name):
        raise ValueError(
            f"知识库名称只能包含中文、字母、数字、下划线和短横线: {name}"
        )


class KnowledgeBaseManager:
    """Manages knowledge base lifecycle and document indexing."""

    def __init__(self):
        self._init_lock = False

    def _ensure_init(self):
        """Lazy-init the DB on first use."""
        if not self._init_lock:
            import handlers.knowledge_base.db as db_mod
            db_mod.init_db(KB_DB_PATH)
            self._init_lock = True

    # -- KB CRUD ---------------------------------------------------------------

    def create_kb(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new knowledge base.

        Args:
            name: KB name (validated).
            description: Optional description.

        Returns:
            Dict with KB info (name, description, embedding_model, doc_count, created_at).

        Raises:
            ValueError: If name is invalid or KB already exists.
        """
        _validate_kb_name(name)
        self._ensure_init()

        import handlers.knowledge_base.db as db_mod

        # Check for duplicate
        existing = db_mod.get_kb_from_db(name)
        if existing:
            raise ValueError(f"知识库已存在: {name}")

        # Create directories
        kb_dir = KB_ROOT_PATH / name
        content_dir = kb_dir / KB_CONTENT_DIR_NAME
        content_dir.mkdir(parents=True, exist_ok=True)

        # Add to DB
        db_mod.add_kb_to_db(name, description, KB_DEFAULT_EMBEDDING_MODEL)

        # Init ChromaDB collection
        self._get_vector_store(name)

        logger.info("Created KB: %s", name)
        return db_mod.get_kb_from_db(name)

    def delete_kb(self, name: str) -> None:
        """Delete a knowledge base and all its data.

        Args:
            name: KB name.

        Raises:
            ValueError: If KB does not exist.
        """
        self._ensure_init()
        import handlers.knowledge_base.db as db_mod

        kb_info = db_mod.get_kb_from_db(name)
        if not kb_info:
            raise ValueError(f"知识库不存在: {name}")

        # Clear vector store
        try:
            vs = self._get_vector_store(name)
            vs.clear()
        except Exception as e:
            logger.warning("Error clearing vector store for KB '%s': %s", name, e)

        # Remove from DB (cascades to knowledge_file and file_chunk)
        db_mod.remove_kb_from_db(name)

        # Remove file directories
        kb_dir = KB_ROOT_PATH / name
        if kb_dir.exists():
            shutil.rmtree(kb_dir, ignore_errors=True)

        logger.info("Deleted KB: %s", name)

    def list_kbs(self) -> List[Dict[str, Any]]:
        """List all knowledge bases."""
        self._ensure_init()
        import handlers.knowledge_base.db as db_mod
        return db_mod.list_kbs_from_db()

    def get_kb(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a knowledge base by name. Returns None if not found."""
        self._ensure_init()
        import handlers.knowledge_base.db as db_mod
        return db_mod.get_kb_from_db(name)

    # -- Document Management --------------------------------------------------

    def add_documents(
        self,
        kb_name: str,
        file_paths: List[Path],
        chunk_size: int = KB_DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = KB_DEFAULT_CHUNK_OVERLAP,
    ) -> int:
        """Add documents to a knowledge base: load -> chunk -> embed -> store.

        Args:
            kb_name: Target KB name.
            file_paths: List of paths to document files.
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            Total number of chunks created.

        Raises:
            ValueError: If KB doesn't exist or file format unsupported.
        """
        self._ensure_init()
        import handlers.knowledge_base.db as db_mod
        from handlers.knowledge_base.document_loader import load_file, ChineseRecursiveTextSplitter
        from handlers.knowledge_base.embedding import EmbeddingModel

        kb_info = db_mod.get_kb_from_db(kb_name)
        if not kb_info:
            raise ValueError(f"知识库不存在: {kb_name}")

        splitter = ChineseRecursiveTextSplitter()
        emb_model = EmbeddingModel()
        vs = self._get_vector_store(kb_name)
        content_dir = KB_ROOT_PATH / kb_name / KB_CONTENT_DIR_NAME
        total_chunks = 0

        for fp in file_paths:
            if not fp.exists():
                logger.warning("File not found, skipping: %s", fp)
                continue

            # Validate extension
            ext = fp.suffix.lower()
            if ext not in KB_SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"不支持的文件格式: {ext}。支持: {', '.join(KB_SUPPORTED_EXTENSIONS)}"
                )

            # Validate size
            file_size = fp.stat().st_size
            if file_size > KB_MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"文件过大: {fp.name} ({file_size / 1024 / 1024:.1f}MB)。最大 {KB_MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB"
                )

            filename = fp.name
            logger.info("Processing file: %s (%d bytes)", filename, file_size)

            # If file already exists in KB, remove old chunks
            old_chunk_ids = db_mod.get_chunk_ids_by_file(kb_name, filename)
            if old_chunk_ids:
                logger.info("Removing %d old chunks for %s", len(old_chunk_ids), filename)
                try:
                    vs.delete_by_ids(old_chunk_ids)
                except Exception as e:
                    logger.warning("Error deleting old chunks: %s", e)
                db_mod.remove_chunk_records_by_file(kb_name, filename)
                db_mod.remove_file_record(kb_name, filename)

            # Copy file to content dir
            dest = content_dir / filename
            shutil.copy2(fp, dest)

            # Load and chunk
            text = load_file(dest)
            chunks = splitter.split_text(text, chunk_size, chunk_overlap)
            if not chunks:
                logger.warning("No text extracted from %s", filename)
                continue

            # Generate embeddings
            embeddings = emb_model.embed(chunks)

            # Prepare metadata and IDs
            metadatas = [
                {"source": filename, "chunk_index": i, "kb_name": kb_name}
                for i in range(len(chunks))
            ]
            chunk_ids = [f"{kb_name}_{filename}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]

            # Store in vector DB
            vs.add(texts=chunks, metadatas=metadatas, ids=chunk_ids, embeddings=embeddings)

            # Update metadata DB
            db_mod.add_file_record(kb_name, filename, ext, file_size, len(chunks))
            db_mod.add_chunk_records(kb_name, filename, chunk_ids)

            total_chunks += len(chunks)
            logger.info("Indexed %s: %d chunks", filename, len(chunks))

        db_mod.update_kb_doc_count(kb_name)
        return total_chunks

    def remove_documents(self, kb_name: str, filenames: List[str]) -> None:
        """Remove documents from a knowledge base.

        Args:
            kb_name: Target KB name.
            filenames: List of filenames to remove.
        """
        self._ensure_init()
        import handlers.knowledge_base.db as db_mod

        kb_info = db_mod.get_kb_from_db(kb_name)
        if not kb_info:
            raise ValueError(f"知识库不存在: {kb_name}")

        vs = self._get_vector_store(kb_name)
        content_dir = KB_ROOT_PATH / kb_name / KB_CONTENT_DIR_NAME

        for filename in filenames:
            # Remove from vector store
            chunk_ids = db_mod.get_chunk_ids_by_file(kb_name, filename)
            if chunk_ids:
                try:
                    vs.delete_by_ids(chunk_ids)
                except Exception as e:
                    logger.warning("Error deleting chunks for %s: %s", filename, e)

            # Remove from metadata DB
            db_mod.remove_chunk_records_by_file(kb_name, filename)
            db_mod.remove_file_record(kb_name, filename)

            # Remove content file
            file_path = content_dir / filename
            if file_path.exists():
                file_path.unlink()

            logger.info("Removed document: %s from KB: %s", filename, kb_name)

        db_mod.update_kb_doc_count(kb_name)

    def list_documents(self, kb_name: str) -> List[Dict[str, Any]]:
        """List all documents in a knowledge base."""
        self._ensure_init()
        import handlers.knowledge_base.db as db_mod

        if not db_mod.get_kb_from_db(kb_name):
            raise ValueError(f"知识库不存在: {kb_name}")

        return db_mod.list_file_records(kb_name)

    # -- Internal helpers -----------------------------------------------------

    def _get_vector_store(self, kb_name: str):
        """Get or create the VectorStore for a KB."""
        from handlers.knowledge_base.vector_store import VectorStore
        persist_dir = KB_ROOT_PATH / kb_name / KB_CHROMA_DIR_NAME
        persist_dir.mkdir(parents=True, exist_ok=True)
        return VectorStore(kb_name, persist_dir)


# -- Singleton ---------------------------------------------------------------

kb_manager = KnowledgeBaseManager()

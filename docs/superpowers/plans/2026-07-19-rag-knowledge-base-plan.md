# RAG Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a RAG knowledge base subsystem to the smart recipe assistant, enabling users to upload recipe documents, auto-chunk/embed/index them, and search via LLM tool calling.

**Architecture:** Bottom-up build of `handlers/knowledge_base/` (7 files: db → embedding → document_loader → vector_store → search → kb_manager → __init__), then wire into tools.py (LLM tool) and app.py (Flask API). Each module depends only on the modules before it in the chain.

**Tech Stack:** ChromaDB (vector store), sentence-transformers + BAAI/bge-small-zh-v1.5 (embeddings), PyMuPDF + python-docx (document parsing), SQLite (metadata), Flask (API).

## Global Constraints

- Project uses Flask 3.1.3, no FastAPI
- Python ≥ 3.10 required
- All KB data stored under `data/knowledge_base/`
- Embedding model: `BAAI/bge-small-zh-v1.5` (local, ~100MB download on first use)
- ChromaDB version: `0.5.23`
- sentence-transformers version: `3.4.1`
- pymupdf version: `1.25.5`
- python-docx version: `1.1.2`
- Supported file types: `.txt`, `.md`, `.pdf`, `.docx`, `.json`
- Max file size: 50MB
- Default chunk parameters: size=500, overlap=50
- Default search parameters: top_k=3, score_threshold=0.3
- KB name validation: Chinese chars, letters, digits, underscores, hyphens only
- Module init must handle first-use model download gracefully (lazy loading)
- Naming conventions: follow existing project patterns (snake_case files, PascalCase classes)
- SQLite foreign keys must be enabled

---

## File Structure

```
NEW:
  handlers/knowledge_base/__init__.py       # Re-exports KBManager, SearchResult
  handlers/knowledge_base/db.py             # SQLite metadata: 3 tables + CRUD
  handlers/knowledge_base/embedding.py       # EmbeddingModel singleton (bge-small-zh-v1.5)
  handlers/knowledge_base/document_loader.py # File→text extraction + ChineseRecursiveTextSplitter
  handlers/knowledge_base/vector_store.py    # ChromaDB wrapper per-kb collection
  handlers/knowledge_base/search.py          # Search orchestration + result formatting
  handlers/knowledge_base/kb_manager.py      # Facade: KB lifecycle + document management
  tests/test_knowledge_base.py               # Unit + integration tests

MODIFIED:
  settings/constant.py     # Add KB_* config constants
  requirements.txt         # Add chromadb, sentence-transformers, pymupdf, python-docx
  tools.py                 # Add search_recipe_knowledge_base to NATIVE_HANDLERS + NATIVE_TOOLS
  app.py                   # Add /api/kb/* Flask routes
```

---

### Task 1: Add KB Configuration Constants

**Files:**
- Modify: `settings/constant.py` (append at end)

**Interfaces:**
- Produces: `KB_ROOT_PATH`, `KB_DB_PATH`, `KB_DEFAULT_EMBEDDING_MODEL`, `KB_DEFAULT_CHUNK_SIZE`, `KB_DEFAULT_CHUNK_OVERLAP`, `KB_DEFAULT_TOP_K`, `KB_DEFAULT_SCORE_THRESHOLD`, `KB_MAX_FILE_SIZE_MB`, `KB_SUPPORTED_EXTENSIONS`, `KB_VALID_NAME_PATTERN`

- [ ] **Step 1: Append KB config constants to settings/constant.py**

Open `settings/constant.py` and append the following block after the last line:

```python
# ── 知识库配置 ──────────────────────────────────────────────────────────

KB_ROOT_PATH = WORKDIR / "data" / "knowledge_base"
KB_DB_PATH = KB_ROOT_PATH / "info.db"
KB_CONTENT_DIR_NAME = "content"
KB_CHROMA_DIR_NAME = "chroma"
KB_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
KB_DEFAULT_CHUNK_SIZE = 500
KB_DEFAULT_CHUNK_OVERLAP = 50
KB_DEFAULT_TOP_K = 3
KB_DEFAULT_SCORE_THRESHOLD = 0.3
KB_MAX_FILE_SIZE_MB = 50
KB_MAX_FILE_SIZE_BYTES = KB_MAX_FILE_SIZE_MB * 1024 * 1024
KB_SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx", ".json")
KB_VALID_NAME_PATTERN = r"^[一-龥a-zA-Z0-9_\-]+$"
```

- [ ] **Step 2: Verify Python can import the new constants**

```bash
python -c "from settings.constant import KB_ROOT_PATH, KB_DEFAULT_CHUNK_SIZE; print(f'KB_ROOT={KB_ROOT_PATH}, CHUNK={KB_DEFAULT_CHUNK_SIZE}')"
```

- [ ] **Step 3: Commit**

```bash
git add settings/constant.py
git commit -m "feat: add KB configuration constants"
```

---

### Task 2: Add Python Dependencies

**Files:**
- Modify: `requirements.txt` (append at end)

**Interfaces:**
- Produces: New packages available for import: `chromadb`, `sentence_transformers`, `fitz` (pymupdf), `docx` (python-docx)

- [ ] **Step 1: Append new dependencies to requirements.txt**

Append at end of `requirements.txt`:

```
# RAG Knowledge Base
chromadb==0.5.23
sentence-transformers==3.4.1
pymupdf==1.25.5
python-docx==1.1.2
```

- [ ] **Step 2: Install the new dependencies**

```bash
pip install chromadb==0.5.23 sentence-transformers==3.4.1 pymupdf==1.25.5 python-docx==1.1.2
```

Expected: All four packages install successfully. This will also download `onnxruntime`, `tokenizers`, `torch` (if not present), `Pillow`, etc. as transitive dependencies.

- [ ] **Step 3: Verify imports work**

```bash
python -c "import chromadb; from sentence_transformers import SentenceTransformer; import fitz; import docx; print('All imports OK')"
```

Expected output: `All imports OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat: add RAG dependencies (chromadb, sentence-transformers, pymupdf, python-docx)"
```

---

### Task 3: Create SQLite Metadata Database (db.py)

**Files:**
- Create: `handlers/knowledge_base/db.py`

**Interfaces:**
- Produces:
  - `db_path: Path | None` — module-level variable set by `init_db()`
  - `init_db(p: Path) -> None` — set module-level db_path, create tables if not exist, enable foreign keys
  - `add_kb_to_db(name: str, description: str, embedding_model: str) -> None`
  - `remove_kb_from_db(name: str) -> None`
  - `list_kbs_from_db() -> list[dict]`
  - `get_kb_from_db(name: str) -> dict | None`
  - `add_file_record(kb_name: str, filename: str, file_ext: str, file_size: int, chunk_count: int) -> None`
  - `remove_file_record(kb_name: str, filename: str) -> None`
  - `list_file_records(kb_name: str) -> list[dict]`
  - `add_chunk_records(kb_name: str, filename: str, chunk_ids: list[str]) -> None`
  - `remove_chunk_records_by_file(kb_name: str, filename: str) -> None`
  - `get_chunk_ids_by_file(kb_name: str, filename: str) -> list[str]`
  - `update_kb_doc_count(kb_name: str) -> None`

- [ ] **Step 1: Write the test placeholder**

Create `tests/test_knowledge_base.py`:

```python
"""Tests for RAG knowledge base subsystem."""
import pytest
```

- [ ] **Step 2: Write db.py**

Create `handlers/knowledge_base/db.py`:

```python
"""SQLite metadata database for knowledge base management."""
import sqlite3
from pathlib import Path

# Module-level: set by init_db() at startup
db_path: Path | None = None


def _connect() -> sqlite3.Connection:
    """Get a connection with foreign keys enabled (uses module-level db_path)."""
    if db_path is None:
        raise RuntimeError("db.db_path not initialized — call init_db() first")
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(p: Path) -> None:
    """Create tables if they don't exist. Sets module-level db_path."""
    global db_path
    db_path = p
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT DEFAULT '',
            embedding_model TEXT DEFAULT 'BAAI/bge-small-zh-v1.5',
            doc_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS knowledge_file (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_name TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_ext TEXT DEFAULT '',
            file_size INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(kb_name, filename),
            FOREIGN KEY (kb_name) REFERENCES knowledge_base(name) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS file_chunk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_name TEXT NOT NULL,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            FOREIGN KEY (kb_name) REFERENCES knowledge_base(name) ON DELETE CASCADE,
            FOREIGN KEY (kb_name, filename) REFERENCES knowledge_file(kb_name, filename) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()


# ── KB CRUD ───────────────────────────────────────────────────────────

def add_kb_to_db(name: str, description: str = "",
                  embedding_model: str = "BAAI/bge-small-zh-v1.5") -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO knowledge_base (name, description, embedding_model) VALUES (?, ?, ?)",
        (name, description, embedding_model),
    )
    conn.commit()
    conn.close()


def remove_kb_from_db(name: str) -> None:
    conn = _connect()
    conn.execute("DELETE FROM knowledge_base WHERE name = ?", (name,))
    conn.commit()
    conn.close()


def list_kbs_from_db() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT name, description, embedding_model, doc_count, created_at "
        "FROM knowledge_base ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_kb_from_db(name: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT name, description, embedding_model, doc_count, created_at "
        "FROM knowledge_base WHERE name = ?", (name,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── File CRUD ─────────────────────────────────────────────────────────

def add_file_record(kb_name: str, filename: str, file_ext: str = "",
                    file_size: int = 0, chunk_count: int = 0) -> None:
    conn = _connect()
    conn.execute(
        """INSERT OR REPLACE INTO knowledge_file (kb_name, filename, file_ext, file_size, chunk_count)
           VALUES (?, ?, ?, ?, ?)""",
        (kb_name, filename, file_ext, file_size, chunk_count),
    )
    conn.commit()
    conn.close()


def remove_file_record(kb_name: str, filename: str) -> None:
    conn = _connect()
    conn.execute(
        "DELETE FROM knowledge_file WHERE kb_name = ? AND filename = ?",
        (kb_name, filename),
    )
    conn.commit()
    conn.close()


def list_file_records(kb_name: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT filename, file_ext, file_size, chunk_count, created_at "
        "FROM knowledge_file WHERE kb_name = ? ORDER BY created_at DESC",
        (kb_name,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Chunk mapping ─────────────────────────────────────────────────────

def add_chunk_records(kb_name: str, filename: str, chunk_ids: list[str]) -> None:
    conn = _connect()
    conn.executemany(
        "INSERT INTO file_chunk (kb_name, filename, chunk_index, chunk_id) VALUES (?, ?, ?, ?)",
        [(kb_name, filename, i, cid) for i, cid in enumerate(chunk_ids)],
    )
    conn.commit()
    conn.close()


def remove_chunk_records_by_file(kb_name: str, filename: str) -> None:
    conn = _connect()
    conn.execute(
        "DELETE FROM file_chunk WHERE kb_name = ? AND filename = ?",
        (kb_name, filename),
    )
    conn.commit()
    conn.close()


def get_chunk_ids_by_file(kb_name: str, filename: str) -> list[str]:
    conn = _connect()
    rows = conn.execute(
        "SELECT chunk_id FROM file_chunk WHERE kb_name = ? AND filename = ? ORDER BY chunk_index",
        (kb_name, filename),
    ).fetchall()
    conn.close()
    return [r["chunk_id"] for r in rows]


def update_kb_doc_count(kb_name: str) -> None:
    conn = _connect()
    count = conn.execute(
        "SELECT COUNT(*) FROM knowledge_file WHERE kb_name = ?", (kb_name,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE knowledge_base SET doc_count = ? WHERE name = ?",
        (count, kb_name),
    )
    conn.commit()
    conn.close()
```

- [ ] **Step 3: Write db unit tests**

Add to `tests/test_knowledge_base.py`:

```python
import tempfile
from pathlib import Path
import sys

# Make handlers.knowledge_base importable
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDB:
    """Tests for handlers/knowledge_base/db.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        # Patch db module's db_path
        import importlib
        import handlers.knowledge_base.db as db_mod
        db_mod.init_db(self.db_path)
        self.db = db_mod
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_db_creates_tables(self):
        conn = self.db._connect()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r["name"] for r in tables}
        assert "knowledge_base" in names
        assert "knowledge_file" in names
        assert "file_chunk" in names
        conn.close()

    def test_add_and_list_kb(self):
        self.db.add_kb_to_db("test_kb", "test desc")
        kbs = self.db.list_kbs_from_db()
        assert len(kbs) == 1
        assert kbs[0]["name"] == "test_kb"
        assert kbs[0]["description"] == "test desc"

    def test_add_duplicate_kb_raises(self):
        self.db.add_kb_to_db("test_kb")
        with pytest.raises(Exception):
            self.db.add_kb_to_db("test_kb")

    def test_remove_kb(self):
        self.db.add_kb_to_db("test_kb")
        self.db.remove_kb_from_db("test_kb")
        assert len(self.db.list_kbs_from_db()) == 0

    def test_get_kb(self):
        self.db.add_kb_to_db("test_kb", "desc")
        kb = self.db.get_kb_from_db("test_kb")
        assert kb["name"] == "test_kb"
        assert self.db.get_kb_from_db("nonexistent") is None

    def test_file_records(self):
        self.db.add_kb_to_db("test_kb")
        self.db.add_file_record("test_kb", "recipe.txt", ".txt", 100, 3)
        files = self.db.list_file_records("test_kb")
        assert len(files) == 1
        assert files[0]["filename"] == "recipe.txt"
        assert files[0]["chunk_count"] == 3

    def test_file_record_replace_on_duplicate(self):
        self.db.add_kb_to_db("test_kb")
        self.db.add_file_record("test_kb", "r.txt", ".txt", 100, 3)
        self.db.add_file_record("test_kb", "r.txt", ".txt", 200, 5)
        files = self.db.list_file_records("test_kb")
        assert len(files) == 1
        assert files[0]["file_size"] == 200
        assert files[0]["chunk_count"] == 5

    def test_chunk_records(self):
        self.db.add_kb_to_db("test_kb")
        self.db.add_file_record("test_kb", "r.txt", ".txt", 100, 0)
        self.db.add_chunk_records("test_kb", "r.txt", ["c1", "c2", "c3"])
        ids = self.db.get_chunk_ids_by_file("test_kb", "r.txt")
        assert ids == ["c1", "c2", "c3"]

    def test_remove_chunk_records(self):
        self.db.add_kb_to_db("test_kb")
        self.db.add_file_record("test_kb", "r.txt")
        self.db.add_chunk_records("test_kb", "r.txt", ["c1", "c2"])
        self.db.remove_chunk_records_by_file("test_kb", "r.txt")
        assert self.db.get_chunk_ids_by_file("test_kb", "r.txt") == []

    def test_cascade_delete_kb_removes_files_and_chunks(self):
        self.db.add_kb_to_db("test_kb")
        self.db.add_file_record("test_kb", "r.txt")
        self.db.add_chunk_records("test_kb", "r.txt", ["c1"])
        self.db.remove_kb_from_db("test_kb")
        assert self.db.list_file_records("test_kb") == []
        assert self.db.get_chunk_ids_by_file("test_kb", "r.txt") == []

    def test_update_doc_count(self):
        self.db.add_kb_to_db("test_kb")
        self.db.add_file_record("test_kb", "a.txt")
        self.db.add_file_record("test_kb", "b.txt")
        self.db.update_kb_doc_count("test_kb")
        kb = self.db.get_kb_from_db("test_kb")
        assert kb["doc_count"] == 2
```

- [ ] **Step 4: Run db tests to verify they pass**

```bash
python -m pytest tests/test_knowledge_base.py::TestDB -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add handlers/knowledge_base/db.py tests/test_knowledge_base.py
git commit -m "feat: add SQLite metadata DB for knowledge base"
```

---

### Task 4: Create Embedding Model (embedding.py)

**Files:**
- Create: `handlers/knowledge_base/embedding.py`

**Interfaces:**
- Produces:
  - `EmbeddingModel` — singleton class with `load()`, `embed(texts) -> list[list[float]]`, `embed_query(text) -> list[float]`, `dimension -> int` property

- [ ] **Step 1: Write embedding.py**

Create `handlers/knowledge_base/embedding.py`:

```python
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
```

- [ ] **Step 2: Write embedding unit tests**

Add to `tests/test_knowledge_base.py`:

```python

class TestEmbeddingModel:
    """Tests for handlers/knowledge_base/embedding.py."""

    def teardown_method(self):
        from handlers.knowledge_base.embedding import EmbeddingModel
        EmbeddingModel.reset()

    def test_singleton(self):
        from handlers.knowledge_base.embedding import EmbeddingModel
        a = EmbeddingModel()
        b = EmbeddingModel()
        assert a is b

    def test_singleton_different_model_names(self):
        from handlers.knowledge_base.embedding import EmbeddingModel
        EmbeddingModel.reset()
        a = EmbeddingModel("BAAI/bge-small-zh-v1.5")
        b = EmbeddingModel("other-model")
        assert a is b  # singleton ignores second model_name

    def test_lazy_loading(self):
        from handlers.knowledge_base.embedding import EmbeddingModel
        EmbeddingModel.reset()
        m = EmbeddingModel()
        assert not m._loaded

    def test_load_and_embed(self):
        from handlers.knowledge_base.embedding import EmbeddingModel
        EmbeddingModel.reset()
        m = EmbeddingModel()
        vectors = m.embed(["测试文本", "第二段文本"])
        assert len(vectors) == 2
        assert len(vectors[0]) == m.dimension
        assert m.dimension == 512

    def test_embed_empty_list(self):
        from handlers.knowledge_base.embedding import EmbeddingModel
        EmbeddingModel.reset()
        m = EmbeddingModel()
        assert m.embed([]) == []

    def test_embed_query(self):
        from handlers.knowledge_base.embedding import EmbeddingModel
        EmbeddingModel.reset()
        m = EmbeddingModel()
        vec = m.embed_query("红烧肉怎么做")
        assert len(vec) == 512
```

- [ ] **Step 3: Run embedding tests**

```bash
python -m pytest tests/test_knowledge_base.py::TestEmbeddingModel -v
```

Expected: 6 tests pass. Note: first run will download the ~100MB model from HuggingFace Hub. Subsequent runs use cached model.

- [ ] **Step 4: Commit**

```bash
git add handlers/knowledge_base/embedding.py tests/test_knowledge_base.py
git commit -m "feat: add embedding model wrapper (bge-small-zh-v1.5 singleton)"
```

---

### Task 5: Create Document Loader and Chinese Text Splitter

**Files:**
- Create: `handlers/knowledge_base/document_loader.py`

**Interfaces:**
- Produces:
  - `load_file(filepath: Path) -> str` — detect format, extract text
  - `ChineseRecursiveTextSplitter` — class with `split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]`
  - `LOADER_EXTENSIONS: dict[str, str]` — maps extension → loader name

- [ ] **Step 1: Write document_loader.py**

Create `handlers/knowledge_base/document_loader.py`:

```python
"""Document loading and Chinese-aware text splitting."""
import json
import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ── Supported formats ──────────────────────────────────────────────────

LOADER_EXTENSIONS: dict[str, str] = {
    ".txt": "text",
    ".md": "markdown",
    ".pdf": "pdf",
    ".docx": "docx",
    ".json": "json",
}


def load_file(filepath: Path) -> str:
    """Load text content from a file based on its extension.

    Args:
        filepath: Path to the file.

    Returns:
        Extracted text content.

    Raises:
        ValueError: If file extension is not supported.
        FileNotFoundError: If file does not exist.
    """
    ext = filepath.suffix.lower()
    if ext not in LOADER_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported: {', '.join(LOADER_EXTENSIONS)}"
        )

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info("Loading %s file: %s", ext, filepath.name)

    if ext in (".txt", ".md"):
        return filepath.read_text(encoding="utf-8")

    if ext == ".pdf":
        return _load_pdf(filepath)

    if ext == ".docx":
        return _load_docx(filepath)

    if ext == ".json":
        return _load_json(filepath)

    raise ValueError(f"Loader not implemented for: {ext}")


def _load_pdf(filepath: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    import fitz  # pymupdf
    doc = fitz.open(str(filepath))
    texts = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            texts.append(text.strip())
    doc.close()
    return "\n\n".join(texts)


def _load_docx(filepath: Path) -> str:
    """Extract text from DOCX using python-docx."""
    from docx import Document
    doc = Document(str(filepath))
    texts = []
    for para in doc.paragraphs:
        if para.text.strip():
            texts.append(para.text.strip())
    # Also extract table text
    for table in doc.tables:
        for row in table.rows:
            row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_texts:
                texts.append(" | ".join(row_texts))
    return "\n".join(texts)


def _load_json(filepath: Path) -> str:
    """Extract text from JSON. Handles list of recipe objects, dict with list values, or plain dict."""
    data = json.loads(filepath.read_text(encoding="utf-8"))

    if isinstance(data, list):
        # List of objects — assume recipe list
        texts = []
        for item in data:
            if isinstance(item, dict):
                texts.append(_dict_to_text(item))
            else:
                texts.append(str(item))
        return "\n\n---\n\n".join(texts)

    if isinstance(data, dict):
        # Check if any value is a list of recipe objects
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                texts = [_dict_to_text(item) for item in value]
                return "\n\n---\n\n".join(texts)
        return _dict_to_text(data)

    return str(data)


def _dict_to_text(d: dict) -> str:
    """Convert a recipe-like dict to readable text."""
    lines = []
    # Title / name fields
    for key in ("name", "title", "菜名", "名称", "recipe_name"):
        if d.get(key):
            lines.append(f"# {d[key]}")
            break

    # Category / cuisine
    for key in ("category", "cuisine", "菜系", "分类"):
        if d.get(key):
            lines.append(f"菜系: {d[key]}")
            break

    # Ingredients
    for key in ("ingredients", "食材", "配料"):
        val = d.get(key)
        if val:
            if isinstance(val, list):
                lines.append(f"食材: {', '.join(str(v) for v in val)}")
            elif isinstance(val, str):
                lines.append(f"食材: {val}")
            break

    # Instructions / steps
    for key in ("instructions", "steps", "做法", "步骤", "烹饪方法"):
        val = d.get(key)
        if val:
            lines.append(f"\n做法:")
            if isinstance(val, list):
                for i, step in enumerate(val, 1):
                    lines.append(f"{i}. {step}")
            elif isinstance(val, str):
                lines.append(val)
            break

    # If nothing recognized, dump all
    if not lines:
        for k, v in d.items():
            lines.append(f"{k}: {v}")

    return "\n".join(lines)


# ── Chinese Recursive Text Splitter ────────────────────────────────────

class ChineseRecursiveTextSplitter:
    """Recursive text splitter with Chinese-aware separators.

    Splits text by trying separators in priority order until each chunk
    fits within chunk_size. Inspired by Langchain-ChatChat's implementation.
    """

    # Separators ordered by priority (paragraph → sentence → phrase)
    _SEPARATORS = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        ".",
        "!",
        "?",
        "；",
        ";",
        "，",
        ",",
        " ",
    ]

    def split_text(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> List[str]:
        """Split text into chunks.

        Args:
            text: The input text to split.
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        return self._split_recursive(text.strip(), chunk_size, chunk_overlap, 0)

    def _split_recursive(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separator_idx: int,
    ) -> List[str]:
        """Recursively split text, trying each separator in priority order."""
        # Base case: text fits in one chunk
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        # Try current separator
        if separator_idx < len(self._SEPARATORS):
            sep = self._SEPARATORS[separator_idx]
            splits = text.split(sep)

            # If separator doesn't actually split (single element), try next
            if len(splits) == 1:
                return self._split_recursive(text, chunk_size, chunk_overlap, separator_idx + 1)

            # Merge splits into chunks
            chunks = []
            current = ""
            for part in splits:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        # Current chunk is full — recursively split it if needed
                        if len(current) > chunk_size:
                            chunks.extend(
                                self._split_recursive(
                                    current, chunk_size, chunk_overlap, separator_idx + 1
                                )
                            )
                        else:
                            chunks.append(current)
                    # Start new chunk — if it's too big, recurse
                    if len(part) > chunk_size:
                        chunks.extend(
                            self._split_recursive(
                                part, chunk_size, chunk_overlap, separator_idx + 1
                            )
                        )
                    else:
                        current = part
            if current.strip():
                if len(current) > chunk_size:
                    chunks.extend(
                        self._split_recursive(
                            current, chunk_size, chunk_overlap, separator_idx + 1
                        )
                    )
                else:
                    chunks.append(current)

            # Apply overlap by merging small final chunks
            if chunk_overlap > 0 and len(chunks) > 1:
                return self._apply_overlap(chunks, chunk_overlap)
            return chunks

        # Last resort: no separator worked — hard split by character count
        return self._hard_split(text, chunk_size, chunk_overlap)

    def _hard_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Hard split by fixed character count (last resort)."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - chunk_overlap
        return chunks

    def _apply_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """Apply overlap by appending tail of previous chunk to next chunk's start."""
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            if len(prev) > overlap:
                curr = prev[-overlap:] + curr
            result.append(curr)
        return result
```

- [ ] **Step 2: Write document_loader unit tests**

Add to `tests/test_knowledge_base.py`:

```python

class TestChineseRecursiveTextSplitter:
    """Tests for ChineseRecursiveTextSplitter."""

    @pytest.fixture
    def splitter(self):
        from handlers.knowledge_base.document_loader import ChineseRecursiveTextSplitter
        return ChineseRecursiveTextSplitter()

    def test_short_text_not_split(self, splitter):
        chunks = splitter.split_text("简短文本", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "简短文本"

    def test_empty_text(self, splitter):
        assert splitter.split_text("") == []
        assert splitter.split_text("   ") == []

    def test_split_by_newline_paragraph(self, splitter):
        text = "第一段内容\n\n第二段内容\n\n第三段内容"
        chunks = splitter.split_text(text, chunk_size=10, chunk_overlap=0)
        assert len(chunks) == 3
        assert "第一段内容" in chunks[0]
        assert "第二段内容" in chunks[1]
        assert "第三段内容" in chunks[2]

    def test_split_by_chinese_period(self, splitter):
        text = "红烧肉是一道著名的家常菜。做法简单味道好。深受大家喜爱。"
        chunks = splitter.split_text(text, chunk_size=12, chunk_overlap=0)
        # Each sentence should be its own chunk roughly
        assert len(chunks) >= 2

    def test_split_by_comma_fallback(self, splitter):
        text = "食材：五花肉，冰糖，老抽，料酒，八角，桂皮，香叶，生姜，大葱"
        chunks = splitter.split_text(text, chunk_size=8, chunk_overlap=0)
        assert len(chunks) > 1

    def test_hard_split_last_resort(self, splitter):
        # A string with no separators
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10  # 260 chars
        chunks = splitter.split_text(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 50

    def test_overlap_applied(self, splitter):
        text = "A\n\nB\n\nC\n\nD\n\nE"
        chunks = splitter.split_text(text, chunk_size=10, chunk_overlap=3)
        assert len(chunks) > 1

    def test_recipe_text(self, splitter):
        text = (
            "红烧肉\n\n"
            "食材：五花肉500g，冰糖30g，老抽15ml，料酒30ml\n\n"
            "做法：\n"
            "1. 五花肉切3cm方块，冷水下锅焯水捞出。\n"
            "2. 锅中放少量油，加入冰糖小火炒至焦糖色。\n"
            "3. 加入五花肉翻炒上色，加入老抽、料酒。\n"
            "4. 加入八角、桂皮、香叶，倒入开水没过肉面。\n"
            "5. 大火烧开转小火炖40分钟，收汁即可。"
        )
        chunks = splitter.split_text(text, chunk_size=200, chunk_overlap=30)
        assert len(chunks) >= 1
        # All chunks should contain meaningful content
        for c in chunks:
            assert len(c) > 0


class TestDocumentLoader:
    """Tests for file loading."""

    @pytest.fixture
    def tmpdir(self):
        import tempfile
        d = tempfile.mkdtemp()
        yield Path(d)
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    def test_load_txt(self, tmpdir):
        from handlers.knowledge_base.document_loader import load_file
        f = tmpdir / "test.txt"
        f.write_text("测试菜谱内容", encoding="utf-8")
        assert load_file(f) == "测试菜谱内容"

    def test_load_md(self, tmpdir):
        from handlers.knowledge_base.document_loader import load_file
        f = tmpdir / "test.md"
        f.write_text("# 红烧肉\n\n食材：五花肉", encoding="utf-8")
        text = load_file(f)
        assert "红烧肉" in text
        assert "五花肉" in text

    def test_load_unsupported_extension(self, tmpdir):
        from handlers.knowledge_base.document_loader import load_file
        f = tmpdir / "test.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(f)

    def test_load_nonexistent_file(self, tmpdir):
        from handlers.knowledge_base.document_loader import load_file
        with pytest.raises(FileNotFoundError):
            load_file(tmpdir / "nonexistent.txt")

    def test_load_json_list(self, tmpdir):
        from handlers.knowledge_base.document_loader import load_file
        f = tmpdir / "recipes.json"
        f.write_text(json.dumps([
            {"name": "红烧肉", "ingredients": ["五花肉", "冰糖"], "steps": ["焯水", "炖"]},
            {"name": "麻婆豆腐", "ingredients": ["豆腐", "牛肉末"], "steps": ["炒", "烧"]},
        ]), encoding="utf-8")
        text = load_file(f)
        assert "红烧肉" in text
        assert "麻婆豆腐" in text
        assert "五花肉" in text

    def test_load_json_dict_with_list(self, tmpdir):
        from handlers.knowledge_base.document_loader import load_file
        f = tmpdir / "recipes.json"
        f.write_text(json.dumps({
            "recipes": [
                {"name": "红烧肉", "ingredients": ["五花肉"]},
            ]
        }), encoding="utf-8")
        text = load_file(f)
        assert "红烧肉" in text

    def test_load_json_plain_dict(self, tmpdir):
        from handlers.knowledge_base.document_loader import load_file
        f = tmpdir / "recipe.json"
        f.write_text(json.dumps({
            "name": "清蒸鱼", "ingredients": ["鱼", "姜", "葱"], "steps": ["蒸10分钟"]
        }), encoding="utf-8")
        text = load_file(f)
        assert "清蒸鱼" in text
        assert "蒸10分钟" in text
```

Note: the PDF and DOCX loaders are tested indirectly via integration tests (they need real binary files).

- [ ] **Step 3: Run document_loader tests**

```bash
python -m pytest tests/test_knowledge_base.py::TestChineseRecursiveTextSplitter tests/test_knowledge_base.py::TestDocumentLoader -v
```

Expected: all 15 tests pass.

- [ ] **Step 4: Commit**

```bash
git add handlers/knowledge_base/document_loader.py tests/test_knowledge_base.py
git commit -m "feat: add document loader and ChineseRecursiveTextSplitter"
```

---

### Task 6: Create Vector Store Wrapper (vector_store.py)

**Files:**
- Create: `handlers/knowledge_base/vector_store.py`

**Interfaces:**
- Produces:
  - `VectorStore(kb_name, persist_dir)` — ChromaDB wrapper per KB
  - Methods: `add(texts, metadatas, ids, embeddings)`, `query(query_embedding, top_k, score_threshold) -> list[dict]`, `delete_by_filter(filter)`, `delete_by_ids(ids)`, `count() -> int`, `clear()`

  Where each result dict has: `{content, metadata, score}`

- [ ] **Step 1: Write vector_store.py**

Create `handlers/knowledge_base/vector_store.py`:

```python
"""ChromaDB vector store wrapper — one collection per knowledge base."""
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ChromaDB client singleton (shared across all KB collections)
_client = None


def _get_client(persist_dir: str) -> "chromadb.PersistentClient":
    """Get or create a shared ChromaDB PersistentClient."""
    global _client
    if _client is None:
        import chromadb
        _client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
    return _client


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

    def clear(self) -> None:
        """Delete the entire collection and re-create empty."""
        self.client.delete_collection(name=self.collection_name)
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared collection '%s'", self.collection_name)
```

- [ ] **Step 2: Write vector store tests (using mocked embedding model)**

Add to `tests/test_knowledge_base.py`:

```python

class TestVectorStore:
    """Tests for VectorStore with ChromaDB."""

    @pytest.fixture
    def store(self):
        import tempfile
        from handlers.knowledge_base.vector_store import VectorStore
        tmpdir = tempfile.mkdtemp()
        store = VectorStore("test_kb", Path(tmpdir))
        yield store
        # Cleanup
        store.clear()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_add_and_count(self, store):
        # Use random embeddings for test (512-dim)
        import random
        random.seed(42)
        embeddings = [[random.random() for _ in range(512)] for _ in range(3)]
        store.add(
            texts=["文本一", "文本二", "文本三"],
            metadatas=[
                {"source": "a.txt", "chunk_index": 0, "kb_name": "test_kb"},
                {"source": "a.txt", "chunk_index": 1, "kb_name": "test_kb"},
                {"source": "b.txt", "chunk_index": 0, "kb_name": "test_kb"},
            ],
            ids=["c1", "c2", "c3"],
            embeddings=embeddings,
        )
        assert store.count() == 3

    def test_query_returns_results(self, store):
        import random
        random.seed(42)
        vec1 = [random.random() for _ in range(512)]
        store.add(
            texts=["红烧肉的做法：五花肉焯水后炖煮"],
            metadatas=[{"source": "recipe.txt", "chunk_index": 0, "kb_name": "test_kb"}],
            ids=["c1"],
            embeddings=[vec1],
        )
        # Query with the same vector should return high similarity
        results = store.query(vec1, top_k=3, score_threshold=0.0)
        assert len(results) >= 1
        assert results[0]["score"] > 0.9  # near-exact match

    def test_query_empty_store(self, store):
        results = store.query([0.1] * 512, top_k=3)
        assert results == []

    def test_delete_by_ids(self, store):
        import random
        random.seed(42)
        embeddings = [[random.random() for _ in range(512)] for _ in range(3)]
        store.add(
            texts=["a", "b", "c"],
            metadatas=[{"source": "x.txt", "chunk_index": i, "kb_name": "test_kb"} for i in range(3)],
            ids=["c1", "c2", "c3"],
            embeddings=embeddings,
        )
        store.delete_by_ids(["c1", "c3"])
        assert store.count() == 1

    def test_delete_by_filter(self, store):
        import random
        random.seed(42)
        embeddings = [[random.random() for _ in range(512)] for _ in range(3)]
        store.add(
            texts=["a", "b", "c"],
            metadatas=[
                {"source": "a.txt", "chunk_index": 0, "kb_name": "test_kb"},
                {"source": "b.txt", "chunk_index": 0, "kb_name": "test_kb"},
                {"source": "a.txt", "chunk_index": 1, "kb_name": "test_kb"},
            ],
            ids=["c1", "c2", "c3"],
            embeddings=embeddings,
        )
        store.delete_by_filter({"source": "a.txt"})
        assert store.count() == 1

    def test_clear(self, store):
        import random
        random.seed(42)
        embeddings = [[random.random() for _ in range(512)] for _ in range(2)]
        store.add(
            texts=["a", "b"],
            metadatas=[{"source": "x.txt", "chunk_index": i, "kb_name": "test_kb"} for i in range(2)],
            ids=["c1", "c2"],
            embeddings=embeddings,
        )
        store.clear()
        assert store.count() == 0
```

- [ ] **Step 3: Run vector store tests**

```bash
python -m pytest tests/test_knowledge_base.py::TestVectorStore -v
```

Expected: all 6 tests pass.

- [ ] **Step 4: Commit**

```bash
git add handlers/knowledge_base/vector_store.py tests/test_knowledge_base.py
git commit -m "feat: add ChromaDB vector store wrapper"
```

---

### Task 7: Create Search Module (search.py)

**Files:**
- Create: `handlers/knowledge_base/search.py`

**Interfaces:**
- Produces:
  - `SearchResult` — dataclass with `content`, `metadata`, `score`
  - `search_knowledge_base(query, kb_name, top_k, score_threshold) -> list[SearchResult]`
  - `format_search_results(results) -> str`

- [ ] **Step 1: Write search.py**

Create `handlers/knowledge_base/search.py`:

```python
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
```

- [ ] **Step 2: Write search unit tests**

Add to `tests/test_knowledge_base.py`:

```python

class TestFormatSearchResults:
    """Tests for search result formatting."""

    def test_empty_results(self):
        from handlers.knowledge_base.search import format_search_results
        text = format_search_results([])
        assert "未找到相关内容" in text

    def test_single_result(self):
        from handlers.knowledge_base.search import format_search_results, SearchResult
        results = [SearchResult(
            content="红烧肉做法",
            metadata={"source": "recipe.pdf", "chunk_index": 0, "kb_name": "test"},
            score=0.92,
        )]
        text = format_search_results(results)
        assert "recipe.pdf" in text
        assert "红烧肉做法" in text
        assert "0.92" in text

    def test_multiple_results_sorted(self):
        from handlers.knowledge_base.search import format_search_results, SearchResult
        results = [
            SearchResult("内容A", {"source": "a.txt"}, 0.85),
            SearchResult("内容B", {"source": "b.txt"}, 0.95),
        ]
        text = format_search_results(results)
        # Results are formatted as-is (sorting is done upstream)
        assert "内容A" in text
        assert "内容B" in text
```

- [ ] **Step 3: Commit (tests will be run in Task 8 after kb_manager is created)**

```bash
git add handlers/knowledge_base/search.py tests/test_knowledge_base.py
git commit -m "feat: add KB search module with result formatting"
```

---

### Task 8: Create Knowledge Base Manager (kb_manager.py + __init__.py)

**Files:**
- Create: `handlers/knowledge_base/kb_manager.py`
- Create: `handlers/knowledge_base/__init__.py`

**Interfaces:**
- Produces:
  - `kb_manager` — singleton `KnowledgeBaseManager` instance
  - `KnowledgeBaseManager` class with methods from design spec §4.1
  - `__init__.py` re-exports `kb_manager`, `KnowledgeBaseManager`, `SearchResult`, `search_knowledge_base`, `format_search_results`

- [ ] **Step 1: Write kb_manager.py**

Create `handlers/knowledge_base/kb_manager.py`:

```python
"""Knowledge Base Manager — facade for KB lifecycle and document management."""
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

    # ── KB CRUD ───────────────────────────────────────────────────────

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

    # ── Document Management ───────────────────────────────────────────

    def add_documents(
        self,
        kb_name: str,
        file_paths: List[Path],
        chunk_size: int = KB_DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = KB_DEFAULT_CHUNK_OVERLAP,
    ) -> int:
        """Add documents to a knowledge base: load → chunk → embed → store.

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
            db_mod.update_kb_doc_count(kb_name)

            total_chunks += len(chunks)
            logger.info("Indexed %s: %d chunks", filename, len(chunks))

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

    # ── Internal helpers ──────────────────────────────────────────────

    def _get_vector_store(self, kb_name: str):
        """Get or create the VectorStore for a KB."""
        from handlers.knowledge_base.vector_store import VectorStore
        persist_dir = KB_ROOT_PATH / kb_name / KB_CHROMA_DIR_NAME
        persist_dir.mkdir(parents=True, exist_ok=True)
        return VectorStore(kb_name, persist_dir)


# ── Singleton ─────────────────────────────────────────────────────────

kb_manager = KnowledgeBaseManager()
```

- [ ] **Step 2: Write __init__.py**

Create `handlers/knowledge_base/__init__.py`:

```python
"""Knowledge Base subsystem for RAG-powered recipe search."""
from handlers.knowledge_base.kb_manager import KnowledgeBaseManager, kb_manager
from handlers.knowledge_base.search import (
    SearchResult,
    search_knowledge_base,
    format_search_results,
)

__all__ = [
    "KnowledgeBaseManager",
    "kb_manager",
    "SearchResult",
    "search_knowledge_base",
    "format_search_results",
]
```

- [ ] **Step 3: Write KB manager integration tests**

Add to `tests/test_knowledge_base.py`:

```python

class TestKnowledgeBaseManager:
    """Integration tests for KB manager (uses real ChromaDB + embedding model)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import tempfile
        import handlers.knowledge_base.db as db_mod
        from settings import constant
        self._orig_kb_root = constant.KB_ROOT_PATH
        self._orig_kb_db = constant.KB_DB_PATH
        self.tmpdir = Path(tempfile.mkdtemp())
        constant.KB_ROOT_PATH = self.tmpdir
        constant.KB_DB_PATH = self.tmpdir / "info.db"
        db_mod.db_path = constant.KB_DB_PATH
        from handlers.knowledge_base.kb_manager import KnowledgeBaseManager
        self.mgr = KnowledgeBaseManager()
        self.mgr._init_lock = False
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        constant.KB_ROOT_PATH = self._orig_kb_root
        constant.KB_DB_PATH = self._orig_kb_db
        db_mod.db_path = self._orig_kb_db

    def test_create_and_list_kb(self):
        info = self.mgr.create_kb("test_kb", "测试知识库")
        assert info["name"] == "test_kb"
        assert info["description"] == "测试知识库"
        kbs = self.mgr.list_kbs()
        assert len(kbs) == 1

    def test_create_duplicate_kb_raises(self):
        self.mgr.create_kb("test_kb")
        with pytest.raises(ValueError, match="已存在"):
            self.mgr.create_kb("test_kb")

    def test_invalid_kb_name(self):
        with pytest.raises(ValueError):
            self.mgr.create_kb("invalid/name")
        with pytest.raises(ValueError):
            self.mgr.create_kb("invalid\\name")
        with pytest.raises(ValueError):
            self.mgr.create_kb("")

    def test_get_kb(self):
        self.mgr.create_kb("test_kb")
        assert self.mgr.get_kb("test_kb") is not None
        assert self.mgr.get_kb("nonexistent") is None

    def test_delete_kb(self):
        self.mgr.create_kb("test_kb")
        self.mgr.delete_kb("test_kb")
        assert len(self.mgr.list_kbs()) == 0

    def test_delete_nonexistent_kb_raises(self):
        with pytest.raises(ValueError, match="不存在"):
            self.mgr.delete_kb("nonexistent")

    def test_add_documents_txt(self):
        self.mgr.create_kb("test_kb")
        f = self.tmpdir / "recipe.txt"
        f.write_text("红烧肉做法：五花肉焯水后加入冰糖炒色，小火炖40分钟。", encoding="utf-8")
        chunk_count = self.mgr.add_documents("test_kb", [f])
        assert chunk_count >= 1
        docs = self.mgr.list_documents("test_kb")
        assert len(docs) == 1
        assert docs[0]["filename"] == "recipe.txt"

    def test_add_documents_md(self):
        self.mgr.create_kb("test_kb")
        f = self.tmpdir / "recipe.md"
        f.write_text("# 麻婆豆腐\n\n## 食材\n豆腐、牛肉末、豆瓣酱\n\n## 做法\n1. 炒肉末\n2. 加豆瓣酱\n3. 烧豆腐", encoding="utf-8")
        chunk_count = self.mgr.add_documents("test_kb", [f])
        assert chunk_count >= 1

    def test_remove_documents(self):
        self.mgr.create_kb("test_kb")
        f = self.tmpdir / "recipe.txt"
        f.write_text("红烧肉做法", encoding="utf-8")
        self.mgr.add_documents("test_kb", [f])
        self.mgr.remove_documents("test_kb", ["recipe.txt"])
        docs = self.mgr.list_documents("test_kb")
        assert len(docs) == 0

    def test_add_documents_unsupported_format(self):
        self.mgr.create_kb("test_kb")
        f = self.tmpdir / "data.bin"
        f.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="不支持"):
            self.mgr.add_documents("test_kb", [f])

    def test_add_documents_kb_not_found(self):
        f = self.tmpdir / "recipe.txt"
        f.write_text("test")
        with pytest.raises(ValueError, match="不存在"):
            self.mgr.add_documents("nonexistent", [f])

    def test_list_documents_kb_not_found(self):
        with pytest.raises(ValueError, match="不存在"):
            self.mgr.list_documents("nonexistent")

    def test_full_workflow_with_search(self):
        """End-to-end: create KB → add doc → search → delete KB."""
        from handlers.knowledge_base.search import search_knowledge_base

        self.mgr.create_kb("test_kb")
        f = self.tmpdir / "recipes.txt"
        f.write_text(
            "红烧肉是一道经典的中式菜肴。主要食材包括五花肉、冰糖、老抽、料酒。"
            "做法是先将五花肉焯水，然后用冰糖炒糖色，加入调料炖煮40分钟。",
            encoding="utf-8",
        )
        self.mgr.add_documents("test_kb", [f])

        results = search_knowledge_base("红烧肉怎么做", kb_name="test_kb", top_k=2)
        assert len(results) >= 1
        assert "五花肉" in results[0].content

        # Search across all KBs
        results_all = search_knowledge_base("红烧肉", top_k=2)
        assert len(results_all) >= 1

        self.mgr.delete_kb("test_kb")
```

- [ ] **Step 4: Run all KB tests**

```bash
python -m pytest tests/test_knowledge_base.py -v
```

Expected: all tests pass (~40 tests across all classes).

- [ ] **Step 5: Commit**

```bash
git add handlers/knowledge_base/kb_manager.py handlers/knowledge_base/__init__.py tests/test_knowledge_base.py
git commit -m "feat: add KnowledgeBaseManager facade + __init__ exports"
```

---

### Task 9: Add search_recipe_knowledge_base to Tools

**Files:**
- Modify: `tools.py`

**Interfaces:**
- Consumes: `search_knowledge_base`, `format_search_results` from `handlers/knowledge_base`
- Produces: New entry in `NATIVE_HANDLERS` and `NATIVE_TOOLS`

- [ ] **Step 1: Add import and handler function to tools.py**

At the top of `tools.py`, add the import after the existing imports (after line 9, `from settings.constant import WORKDIR`):

```python
from handlers.knowledge_base.search import search_knowledge_base, format_search_results
```

Add the handler function before the `NATIVE_HANDLERS` dict (before line 235, `NATIVE_HANDLERS = {`):

```python
def search_recipe_knowledge_base(query: str, kb_name: str | None = None) -> dict:
    """搜索本地菜谱知识库，查找用户上传的私人或本地菜谱内容。

    当用户询问中餐、家常菜、或提到"我的菜谱""本地菜谱""家传"等关键词时优先使用。

    Args:
        query: 搜索查询，如"红烧肉做法"、"川菜麻婆豆腐"。
        kb_name: 指定知识库名称（可选，不指定则搜索所有知识库）。

    Returns:
        {"results": [{"content": ..., "source": ..., "score": ...}], "total": N}
    """
    try:
        results = search_knowledge_base(query, kb_name=kb_name)
        return {
            "results": [
                {
                    "content": r.content,
                    "source": r.metadata.get("source", "未知"),
                    "score": r.score,
                }
                for r in results
            ],
            "total": len(results),
        }
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception("搜索知识库失败")
        return {"error": str(e), "results": [], "total": 0}
```

- [ ] **Step 2: Add handler to NATIVE_HANDLERS dict**

In `NATIVE_HANDLERS`, add after the existing meal/drink handlers (after the `"random_cocktail": random_cocktail,` line):

```python
    "search_recipe_knowledge_base": lambda **kw: search_recipe_knowledge_base(
        kw["query"], kw.get("kb_name")
    ),
```

- [ ] **Step 3: Add tool definition to NATIVE_TOOLS list**

In `NATIVE_TOOLS`, add after the last tool definition (`random_cocktail` block ends with `}`):

```python
    {
        "name": "search_recipe_knowledge_base",
        "description": "搜索本地菜谱知识库，查找用户上传的私房菜谱、家传菜谱或本地导入的菜谱文档。当用户询问中餐、家常菜、或提到'我的菜谱''本地菜谱''家传'等关键词时优先使用。与TheMealDB的search_meals互补——中式/私人菜谱走知识库，西餐/鸡尾酒走TheMealDB。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询，如'红烧肉做法'、'川菜麻婆豆腐'"
                },
                "kb_name": {
                    "type": "string",
                    "description": "指定知识库名称，不指定则搜索所有知识库"
                }
            },
            "required": ["query"]
        }
    },
```

- [ ] **Step 4: Verify tools.py compiles**

```bash
python -c "from tools import NATIVE_HANDLERS, NATIVE_TOOLS; print(f'Handlers: {len(NATIVE_HANDLERS)}, Tools: {len(NATIVE_TOOLS)}'); assert 'search_recipe_knowledge_base' in NATIVE_HANDLERS; print('OK')"
```

Expected output: `Handlers: 20, Tools: 20` (count may vary) and `OK`.

- [ ] **Step 5: Commit**

```bash
git add tools.py
git commit -m "feat: add search_recipe_knowledge_base tool for LLM"
```

---

### Task 10: Add Flask API Endpoints for KB Management

**Files:**
- Modify: `app.py`

**Interfaces:**
- Produces: 6 new Flask routes for KB CRUD + document management

- [ ] **Step 1: Add import and init KB manager in app.py**

After `from tools import NATIVE_TOOLS` (line 18), add:

```python
from handlers.knowledge_base import kb_manager
```

After `_agent = RecipeAgent(_client, _tools)` (line 29), add:

```python
# ── 初始化知识库 ──────────────────────────────────────────────────────
KB_ROOT_PATH.mkdir(parents=True, exist_ok=True)  # KB_ROOT_PATH already imported or create inline
```

Actually, KB_ROOT_PATH should be imported. Update the import line at line 17 `from settings.constant import MODEL` to:

```python
from settings.constant import MODEL, KB_ROOT_PATH, KB_MAX_FILE_SIZE_BYTES, KB_SUPPORTED_EXTENSIONS
```

Wait — re-check: line 17 already has `from settings.constant import MODEL`. I'll modify it.

Change line 17 from:
```python
from settings.constant import MODEL
```
to:
```python
from settings.constant import MODEL, KB_ROOT_PATH, KB_MAX_FILE_SIZE_BYTES, KB_SUPPORTED_EXTENSIONS
```

Add after line 29 (`_agent = RecipeAgent(_client, _tools)`):
```python
KB_ROOT_PATH.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Add KB API routes**

After the `health()` route (after line 82 `return jsonify({"status": "ok"})`), add the following routes:

```python
# ── 知识库管理 API ─────────────────────────────────────────────────────

@app.route("/api/kb/create", methods=["POST"])
def kb_create():
    """创建知识库。请求体: {"name": "...", "description": "..."}"""
    try:
        data = request.get_json(force=True)
        if not data or "name" not in data:
            return jsonify({"error": "缺少知识库名称"}), 400
        info = kb_manager.create_kb(data["name"], data.get("description", ""))
        return jsonify({"status": "ok", "kb": info})
    except ValueError as e:
        return jsonify({"error": str(e)}), 409 if "已存在" in str(e) else 400
    except Exception as e:
        logging.exception("创建知识库失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kb/list", methods=["GET"])
def kb_list():
    """列出所有知识库。"""
    try:
        kbs = kb_manager.list_kbs()
        return jsonify({"status": "ok", "knowledge_bases": kbs})
    except Exception as e:
        logging.exception("列出知识库失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kb/<name>", methods=["DELETE"])
def kb_delete(name):
    """删除知识库。"""
    try:
        kb_manager.delete_kb(name)
        return jsonify({"status": "ok", "message": f"知识库 {name} 已删除"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception("删除知识库失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kb/<name>/upload", methods=["POST"])
def kb_upload(name):
    """上传文档到知识库。multipart/form-data, files 字段。"""
    try:
        if "files" not in request.files:
            return jsonify({"error": "缺少 files 字段"}), 400

        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "未选择文件"}), 400

        # Validate file sizes and extensions
        for f in files:
            if f.filename == "":
                continue
            ext = Path(f.filename).suffix.lower()
            if ext not in KB_SUPPORTED_EXTENSIONS:
                return jsonify({
                    "error": f"不支持的文件格式: {ext}。支持: {', '.join(KB_SUPPORTED_EXTENSIONS)}"
                }), 400
            # Read file content to check size
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(0)
            if size > KB_MAX_FILE_SIZE_BYTES:
                return jsonify({
                    "error": f"文件 {f.filename} 过大 ({size / 1024 / 1024:.1f}MB)，"
                             f"最大 {KB_MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB"
                }), 413

        # Save files to temp location
        import tempfile
        saved_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for f in files:
                if f.filename == "":
                    continue
                dest = tmp_path / f.filename
                f.save(str(dest))
                saved_paths.append(dest)

            # Process documents
            import time
            t0 = time.time()
            chunk_count = kb_manager.add_documents(name, saved_paths)
            duration = time.time() - t0

        return jsonify({
            "status": "ok",
            "message": f"成功处理 {len(saved_paths)} 个文件，生成 {chunk_count} 个切片",
            "file_count": len(saved_paths),
            "chunk_count": chunk_count,
            "duration_sec": round(duration, 2),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404 if "不存在" in str(e) else 400
    except Exception as e:
        logging.exception("上传文档失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kb/<name>/docs", methods=["GET"])
def kb_list_docs(name):
    """列出知识库中的所有文档。"""
    try:
        docs = kb_manager.list_documents(name)
        return jsonify({"status": "ok", "documents": docs})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception("列出文档失败")
        return jsonify({"error": str(e)}), 500


@app.route("/api/kb/<name>/docs", methods=["DELETE"])
def kb_delete_docs(name):
    """删除知识库中的文档。请求体: {"filenames": ["a.pdf", "b.txt"]}"""
    try:
        data = request.get_json(force=True)
        if not data or "filenames" not in data:
            return jsonify({"error": "缺少 filenames 字段"}), 400
        kb_manager.remove_documents(name, data["filenames"])
        return jsonify({"status": "ok", "message": f"已删除 {len(data['filenames'])} 个文档"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception("删除文档失败")
        return jsonify({"error": str(e)}), 500
```

- [ ] **Step 3: Verify app.py compiles and imports correctly**

```bash
python -c "from app import app; print('App imported OK, routes:', len([r for r in app.url_map.iter_rules()]))"
```

Expected: App imports OK with new routes registered.

- [ ] **Step 4: Write Flask API endpoint tests**

Add to `tests/test_knowledge_base.py`:

```python

class TestFlaskKBAPI:
    """Tests for Flask KB management endpoints."""

    @pytest.fixture
    def client(self):
        import tempfile
        import handlers.knowledge_base.db as db_mod
        from settings import constant
        # Reset paths BEFORE any kb_manager method is called
        import handlers.knowledge_base.db as db_mod
        from settings import constant
        self._orig_kb_root = constant.KB_ROOT_PATH
        self._orig_kb_db = constant.KB_DB_PATH
        self.tmpdir = Path(tempfile.mkdtemp())
        constant.KB_ROOT_PATH = self.tmpdir
        constant.KB_DB_PATH = self.tmpdir / "info.db"
        db_mod.init_db(constant.KB_DB_PATH)  # Init with test path

        # Replace kb_manager singleton with fresh instance for test isolation
        import handlers.knowledge_base.kb_manager as kb_mod
        from handlers.knowledge_base.kb_manager import KnowledgeBaseManager
        kb_mod.kb_manager = KnowledgeBaseManager()

        from app import app
        app.config["TESTING"] = True
        # Update app's kb_manager reference to use test singleton
        import app as app_module
        app_module.kb_manager = kb_mod.kb_manager
        with app.test_client() as client:
            yield client

        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        constant.KB_ROOT_PATH = self._orig_kb_root
        constant.KB_DB_PATH = self._orig_kb_db
        db_mod.db_path = self._orig_kb_db

    def test_create_kb(self, client):
        resp = client.post("/api/kb/create",
                          json={"name": "test_kb", "description": "测试"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["kb"]["name"] == "test_kb"

    def test_create_duplicate_kb(self, client):
        client.post("/api/kb/create", json={"name": "test_kb"})
        resp = client.post("/api/kb/create", json={"name": "test_kb"})
        assert resp.status_code == 409

    def test_create_kb_invalid_name(self, client):
        resp = client.post("/api/kb/create", json={"name": "bad/name"})
        assert resp.status_code in (400, 409)

    def test_list_kbs(self, client):
        client.post("/api/kb/create", json={"name": "kb1"})
        client.post("/api/kb/create", json={"name": "kb2"})
        resp = client.get("/api/kb/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["knowledge_bases"]) >= 2

    def test_delete_kb(self, client):
        client.post("/api/kb/create", json={"name": "test_kb"})
        resp = client.delete("/api/kb/test_kb")
        assert resp.status_code == 200
        # Verify deleted
        resp2 = client.get("/api/kb/list")
        assert len(resp2.get_json()["knowledge_bases"]) == 0

    def test_delete_nonexistent_kb(self, client):
        resp = client.delete("/api/kb/nonexistent")
        assert resp.status_code == 404

    def test_upload_txt(self, client):
        client.post("/api/kb/create", json={"name": "test_kb"})
        data = {"files": (BytesIO("红烧肉做法测试".encode("utf-8")), "recipe.txt")}
        resp = client.post("/api/kb/test_kb/upload",
                          data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["chunk_count"] >= 1

    def test_upload_unsupported_format(self, client):
        client.post("/api/kb/create", json={"name": "test_kb"})
        data = {"files": (BytesIO(b"\x00\x01\x02"), "data.bin")}
        resp = client.post("/api/kb/test_kb/upload",
                          data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_upload_to_nonexistent_kb(self, client):
        data = {"files": (BytesIO(b"test"), "recipe.txt")}
        resp = client.post("/api/kb/nonexistent/upload",
                          data=data, content_type="multipart/form-data")
        assert resp.status_code == 404

    def test_list_docs(self, client):
        client.post("/api/kb/create", json={"name": "test_kb"})
        client.post("/api/kb/test_kb/upload",
                   data={"files": (BytesIO("测试内容".encode("utf-8")), "r.txt")},
                   content_type="multipart/form-data")
        resp = client.get("/api/kb/test_kb/docs")
        assert resp.status_code == 200
        docs = resp.get_json()["documents"]
        assert len(docs) == 1
        assert docs[0]["filename"] == "r.txt"

    def test_list_docs_nonexistent_kb(self, client):
        resp = client.get("/api/kb/nonexistent/docs")
        assert resp.status_code == 404

    def test_delete_docs(self, client):
        client.post("/api/kb/create", json={"name": "test_kb"})
        client.post("/api/kb/test_kb/upload",
                   data={"files": (BytesIO("测试".encode("utf-8")), "r.txt")},
                   content_type="multipart/form-data")
        resp = client.delete("/api/kb/test_kb/docs",
                            json={"filenames": ["r.txt"]})
        assert resp.status_code == 200
        # Verify deleted
        resp2 = client.get("/api/kb/test_kb/docs")
        assert len(resp2.get_json()["documents"]) == 0
```

Don't forget to add the required imports at the top of the test file. Update the top of `tests/test_knowledge_base.py` to include:

```python
"""Tests for RAG knowledge base subsystem."""
import json
import pytest
from io import BytesIO
from pathlib import Path
```

- [ ] **Step 5: Run API endpoint tests**

```bash
python -m pytest tests/test_knowledge_base.py::TestFlaskKBAPI -v
```

Expected: all 12 API tests pass.

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest tests/test_knowledge_base.py -v
```

Expected: all tests pass (50+ tests across all test classes).

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_knowledge_base.py
git commit -m "feat: add Flask API endpoints for KB management"
```

---

### Task 11: Manual Verification (no test file — run once manually)

**Goal:** Confirm the full tool chain works end-to-end with real data.

- [ ] **Step 1: Start the Flask app**

```bash
python app.py
```

Expected: App starts on port 5000 without import errors.

- [ ] **Step 2: Create a KB via curl (separate terminal)**

```bash
curl -X POST http://localhost:5000/api/kb/create \
  -H "Content-Type: application/json" \
  -d '{"name": "my_recipes", "description": "我的私房菜谱"}'
```

Expected: `{"kb": {"name": "my_recipes", ...}, "status": "ok"}`

- [ ] **Step 3: Upload a test document**

```bash
echo "# 宫保鸡丁
食材：鸡胸肉、花生米、干辣椒、花椒、葱、姜、蒜
调料：酱油、醋、糖、淀粉、料酒
做法：
1. 鸡胸肉切丁，加料酒、淀粉腌制15分钟
2. 调汁：酱油、醋、糖、淀粉加水调匀
3. 热锅凉油，下干辣椒花椒爆香
4. 加入鸡丁翻炒至变色
5. 倒入调好的汁，加花生米翻炒均匀
6. 出锅装盘" > /tmp/gongbao_recipe.md

curl -X POST http://localhost:5000/api/kb/my_recipes/upload \
  -F "files=@/tmp/gongbao_recipe.md"
```

Expected: Returns `chunk_count` >= 1, status `ok`.

- [ ] **Step 4: List KBs and docs**

```bash
curl http://localhost:5000/api/kb/list
curl http://localhost:5000/api/kb/my_recipes/docs
```

Expected: Shows the KB and document.

- [ ] **Step 5: Test tool directly in Python**

```bash
python -c "
from handlers.knowledge_base import search_knowledge_base, format_search_results
results = search_knowledge_base('宫保鸡丁怎么做', kb_name='my_recipes')
print(format_search_results(results))
"
```

Expected: Prints formatted search results containing 宫保鸡丁 recipe content.

- [ ] **Step 6: Verify ChromaDB persistence**

Stop the Flask app (Ctrl+C), restart it, and run the search again — results should be the same (data persisted).

- [ ] **Step 7: Clean up test KB**

```bash
curl -X DELETE http://localhost:5000/api/kb/my_recipes
```

- [ ] **Step 8: Commit any remaining changes**

```bash
git status
git add -A
git commit -m "chore: finalize RAG knowledge base integration"
```

---

## Verification Checklist

Before declaring this plan complete, verify:

1. `python -c "from handlers.knowledge_base import kb_manager, search_knowledge_base, format_search_results"` — no import errors
2. `python -m pytest tests/test_knowledge_base.py -v` — all tests pass
3. Flask app starts without errors: `python app.py`
4. All 6 KB API endpoints work via curl
5. `search_recipe_knowledge_base` is registered in `NATIVE_HANDLERS` and `NATIVE_TOOLS`
6. ChromaDB data persists across app restarts
7. KB name validation blocks path traversal attempts
8. Unsupported file formats return 400 error

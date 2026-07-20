"""Tests for RAG knowledge base subsystem."""
import importlib
import json
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import pytest

# Make handlers.knowledge_base importable
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDB:
    """Tests for handlers/knowledge_base/db.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        import handlers.knowledge_base.db as db_mod
        db_mod.init_db(self.db_path)
        self.db = db_mod
        yield
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
        a = EmbeddingModel("qwen3-embedding:0.6b")
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
        assert m.dimension > 0  # auto-detected from API response

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
        assert len(vec) == m.dimension
        assert m.dimension > 0


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


class TestVectorStore:
    """Tests for VectorStore with ChromaDB."""

    @pytest.fixture
    def store(self):
        from handlers.knowledge_base.vector_store import VectorStore
        tmpdir = tempfile.mkdtemp()
        store = VectorStore("test_kb", Path(tmpdir))
        yield store
        # Cleanup
        store.clear()
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_add_and_count(self, store):
        import random
        random.seed(42)
        embeddings = [[random.random() for _ in range(1024)] for _ in range(3)]
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
        vec1 = [random.random() for _ in range(1024)]
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
        results = store.query([0.1] * 1024, top_k=3)
        assert results == []

    def test_delete_by_ids(self, store):
        import random
        random.seed(42)
        embeddings = [[random.random() for _ in range(1024)] for _ in range(3)]
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
        embeddings = [[random.random() for _ in range(1024)] for _ in range(3)]
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
        embeddings = [[random.random() for _ in range(1024)] for _ in range(2)]
        store.add(
            texts=["a", "b"],
            metadatas=[{"source": "x.txt", "chunk_index": i, "kb_name": "test_kb"} for i in range(2)],
            ids=["c1", "c2"],
            embeddings=embeddings,
        )
        store.clear()
        assert store.count() == 0


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


class TestKnowledgeBaseManager:
    """Integration tests for KB manager (uses real ChromaDB + embedding model)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import sys as _sys
        import tempfile
        import handlers.knowledge_base.db as db_mod
        from settings import constant
        self._orig_kb_root = constant.KB_ROOT_PATH
        self._orig_kb_db = constant.KB_DB_PATH
        self.tmpdir = Path(tempfile.mkdtemp())
        constant.KB_ROOT_PATH = self.tmpdir
        constant.KB_DB_PATH = self.tmpdir / "info.db"
        db_mod.db_path = constant.KB_DB_PATH
        # Override kb_manager module-level constants.
        # NOTE: __init__.py shadows the module with the singleton instance,
        # so we access the module via sys.modules.
        km_mod = _sys.modules["handlers.knowledge_base.kb_manager"]
        km_mod.KB_ROOT_PATH = self.tmpdir
        km_mod.KB_DB_PATH = self.tmpdir / "info.db"
        # Reset VectorStore client singleton to force fresh ChromaDB
        import handlers.knowledge_base.vector_store as vs_mod
        vs_mod._client = None
        from handlers.knowledge_base.kb_manager import KnowledgeBaseManager
        self.mgr = KnowledgeBaseManager()
        self.mgr._init_lock = False
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        constant.KB_ROOT_PATH = self._orig_kb_root
        constant.KB_DB_PATH = self._orig_kb_db
        db_mod.db_path = self._orig_kb_db
        km_mod.KB_ROOT_PATH = self._orig_kb_root
        km_mod.KB_DB_PATH = self._orig_kb_db

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
        """End-to-end: create KB -> add doc -> search -> delete KB."""
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


class TestFlaskKBAPI:
    """Tests for Flask KB management endpoints."""

    @pytest.fixture
    def client(self):
        import sys as _sys
        import handlers.knowledge_base.db as db_mod
        from settings import constant
        # Reset paths BEFORE any kb_manager method is called
        self._orig_kb_root = constant.KB_ROOT_PATH
        self._orig_kb_db = constant.KB_DB_PATH
        self.tmpdir = Path(tempfile.mkdtemp())
        constant.KB_ROOT_PATH = self.tmpdir
        constant.KB_DB_PATH = self.tmpdir / "info.db"
        db_mod.init_db(constant.KB_DB_PATH)  # Init with test path

        # Mock embedding model to avoid HF download (tests need CRUD, not search)
        import random as _random
        _random.seed(42)
        import handlers.knowledge_base.embedding as emb_mod
        orig_new = emb_mod.EmbeddingModel.__new__
        orig_load = emb_mod.EmbeddingModel.load
        def _mock_new(cls, model_name="BAAI/bge-small-zh-v1.5"):
            instance = super(emb_mod.EmbeddingModel, cls).__new__(cls)
            instance._model_name = model_name
            instance._loaded = True
            instance._model = None
            return instance
        def _mock_load(self):
            pass
        emb_mod.EmbeddingModel.__new__ = _mock_new
        emb_mod.EmbeddingModel.load = _mock_load
        # Patch embed methods to return fixed-dim random vectors
        _orig_embed = emb_mod.EmbeddingModel.embed
        _orig_dim = emb_mod.EmbeddingModel.dimension.fget if isinstance(
            emb_mod.EmbeddingModel.__dict__.get('dimension'), property
        ) else None
        def _mock_embed(self, texts):
            if not texts:
                return []
            return [[_random.random() for _ in range(1024)] for _ in range(len(texts))]
        def _mock_dim(self):
            return 1024
        emb_mod.EmbeddingModel.embed = _mock_embed
        emb_mod.EmbeddingModel.dimension = property(_mock_dim)
        emb_mod.EmbeddingModel.reset = lambda: None

        # Replace kb_manager singleton with fresh instance for test isolation.
        # NOTE: import ... as kb_mod resolves to the singleton (not the module)
        # because __init__.py shadows the module name. Use sys.modules instead.
        km_mod = _sys.modules["handlers.knowledge_base.kb_manager"]
        from handlers.knowledge_base.kb_manager import KnowledgeBaseManager
        # Override module-level constants so _ensure_init uses test paths
        km_mod.KB_ROOT_PATH = self.tmpdir
        km_mod.KB_DB_PATH = self.tmpdir / "info.db"
        km_mod.kb_manager = KnowledgeBaseManager()
        # Also update the package-level reference
        import handlers.knowledge_base as kb_pkg
        kb_pkg.kb_manager = km_mod.kb_manager

        from app import app
        app.config["TESTING"] = True
        # Update app's kb_manager reference to use test singleton
        import app as app_module
        app_module.kb_manager = km_mod.kb_manager
        with app.test_client() as client:
            yield client

        shutil.rmtree(self.tmpdir, ignore_errors=True)
        constant.KB_ROOT_PATH = self._orig_kb_root
        constant.KB_DB_PATH = self._orig_kb_db
        db_mod.db_path = self._orig_kb_db
        if km_mod:
            km_mod.KB_ROOT_PATH = self._orig_kb_root
            km_mod.KB_DB_PATH = self._orig_kb_db

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


# ══════════════════════════════════════════════════════════════════════
# BM25 + Reranker + Hybrid Search tests
# ══════════════════════════════════════════════════════════════════════

class TestBM25:
    """Tests for BM25 sparse retrieval."""

    def test_tokenize(self):
        from handlers.knowledge_base.bm25 import _tokenize
        tokens = _tokenize("红烧肉的做法")
        assert len(tokens) >= 2
        assert "红烧肉" in tokens or "红烧" in tokens

    def test_build_index(self):
        from handlers.knowledge_base.bm25 import build_bm25_from_docs
        docs = ["红烧肉是一道著名的家常菜", "麻婆豆腐是四川名菜"]
        idx = build_bm25_from_docs(docs)
        assert idx is not None

    def test_build_empty_index(self):
        from handlers.knowledge_base.bm25 import build_bm25_from_docs
        assert build_bm25_from_docs([]) is None

    def test_bm25_search(self):
        from handlers.knowledge_base.bm25 import build_bm25_from_docs, bm25_search
        docs = [
            "红烧肉：五花肉焯水后加冰糖炒色",
            "麻婆豆腐：豆腐加牛肉末和豆瓣酱",
            "清蒸鱼：鱼加姜葱蒸10分钟",
        ]
        idx = build_bm25_from_docs(docs)
        results = bm25_search(idx, "红烧肉做法", top_k=2)
        assert len(results) >= 1
        # First doc should be about 红烧肉
        assert results[0][0] == 0

    def test_bm25_empty_query(self):
        from handlers.knowledge_base.bm25 import build_bm25_from_docs, bm25_search
        docs = ["测试文档"]
        idx = build_bm25_from_docs(docs)
        results = bm25_search(idx, "", top_k=5)
        assert results == []


class TestReranker:
    """Tests for Cross-Encoder reranker (falls back gracefully)."""

    def test_singleton(self):
        from handlers.knowledge_base.reranker import RerankerModel
        RerankerModel.reset()
        a = RerankerModel()
        b = RerankerModel()
        assert a is b

    def test_rerank_fallback(self):
        """Without HF access, reranker falls back to identity ranking."""
        from handlers.knowledge_base.reranker import RerankerModel
        RerankerModel.reset()
        m = RerankerModel()
        docs = ["doc one", "doc two", "doc three"]
        results = m.rerank("test query", docs)
        assert len(results) == len(docs)
        indices = [r[0] for r in results]
        assert indices == list(range(len(docs)))  # identity order preserved

    def test_rerank_empty(self):
        from handlers.knowledge_base.reranker import RerankerModel
        RerankerModel.reset()
        m = RerankerModel()
        assert m.rerank("query", []) == []

    def test_available_property(self):
        from handlers.knowledge_base.reranker import RerankerModel
        RerankerModel.reset()
        m = RerankerModel()
        # available property works (True if model loaded, False if not)
        assert isinstance(m.available, bool)


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion logic."""

    def test_rrf_fusion_combines_dense_and_sparse(self):
        from handlers.knowledge_base.search import _rrf_fusion, SearchResult

        dense = [
            SearchResult(content="doc A", metadata={"kb_name": "kb1", "source": "a.txt", "chunk_index": 0}, score=0.9),
            SearchResult(content="doc B", metadata={"kb_name": "kb1", "source": "a.txt", "chunk_index": 1}, score=0.5),
        ]
        sparse = [(2, 2.5)]  # index=2 → doc C
        all_docs = [
            {"content": "doc A", "metadata": {"kb_name": "kb1", "source": "a.txt", "chunk_index": 0}},
            {"content": "doc B", "metadata": {"kb_name": "kb1", "source": "a.txt", "chunk_index": 1}},
            {"content": "doc C", "metadata": {"kb_name": "kb1", "source": "b.txt", "chunk_index": 0}},
        ]

        result = _rrf_fusion(dense, sparse, all_docs)
        assert len(result) >= 2  # doc A + doc B (from dense) + doc C (from sparse)
        contents = {r.content for r in result}
        assert "doc A" in contents
        assert "doc C" in contents  # should be included from BM25

    def test_rrf_sparse_brings_new_docs(self):
        from handlers.knowledge_base.search import _rrf_fusion, SearchResult

        dense = [
            SearchResult(content="doc A", metadata={"kb_name": "kb1", "source": "a.txt", "chunk_index": 0}, score=0.9),
        ]
        sparse = [(1, 3.0)]  # doc B → not in dense
        all_docs = [
            {"content": "doc A", "metadata": {"kb_name": "kb1", "source": "a.txt", "chunk_index": 0}},
            {"content": "doc B", "metadata": {"kb_name": "kb1", "source": "b.txt", "chunk_index": 0}},
        ]

        result = _rrf_fusion(dense, sparse, all_docs)
        assert len(result) == 2  # both doc A and doc B

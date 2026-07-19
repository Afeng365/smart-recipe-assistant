"""Tests for RAG knowledge base subsystem."""
import importlib
import json
import shutil
import sys
import tempfile
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

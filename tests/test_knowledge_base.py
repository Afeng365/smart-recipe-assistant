"""Tests for RAG knowledge base subsystem."""
import importlib
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

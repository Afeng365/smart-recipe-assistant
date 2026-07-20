"""SQLite metadata database for knowledge base management."""
import sqlite3
from pathlib import Path

# Module-level: set by init_db() at startup
db_path: Path | None = None


def _connect() -> sqlite3.Connection:
    """Get a connection with foreign keys enabled (uses module-level db_path)."""
    if db_path is None:
        raise RuntimeError("db.db_path not initialized -- call init_db() first")
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
            embedding_model TEXT DEFAULT 'qwen3-embedding:0.6b',
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


# -- KB CRUD ---------------------------------------------------------------

def add_kb_to_db(name: str, description: str = "",
                  embedding_model: str = "qwen3-embedding:0.6b") -> None:
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


# -- File CRUD -------------------------------------------------------------

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


# -- Chunk mapping ---------------------------------------------------------

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

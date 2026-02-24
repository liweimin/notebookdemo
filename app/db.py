import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from app.config import settings


def _connect() -> sqlite3.Connection:
    db_path = Path(settings.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS notebooks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                notebook_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                notebook_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            """
        )


def list_notebooks() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, created_at
            FROM notebooks
            ORDER BY datetime(created_at) DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def create_notebook(name: str) -> dict[str, Any]:
    notebook_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO notebooks (id, name) VALUES (?, ?)",
            (notebook_id, name.strip()),
        )
        row = conn.execute(
            "SELECT id, name, created_at FROM notebooks WHERE id = ?",
            (notebook_id,),
        ).fetchone()
    return dict(row)


def notebook_exists(notebook_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM notebooks WHERE id = ? LIMIT 1",
            (notebook_id,),
        ).fetchone()
    return row is not None


def add_document(
    notebook_id: str,
    filename: str,
    raw_text: str,
    chunk_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    document_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, notebook_id, filename, raw_text)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, notebook_id, filename, raw_text),
        )

        chunk_rows = [
            (
                str(uuid.uuid4()),
                notebook_id,
                document_id,
                payload["chunk_index"],
                payload["content"],
                json.dumps(payload["embedding"], ensure_ascii=False),
            )
            for payload in chunk_payloads
        ]
        conn.executemany(
            """
            INSERT INTO chunks (id, notebook_id, document_id, chunk_index, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )

        row = conn.execute(
            """
            SELECT d.id, d.filename, d.created_at, COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            WHERE d.id = ?
            GROUP BY d.id
            """,
            (document_id,),
        ).fetchone()
    return dict(row)


def list_documents(notebook_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.filename, d.created_at, COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            WHERE d.notebook_id = ?
            GROUP BY d.id
            ORDER BY datetime(d.created_at) DESC
            """,
            (notebook_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_notebook_chunks(notebook_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.embedding,
                d.filename
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.notebook_id = ?
            ORDER BY d.created_at DESC, c.chunk_index ASC
            """,
            (notebook_id,),
        ).fetchall()

    output = [dict(row) for row in rows]
    for row in output:
        row["embedding"] = json.loads(row["embedding"])
    return output

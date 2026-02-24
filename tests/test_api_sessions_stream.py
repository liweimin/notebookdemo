from pathlib import Path

from fastapi.testclient import TestClient

from app import main as main_module
from app.config import settings


def _setup_runtime(tmp_path: Path) -> None:
    settings.database_path = str(tmp_path / "test.db")
    settings.llm_provider = "local"
    settings.embedding_backend = "local"
    settings.chat_memory_messages = 12


def _create_notebook(client: TestClient, name: str) -> str:
    res = client.post("/api/notebooks", json={"name": name})
    assert res.status_code == 200
    return res.json()["id"]


def _upload_text_doc(client: TestClient, notebook_id: str, text: str, filename: str = "doc.txt") -> None:
    files = [("files", (filename, text.encode("utf-8"), "text/plain"))]
    res = client.post(f"/api/notebooks/{notebook_id}/documents", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body and body[0]["chunk_count"] >= 1


def _create_session(client: TestClient, notebook_id: str) -> str:
    res = client.post(f"/api/notebooks/{notebook_id}/sessions", json={"title": "test"})
    assert res.status_code == 200
    return res.json()["id"]


def test_multi_turn_session_memory_persists_messages(tmp_path: Path) -> None:
    _setup_runtime(tmp_path)
    with TestClient(main_module.app) as client:
        notebook_id = _create_notebook(client, "n1")
        _upload_text_doc(
            client,
            notebook_id,
            "System goal is source-grounded answers with citations. "
            "Architecture is FastAPI and SQLite.",
        )
        session_id = _create_session(client, notebook_id)

        q1 = client.post(
            f"/api/notebooks/{notebook_id}/ask",
            json={"question": "系统目标是什么？", "session_id": session_id},
        )
        assert q1.status_code == 200
        j1 = q1.json()
        assert j1["session_id"] == session_id
        assert len(j1["citations"]) >= 1

        q2 = client.post(
            f"/api/notebooks/{notebook_id}/ask",
            json={"question": "再说下架构。", "session_id": session_id},
        )
        assert q2.status_code == 200
        j2 = q2.json()
        assert j2["session_id"] == session_id
        assert len(j2["citations"]) >= 1

        messages = client.get(f"/api/sessions/{session_id}/messages")
        assert messages.status_code == 200
        rows = messages.json()
        assert len(rows) == 4
        assert [row["role"] for row in rows] == ["user", "assistant", "user", "assistant"]


def test_stream_endpoint_returns_done_event_and_persists(tmp_path: Path) -> None:
    _setup_runtime(tmp_path)
    with TestClient(main_module.app) as client:
        notebook_id = _create_notebook(client, "n2")
        _upload_text_doc(client, notebook_id, "Streaming endpoint test content.")
        session_id = _create_session(client, notebook_id)

        with client.stream(
            "POST",
            f"/api/notebooks/{notebook_id}/ask/stream",
            json={"question": "总结一下", "session_id": session_id},
        ) as response:
            assert response.status_code == 200
            content = "".join(response.iter_text())

        assert "event: meta" in content
        assert "event: token" in content
        assert "event: done" in content

        messages = client.get(f"/api/sessions/{session_id}/messages")
        assert messages.status_code == 200
        rows = messages.json()
        assert len(rows) == 2
        assert rows[0]["role"] == "user"
        assert rows[1]["role"] == "assistant"


def test_session_isolation_between_notebooks(tmp_path: Path) -> None:
    _setup_runtime(tmp_path)
    with TestClient(main_module.app) as client:
        n1 = _create_notebook(client, "n1")
        n2 = _create_notebook(client, "n2")
        _upload_text_doc(client, n1, "Notebook one text.")
        _upload_text_doc(client, n2, "Notebook two text.")
        session_n1 = _create_session(client, n1)

        wrong = client.post(
            f"/api/notebooks/{n2}/ask",
            json={"question": "test", "session_id": session_n1},
        )
        assert wrong.status_code == 400
        assert "Session 不属于当前 Notebook" in wrong.json()["detail"]

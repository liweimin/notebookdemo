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


def _upload_text_doc(
    client: TestClient,
    notebook_id: str,
    text: str,
    filename: str = "doc.txt",
) -> dict:
    files = [("files", (filename, text.encode("utf-8"), "text/plain"))]
    res = client.post(f"/api/notebooks/{notebook_id}/documents", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body and body[0]["chunk_count"] >= 1
    return body[0]


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


def test_session_auto_naming_and_rename_endpoint(tmp_path: Path) -> None:
    _setup_runtime(tmp_path)
    with TestClient(main_module.app) as client:
        notebook_id = _create_notebook(client, "rename")
        _upload_text_doc(client, notebook_id, "Session naming content.")
        session = client.post(f"/api/notebooks/{notebook_id}/sessions", json={"title": ""}).json()
        session_id = session["id"]

        ask = client.post(
            f"/api/notebooks/{notebook_id}/ask",
            json={"question": "这是第一轮问题，用来自动命名会话。", "session_id": session_id},
        )
        assert ask.status_code == 200

        sessions = client.get(f"/api/notebooks/{notebook_id}/sessions")
        assert sessions.status_code == 200
        matched = [item for item in sessions.json() if item["id"] == session_id][0]
        assert matched["title"] != ""

        rename = client.patch(f"/api/sessions/{session_id}", json={"title": "项目讨论"})
        assert rename.status_code == 200
        assert rename.json()["title"] == "项目讨论"


def test_metadata_filtering_by_filename_and_document_ids(tmp_path: Path) -> None:
    _setup_runtime(tmp_path)
    with TestClient(main_module.app) as client:
        notebook_id = _create_notebook(client, "filter")
        d1 = _upload_text_doc(client, notebook_id, "Apple report content.", filename="apple_report.txt")
        d2 = _upload_text_doc(client, notebook_id, "Banana report content.", filename="banana_report.txt")
        session_id = _create_session(client, notebook_id)

        by_filename = client.post(
            f"/api/notebooks/{notebook_id}/ask",
            json={
                "question": "总结报告内容",
                "session_id": session_id,
                "filename_contains": "banana",
            },
        )
        assert by_filename.status_code == 200
        for citation in by_filename.json()["citations"]:
            assert "banana" in citation["filename"].lower()

        by_doc = client.post(
            f"/api/notebooks/{notebook_id}/ask",
            json={
                "question": "总结报告内容",
                "session_id": session_id,
                "document_ids": [d1["id"]],
            },
        )
        assert by_doc.status_code == 200
        citation_doc_ids = {item["document_id"] for item in by_doc.json()["citations"]}
        assert citation_doc_ids == {d1["id"]}
        assert d2["id"] not in citation_doc_ids

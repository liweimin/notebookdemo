from pathlib import Path
import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import (
    add_document,
    add_chat_message,
    create_chat_session,
    create_notebook,
    get_chat_session,
    get_notebook_chunks,
    init_db,
    list_chat_messages,
    list_chat_sessions,
    list_documents,
    list_notebooks,
    notebook_exists,
    session_belongs_to_notebook,
    update_chat_session_title,
)
from app.llm_provider import get_provider_runtime
from app.models import (
    AskRequest,
    AskResponse,
    ChatMessageOut,
    ChatSessionRename,
    ChatSessionCreate,
    ChatSessionOut,
    CitationOut,
    DocumentOut,
    LLMRuntimeOut,
    NotebookCreate,
    NotebookOut,
)
from app.rag import (
    embed_texts,
    extract_text,
    generate_answer,
    retrieve_top_chunks,
    stream_generate_answer,
    split_text,
)

app = FastAPI(title="Mini NotebookLM", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@app.on_event("startup")
def on_startup() -> None:
    init_db()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/llm/runtime", response_model=LLMRuntimeOut)
def llm_runtime_api() -> LLMRuntimeOut:
    runtime = get_provider_runtime()
    return LLMRuntimeOut(
        provider=runtime.provider,
        api_key_configured=runtime.api_key_configured,
        base_url=runtime.base_url,
        chat_model=runtime.chat_model,
        embedding_model=runtime.embedding_model,
        embedding_backend=runtime.embedding_backend,
    )


@app.post("/api/notebooks", response_model=NotebookOut)
def create_notebook_api(payload: NotebookCreate) -> NotebookOut:
    created = create_notebook(payload.name)
    return NotebookOut(**created)


@app.get("/api/notebooks", response_model=list[NotebookOut])
def list_notebooks_api() -> list[NotebookOut]:
    return [NotebookOut(**row) for row in list_notebooks()]


@app.post("/api/notebooks/{notebook_id}/documents", response_model=list[DocumentOut])
async def upload_documents(notebook_id: str, files: list[UploadFile] = File(...)) -> list[DocumentOut]:
    if not notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook 不存在。")

    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件。")

    inserted_docs: list[DocumentOut] = []
    for upload in files:
        text = await extract_text(upload)
        if not text.strip():
            raise HTTPException(status_code=400, detail=f"{upload.filename} 没有可提取的文本。")

        chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise HTTPException(status_code=400, detail=f"{upload.filename} 切分后没有有效内容。")

        vectors, _ = embed_texts(chunks)
        payloads = [
            {
                "chunk_index": idx,
                "content": chunk,
                "embedding": vectors[idx],
            }
            for idx, chunk in enumerate(chunks)
        ]
        saved = add_document(
            notebook_id=notebook_id,
            filename=upload.filename or "untitled.txt",
            raw_text=text,
            chunk_payloads=payloads,
        )
        inserted_docs.append(DocumentOut(**saved))
    return inserted_docs


@app.get("/api/notebooks/{notebook_id}/documents", response_model=list[DocumentOut])
def list_documents_api(notebook_id: str) -> list[DocumentOut]:
    if not notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook 不存在。")
    return [DocumentOut(**row) for row in list_documents(notebook_id)]


@app.post("/api/notebooks/{notebook_id}/sessions", response_model=ChatSessionOut)
def create_session_api(notebook_id: str, payload: ChatSessionCreate) -> ChatSessionOut:
    if not notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook 不存在。")
    created = create_chat_session(notebook_id, title=(payload.title or "").strip())
    return ChatSessionOut(**created)


@app.get("/api/notebooks/{notebook_id}/sessions", response_model=list[ChatSessionOut])
def list_sessions_api(notebook_id: str) -> list[ChatSessionOut]:
    if not notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook 不存在。")
    return [ChatSessionOut(**row) for row in list_chat_sessions(notebook_id)]


@app.patch("/api/sessions/{session_id}", response_model=ChatSessionOut)
def rename_session_api(session_id: str, payload: ChatSessionRename) -> ChatSessionOut:
    session = get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session 不存在。")
    updated = update_chat_session_title(session_id, payload.title.strip())
    if not updated:
        raise HTTPException(status_code=404, detail="Session 不存在。")
    return ChatSessionOut(**updated)


@app.get("/api/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
def list_session_messages_api(session_id: str) -> list[ChatMessageOut]:
    session = get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session 不存在。")
    return [ChatMessageOut(**row) for row in list_chat_messages(session_id, limit=200)]


def _resolve_session_id(notebook_id: str, requested_session_id: str | None) -> str:
    if requested_session_id:
        if not session_belongs_to_notebook(notebook_id, requested_session_id):
            raise HTTPException(status_code=400, detail="Session 不属于当前 Notebook。")
        return requested_session_id
    created = create_chat_session(notebook_id, title="新会话")
    return created["id"]


def _auto_name_session_if_needed(session_id: str, question: str) -> None:
    session = get_chat_session(session_id)
    if not session:
        return
    if (session.get("title") or "").strip():
        return
    title = question.strip()[: settings.session_title_auto_chars]
    if title:
        update_chat_session_title(session_id, title)


def _build_citations(top_chunks: list[dict]) -> list[CitationOut]:
    return [
        CitationOut(
            index=idx,
            document_id=chunk["document_id"],
            filename=chunk["filename"],
            snippet=chunk["content"][:300].replace("\n", " ").strip(),
        )
        for idx, chunk in enumerate(top_chunks, start=1)
    ]


def _prepare_ask_context(notebook_id: str, payload: AskRequest) -> dict:
    if not notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook 不存在。")

    runtime = get_provider_runtime()
    chunks = get_notebook_chunks(notebook_id)
    if not chunks:
        raise HTTPException(status_code=400, detail="这个 Notebook 还没有文档。")

    session_id = _resolve_session_id(notebook_id, payload.session_id)
    history_messages = list_chat_messages(
        session_id=session_id,
        limit=settings.chat_memory_messages * 2,
    )
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history_messages
        if msg["role"] in {"user", "assistant"}
    ]

    top_chunks, retrieval_fallback, embedding_mode = retrieve_top_chunks(
        query=payload.question,
        chunks=chunks,
        top_k=settings.retrieval_top_k,
        document_ids=payload.document_ids,
        filename_contains=payload.filename_contains,
    )
    if not top_chunks:
        raise HTTPException(status_code=400, detail="过滤条件下没有可检索内容。")
    add_chat_message(
        session_id=session_id,
        notebook_id=notebook_id,
        role="user",
        content=payload.question,
    )
    _auto_name_session_if_needed(session_id, payload.question)
    citations = _build_citations(top_chunks)
    return {
        "runtime": runtime,
        "session_id": session_id,
        "history": history,
        "top_chunks": top_chunks,
        "citations": citations,
        "retrieval_fallback": retrieval_fallback,
        "embedding_mode": embedding_mode,
    }


def _persist_assistant_message(
    session_id: str,
    notebook_id: str,
    answer: str,
    citations: list[CitationOut],
    runtime,
    generation_mode: str,
    embedding_mode: str,
    used_fallback: bool,
) -> None:
    add_chat_message(
        session_id=session_id,
        notebook_id=notebook_id,
        role="assistant",
        content=answer,
        citations=[item.model_dump() for item in citations],
        metadata={
            "llm_provider": runtime.provider,
            "generation_mode": generation_mode,
            "embedding_mode": embedding_mode,
            "chat_model": runtime.chat_model,
            "embedding_model": runtime.embedding_model,
            "used_fallback": used_fallback,
        },
    )


def _run_ask(notebook_id: str, payload: AskRequest) -> AskResponse:
    context = _prepare_ask_context(notebook_id, payload)
    answer, generation_fallback, generation_mode = generate_answer(
        question=payload.question,
        chunks=context["top_chunks"],
        history=context["history"],
    )
    response = AskResponse(
        session_id=context["session_id"],
        answer=answer,
        citations=context["citations"],
        used_fallback=context["retrieval_fallback"] or generation_fallback,
        llm_provider=context["runtime"].provider,
        generation_mode=generation_mode,
        embedding_mode=context["embedding_mode"],
        chat_model=context["runtime"].chat_model,
        embedding_model=context["runtime"].embedding_model,
    )
    _persist_assistant_message(
        session_id=context["session_id"],
        notebook_id=notebook_id,
        answer=answer,
        citations=context["citations"],
        runtime=context["runtime"],
        generation_mode=generation_mode,
        embedding_mode=context["embedding_mode"],
        used_fallback=response.used_fallback,
    )
    return response


@app.post("/api/notebooks/{notebook_id}/ask", response_model=AskResponse)
def ask_api(notebook_id: str, payload: AskRequest) -> AskResponse:
    return _run_ask(notebook_id, payload)


@app.post("/api/notebooks/{notebook_id}/ask/stream")
def ask_stream_api(notebook_id: str, payload: AskRequest) -> StreamingResponse:
    context = _prepare_ask_context(notebook_id, payload)
    token_iter, stream_state = stream_generate_answer(
        question=payload.question,
        chunks=context["top_chunks"],
        history=context["history"],
    )

    def _event(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _iter_events():
        answer_parts: list[str] = []
        yield _event(
            "meta",
            {
                "session_id": context["session_id"],
                "llm_provider": context["runtime"].provider,
                "generation_mode": stream_state.get("generation_mode", "local"),
                "embedding_mode": context["embedding_mode"],
                "chat_model": context["runtime"].chat_model,
                "embedding_model": context["runtime"].embedding_model,
            },
        )
        try:
            for piece in token_iter:
                if piece:
                    answer_parts.append(piece)
                    yield _event("token", {"delta": piece})
        except Exception:
            stream_state["used_fallback"] = True
            stream_state["generation_mode"] = "local"
            fallback_note = "流式生成中断，请重试。"
            answer_parts.append(fallback_note)
            yield _event("token", {"delta": fallback_note})

        answer_text = "".join(answer_parts).strip() or "资料不足。"
        response = AskResponse(
            session_id=context["session_id"],
            answer=answer_text,
            citations=context["citations"],
            used_fallback=context["retrieval_fallback"] or stream_state.get("used_fallback", False),
            llm_provider=context["runtime"].provider,
            generation_mode=stream_state.get("generation_mode", "local"),
            embedding_mode=context["embedding_mode"],
            chat_model=context["runtime"].chat_model,
            embedding_model=context["runtime"].embedding_model,
        )
        _persist_assistant_message(
            session_id=context["session_id"],
            notebook_id=notebook_id,
            answer=answer_text,
            citations=context["citations"],
            runtime=context["runtime"],
            generation_mode=response.generation_mode,
            embedding_mode=context["embedding_mode"],
            used_fallback=response.used_fallback,
        )
        yield _event("done", response.model_dump())

    return StreamingResponse(
        _iter_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import (
    add_document,
    create_notebook,
    get_notebook_chunks,
    init_db,
    list_documents,
    list_notebooks,
    notebook_exists,
)
from app.llm_provider import get_provider_runtime
from app.models import (
    AskRequest,
    AskResponse,
    CitationOut,
    DocumentOut,
    LLMRuntimeOut,
    NotebookCreate,
    NotebookOut,
)
from app.rag import embed_texts, extract_text, generate_answer, retrieve_top_chunks, split_text

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


@app.post("/api/notebooks/{notebook_id}/ask", response_model=AskResponse)
def ask_api(notebook_id: str, payload: AskRequest) -> AskResponse:
    if not notebook_exists(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook 不存在。")

    runtime = get_provider_runtime()
    chunks = get_notebook_chunks(notebook_id)
    if not chunks:
        raise HTTPException(status_code=400, detail="这个 Notebook 还没有文档。")

    top_chunks, retrieval_fallback, embedding_mode = retrieve_top_chunks(
        query=payload.question,
        chunks=chunks,
        top_k=settings.retrieval_top_k,
    )
    answer, generation_fallback, generation_mode = generate_answer(payload.question, top_chunks)

    citations = [
        CitationOut(
            index=idx,
            document_id=chunk["document_id"],
            filename=chunk["filename"],
            snippet=chunk["content"][:300].replace("\n", " ").strip(),
        )
        for idx, chunk in enumerate(top_chunks, start=1)
    ]
    return AskResponse(
        answer=answer,
        citations=citations,
        used_fallback=retrieval_fallback or generation_fallback,
        llm_provider=runtime.provider,
        generation_mode=generation_mode,
        embedding_mode=embedding_mode,
        chat_model=runtime.chat_model,
        embedding_model=runtime.embedding_model,
    )

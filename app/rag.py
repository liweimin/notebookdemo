import hashlib
import math
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader

from app.config import settings
from app.llm_provider import get_provider_runtime, create_provider_client


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower())


def _local_embedding(text: str, dim: int = 384) -> list[float]:
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % dim
        vec[idx] += 1.0

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    return sum(x * y for x, y in zip(a, b))


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", text)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(normalized)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        candidate = normalized[start:end]

        if end < text_len:
            sentence_break = max(candidate.rfind("\n"), candidate.rfind("。"), candidate.rfind(". "))
            if sentence_break > int(chunk_size * 0.55):
                end = start + sentence_break + 1
                candidate = normalized[start:end]

        cleaned = candidate.strip()
        if cleaned:
            chunks.append(cleaned)

        if end >= text_len:
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def _extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


async def extract_text(upload_file: UploadFile) -> str:
    filename = upload_file.filename or "untitled.txt"
    suffix = Path(filename).suffix.lower()
    file_bytes = await upload_file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail=f"{filename} 是空文件。")

    if suffix in {".txt", ".md", ".markdown", ".csv", ".log"}:
        return file_bytes.decode("utf-8", errors="ignore").strip()
    if suffix == ".pdf":
        return _extract_text_from_pdf_bytes(file_bytes)

    raise HTTPException(
        status_code=400,
        detail=f"不支持文件类型 {suffix or '[no extension]'}，目前支持 txt/md/csv/log/pdf。",
    )


def embed_texts(texts: list[str], local_dim: int = 384) -> tuple[list[list[float]], bool]:
    if not texts:
        return [], True

    runtime = get_provider_runtime()
    if runtime.embedding_backend == "local":
        return [_local_embedding(text, dim=local_dim) for text in texts], True
    client = create_provider_client()
    if client is None:
        return [_local_embedding(text, dim=local_dim) for text in texts], True

    try:
        vectors: list[list[float]] = []
        step = 64
        for idx in range(0, len(texts), step):
            batch = texts[idx : idx + step]
            response = client.embeddings.create(model=runtime.embedding_model, input=batch)
            vectors.extend([item.embedding for item in response.data])
        return vectors, False
    except Exception:
        return [_local_embedding(text, dim=local_dim) for text in texts], True


def retrieve_top_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], bool, str]:
    target_dim = len(chunks[0]["embedding"]) if chunks and chunks[0]["embedding"] else 384
    query_embeddings, used_fallback = embed_texts([query], local_dim=target_dim)
    embedding_mode = "local" if used_fallback else "provider"
    query_embedding = query_embeddings[0]
    scored = []
    for chunk in chunks:
        score = _cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]], used_fallback, embedding_mode


def _build_context(chunks: list[dict[str, Any]]) -> str:
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        snippet = chunk["content"].strip().replace("\n", " ")
        lines.append(
            f"[{idx}] 文件: {chunk['filename']} | 片段: {snippet}"
        )
    return "\n".join(lines)


def _build_history(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    window = history[-settings.chat_memory_messages :]
    lines = []
    for item in window:
        role = item.get("role", "").lower().strip()
        role_text = "用户" if role == "user" else "助手"
        content = (item.get("content") or "").replace("\n", " ").strip()
        if content:
            lines.append(f"{role_text}: {content}")
    return "\n".join(lines)


def _local_summary(chunks: list[dict[str, Any]], prefix: str) -> str:
    summary_lines = [prefix]
    for idx, chunk in enumerate(chunks, start=1):
        snippet = chunk["content"].replace("\n", " ").strip()
        summary_lines.append(f"[{idx}] {chunk['filename']}: {snippet[:260]}...")
    return "\n".join(summary_lines)


def chunk_stream_text(text: str, chunk_size: int = 22) -> list[str]:
    if not text:
        return []
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> tuple[str, bool, str]:
    if not chunks:
        return "当前笔记本没有可用内容。", True, "local"

    runtime = get_provider_runtime()
    client = create_provider_client()
    provider_name = runtime.provider.upper()
    if client is None:
        if runtime.provider == "local":
            return (
                _local_summary(chunks, "当前处于本地模式（LLM_PROVIDER=local），以下是相关片段摘要："),
                True,
                "local",
            )
        return (
            _local_summary(
                chunks,
                f"{provider_name} 未配置 API Key，自动切换本地摘要模式：",
            ),
            True,
            "local",
        )

    context = _build_context(chunks)
    history_text = _build_history(history or [])
    system_prompt = (
        "你是一个 NotebookLM 风格的研究助手。"
        "必须只基于给定上下文回答，不要编造。"
        "回答请使用中文，并在关键结论后附上 [1] 这种引用编号。"
        "若信息不足，明确说“资料不足”。"
    )
    prompt_parts = []
    if history_text:
        prompt_parts.append(f"历史对话（供上下文延续，不可覆盖文档事实）：\n{history_text}")
    prompt_parts.append(f"问题：{question}")
    prompt_parts.append(f"可用上下文：\n{context}")
    prompt_parts.append("请给出结构化回答（简短要点即可）。")
    user_prompt = "\n\n".join(prompt_parts)

    try:
        completion = client.chat.completions.create(
            model=runtime.chat_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = completion.choices[0].message.content or "资料不足。"
        return answer.strip(), False, "provider"
    except Exception:
        return (
            _local_summary(chunks, f"{provider_name} 调用失败，自动切换到本地摘要模式："),
            True,
            "local",
        )

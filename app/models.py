from pydantic import BaseModel, Field


class NotebookCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class NotebookOut(BaseModel):
    id: str
    name: str
    created_at: str


class DocumentOut(BaseModel):
    id: str
    filename: str
    created_at: str
    chunk_count: int


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)


class CitationOut(BaseModel):
    index: int
    document_id: str
    filename: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    used_fallback: bool
    llm_provider: str
    generation_mode: str
    embedding_mode: str
    chat_model: str
    embedding_model: str


class LLMRuntimeOut(BaseModel):
    provider: str
    api_key_configured: bool
    base_url: str | None
    chat_model: str
    embedding_model: str
    embedding_backend: str

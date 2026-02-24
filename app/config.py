from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "local"

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    zhipu_api_key: str | None = None

    openai_base_url: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"

    openai_chat_model: str = "gpt-4o-mini"
    gemini_chat_model: str = "gemini-2.5-flash"
    zhipu_chat_model: str = "glm-4-plus"

    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "gemini-embedding-001"
    zhipu_embedding_model: str = "embedding-3"

    # Optional global overrides for any provider.
    chat_model_override: str | None = None
    embedding_model_override: str | None = None

    # provider or local. local means always use local hash embeddings for retrieval.
    embedding_backend: str = "provider"

    database_path: str = "data/notebookllm.db"
    chunk_size: int = 900
    chunk_overlap: int = 140
    retrieval_top_k: int = 6
    chat_memory_messages: int = 12

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

from dataclasses import dataclass

from openai import OpenAI

from app.config import settings

SUPPORTED_PROVIDERS = {"local", "openai", "gemini", "zhipu"}


@dataclass(frozen=True)
class ProviderRuntime:
    provider: str
    api_key_configured: bool
    base_url: str | None
    chat_model: str
    embedding_model: str
    embedding_backend: str


def _normalize_provider(raw_provider: str) -> str:
    normalized = (raw_provider or "local").strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        return "local"
    return normalized


def _normalize_embedding_backend(raw_backend: str) -> str:
    normalized = (raw_backend or "provider").strip().lower()
    return "provider" if normalized not in {"provider", "local"} else normalized


def _provider_api_key(provider: str) -> str | None:
    if provider == "openai":
        return settings.openai_api_key
    if provider == "gemini":
        return settings.gemini_api_key
    if provider == "zhipu":
        return settings.zhipu_api_key
    return None


def _provider_base_url(provider: str) -> str | None:
    if provider == "openai":
        return settings.openai_base_url
    if provider == "gemini":
        return settings.gemini_base_url
    if provider == "zhipu":
        return settings.zhipu_base_url
    return None


def _provider_chat_model(provider: str) -> str:
    if settings.chat_model_override:
        return settings.chat_model_override
    if provider == "openai":
        return settings.openai_chat_model
    if provider == "gemini":
        return settings.gemini_chat_model
    if provider == "zhipu":
        return settings.zhipu_chat_model
    return "local-summary"


def _provider_embedding_model(provider: str) -> str:
    if settings.embedding_model_override:
        return settings.embedding_model_override
    if provider == "openai":
        return settings.openai_embedding_model
    if provider == "gemini":
        return settings.gemini_embedding_model
    if provider == "zhipu":
        return settings.zhipu_embedding_model
    return "local-hash-embedding"


def get_provider_runtime() -> ProviderRuntime:
    provider = _normalize_provider(settings.llm_provider)
    api_key = _provider_api_key(provider)
    return ProviderRuntime(
        provider=provider,
        api_key_configured=bool(api_key),
        base_url=_provider_base_url(provider),
        chat_model=_provider_chat_model(provider),
        embedding_model=_provider_embedding_model(provider),
        embedding_backend=_normalize_embedding_backend(settings.embedding_backend),
    )


def create_provider_client() -> OpenAI | None:
    runtime = get_provider_runtime()
    if runtime.provider == "local" or not runtime.api_key_configured:
        return None
    if runtime.base_url:
        return OpenAI(api_key=_provider_api_key(runtime.provider), base_url=runtime.base_url)
    return OpenAI(api_key=_provider_api_key(runtime.provider))

from app.config import settings
from app.llm_provider import ProviderRuntime
from app.rag import retrieve_top_chunks, stream_generate_answer
import app.rag as rag_module


def test_hybrid_retrieval_with_rerank_promotes_keyword_relevance(monkeypatch):
    original_vector_weight = settings.retrieval_vector_weight
    original_rerank_weight = settings.retrieval_rerank_weight
    original_candidate_k = settings.retrieval_candidate_k
    try:
        settings.retrieval_vector_weight = 0.85
        settings.retrieval_rerank_weight = 0.7
        settings.retrieval_candidate_k = 10

        monkeypatch.setattr(rag_module, "embed_texts", lambda texts, local_dim=384: ([[1.0, 0.0]], False))

        chunks = [
            {
                "id": "c1",
                "document_id": "d1",
                "chunk_index": 0,
                "content": "General architecture overview without query keywords.",
                "embedding": [1.0, 0.0],
                "filename": "general.txt",
            },
            {
                "id": "c2",
                "document_id": "d2",
                "chunk_index": 0,
                "content": "banana smoothie recipe and banana smoothie nutrition details.",
                "embedding": [0.2, 0.98],
                "filename": "banana.txt",
            },
        ]

        top_chunks, _, _ = retrieve_top_chunks("banana smoothie", chunks, top_k=1)
        assert len(top_chunks) == 1
        assert top_chunks[0]["document_id"] == "d2"
    finally:
        settings.retrieval_vector_weight = original_vector_weight
        settings.retrieval_rerank_weight = original_rerank_weight
        settings.retrieval_candidate_k = original_candidate_k


def test_stream_generate_answer_uses_provider_token_stream(monkeypatch):
    class FakeDelta:
        def __init__(self, content: str):
            self.content = content

    class FakeChoice:
        def __init__(self, content: str):
            self.delta = FakeDelta(content)

    class FakeChunk:
        def __init__(self, content: str):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            assert kwargs.get("stream") is True
            return [FakeChunk("你好"), FakeChunk("，世界")]

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(
        rag_module,
        "get_provider_runtime",
        lambda: ProviderRuntime(
            provider="gemini",
            api_key_configured=True,
            base_url="https://example.com",
            chat_model="fake-model",
            embedding_model="fake-embedding",
            embedding_backend="provider",
        ),
    )
    monkeypatch.setattr(rag_module, "create_provider_client", lambda: FakeClient())

    chunks = [
        {
            "id": "c1",
            "document_id": "d1",
            "chunk_index": 0,
            "content": "streaming context",
            "embedding": [1.0, 0.0],
            "filename": "ctx.txt",
        }
    ]
    token_iter, state = stream_generate_answer("say hi", chunks, history=[])
    text = "".join(list(token_iter))
    assert text == "你好，世界"
    assert state["generation_mode"] == "provider"
    assert state["used_fallback"] is False

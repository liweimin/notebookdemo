# Agent Memory (Minimal Handoff)

This file is the minimal persistent context for future coding agents when chat context is missing.

## 1. Product Goal
- Build a NotebookLM-like app:
  - Upload documents
  - Ask grounded questions
  - Return answers with citations

## 2. Current Status (2026-02-25)
- Backend: FastAPI + SQLite + local static frontend.
- Core flow implemented:
  - notebook CRUD (basic)
  - document upload (txt/md/csv/log/pdf)
  - chunking + hybrid retrieval (vector + keyword) + reranking
  - metadata filtering for retrieval (`filename_contains`, `document_ids`)
  - answer generation with citations
  - chat session memory (multi-turn per notebook)
  - streaming answer API (SSE)
- Multi-provider LLM runtime implemented:
  - `local`, `openai`, `gemini`, `zhipu`
  - runtime API: `GET /api/llm/runtime`
- Frontend:
  - stream rendering in ask flow
  - session switch/create/rename and message history panel
  - filename filter input for retrieval metadata filter
- Automated tests:
  - `pytest` API tests for sessions, stream endpoint, session isolation
  - retrieval rerank + provider-stream unit tests
- CI:
  - GitHub Actions workflow runs `pytest` + browser simulation smoke check

## 3. Key Files
- `app/main.py`: API endpoints + app wiring.
- `app/rag.py`: extraction/chunk/retrieval/answer pipeline.
- `app/llm_provider.py`: provider routing (key/base_url/model/backend).
- `app/db.py`: sqlite schema + persistence.
- `app/static/index.html`, `app/static/app.js`: streaming UI + session memory UX.
- `.env.example`: runtime configuration template.
- `scripts/browser_simulation.py`: browser E2E simulation + screenshots/report.
- `tests/test_api_sessions_stream.py`: API regression tests.
- `tests/test_rag_hybrid_stream.py`: retrieval/rerank + provider streaming tests.
- `.github/workflows/ci.yml`: CI pipeline.

## 4. Runtime Config Rules
- App reads only `.env` (not `.env.example`).
- Critical envs:
  - `LLM_PROVIDER`
  - `${PROVIDER}_API_KEY`
  - `${PROVIDER}_CHAT_MODEL`
  - `${PROVIDER}_EMBEDDING_MODEL`
  - `EMBEDDING_BACKEND` (`provider` or `local`)
- Never commit `.env` or secrets.

## 5. Known Good Gemini Setup
Use this for fully remote generation + embedding:

```env
LLM_PROVIDER=gemini
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_BACKEND=provider
```

Expected result in ask response:
- `generation_mode=provider`
- `embedding_mode=provider`
- `used_fallback=false`

## 6. Smoke Test Commands
Start app:

```bash
uvicorn app.main:app --reload
```

Runtime check:

```bash
curl http://127.0.0.1:8000/api/llm/runtime
```

Browser E2E:

```bash
python scripts/browser_simulation.py
```

API regression tests:

```bash
pytest -q
```

## 7. Invariants (Do Not Break)
- If remote provider fails, app must gracefully fallback to local mode (no crash).
- Ask response must include citations and runtime mode fields.
- UI engine tag should reflect provider + generation mode + embedding mode.
- Session must be isolated per notebook (cross-notebook session_id should fail).
- Stream endpoint must emit `meta -> token* -> done` events.
- Retrieval should support filter + hybrid + rerank while preserving fallback behavior.

## 8. Next Priority Backlog
1. Add provider-native citation alignment in stream mode (token-time citation hints).
2. Add session list UI with explicit session switcher (not just latest/default).
3. Add retrieval diagnostics endpoint (scores, chosen chunks, filter stats).
4. Replace deprecated FastAPI `on_event` startup with lifespan handler.

## 9. Handoff Prompt Template
When starting a new agent turn, provide:
- Target feature
- Current `.env` provider choice
- Output expectation (`provider` vs `local`)
- Whether to run browser simulation

Example:
"Continue from AGENT_MEMORY.md. Implement streaming answers for current Gemini provider setup and keep fallback logic unchanged."

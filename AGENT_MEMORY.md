# Agent Memory (Minimal Handoff)

This file is the minimal persistent context for future coding agents when chat context is missing.

## 1. Product Goal
- Build a NotebookLM-like app:
  - Upload documents
  - Ask grounded questions
  - Return answers with citations

## 2. Current Status (2026-02-24)
- Backend: FastAPI + SQLite + local static frontend.
- Core flow implemented:
  - notebook CRUD (basic)
  - document upload (txt/md/csv/log/pdf)
  - chunking + vector retrieval
  - answer generation with citations
  - chat session memory (multi-turn per notebook)
  - streaming answer API (SSE)
- Multi-provider LLM runtime implemented:
  - `local`, `openai`, `gemini`, `zhipu`
  - runtime API: `GET /api/llm/runtime`
- Frontend:
  - stream rendering in ask flow
  - session switch/create and message history panel
- Automated tests:
  - `pytest` API tests for sessions, stream endpoint, session isolation

## 3. Key Files
- `app/main.py`: API endpoints + app wiring.
- `app/rag.py`: extraction/chunk/retrieval/answer pipeline.
- `app/llm_provider.py`: provider routing (key/base_url/model/backend).
- `app/db.py`: sqlite schema + persistence.
- `app/static/index.html`, `app/static/app.js`: streaming UI + session memory UX.
- `.env.example`: runtime configuration template.
- `scripts/browser_simulation.py`: browser E2E simulation + screenshots/report.
- `tests/test_api_sessions_stream.py`: API regression tests.

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

## 8. Next Priority Backlog
1. Improve retrieval quality:
   - metadata filtering
   - reranking
   - hybrid retrieval (keyword + vector)
2. Add true provider token streaming (current stream is answer chunk streaming).
3. Add chat session naming + rename endpoint.
4. Add CI workflow to run `pytest` and browser simulation smoke checks.

## 9. Handoff Prompt Template
When starting a new agent turn, provide:
- Target feature
- Current `.env` provider choice
- Output expectation (`provider` vs `local`)
- Whether to run browser simulation

Example:
"Continue from AGENT_MEMORY.md. Implement streaming answers for current Gemini provider setup and keep fallback logic unchanged."

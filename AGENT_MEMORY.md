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
- Multi-provider LLM runtime implemented:
  - `local`, `openai`, `gemini`, `zhipu`
  - runtime API: `GET /api/llm/runtime`

## 3. Key Files
- `app/main.py`: API endpoints + app wiring.
- `app/rag.py`: extraction/chunk/retrieval/answer pipeline.
- `app/llm_provider.py`: provider routing (key/base_url/model/backend).
- `app/db.py`: sqlite schema + persistence.
- `app/static/index.html`, `app/static/app.js`: UI and provider mode tag display.
- `.env.example`: runtime configuration template.
- `scripts/browser_simulation.py`: browser E2E simulation + screenshots/report.

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

## 7. Invariants (Do Not Break)
- If remote provider fails, app must gracefully fallback to local mode (no crash).
- Ask response must include citations and runtime mode fields.
- UI engine tag should reflect provider + generation mode + embedding mode.

## 8. Next Priority Backlog
1. Add streaming answer API + frontend stream rendering.
2. Add chat session memory (multi-turn per notebook).
3. Improve retrieval quality:
   - metadata filtering
   - reranking
   - hybrid retrieval (keyword + vector)
4. Add automated tests:
   - API tests (pytest)
   - regression checks for fallback behavior.

## 9. Handoff Prompt Template
When starting a new agent turn, provide:
- Target feature
- Current `.env` provider choice
- Output expectation (`provider` vs `local`)
- Whether to run browser simulation

Example:
"Continue from AGENT_MEMORY.md. Implement streaming answers for current Gemini provider setup and keep fallback logic unchanged."

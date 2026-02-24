# Mini NotebookLM (MVP)

一个可本地运行的 NotebookLM 风格应用，支持：

- 创建 Notebook
- 上传文档（`txt/md/csv/log/pdf`）
- 自动切分 + 向量检索
- 基于文档问答并返回引用片段
- 支持会话记忆（同 Notebook 下多轮对话）
- 支持流式回答（SSE）
- 可切换 LLM 提供商：`OpenAI / Gemini / Zhipu / Local`

上下文延续文档（给后续编程 agent）：`AGENT_MEMORY.md`

## 1. 启动方式

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

启动后访问：`http://127.0.0.1:8000`

## 2. LLM 提供商配置（OpenAI / Gemini / Zhipu）

默认是 `LLM_PROVIDER=local`，不依赖任何 Key 也能用（会走本地摘要模式）。

### 2.1 OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 2.2 Gemini Pro（OpenAI 兼容接口）

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

### 2.3 智谱（OpenAI 兼容接口）

```env
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=your_zhipu_key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4/
ZHIPU_CHAT_MODEL=glm-4.7
ZHIPU_EMBEDDING_MODEL=embedding-3
EMBEDDING_BACKEND=local
```

说明：

- 不需要把 Key 发在聊天里，只要配置到本机 `.env` 即可。
- `EMBEDDING_BACKEND=provider` 表示检索向量走远程模型；设为 `local` 可强制用本地向量。
- 任一远程调用失败时，会自动 fallback 到本地模式，服务不会中断。

## 3. 项目结构

```text
app/
  main.py         # FastAPI 入口与接口
  db.py           # SQLite 数据访问
  rag.py          # 切分、向量化、检索、回答生成
  models.py       # Pydantic 模型
  static/
    index.html
    styles.css
    app.js
```

## 4. 接口概览

- `POST /api/notebooks` 创建 Notebook
- `GET /api/notebooks` 列出 Notebook
- `POST /api/notebooks/{id}/documents` 上传文档
- `GET /api/notebooks/{id}/documents` 查看文档
- `POST /api/notebooks/{id}/sessions` 创建会话
- `GET /api/notebooks/{id}/sessions` 列出会话
- `GET /api/sessions/{session_id}/messages` 查看会话历史
- `POST /api/notebooks/{id}/ask` 基于文档提问
- `POST /api/notebooks/{id}/ask/stream` 流式提问（SSE）
- `GET /api/llm/runtime` 查看当前 LLM 运行配置与模式

## 5. 自动化测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 6. 浏览器模拟测试（可视化过程）

这不是纯接口测试，而是会实际打开浏览器自动操作页面（创建 Notebook、上传文档、提问）。

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
python scripts/browser_simulation.py
```

运行后会生成：

- `artifacts/browser-sim/index.html`：可视化步骤页（每一步截图 + 结果）
- `artifacts/browser-sim/run.json`：原始执行记录

实时看浏览器自动操作（可见窗口）：

```bash
python scripts/browser_simulation.py --headed --slow-mo 500 --step-delay-ms 1000 --hold-seconds 15
```

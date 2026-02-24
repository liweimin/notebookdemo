const notebookListEl = document.getElementById("notebookList");
const documentListEl = document.getElementById("documentList");
const answerOutputEl = document.getElementById("answerOutput");
const citationListEl = document.getElementById("citationList");
const messageListEl = document.getElementById("messageList");
const activeNotebookTitleEl = document.getElementById("activeNotebookTitle");
const engineModeTagEl = document.getElementById("engineModeTag");
const sessionTagEl = document.getElementById("sessionTag");

const createNotebookForm = document.getElementById("createNotebookForm");
const notebookNameInput = document.getElementById("notebookNameInput");
const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const askForm = document.getElementById("askForm");
const questionInput = document.getElementById("questionInput");
const refreshNotebooksBtn = document.getElementById("refreshNotebooksBtn");
const refreshDocsBtn = document.getElementById("refreshDocsBtn");
const newSessionBtn = document.getElementById("newSessionBtn");

let activeNotebookId = null;
let activeSessionId = null;
let isStreaming = false;

function providerLabel(provider) {
  const normalized = (provider || "local").toLowerCase();
  if (normalized === "openai") return "OpenAI";
  if (normalized === "gemini") return "Gemini";
  if (normalized === "zhipu") return "Zhipu";
  return "Local";
}

function updateEngineTagFromRuntime(runtime) {
  const provider = providerLabel(runtime.provider);
  const genMode = runtime.api_key_configured && runtime.provider !== "local" ? "远程生成" : "本地生成";
  const embMode = runtime.embedding_backend === "local" ? "本地向量" : "远程向量";
  engineModeTagEl.textContent = `${provider} | ${genMode} | ${embMode}`;
}

function updateEngineTagFromAskResult(result) {
  const provider = providerLabel(result.llm_provider);
  const genMode = result.generation_mode === "provider" ? "远程生成" : "本地生成";
  const embMode = result.embedding_mode === "provider" ? "远程向量" : "本地向量";
  engineModeTagEl.textContent = `${provider} | ${genMode} | ${embMode}`;
}

function setStatus(message, isError = false) {
  answerOutputEl.textContent = message;
  answerOutputEl.classList.toggle("error", isError);
}

function updateSessionTag() {
  if (!activeSessionId) {
    sessionTagEl.textContent = "会话未创建";
    return;
  }
  sessionTagEl.textContent = `会话: ${activeSessionId.slice(0, 8)}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    let detail = "请求失败";
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {
      // ignore
    }
    throw new Error(detail);
  }
  return response.json();
}

function clearAnswer() {
  setStatus("还没有回答。");
  citationListEl.innerHTML = "";
}

function renderNotebooks(notebooks) {
  notebookListEl.innerHTML = "";
  if (!notebooks.length) {
    notebookListEl.innerHTML = "<li class='muted'>暂无 Notebook。</li>";
    return;
  }

  notebooks.forEach((item) => {
    const li = document.createElement("li");
    if (item.id === activeNotebookId) {
      li.classList.add("active");
    }
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.name;
    button.addEventListener("click", async () => {
      activeNotebookId = item.id;
      activeSessionId = null;
      activeNotebookTitleEl.textContent = `当前：${item.name}`;
      clearAnswer();
      await loadNotebooks();
      await loadDocuments();
      await ensureSession(false);
    });
    li.appendChild(button);
    notebookListEl.appendChild(li);
  });
}

function renderDocuments(docs) {
  documentListEl.innerHTML = "";
  if (!docs.length) {
    documentListEl.innerHTML = "<li class='muted'>当前 Notebook 还没有文档。</li>";
    return;
  }

  docs.forEach((doc) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${doc.filename}</strong><br/><span class="muted">${doc.chunk_count} chunks</span>`;
    documentListEl.appendChild(li);
  });
}

function renderCitations(citations) {
  citationListEl.innerHTML = "";
  citations.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `[${item.index}] ${item.filename} - ${item.snippet}`;
    citationListEl.appendChild(li);
  });
}

function renderMessages(messages) {
  messageListEl.innerHTML = "";
  if (!messages.length) {
    messageListEl.innerHTML = "<li class='muted'>暂无历史消息。</li>";
    return;
  }
  messages.forEach((msg) => {
    const li = document.createElement("li");
    li.classList.add(msg.role === "assistant" ? "assistant" : "user");
    li.innerHTML = `<div class="message-role">${msg.role === "assistant" ? "助手" : "用户"}</div>${msg.content}`;
    messageListEl.appendChild(li);
  });
}

async function loadNotebooks() {
  const notebooks = await api("/api/notebooks");
  if (!notebooks.length) {
    activeNotebookId = null;
    activeSessionId = null;
    activeNotebookTitleEl.textContent = "请选择 Notebook";
    updateSessionTag();
    renderNotebooks([]);
    return;
  }

  if (!activeNotebookId || !notebooks.some((item) => item.id === activeNotebookId)) {
    activeNotebookId = notebooks[0].id;
    activeNotebookTitleEl.textContent = `当前：${notebooks[0].name}`;
  }
  renderNotebooks(notebooks);
}

async function loadDocuments() {
  if (!activeNotebookId) {
    renderDocuments([]);
    return;
  }
  const docs = await api(`/api/notebooks/${activeNotebookId}/documents`);
  renderDocuments(docs);
}

async function loadMessages() {
  if (!activeSessionId) {
    renderMessages([]);
    return;
  }
  const messages = await api(`/api/sessions/${activeSessionId}/messages`);
  renderMessages(messages);
}

async function ensureSession(forceCreate = false) {
  if (!activeNotebookId) {
    activeSessionId = null;
    updateSessionTag();
    return;
  }
  if (!forceCreate && activeSessionId) {
    updateSessionTag();
    await loadMessages();
    return;
  }

  if (!forceCreate) {
    const sessions = await api(`/api/notebooks/${activeNotebookId}/sessions`);
    if (sessions.length) {
      activeSessionId = sessions[0].id;
      updateSessionTag();
      await loadMessages();
      return;
    }
  }

  const created = await api(`/api/notebooks/${activeNotebookId}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "" }),
  });
  activeSessionId = created.id;
  updateSessionTag();
  await loadMessages();
}

function parseSseBlocks(buffer) {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() || "";
  const parsed = [];
  blocks.forEach((block) => {
    const lines = block.split(/\r?\n/);
    let event = "message";
    const dataLines = [];
    lines.forEach((line) => {
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    });
    parsed.push({ event, data: dataLines.join("\n") });
  });
  return { parsed, rest };
}

async function askWithStream(question) {
  const response = await fetch(`/api/notebooks/${activeNotebookId}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: activeSessionId }),
  });
  if (!response.ok) {
    let detail = "请求失败";
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {
      // ignore
    }
    throw new Error(detail);
  }
  if (!response.body) {
    throw new Error("浏览器不支持流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let answerText = "";
  let finalResult = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { parsed, rest } = parseSseBlocks(buffer);
    buffer = rest;
    parsed.forEach((item) => {
      if (!item.data) return;
      if (item.event === "token") {
        const payload = JSON.parse(item.data);
        answerText += payload.delta || "";
        setStatus(answerText);
      } else if (item.event === "done") {
        finalResult = JSON.parse(item.data);
      } else if (item.event === "error") {
        const payload = JSON.parse(item.data);
        throw new Error(payload.detail || "流式请求失败");
      }
    });
  }

  if (!finalResult) {
    throw new Error("流式结果不完整。");
  }
  return finalResult;
}

async function loadRuntime() {
  const runtime = await api("/api/llm/runtime");
  updateEngineTagFromRuntime(runtime);
}

createNotebookForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = notebookNameInput.value.trim();
  if (!name) return;
  try {
    await api("/api/notebooks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    notebookNameInput.value = "";
    await loadNotebooks();
    await loadDocuments();
    await ensureSession(false);
  } catch (error) {
    setStatus(error.message, true);
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeNotebookId) {
    setStatus("请先创建或选择一个 Notebook。", true);
    return;
  }
  if (!fileInput.files.length) {
    setStatus("请选择要上传的文件。", true);
    return;
  }

  const formData = new FormData();
  Array.from(fileInput.files).forEach((file) => formData.append("files", file));

  try {
    setStatus("正在上传并索引文档...");
    await api(`/api/notebooks/${activeNotebookId}/documents`, {
      method: "POST",
      body: formData,
    });
    fileInput.value = "";
    await loadDocuments();
    setStatus("文档处理完成，可以提问了。");
  } catch (error) {
    setStatus(error.message, true);
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeNotebookId) {
    setStatus("请先创建或选择一个 Notebook。", true);
    return;
  }
  if (isStreaming) {
    setStatus("上一条回答仍在生成，请稍候。", true);
    return;
  }
  const question = questionInput.value.trim();
  if (!question) {
    setStatus("请输入问题。", true);
    return;
  }

  try {
    await ensureSession(false);
    isStreaming = true;
    setStatus("正在检索并流式生成回答...");
    renderCitations([]);
    const result = await askWithStream(question);
    activeSessionId = result.session_id || activeSessionId;
    updateSessionTag();
    setStatus(result.answer || "资料不足。");
    renderCitations(result.citations || []);
    updateEngineTagFromAskResult(result);
    await loadMessages();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    isStreaming = false;
  }
});

newSessionBtn.addEventListener("click", async () => {
  if (!activeNotebookId) {
    setStatus("请先创建或选择一个 Notebook。", true);
    return;
  }
  try {
    await ensureSession(true);
    clearAnswer();
    setStatus("已创建新会话。");
  } catch (error) {
    setStatus(error.message, true);
  }
});

refreshNotebooksBtn.addEventListener("click", async () => {
  try {
    await loadNotebooks();
    await ensureSession(false);
  } catch (error) {
    setStatus(error.message, true);
  }
});

refreshDocsBtn.addEventListener("click", async () => {
  try {
    await loadDocuments();
  } catch (error) {
    setStatus(error.message, true);
  }
});

async function boot() {
  try {
    await loadRuntime();
    await loadNotebooks();
    await loadDocuments();
    await ensureSession(false);
  } catch (error) {
    setStatus(error.message, true);
  }
}

boot();

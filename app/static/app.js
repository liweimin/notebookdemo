const notebookListEl = document.getElementById("notebookList");
const documentListEl = document.getElementById("documentList");
const answerOutputEl = document.getElementById("answerOutput");
const citationListEl = document.getElementById("citationList");
const activeNotebookTitleEl = document.getElementById("activeNotebookTitle");
const engineModeTagEl = document.getElementById("engineModeTag");

const createNotebookForm = document.getElementById("createNotebookForm");
const notebookNameInput = document.getElementById("notebookNameInput");
const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const askForm = document.getElementById("askForm");
const questionInput = document.getElementById("questionInput");
const refreshNotebooksBtn = document.getElementById("refreshNotebooksBtn");
const refreshDocsBtn = document.getElementById("refreshDocsBtn");

let activeNotebookId = null;

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
    button.addEventListener("click", () => {
      activeNotebookId = item.id;
      activeNotebookTitleEl.textContent = `当前：${item.name}`;
      clearAnswer();
      loadNotebooks();
      loadDocuments();
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

async function loadNotebooks() {
  const notebooks = await api("/api/notebooks");
  if (!activeNotebookId && notebooks.length) {
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
  const question = questionInput.value.trim();
  if (!question) {
    setStatus("请输入问题。", true);
    return;
  }

  try {
    setStatus("正在检索并生成回答...");
    const result = await api(`/api/notebooks/${activeNotebookId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    setStatus(result.answer);
    renderCitations(result.citations || []);
    updateEngineTagFromAskResult(result);
  } catch (error) {
    setStatus(error.message, true);
  }
});

refreshNotebooksBtn.addEventListener("click", async () => {
  try {
    await loadNotebooks();
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
  } catch (error) {
    setStatus(error.message, true);
  }
}

boot();

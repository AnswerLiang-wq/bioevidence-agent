"use strict";

const state = {
  mode: "standard",
  sessionId: makeSessionId(),
  taskId: null,
  tasks: [],
  result: null,
  decisions: new Map(),
  startedAt: null,
  timerHandle: null,
  timeoutLogged: false,
};

const elements = {
  modeButtons: [...document.querySelectorAll(".mode-button")],
  modeLabel: document.querySelector("#mode-label"),
  standardFields: document.querySelector("#standard-fields"),
  liveFields: document.querySelector("#live-fields"),
  taskSelect: document.querySelector("#task-select"),
  taskDescription: document.querySelector("#task-description"),
  question: document.querySelector("#question"),
  safetyConfirm: document.querySelector("#safety-confirm"),
  message: document.querySelector("#message"),
  searchButton: document.querySelector("#search-button"),
  resultsSection: document.querySelector("#results-section"),
  exportSection: document.querySelector("#export-section"),
  cards: document.querySelector("#cards"),
  resultCount: document.querySelector("#result-count"),
  resultMode: document.querySelector("#result-mode"),
  trace: document.querySelector("#trace"),
  acceptedCount: document.querySelector("#accepted-count"),
  packStatus: document.querySelector("#pack-status"),
  synthesis: document.querySelector("#synthesis"),
  exportJson: document.querySelector("#export-json"),
  exportMarkdown: document.querySelector("#export-markdown"),
  exportStatus: document.querySelector("#export-status"),
  timer: document.querySelector("#timer"),
  timerStatus: document.querySelector("#timer-status"),
  timerCard: document.querySelector(".timer-card"),
  steps: [...document.querySelectorAll(".step")],
};

initialize();

async function initialize() {
  bindEvents();
  try {
    const payload = await api("/api/standard-tasks", {method: "GET"});
    state.tasks = Array.isArray(payload.tasks) ? payload.tasks : [];
    renderTaskOptions();
  } catch (error) {
    showMessage(safeError(error), "error");
    elements.searchButton.disabled = true;
  }
}

function bindEvents() {
  elements.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  elements.taskSelect.addEventListener("change", updateTaskDescription);
  elements.searchButton.addEventListener("click", runSearch);
  elements.exportJson.addEventListener("click", () => exportPack("json"));
  elements.exportMarkdown.addEventListener("click", () => exportPack("markdown"));
}

function renderTaskOptions() {
  elements.taskSelect.replaceChildren();
  state.tasks.forEach((task) => {
    const option = document.createElement("option");
    option.value = String(task.id || "");
    option.textContent = String(task.title || task.id || "未命名任务");
    elements.taskSelect.append(option);
  });
  updateTaskDescription();
}

function updateTaskDescription() {
  const selected = state.tasks.find((task) => task.id === elements.taskSelect.value);
  elements.taskDescription.textContent = selected
    ? `${selected.description} · ${selected.candidate_count} 篇候选来源`
    : "暂无可用标准任务。";
}

function setMode(mode) {
  if (!new Set(["standard", "live"]).has(mode)) return;
  state.mode = mode;
  elements.modeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  elements.standardFields.hidden = mode !== "standard";
  elements.liveFields.hidden = mode !== "live";
  elements.modeLabel.textContent = mode === "standard" ? "标准任务" : "Live research";
  hideMessage();
}

async function runSearch() {
  if (!elements.safetyConfirm.checked) {
    showMessage("请先确认任务不含患者、个人、未公开或机密信息。", "error");
    return;
  }
  if (state.mode === "live" && elements.question.value.trim().length < 8) {
    showMessage("请输入至少 8 个字符的具体科研问题。", "error");
    return;
  }
  const sessionStart = startTimerIfNeeded();
  setBusy(elements.searchButton, true, "正在查询 PubMed…");
  hideMessage();
  state.taskId = state.mode === "standard"
    ? elements.taskSelect.value
    : "live-research";
  await sessionStart;
  await logEvent("search_started", {mode: state.mode});
  try {
    const payload = await api("/api/search", {
      method: "POST",
      body: {
        session_id: state.sessionId,
        mode: state.mode,
        task_id: state.taskId,
        question: state.mode === "live" ? elements.question.value.trim() : null,
      },
    });
    state.result = payload;
    state.taskId = payload.task_id;
    state.decisions = new Map(
      payload.cards.map((card) => [card.card_id, defaultDecision(card.card_id)]),
    );
    renderResults(payload);
    await logEvent("search_completed", {
      mode: state.mode,
      card_count: payload.cards.length,
      result_status: payload.status,
    });
    showMessage(
      payload.status === "partial"
        ? "仅生成少于 3 张可引用卡片；你仍可审查并导出不完整证据包。"
        : "证据卡已生成。请逐张打开来源并作出你的判断。",
      payload.status === "partial" ? "info" : "success",
    );
  } catch (error) {
    const code = error && error.code ? error.code : "unknown";
    await logEvent("search_failed", {error_code: String(code).slice(0, 80)});
    await logEvent("error_shown", {error_code: String(code).slice(0, 80)});
    showMessage(safeError(error), "error");
  } finally {
    setBusy(elements.searchButton, false);
  }
}

function renderResults(result) {
  elements.cards.replaceChildren();
  result.cards.forEach((card) => elements.cards.append(createCard(card)));
  elements.resultCount.textContent = `${result.cards.length} cards`;
  elements.resultMode.textContent = `${result.retrieval.local_ranker.toUpperCase()} · ${result.mode}`;
  elements.trace.textContent = JSON.stringify(
    {retrieval: result.retrieval, tool_trace: result.tool_trace},
    null,
    2,
  );
  elements.resultsSection.hidden = false;
  elements.exportSection.hidden = false;
  elements.exportStatus.textContent = "尚未导出。导出成功不代表科学结论正确或用户价值已验证。";
  setActiveStep(2);
  updateAcceptedCount();
  elements.resultsSection.scrollIntoView({behavior: "smooth", block: "start"});
}

function createCard(card) {
  const article = document.createElement("article");
  article.className = "evidence-card";
  article.dataset.cardId = card.card_id;

  const topline = element("div", "card-topline");
  topline.append(
    textElement("span", "rank-label", `EVIDENCE ${card.display_rank}`),
    textElement("span", "review-label", card.system_suggestion.label),
  );

  const title = textElement("h3", "", card.title);
  const meta = element("div", "card-meta");
  [
    `PMID ${card.pmid}`,
    card.year ? String(card.year) : "年份未报告",
    card.journal || "期刊未报告",
    card.publication_types.length
      ? card.publication_types.slice(0, 2).join(" / ")
      : "文献类型未报告",
    `BM25 #${card.retrieval.rank}`,
  ].forEach((item) => meta.append(textElement("span", "", item)));
  const source = document.createElement("a");
  source.className = "source-link";
  source.href = card.source_url;
  source.target = "_blank";
  source.rel = "noopener noreferrer";
  source.textContent = "打开 PubMed 来源 ↗";
  source.addEventListener("click", () => logEvent("source_opened", {}, card.card_id));
  meta.append(source);

  const snippet = element("div", "snippet-block");
  snippet.append(
    textElement("p", "", card.snippet.text),
    textElement(
      "small",
      "",
      `${card.snippet.section} · chars ${card.snippet.start_char}–${card.snippet.end_char} · snippet ${card.snippet.snippet_sha256}`,
    ),
    textElement("small", "", `normalized record ${card.record_sha256}`),
  );

  const controls = element("div", "card-controls");
  controls.append(
    choiceField(
      "纳入决定",
      card,
      "action",
      [
        ["accepted", "接受"],
        ["excluded", "排除"],
        ["undecided", "未决定"],
      ],
      "card_action",
    ),
    choiceField(
      "你判断的方向",
      card,
      "user_direction",
      [
        ["supports", "支持"],
        ["opposes", "反对"],
        ["unclear", "不明确"],
      ],
      "card_action",
    ),
    usefulField(card),
    noteField(card),
  );

  article.append(topline, title, meta, snippet, controls);
  return article;
}

function choiceField(label, card, key, options, eventType) {
  const field = element("div", "control-field");
  field.append(textElement("label", "", label));
  const group = element("div", "direction-group");
  const current = state.decisions.get(card.card_id)[key];
  options.forEach(([value, text]) => {
    const button = textElement("button", "choice-button", text);
    button.type = "button";
    button.dataset.value = value;
    button.classList.toggle("active", value === current);
    button.addEventListener("click", async () => {
      state.decisions.get(card.card_id)[key] = value;
      [...group.children].forEach((child) => child.classList.toggle(
        "active",
        child.dataset.value === value,
      ));
      updateAcceptedCount();
      await logEvent(eventType, {field: key, value}, card.card_id);
    });
    group.append(button);
  });
  field.append(group);
  return field;
}

function usefulField(card) {
  const field = element("div", "control-field");
  field.append(textElement("label", "", "是否有助于完成任务"));
  const group = element("div", "useful-toggle");
  [[true, "有帮助"], [false, "没帮助"]].forEach(([value, text]) => {
    const button = textElement("button", "choice-button", text);
    button.type = "button";
    button.dataset.value = String(value);
    button.classList.toggle("active", value === false);
    button.addEventListener("click", async () => {
      state.decisions.get(card.card_id).useful = value;
      [...group.children].forEach((child) => child.classList.toggle(
        "active",
        child.dataset.value === String(value),
      ));
      await logEvent("card_useful", {useful: value}, card.card_id);
    });
    group.append(button);
  });
  field.append(group);
  return field;
}

function noteField(card) {
  const field = element("div", "note-field");
  const label = document.createElement("label");
  const id = `note-${card.card_id}`;
  label.htmlFor = id;
  label.textContent = "核验备注（仅保存在导出文件，不进入事件日志）";
  const textarea = document.createElement("textarea");
  textarea.id = id;
  textarea.rows = 2;
  textarea.maxLength = 1200;
  textarea.placeholder = "记录人群、终点、时间、效应量或需要查全文的位置。";
  textarea.addEventListener("input", () => {
    state.decisions.get(card.card_id).note = textarea.value;
  });
  textarea.addEventListener("change", () => logEvent(
    "note_changed",
    {has_content: Boolean(textarea.value.trim())},
    card.card_id,
  ));
  field.append(label, textarea);
  return field;
}

function defaultDecision(cardId) {
  return {
    card_id: cardId,
    action: "undecided",
    user_direction: "unclear",
    useful: false,
    note: "",
  };
}

function updateAcceptedCount() {
  const count = [...state.decisions.values()].filter(
    (decision) => decision.action === "accepted",
  ).length;
  elements.acceptedCount.textContent = String(count);
  if (count > 0) setActiveStep(3);
}

async function exportPack(format) {
  if (!state.result) {
    showMessage("请先生成证据卡。", "error");
    return;
  }
  const button = format === "json" ? elements.exportJson : elements.exportMarkdown;
  setBusy(button, true, "正在生成…");
  await logEvent("export_started", {format});
  try {
    const payload = await api("/api/export", {
      method: "POST",
      body: {
        session_id: state.sessionId,
        decisions: [...state.decisions.values()],
        pack_status: elements.packStatus.value,
        synthesis: elements.synthesis.value,
      },
    });
    const content = format === "json"
      ? `${JSON.stringify(payload.json, null, 2)}\n`
      : payload.markdown;
    const extension = format === "json" ? "json" : "md";
    downloadText(
      `bioevidence-${state.taskId || "evidence-pack"}.${extension}`,
      content,
      format === "json" ? "application/json" : "text/markdown",
    );
    await logEvent("export_completed", {
      format,
      accepted_count: payload.json.audit.accepted_count,
    });
    stopTimerAfterExport();
    elements.exportStatus.textContent = `${format.toUpperCase()} 证据包已生成并触发下载；请核对下载文件与 Hash。`;
    showMessage("证据包已导出到本机下载目录。", "success");
  } catch (error) {
    const code = error && error.code ? error.code : "unknown";
    await logEvent("error_shown", {error_code: String(code).slice(0, 80)});
    showMessage(safeError(error), "error");
  } finally {
    setBusy(button, false);
  }
}

async function startTimerIfNeeded() {
  if (state.startedAt !== null) return;
  state.startedAt = Date.now();
  elements.timerStatus.textContent = "任务进行中";
  await logEvent("session_started", {mode: state.mode});
  state.timerHandle = window.setInterval(updateTimer, 1000);
  updateTimer();
}

function updateTimer() {
  const elapsed = elapsedMs();
  const remaining = Math.max(0, 12 * 60 * 1000 - elapsed);
  const minutes = Math.floor(remaining / 60000);
  const seconds = Math.floor((remaining % 60000) / 1000);
  elements.timer.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  elements.timerCard.classList.toggle("warning", remaining <= 2 * 60 * 1000);
  if (remaining === 0) {
    window.clearInterval(state.timerHandle);
    elements.timerStatus.textContent = "已到 12 分钟";
    if (!state.timeoutLogged) {
      state.timeoutLogged = true;
      logEvent("timeout", {threshold_minutes: 12});
    }
  }
}

function stopTimerAfterExport() {
  if (state.timerHandle !== null) {
    window.clearInterval(state.timerHandle);
    state.timerHandle = null;
  }
  elements.timerStatus.textContent = "证据包已导出";
}

async function logEvent(eventType, metadata = {}, cardId = null) {
  if (state.startedAt === null && eventType !== "session_started") return;
  const body = {
    session_id: state.sessionId,
    event_type: eventType,
    task_id: state.taskId,
    card_id: cardId,
    elapsed_ms: elapsedMs(),
    metadata,
  };
  try {
    await api("/api/events", {method: "POST", body});
  } catch (_error) {
    // Metrics are best-effort and never block the research workflow.
  }
}

function elapsedMs() {
  return state.startedAt === null ? 0 : Math.max(0, Date.now() - state.startedAt);
}

async function api(path, options) {
  const request = {method: options.method, headers: {Accept: "application/json"}};
  if (options.body !== undefined) {
    request.headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, request);
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    throw {code: "invalid_server_response", message: "本地服务返回了无法解析的响应。"};
  }
  if (!response.ok) {
    throw payload.error || {code: "request_failed", message: "请求失败。"};
  }
  return payload;
}

function showMessage(text, kind) {
  elements.message.textContent = text;
  elements.message.className = `message ${kind === "error" ? "error" : kind === "info" ? "info" : ""}`;
  elements.message.hidden = false;
}

function hideMessage() {
  elements.message.hidden = true;
  elements.message.textContent = "";
}

function safeError(error) {
  if (error && typeof error.message === "string") return error.message;
  return "发生未预期错误，请保留本次记录并停止任务。";
}

function setBusy(button, busy, text = "") {
  if (!button.dataset.originalText) button.dataset.originalText = button.textContent;
  button.disabled = busy;
  button.textContent = busy ? text : button.dataset.originalText;
}

function setActiveStep(number) {
  elements.steps.forEach((step) => {
    step.classList.toggle("active", Number(step.dataset.step) === number);
  });
}

function downloadText(filename, content, type) {
  const blob = new Blob([content], {type: `${type};charset=utf-8`});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function element(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function textElement(tag, className, text) {
  const node = element(tag, className);
  node.textContent = String(text ?? "");
  return node;
}

function makeSessionId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  const random = Math.random().toString(36).slice(2, 14);
  return `session-${Date.now().toString(36)}-${random}`;
}

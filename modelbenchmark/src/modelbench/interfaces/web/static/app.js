/* oMLX performance benchmark console — no external dependencies. */
const state = {
  overview: null,
  models: new Set(),
  testset: null,
  scenarios: new Map(),   // scenario -> {selected, count, samples, total, selectedIds, manual, loading}
  sampleModal: null,      // {scenario, testset, filter}
  openTestsets: new Set(),
  job: null,
  poll: null,
  estimate: null,
  detail: null,
  sampleFilter: { q: "", model: "", dataset: "", status: "", page: 0 },
  samplesPage: null,      // {total, offset, limit, rows} served by /samples
  samplesSeq: 0,          // guards against out-of-order pagination responses
  samplesLoaded: false,
};

const PAGE_SIZE = 100;
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));
const fmt = (v, digits = 2) => (v == null || Number.isNaN(Number(v)) ? "-" : Number(v).toFixed(digits));
// Declared context bucket: 1024/8192/32768 -> 1K/8K/32K; null -> native.
const bucketLabel = (v) => (v ? `${Math.round(Number(v) / 1024)}K` : "native");
// Peak memory: phys_footprint peak in GB (GiB); old runs only have RSS MB.
const memGb = (perf, stat = "mean") => {
  const gb = perf?.peak_memory_gb?.[stat];
  if (gb != null) return gb;
  const mb = perf?.peak_rss_mb?.[stat];
  return mb != null ? mb / 1024 : null;
};
const memCell = (memory) => {
  if (!memory) return "-";
  const base = memory.baseline_gb;
  const idle = memory.idle_gb;
  const load = memory.load_gb;
  if (base == null && idle == null) return "-";
  return `${base ?? "-"} → ${idle ?? "-"}${load != null ? ` (+${load})` : ""}`;
};
const sampleMemGb = (sample) =>
  sample?.peak_memory_gb != null
    ? sample.peak_memory_gb
    : (sample?.peak_rss_mb != null ? sample.peak_rss_mb / 1024 : null);
const int = (v) => (v == null || v === "" ? "-" : String(v));
// Skipped/warning entries are structured objects in schema 3.0 (strings before).
const skipText = (item) => {
  if (item == null) return "";
  if (typeof item === "string") return item;
  const where = [item.scenario, item.dataset, item.sample_id].filter(Boolean).join(" · ");
  const reason = item.reason || item.error || JSON.stringify(item);
  return where ? `${where}：${reason}` : String(reason);
};
// oMLX feature flags may be strings or {key,...} blocks depending on schema.
const flagsText = (value) => {
  if (!value) return "无";
  if (Array.isArray(value)) {
    return value.map((flag) => (flag && flag.key) || flag).filter(Boolean).join(", ") || "无";
  }
  if (typeof value === "object") {
    const keys = Object.entries(value)
      .filter(([, enabled]) => enabled)
      .map(([key]) => key);
    return keys.join(", ") || "无";
  }
  return String(value);
};

function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.className = `toast${isError ? " err" : ""}`;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.add("hidden"), 3200);
}

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function setPill(el, text, cls = "muted") {
  el.textContent = text;
  el.className = `pill ${cls}`;
}

/* ------------------------------------------------------------------ */
/* overview                                                           */
/* ------------------------------------------------------------------ */

function renderOverview(data) {
  state.overview = data;
  const server = data.server || {};
  if (server.error) setPill($("#server-pill"), "oMLX 未连接", "err");
  else setPill($("#server-pill"), "oMLX 已连接", "ok");
  const catalog = data.catalog || {};
  setPill($("#catalog-pill"), `索引 ${catalog.n_runs ?? 0} runs`, "soft");
  $("#catalog-info").textContent = catalog.path
    ? `${catalog.path} · ${catalog.n_runs ?? 0} runs`
    : "output/index.json";

  const runDefaults = data.run_defaults || {};
  if (runDefaults.warmup_per_model != null) {
    // One switch for the default (model-level) warmup; the diagnostic
    // per-cell count is intentionally not a UI parameter.
    $("#warmup-enabled").checked = Number(runDefaults.warmup_per_model) > 0;
  }
  renderModels(data.models || [], data.model_sync || null);
  if (!state.testset) {
    const testsets = data.testsets || [];
    const first = testsets.find((ts) => ts.name === "performance") || testsets[0];
    if (first) {
      state.testset = first.name;
      state.openTestsets.add(first.name);
      resetScenarioState(first);
    }
  }
  renderTestsets();
  renderRuns(data.runs || []);
  updateEstimate();
}

function renderModels(models, sync) {
  const container = $("#models");
  container.innerHTML = models.map((m) => {
    const available = m.on_server !== false;
    const checked = available && state.models.has(m.id) ? "checked" : "";
    const dot = m.loaded === true ? "on" : "";
    const moe = m.load_settings && m.load_settings.moe_expert_offload_enabled
      ? `<span class="badge" title="MoE expert offload">MoE</span>` : "";
    let stateBadge = "";
    if (m.on_server === false) {
      stateBadge = `<span class="badge warn" title="oMLX 服务端已不存在；点「同步清单」移除">服务端不存在</span>`;
    } else if (m.on_server == null) {
      stateBadge = `<span class="badge" title="无法确认（oMLX 未连接）">未确认</span>`;
    }
    return `<label class="model-item ${available ? "" : "off"}" title="${esc(m.id)}">
      <input type="checkbox" value="${esc(m.id)}" ${checked} ${available ? "" : "disabled"}>
      <span class="dot ${dot}" title="${m.loaded ? "已加载" : "未加载"}"></span>
      <span class="model-name">${esc(m.alias)}</span>${moe}${stateBadge}
    </label>`;
  }).join("") || `<div class="empty">没有可用模型</div>`;
  container.querySelectorAll("input").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) state.models.add(input.value);
      else state.models.delete(input.value);
      $("#model-count").textContent = `已选 ${state.models.size}`;
      updateEstimate();
    });
  });
  // A stale selection must not survive a refresh: those models will 404.
  const selectable = new Set(models.filter((m) => m.on_server !== false).map((m) => m.id));
  let pruned = 0;
  for (const id of [...state.models]) {
    if (!selectable.has(id)) {
      state.models.delete(id);
      pruned += 1;
    }
  }
  if (state.models.size === 0 && selectable.size) {
    const first = models.find((m) => selectable.has(m.id));
    state.models.add(first.id);
    const input = Array.from(container.querySelectorAll("input")).find((item) => item.value === first.id);
    if (input) input.checked = true;
  }
  $("#model-count").textContent = `已选 ${state.models.size}`;
  renderModelSync(sync || state.overview?.model_sync || null);
  if (pruned) {
    toast(`已忽略 ${pruned} 个 oMLX 服务端不存在的模型`);
    updateEstimate();
  }
}

function renderModelSync(sync) {
  const el = $("#model-sync");
  const syncBtn = $("#sync-models");
  if (!el) return;
  if (!sync) {
    el.innerHTML = "";
    if (syncBtn) syncBtn.disabled = false;
    return;
  }
  if (syncBtn) syncBtn.disabled = !sync.error && sync.in_sync === true;
  if (sync.error) {
    el.innerHTML = `<div class="sync-warn">无法检索 oMLX 模型列表：${esc(sync.error)}（仍显示本地清单）</div>`;
    return;
  }
  const missing = sync.configured_missing || [];
  const extra = sync.server_only || [];
  const lines = [
    `<div>oMLX ${sync.server_model_count ?? 0} 个模型 · 清单 ${sync.configured_model_count ?? 0} 条`
      + `${sync.in_sync ? ` · <span class="sync-ok">已一致</span>` : ""}</div>`,
  ];
  if (missing.length) {
    lines.push(
      `<div class="sync-warn">清单中 ${missing.length} 个在服务端不存在：${missing.map(esc).join("、")}</div>`
    );
  }
  if (extra.length) {
    lines.push(
      `<div class="sync-new">服务端新增 ${extra.length} 个未登记：${extra.map(esc).join("、")}</div>`
    );
  }
  lines.push(`<div class="sync-path">清单文件：<code>${esc(sync.models_path || "models.yaml")}</code></div>`);
  el.innerHTML = lines.join("");
}

async function refreshModels() {
  const btn = $("#refresh-models");
  if (btn) btn.disabled = true;
  try {
    const data = await api("/api/models");
    state.overview = {
      ...(state.overview || {}),
      models: data.models || [],
      model_sync: data.model_sync || null,
    };
    renderModels(data.models || [], data.model_sync || null);
    updateEstimate();
    const sync = data.model_sync || {};
    if (sync.error) {
      toast(`模型列表刷新失败：${sync.error}`, true);
    } else if (sync.in_sync) {
      toast(`模型列表已刷新（oMLX ${sync.server_model_count} 个，与清单一致）`);
    } else {
      toast(
        `模型列表已刷新：${(sync.configured_missing || []).length} 个失效 / `
        + `${(sync.server_only || []).length} 个未登记`
      );
    }
  } catch (err) {
    toast(`模型列表刷新失败：${err.message}`, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function syncModels() {
  const btn = $("#sync-models");
  if (btn) btn.disabled = true;
  try {
    const preview = await api("/api/models/sync", { method: "POST", body: "{}" });
    if (!preview.changed) {
      toast("模型清单已是最新");
      return;
    }
    const lines = [];
    if ((preview.added || []).length) lines.push(`新增：${preview.added.join("、")}`);
    if ((preview.removed || []).length) lines.push(`移除：${preview.removed.join("、")}`);
    if (!window.confirm(`将写入 ${preview.path}\n\n${lines.join("\n")}\n\n继续？`)) return;
    const res = await api("/api/models/sync", {
      method: "POST",
      body: JSON.stringify({ apply: true }),
    });
    toast(`清单已同步：新增 ${(res.added || []).length} / 移除 ${(res.removed || []).length}`);
    await loadOverview();
  } catch (err) {
    toast(`同步失败：${err.message}`, true);
  } finally {
    // Recompute from the latest catalog state, so a successful sync keeps the
    // button disabled ("已一致") instead of re-enabling it blindly.
    renderModelSync(state.overview?.model_sync || null);
  }
}

/* ------------------------------------------------------------------ */
/* test-set directory tree                                            */
/* ------------------------------------------------------------------ */

function selectedContextWindows() {
  return [...document.querySelectorAll("#context-windows input:checked")]
    .map((input) => Number(input.value))
    .filter((value) => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b);
}

function contextFilterLabel(windows) {
  if (!windows || !windows.length) return "全部（不筛选）";
  const k = (v) => `${(v / 1024).toFixed(1).replace(/\.0$/, "")}K`;
  return windows.map((w) => {
    const low = w === 1024 ? 0 : w - 512;
    const high = w === 1024 ? 1536 : w + 512;
    return `${k(w)}:${low === 0 ? "0" : k(low)}–${k(high)}`;
  }).join(", ");
}

function contextWindowChips() {
  const testset = currentTestset();
  const buckets = (testset?.buckets || []).filter((v) => Number.isFinite(v) && v > 0);
  return (buckets.length ? [...new Set(buckets)] : [1024, 8192, 32768])
    .sort((a, b) => a - b);
}

function renderContextWindows() {
  const container = $("#context-windows");
  if (!container) return;
  const selected = new Set(selectedContextWindows());
  const k = (v) => (v === 0 ? "0" : `${(v / 1024).toFixed(1).replace(/\.0$/, "")}K`);
  container.innerHTML = contextWindowChips().map((w) => {
    const low = w === 1024 ? 0 : Math.max(0, w - 512);
    const high = w === 1024 ? 1536 : w + 512;
    return `<label><input type="checkbox" value="${w}" ${selected.has(w) ? "checked" : ""}> ${k(w)} <span>${low === 0 ? "0" : k(low)}～${k(high)}</span></label>`;
  }).join("") || `<span class="hint">该测试集没有可筛选的档位</span>`;
  container.querySelectorAll("input").forEach((input) => {
    input.addEventListener("change", updateEstimate);
  });
}

function currentTestset() {
  return (state.overview?.testsets || []).find((t) => t.name === state.testset);
}

function resetScenarioState(testset) {
  state.scenarios = new Map();
  for (const scenario of testset.scenarios || []) {
    state.scenarios.set(scenario.name, {
      selected: true,
      count: "",
      loading: false,
      samples: null,
      total: null,
      selectedIds: new Set(),
      manual: false,
      filter: "",
    });
  }
}

function ensureTestset(name) {
  if (state.testset === name) return currentTestset();
  const ts = (state.overview?.testsets || []).find((item) => item.name === name);
  if (!ts) return null;
  state.testset = name;
  state.openTestsets.add(name);
  resetScenarioState(ts);
  renderTestsets();
  updateEstimate();
  return ts;
}

function renderTestsets() {
  const container = $("#testsets");
  const testsets = state.overview?.testsets || [];
  if (!testsets.length) {
    container.innerHTML = `<div class="empty">dataset/test/ 下没有测试集</div>`;
    return;
  }
  container.innerHTML = testsets.map((ts) => {
    const active = ts.name === state.testset;
    const scenarios = ts.scenarios || [];
    const nSamples = scenarios.reduce((sum, s) => sum + (s.samples_total || 0), 0);
    const nCells = scenarios.reduce((sum, s) => sum + (s.cells_total || 0), 0);
    return `<label class="testset-item ${active ? "active" : ""}" data-testset="${esc(ts.name)}">
      <input type="checkbox" data-testset-check ${active ? "checked" : ""} title="选择该测试集">
      <span class="testset-main">
        <span class="testset-title">${esc(ts.title || ts.name)}</span>
        <span class="testset-meta">${esc(ts.directory || "")} · ${scenarios.length} 场景 · ${nSamples} 样本 · ${nCells} 测试项</span>
      </span>
    </label>`;
  }).join("");
  container.querySelectorAll("[data-testset-check]").forEach((input) => {
    input.addEventListener("change", () => {
      const name = input.closest("[data-testset]").dataset.testset;
      if (!input.checked) {
        state.testset = "";
      } else {
        state.testset = name;
        const ts = testsets.find((item) => item.name === name);
        if (ts) resetScenarioState(ts);
      }
      renderTestsets();
      updateEstimate();
    });
  });
  renderContextWindows();
}

/* ------------------------------------------------------------------ */
/* sample browser modal                                               */
/* ------------------------------------------------------------------ */

function filteredModalSamples(st) {
  const filter = (state.sampleModal?.filter || "").trim().toLowerCase();
  const samples = st?.samples || [];
  if (!filter) return samples;
  return samples.filter((sample) =>
    sample.sample_id.toLowerCase().includes(filter) ||
    (sample.question || "").toLowerCase().includes(filter) ||
    (sample.subtask || "").toLowerCase().includes(filter)
  );
}

function openSampleBrowser(scenario) {
  const testset = currentTestset();
  const spec = (testset?.scenarios || []).find((item) => item.name === scenario);
  const st = state.scenarios.get(scenario);
  if (!testset || !spec || !st) return;
  state.sampleModal = { scenario, testset: testset.name, filter: "" };
  $("#sample-modal").classList.remove("hidden");
  $("#sample-modal-title").textContent = `${scenario} · 样本浏览`;
  $("#sample-modal-sub").textContent = `${testset.directory} · 数据集 ${spec.dataset}`;
  $("#sample-modal-search").value = "";
  renderSampleModal();
  if (st.samples == null && !st.loading) fetchScenarioSamples(st, spec);
}

async function fetchScenarioSamples(st, spec) {
  const modal = state.sampleModal;
  if (!modal) return;
  st.loading = true;
  renderSampleModal();
  try {
    const data = await api(
      `/api/testsets/${encodeURIComponent(modal.testset)}/scenarios/` +
      `${encodeURIComponent(modal.scenario)}/samples?limit=1000`
    );
    st.samples = data.samples || [];
    st.total = data.total ?? st.samples.length;
    st.loading = false;
    renderSampleModal();
  } catch (err) {
    st.loading = false;
    renderSampleModal();
    toast(`加载样本失败：${err.message}`, true);
  }
}

function renderSampleModal() {
  const modal = state.sampleModal;
  if (!modal) return;
  const st = state.scenarios.get(modal.scenario);
  if (!st) return;
  const list = $("#sample-modal-list");
  if (st.loading) {
    list.innerHTML = `<tr><td colspan="4" class="empty">正在加载样本…</td></tr>`;
    $("#sample-modal-count").textContent = "加载中…";
    $("#sample-modal-selected").textContent = "";
    return;
  }
  const samples = filteredModalSamples(st);
  list.innerHTML = samples.map((sample) => `<tr>
    <td><input type="checkbox" data-modal-sample="${esc(sample.sample_id)}"
      ${st.selectedIds.has(sample.sample_id) ? "checked" : ""}></td>
    <td class="mono">${esc(sample.sample_id)}</td>
    <td class="sample-question" title="${esc(sample.question || "")}">${esc(sample.question || "")}</td>
    <td>${int(sample.context_chars)}</td>
  </tr>`).join("") || `<tr><td colspan="4" class="empty">没有匹配样本</td></tr>`;
  const all = st.samples || [];
  const allChecked = all.length > 0 && all.every((sample) => st.selectedIds.has(sample.sample_id));
  $("#sample-modal-checkall").checked = allChecked;
  $("#sample-modal-checkall").indeterminate = !allChecked && st.selectedIds.size > 0;
  $("#sample-modal-count").textContent =
    `共 ${st.total ?? all.length} 个样本 · 显示 ${samples.length}`;
  $("#sample-modal-selected").textContent = st.manual
    ? `已手选 ${st.selectedIds.size} 个样本（优先于样本数）`
    : "未手选：按样本数/全部执行";
  list.querySelectorAll("[data-modal-sample]").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) st.selectedIds.add(input.dataset.modalSample);
      else st.selectedIds.delete(input.dataset.modalSample);
      st.manual = st.selectedIds.size > 0;
      updateManualBadge(modal.scenario, st);
      renderSampleModal();
    });
  });
}

function updateManualBadge(scenario, st) {
  const card = document.querySelector(`[data-scenario="${scenario}"]`);
  const badge = card?.querySelector("[data-manual-badge]");
  if (badge) {
    badge.textContent = `手选 ${st.selectedIds.size}`;
    badge.classList.toggle("hidden", !st.manual);
  }
  updateEstimate();
}

function closeSampleModal() {
  $("#sample-modal").classList.add("hidden");
  state.sampleModal = null;
}

function selectedScenarioNames() {
  return (currentTestset()?.scenarios || [])
    .map((scenario) => scenario.name)
    .filter((name) => state.scenarios.get(name)?.selected);
}

function scenarioPlan(name) {
  const testset = currentTestset();
  return (testset?.scenarios || []).find((item) => item.name === name) || null;
}

let estimateSeq = 0;

function estimatePayload(names) {
  const payload = {
    models: [...state.models],
    testset: state.testset,
    scenarios: names,
  };
  const globalSamples = parseInt($("#samples").value, 10);
  if (globalSamples) payload.samples = globalSamples;
  const repeats = parseInt($("#repeats").value, 10);
  if (repeats) payload.repeats = repeats;
  const windows = selectedContextWindows();
  if (windows.length) payload.context_windows = windows;
  const scenarioSamples = {};
  const sampleIds = {};
  for (const name of names) {
    const st = state.scenarios.get(name);
    if (!st) continue;
    if (st.manual && st.selectedIds.size) sampleIds[name] = [...st.selectedIds];
    else if (st.count) scenarioSamples[name] = parseInt(st.count, 10);
  }
  if (Object.keys(scenarioSamples).length) payload.scenario_samples = scenarioSamples;
  if (Object.keys(sampleIds).length) payload.sample_ids = sampleIds;
  return payload;
}

function renderEstimateText(models, names, estimate, error) {
  const windows = selectedContextWindows();
  const parts = [`${models} 模型`, `${names.length} 数据集`];
  if (error) {
    parts.push("测试项开始后确定");
    parts.push(`预计 估算失败：${error}`);
  } else if (estimate && estimate.cells_total != null) {
    parts.push(`计划 ${estimate.cells_total} 个测试项`);
    if (estimate.samples_total != null) parts.push(`预计 ${estimate.samples_total} 次测量`);
    else parts.push("预计 样本数开始后确定");
  } else {
    parts.push("测试项开始后确定（默认 44）");
    parts.push("预计 样本数开始后确定");
  }
  if (windows.length) parts.push(`上下文筛选 ${contextFilterLabel(windows)}`);
  $("#estimate").textContent = parts.join(" · ");
}

async function updateEstimate() {
  const testset = currentTestset();
  if (!testset) {
    $("#start").disabled = true;
    $("#estimate").textContent = "未选择测试集（勾选上方测试集后显示计划）";
    return;
  }
  const names = selectedScenarioNames();
  const models = state.models.size;
  $("#start").disabled = models === 0 || names.length === 0;
  if (!models || !names.length) {
    $("#estimate").textContent = "选择模型与数据集后显示计划";
    return;
  }
  const seq = ++estimateSeq;
  const payload = estimatePayload(names);
  try {
    const data = await api("/api/estimate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (seq !== estimateSeq) return;
    const estimate = data.estimate || data;
    state.estimate = estimate.samples_total ?? null;
    renderEstimateText(models, names, estimate, null);
  } catch (err) {
    if (seq !== estimateSeq) return;
    state.estimate = null;
    renderEstimateText(models, names, null, err.message);
  }
}

async function startRun() {
  const names = selectedScenarioNames();
  if (!state.models.size || !names.length) return;
  const payload = {
    models: [...state.models],
    testset: state.testset,
    scenarios: names,
    cache_mode: $("#cache-mode").value,
    thinking_mode: $("#thinking-mode").value,
  };
  const globalSamples = parseInt($("#samples").value, 10);
  const maxTokens = parseInt($("#max-tokens").value, 10);
  const repeats = parseInt($("#repeats").value, 10);
  if (globalSamples) payload.samples = globalSamples;
  if (maxTokens) payload.max_tokens = maxTokens;
  if (repeats) payload.repeats = repeats;
  // On = default model-level warmup (1 discarded request after load);
  // off = no warmup, i.e. a non-default measurement protocol.
  payload.warmup_per_model = $("#warmup-enabled").checked ? 1 : 0;
  const contextWindows = selectedContextWindows();
  if (contextWindows.length) payload.context_windows = contextWindows;
  const scenarioSamples = {};
  const sampleIds = {};
  for (const name of names) {
    const st = state.scenarios.get(name);
    if (!st) continue;
    if (st.manual && st.selectedIds.size) sampleIds[name] = [...st.selectedIds];
    else if (st.count) scenarioSamples[name] = parseInt(st.count, 10);
  }
  if (Object.keys(scenarioSamples).length) payload.scenario_samples = scenarioSamples;
  if (Object.keys(sampleIds).length) payload.sample_ids = sampleIds;
  try {
    $("#start").disabled = true;
    const res = await api("/api/run", { method: "POST", body: JSON.stringify(payload) });
    toast(`已启动 ${res.run_id}`);
    await pollJob(res.job_id);
  } catch (err) {
    toast(`启动失败：${err.message}`, true);
    $("#start").disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* job status                                                         */
/* ------------------------------------------------------------------ */

function renderJobProgress(job) {
  const progress = job.progress || {};
  const plan = job.plan || {};
  const modelsTotal = progress.models_total ?? plan.models_total ?? 0;
  const modelsDone = progress.models_done ?? 0;
  const cellsTotal = progress.cells_total ?? plan.cells_total ?? 0;
  const cellsDone = progress.cells_done ?? 0;
  const cell = progress.cell || {};
  const cellDone = cell.done ?? 0;
  const cellTotal = cell.total ?? 0;
  const samplesTotal = progress.samples_total ?? plan.samples_total;
  const chips = [
    `模型 ${modelsDone}/${modelsTotal}`,
    `测试项 ${cellsDone}/${cellsTotal}`,
    `已测样本 ${job.records ?? 0}${samplesTotal ? `/${samplesTotal}` : ""}`,
  ];
  if (cellTotal) {
    chips.push(`本项 ${cellDone}/${cellTotal}`);
  }
  $("#job-stats").innerHTML = chips.map((text) => `<span class="stat-chip">${esc(text)}</span>`).join("");
  const percent = job.status === "done" ? 100 : progress.percent;
  mutationSetBar(percent, job.status);
}

function mutationSetBar(percent, status) {
  const bar = $("#job-progress");
  if (percent == null) {
    bar.style.width = status === "done" ? "100%" : "35%";
    bar.style.opacity = status === "done" ? "1" : ".55";
  } else {
    bar.style.opacity = "1";
    bar.style.width = `${percent}%`;
  }
}

function renderJobResults(job) {
  const container = $("#job-results");
  const runs = (job.summary && job.summary.model_runs) || [];
  if (!runs.length) {
    container.innerHTML = "";
    container.classList.add("hidden");
    return;
  }
  container.classList.remove("hidden");
  container.innerHTML = `<span class="hint">结果：</span>` + runs.map((run) =>
    `<button class="btn mini ghost" data-run-result="${esc(run.run_id)}">${esc(run.model)}</button>`
  ).join("");
  container.querySelectorAll("[data-run-result]").forEach((button) =>
    button.addEventListener("click", () => openRun(button.dataset.runResult)));
}

function renderJob(job) {
  state.job = job;
  $("#job-card").classList.remove("hidden");
  $("#job-run").textContent = job.run_id || "-";
  $("#job-records").textContent = job.records ?? 0;
  const map = {
    queued: ["排队中", "muted"], running: ["运行中", "run"], cancelling: ["正在取消", "run"],
    done: ["完成", "ok"], failed: ["失败", "err"], cancelled: ["已取消", "muted"],
  };
  const [text, cls] = map[job.status] || [job.status, "muted"];
  setPill($("#job-pill"), text, cls);
  $("#cancel").classList.toggle("hidden", !["running", "queued", "cancelling"].includes(job.status));
  const cur = job.current || {};
  const progress = job.progress || {};
  if (cur.event === "record") {
    $("#job-current").textContent =
      `${cur.model} · ${cur.dataset || cur.scenario} · ${cur.sample_id} · r${cur.repeat} · ` +
      `输入 ${cur.input_context_tokens ?? "-"} token · TTFT ${cur.ttft_ms ?? "-"}ms · ${cur.gen_tps ?? "-"} tok/s`;
  } else if (cur.event === "cell_start") {
    $("#job-current").textContent =
      `开始测试项 ${cur.cell_index}/${cur.cell_total}：${cur.model} · ${cur.scenario} (${cur.dataset}) · ` +
      `${cur.samples} 样本 × ${cur.repeats} 轮`;
  } else if (cur.event === "cell_done") {
    $("#job-current").textContent =
      `完成测试项 ${cur.cell_index}/${cur.cell_total}：${cur.model} · ${cur.scenario}`;
  } else if (cur.event === "batch_model_start") {
    $("#job-current").textContent =
      `开始模型 ${cur.index}/${cur.total}：${cur.model} · 输出 ${cur.run_id}`;
  } else if (cur.event === "batch_model_done") {
    $("#job-current").textContent = `完成模型 ${cur.index}/${cur.total}：${cur.model} · ${cur.run_id}`;
  } else if (cur.event === "model_start") {
    $("#job-current").textContent = `加载模型 ${cur.index}/${cur.total}：${cur.model}`;
  } else if (cur.event === "model_done") {
    $("#job-current").textContent = `模型完成 ${progress.models_done ?? 0}/${progress.models_total ?? "-"}：${cur.model}`;
  } else if (job.status === "queued") {
    $("#job-current").textContent = cur.message || "排队中…（同一 oMLX 实例串行执行）";
  }
  if (job.error) $("#job-current").textContent = job.error;
  else if (job.status === "cancelled") $("#job-current").textContent = "已取消";
  renderJobProgress(job);
  renderJobResults(job);
  $("#job-log").textContent = job.log || "";
  if (["done", "failed", "cancelled"].includes(job.status)) {
    $("#start").disabled = false;
    if (state.poll) { clearInterval(state.poll); state.poll = null; }
    loadOverview().catch(() => {});
  }
}

async function pollJob(jobId) {
  const tick = async () => {
    try {
      renderJob(await api(`/api/jobs/${jobId}`));
    } catch (err) {
      if (state.poll) { clearInterval(state.poll); state.poll = null; }
    }
  };
  await tick();
  state.poll = setInterval(tick, 1200);
}

async function cancelRun() {
  if (!state.job) return;
  try {
    await api(`/api/jobs/${state.job.id}/cancel`, { method: "POST", body: "{}" });
    toast("已请求取消");
  } catch (err) {
    toast(`取消失败：${err.message}`, true);
  }
}

/* ------------------------------------------------------------------ */
/* runs list                                                          */
/* ------------------------------------------------------------------ */

function renderRuns(runs) {
  const container = $("#runs");
  if (!runs.length) {
    container.innerHTML = `<div class="empty">暂无运行记录</div>`;
    return;
  }
  const groups = new Map();
  for (const run of runs) {
    const key = run.batch_id ? `batch:${run.batch_id}` : `run:${run.run_id}`;
    if (!groups.has(key)) groups.set(key, { batchId: run.batch_id, runs: [] });
    groups.get(key).runs.push(run);
  }
  container.innerHTML = [...groups.values()].map(renderRunGroup).join("");
  container.querySelectorAll("[data-run]").forEach((card) => {
    const runId = card.dataset.run;
    card.querySelector("[data-open]").addEventListener("click", () => openRun(runId));
    card.querySelector("[data-detail]").addEventListener("click", () => openRun(runId));
    card.querySelector("[data-files]").addEventListener("click", () => openRun(runId, "files"));
    card.querySelector("[data-finder]").addEventListener("click", () => revealRun(runId));
  });
}

function renderRunGroup(group) {
  const runs = group.runs;
  const first = runs[0];
  if (!group.batchId) {
    return `<div class="run-group">${runs.map((run) => renderRunRow(run)).join("")}</div>`;
  }
  const totalSamples = runs.reduce((sum, run) => sum + (run.n_samples || 0), 0);
  const totalDuration = runs.reduce((sum, run) => sum + (run.duration_s || 0), 0);
  const skipped = runs.reduce((sum, run) => sum + (run.n_skipped || 0), 0);
  const created = (first.created_at || "").replace("T", " ").slice(0, 19);
  return `<div class="run-group batch">
    <div class="batch-head">
      <div class="batch-title">
        <span class="batch-id">${esc(group.batchId)}</span>
        <span class="pill soft">${esc(first.method || "performance")}</span>
        ${skipped ? `<span class="pill tag-skip">跳过 ${skipped}</span>` : ""}
      </div>
      <div class="run-stats">${runs.length} 个模型 · 样本 ${totalSamples || "-"} · ${totalDuration ? totalDuration.toFixed(1) + "s" : "-"} · ${esc(created)}</div>
    </div>
    <div class="batch-models">${runs.map((run) => renderRunRow(run, true)).join("")}</div>
  </div>`;
}

function renderRunRow(run, inBatch = false) {
  const measured = run.n_models_measured ?? (run.models_measured || []).length;
  const selected = run.n_models_selected ?? (run.models_selected || []).length;
  const tags = [];
  if (run.n_skipped) tags.push(`<span class="pill tag-skip" title="${esc((run.skipped || []).join('\n'))}">跳过 ${run.n_skipped}</span>`);
  if (run.cancelled) tags.push(`<span class="pill tag-cancel">已取消</span>`);
  if (run.legacy) tags.push(`<span class="pill tag-legacy">旧 schema</span>`);
  const modelName = run.model || (run.models_measured || []).join(", ") || "-";
  const sizeBadge = run.model_size_gb != null
    ? `<span class="badge" title="模型权重估算大小（oMLX estimated_size，GiB）">${fmt(run.model_size_gb, 2)} GB</span>`
    : "";
  const created = (run.created_at || "").replace("T", " ").slice(0, 19);
  return `<div class="run ${inBatch ? "in-batch" : ""}" data-run="${esc(run.run_id)}">
    <div class="run-main" data-open>
      <div class="run-model-line">
        <span class="run-model">${esc(modelName)}</span>${sizeBadge}
        ${!inBatch ? `<span class="pill soft">${esc(run.method || run.mode || "performance")}</span>` : ""}
        ${tags.join("")}
      </div>
      <div class="run-id">${esc(run.run_id)}</div>
      <div class="run-meta">
        ${inBatch ? "" : `<span class="badge">模型 ${measured}/${selected}</span>`}
        <span class="run-stats">样本 ${int(run.n_samples)} · 耗时 ${run.duration_s ?? "-"}s · ${esc(run.bytes_total_human || "-")} · ${esc(created)}</span>
      </div>
    </div>
    <div class="run-actions">
      <button class="btn mini ghost" data-detail>详情</button>
      <button class="btn mini ghost" data-files>原始文件</button>
      <button class="btn mini ghost" data-finder>Finder</button>
    </div>
  </div>`;
}

async function revealRun(runId) {
  try {
    const res = await api(`/api/runs/${encodeURIComponent(runId)}/reveal`, { method: "POST", body: "{}" });
    toast(`已打开 ${res.path}`);
  } catch (err) {
    toast(`打开失败：${err.message}`, true);
  }
}

/* ------------------------------------------------------------------ */
/* run detail modal                                                   */
/* ------------------------------------------------------------------ */

async function openRun(runId, focus = "overall") {
  try {
    const data = await api(`/api/runs/${encodeURIComponent(runId)}`);
    state.detail = data;
    renderDetail(data);
    switchTab(focus === "files" ? "overall" : focus);
    $("#modal").classList.remove("hidden");
    if (focus === "files") {
      document.querySelector(".file-bar")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  } catch (err) {
    toast(`无法读取 run：${err.message}`, true);
  }
}

function renderDetail(data) {
  const summary = data.summary || {};
  const perf = summary.perf || {};
  const run = summary.run || {};
  $("#modal-title").textContent = `Run ${summary.run_id || ""}`;
  const batch = summary.batch || {};
  $("#modal-sub").textContent =
    `${batch.method || summary.mode || "performance"} · 模型 ${batch.model_id || batch.model || ""} · ` +
    `样本 ${summary.n_samples ?? "-"} · cells ${summary.n_cells ?? "-"} · ` +
    `耗时 ${summary.duration_s ?? "-"}s · 样本数/数据集 ${perf.samples_per_dataset ?? "全部"} · ` +
    `max_tokens ${perf.max_tokens ?? "-"} · decode guard ${perf.decode_guard === true ? "开" : perf.decode_guard === false ? "关" : "未知"} · ` +
    `上下文 ${contextFilterLabel(perf.context_windows || [])} · ` +
    `预热 ${run.warmup_per_model ?? "-"} 次/模型 · 缓存 ${run.cache_mode || "-"} · ` +
    `ISL入带 ${summary.n_isl_in_band ?? 0} · ISL出带 ${summary.n_isl_out_of_band ?? 0} · TG无效 ${summary.n_gen_tps_invalid ?? 0}`;
  const skipped = data.skipped || [];
  const banner = $("#modal-skipped");
  if (skipped.length) {
    banner.classList.remove("hidden");
    banner.innerHTML = `<b>本次有 ${skipped.length} 项被跳过/失败：</b><br>` +
      skipped.map((item) => `<code>${esc(skipText(item))}</code>`).join("<br>");
  } else {
    banner.classList.add("hidden");
  }
  const warnings = data.warnings || [];
  const warningBanner = $("#modal-warnings");
  if (warnings.length) {
    warningBanner.classList.remove("hidden");
    warningBanner.innerHTML = `<b>警告（${warnings.length}）：</b><br>` +
      warnings.map((item) => `<code>${esc(skipText(item))}</code>`).join("<br>");
  } else {
    warningBanner.classList.add("hidden");
  }
  $("#modal-finder").onclick = () => revealRun(summary.run_id);
  $("#modal-copy").onclick = async () => {
    try {
      await navigator.clipboard.writeText(data.run_dir || "");
      toast("已复制运行目录路径");
    } catch {
      toast(data.run_dir || "", false);
    }
  };
  renderFileBar(data);
  renderOverall(summary, data);
  renderCells(summary);
  renderSampleFilters(data);
  // Per-sample rows are paginated from the server (samples.csv); they load on
  // first visit to the tab instead of bloating every run-detail response.
  state.samplesPage = null;
  state.samplesLoaded = false;
  renderEnvironment(data);
  $("#modal-report").textContent = data.report || "(无 report.md)";
}

function renderFileBar(data) {
  const summary = data.summary || {};
  const runId = summary.run_id;
  const files = data.files || [];
  const links = files.map((file) => {
    const download = file.hidden === true || file.size > 2 * 1024 * 1024;
    const href = `/api/runs/${encodeURIComponent(runId)}/file?path=${encodeURIComponent(file.name)}${download ? "&download=1" : ""}`;
    const size = file.size_human || (file.size != null ? `${file.size} B` : "");
    return `<a class="file-btn" href="${href}" target="_blank" rel="noopener" title="${esc(file.name)}">${esc(file.name)}<span class="size">${esc(size)}</span></a>`;
  }).join("");
  $("#modal-files").innerHTML =
    `<span class="hint" style="align-self:center">原始输出</span>${links}`;
}

function renderOverall(summary, data) {
  const legacy = data.legacy === true || summary.legacy === true;
  const rows = (summary.overall || []).map((entry) => {
    const p = entry.perf || {};
    const tpsN = p.gen_tps?.n;
    const tps = fmt(p.gen_tps?.mean) + (tpsN != null && tpsN < entry.n_valid ? ` (${tpsN}/${entry.n_valid})` : "");
    return `<tr>
      <td>${esc(entry.model)}</td>
      <td>${entry.context_length != null ? bucketLabel(entry.context_length) : `<span class="pill soft" title="没有声明的上下文档位：该行是跨全部上下文的合并值，不要与按档位结果混比">全部*</span>`}</td>
      <td>${int(entry.n_datasets)}</td>
      <td>${int(entry.n_samples)}</td>
      <td>${int(entry.n_valid)}</td>
      <td title="finish_reason=length（达到 max_tokens）；这些样本正常参与全部平均">${int(entry.n_length_capped)}</td>
      <td title="TG TPS 被抑制的样本数">${int(entry.n_gen_tps_invalid)}</td>
      <td title="target tokenizer 实际 prompt 在 ±10% 带内的样本数；只有这些进入平均">${int(entry.n_isl_in_band)}</td>
      <td title="actual 在 ±10% 带外、仍测量但不参与平均的样本数">${int(entry.n_isl_out_of_band)}</td>
      <td>${fmt(p.ttft_ms?.mean)}</td>
      <td>${tps}</td>
      <td>${fmt(memGb(p))}</td>
      <td>${fmt(p.input_context_tokens?.mean, 0)}</td>
      <td title="insufficient_samples = 入带样本 <8（默认每 cell 10）；--samples<8 的冒烟 run 必然触发，不代表数据异常">${entry.insufficient_samples ? `<span class="pill warn">insufficient_samples</span>` : entry.unstable ? `<span class="pill warn">unstable</span>` : `<span class="pill ok">ok</span>`}</td>
    </tr>`;
  });
  if (!rows.length) {
    rows.push(`<tr><td colspan="14" class="empty">${legacy ? "旧 schema run：没有 overall 数据" : "没有整体结果"}</td></tr>`);
  }
  $("#overall-body").innerHTML = rows.join("");
}

function renderCells(summary) {
  const rows = (summary.cells || []).map((cell) => {
    const p = cell.perf || {};
    return `<tr>
      <td>${esc(cell.model)}</td>
      <td>${esc(cell.dataset)}</td>
      <td>${bucketLabel(cell.context_length)}</td>
      <td>${int(cell.n_valid)}/${int((cell.n_samples || 0) * Math.max(cell.n_repeats || 1, 1))}</td>
      <td>${int(cell.n_length_capped)}</td>
      <td title="TG TPS 被抑制的样本数">${int(cell.n_gen_tps_invalid)}</td>
      <td title="in-band 样本数（进入平均）">${int(cell.n_isl_in_band)}</td>
      <td title="out-of-band 样本数（不进入平均）">${int(cell.n_isl_out_of_band)}</td>
      <td>${fmt(p.ttft_ms?.mean)}</td>
      <td>${fmt(p.gen_tps?.mean)}</td>
      <td>${fmt(memGb(p, "max"))}</td>
      <td>${fmt(p.input_context_tokens?.mean, 0)}</td>
      <td title="insufficient_samples = 入带样本 <8；小样本冒烟 run 会触发">${cell.insufficient_samples ? `<span class="pill warn">insufficient_samples</span>` : `<span class="pill ok">ok</span>`}</td>
    </tr>`;
  });
  $("#cells-body").innerHTML = rows.join("") || `<tr><td colspan="13" class="empty">没有数据集结果</td></tr>`;
}

function renderSampleFilters(data) {
  // Filter options come from the samples.csv facet list (or the cell table as a
  // fallback); the summary itself no longer carries any per-sample rows.
  const facets = data.sample_facets || {};
  const fromCells = (key) =>
    [...new Set(((data.summary || {}).cells || []).map((cell) => cell[key]).filter(Boolean))];
  const models = (facets.models && facets.models.length ? facets.models : fromCells("model"))
    .slice().sort();
  const datasets = (facets.datasets && facets.datasets.length ? facets.datasets : fromCells("dataset"))
    .slice().sort();
  $("#sample-model").innerHTML = `<option value="">全部模型</option>` +
    models.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join("");
  $("#sample-dataset").innerHTML = `<option value="">全部数据集</option>` +
    datasets.map((d) => `<option value="${esc(d)}">${esc(d)}</option>`).join("");
  state.sampleFilter = { q: "", model: "", dataset: "", status: "", page: 0 };
  $("#sample-search").value = "";
  $("#sample-model").value = "";
  $("#sample-dataset").value = "";
  $("#sample-status").value = "";
}

async function renderSamples() {
  const summary = state.detail?.summary || {};
  const runId = summary.run_id;
  if (!runId) return;
  const filter = state.sampleFilter;
  state.samplesLoaded = true;
  const seq = ++state.samplesSeq;
  const params = new URLSearchParams({
    offset: String(Math.max(0, filter.page) * PAGE_SIZE),
    limit: String(PAGE_SIZE),
  });
  if (filter.model) params.set("model", filter.model);
  if (filter.dataset) params.set("dataset", filter.dataset);
  if (filter.status) params.set("status", filter.status);
  const query = filter.q.trim();
  if (query) params.set("q", query);
  $("#samples-body").innerHTML = `<tr><td colspan="10" class="empty">正在加载逐样本数据…</td></tr>`;
  try {
    const data = await api(
      `/api/runs/${encodeURIComponent(runId)}/samples?${params.toString()}`
    );
    if (seq !== state.samplesSeq) return;
    state.samplesPage = data;
    const total = Number(data.total || 0);
    const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    if (filter.page >= pages) {
      // A filter change shrank the result set: clamp to the last page once.
      filter.page = pages - 1;
      renderSamples();
      return;
    }
    const rows = data.rows || [];
    $("#samples-body").innerHTML = rows.map((sample) => `<tr>
      <td>${esc(sample.model)}</td>
      <td>${esc(sample.dataset)}</td>
      <td title="${esc(sample.sample_id)}">${esc(sample.sample_id)}${sample.truncated ? " (truncated)" : ""}</td>
      <td>${int(sample.input_context_tokens)}</td>
      <td>${int(sample.input_context_chars)}</td>
      <td>${fmt(sample.ttft_ms)}</td>
      <td>${fmt(sample.gen_tps)}</td>
      <td>${fmt(sampleMemGb(sample))}</td>
      <td>${esc(sample.finish_reason || "-")}${sample.length_capped ? ' <span class="pill soft">触顶</span>' : ""}</td>
      <td>${esc(sample.status)}</td>
    </tr>`).join("") || `<tr><td colspan="10" class="empty">${total ? "该页没有数据" : "没有逐样本结果"}</td></tr>`;
    $("#sample-count").textContent = `${total} 条`;
    $("#sample-page").textContent = `第 ${filter.page + 1}/${pages} 页`;
    $("#sample-prev").disabled = filter.page <= 0;
    $("#sample-next").disabled = filter.page >= pages - 1;
  } catch (err) {
    if (seq !== state.samplesSeq) return;
    state.samplesPage = null;
    state.samplesLoaded = false;  // allow a retry on the next tab activation
    $("#samples-body").innerHTML =
      `<tr><td colspan="10" class="empty">加载逐样本数据失败：${esc(err.message)}</td></tr>`;
    $("#sample-count").textContent = "-";
    $("#sample-page").textContent = "";
    $("#sample-prev").disabled = true;
    $("#sample-next").disabled = true;
    toast(`加载逐样本失败：${err.message}`, true);
  }
}

function renderEnvironment(data) {
  // Schema 3.0 keeps the single environment snapshot in run.json; the retired
  // per-file environment snapshots no longer exist.
  const runJson = data.run || {};
  const env = runJson.env || {};
  const hardware = env.hardware || {};
  const omlx = env.omlx || {};
  const upload = hardware.upload_fields || {};
  const perf = (data.summary || {}).perf || {};
  const run = (data.summary || {}).run || {};
  const modelEnv = env.model_env || {};
  // ``env.omlx.models`` is the rich, id-keyed snapshot (settings/flags); the
  // per-record ``model_env`` blocks are the fallback when oMLX admin calls fail.
  const modelSource = Object.keys(omlx.models || {}).length
    ? Object.entries(omlx.models)
    : Object.entries(modelEnv).map(([digest, info]) => [info?.model_name || info?.id || digest, info || {}]);
  const modelRows = modelSource.map(([id, info]) => {
    const settings = info.settings || {};
    return `<tr>
      <td>${esc(id)}</td>
      <td>${esc(info.engine_type || "-")}</td>
      <td>${esc(info.quantization || "-")}</td>
      <td>${esc(info.context_length ?? info.max_context_window ?? "-")}</td>
      <td>${esc(settings.moe_expert_offload_enabled ? "on" : "off")}</td>
      <td>${memCell(info.memory)}</td>
      <td>${esc(flagsText(info.feature_flags))}</td>
    </tr>`;
  }).join("");
  const kv = (obj) => Object.entries(obj || {}).map(([k, v]) =>
    `<dt>${esc(k)}</dt><dd>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</dd>`).join("");
  $("#environment-body").innerHTML = `
    <div class="env-block">
      <h3>硬件</h3><dl class="kv">${kv({
        chip: `${upload.chip_name || ""} ${upload.chip_variant || ""}`.trim(),
        memory_gb: upload.memory_gb,
        gpu_cores: upload.gpu_cores,
        os_version: upload.os_version,
        omlx_version: upload.omlx_version,
        cores: hardware.cpu?.logical_cores,
      })}</dl>
    </div>
    <div class="env-block">
      <h3>性能参数</h3><dl class="kv">${kv({
        samples_per_dataset: perf.samples_per_dataset ?? "全部",
        max_tokens: perf.max_tokens,
        model_size_gb: data.model_size_gb,
        context_filter: contextFilterLabel(perf.context_windows || []),
        repeats: run.repeats,
        warmup_per_model: run.warmup_per_model,
        warmup_requests_per_cell: run.warmup_requests,
        cache_mode: run.cache_mode,
        thinking_mode: run.thinking_mode,
      })}</dl>
    </div>
    <div class="env-block">
      <h3>加速特性（应为空）</h3><dl class="kv">${kv({
        feature_flags: flagsText(omlx.feature_flags),
        default_config: omlx.default_config,
        omlx_version: omlx.version ?? omlx.omlx_version,
        notes: Array.isArray(omlx.notes) ? (omlx.notes.join("; ") || "无") : (omlx.notes || "无"),
      })}</dl>
    </div>
    <div class="env-block" style="grid-column:1/-1">
      <h3>模型生效设置</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>模型</th><th>engine</th><th>量化</th><th>上下文</th><th>MoE offload</th><th title="oMLX 进程 phys_footprint：模型加载前基线 → 加载后（GB，差值=模型加载增量）">内存基线→加载(GB)</th><th>feature flags</th></tr></thead>
        <tbody>${modelRows || `<tr><td colspan="7" class="empty">没有模型配置快照</td></tr>`}</tbody>
      </table></div>
    </div>`;
}

function switchTab(name) {
  $$("#modal-tabs .tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  $$(".tab-panel").forEach((panel) => panel.classList.add("hidden"));
  const panel = $(`#tab-${name}`);
  if (panel) panel.classList.remove("hidden");
  // Pagination is lazy: the first visit to the per-sample tab fetches page 1.
  if (name === "samples" && state.detail && !state.samplesLoaded) {
    renderSamples();
  }
}

function closeModal() {
  $("#modal").classList.add("hidden");
  state.detail = null;
  state.samplesPage = null;
  state.samplesLoaded = false;
}

/* ------------------------------------------------------------------ */
/* wiring                                                             */
/* ------------------------------------------------------------------ */

async function loadOverview() {
  renderOverview(await api("/api/overview"));
}

async function restoreActiveJob() {
  // A page refresh must re-attach to a queued/running job (USER_GUIDE §3).
  try {
    const data = await api("/api/jobs");
    const active = (data.jobs || []).find((job) =>
      ["queued", "running", "cancelling"].includes(job.status));
    if (!active) return;
    if (state.poll) { clearInterval(state.poll); state.poll = null; }
    renderJob(active);
    await pollJob(active.id);
  } catch (err) {
    // Job recovery is best-effort; the page still works without it.
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("#refresh").addEventListener("click", () => loadOverview().catch((err) => toast(err.message, true)));
  $("#refresh-models").addEventListener("click", () => refreshModels());
  $("#sync-models").addEventListener("click", () => syncModels());
  $("#open-output").addEventListener("click", async () => {
    try { const res = await api("/api/output/reveal", { method: "POST", body: "{}" }); toast(`已打开 ${res.path}`); }
    catch (err) { toast(`打开失败：${err.message}`, true); }
  });
  $("#open-testset").addEventListener("click", async () => {
    try { const res = await api("/api/testsets/reveal", { method: "POST", body: "{}" }); toast(`已打开 ${res.path}`); }
    catch (err) { toast(`打开失败：${err.message}`, true); }
  });
  $("#rebuild-index").addEventListener("click", async () => {
    try {
      const res = await api("/api/index/rebuild", { method: "POST", body: "{}" });
      toast(`索引已重建：${res.n_runs} runs`);
      await loadOverview();
    } catch (err) { toast(`重建失败：${err.message}`, true); }
  });
  $("#start").addEventListener("click", startRun);
  $("#cancel").addEventListener("click", cancelRun);
  ["samples", "max-tokens", "repeats"].forEach((id) =>
    $(`#${id}`).addEventListener("input", updateEstimate));
  document.querySelectorAll("[data-models]").forEach((button) => button.addEventListener("click", () => {
    const all = button.dataset.models === "all";
    state.models.clear();
    $("#models").querySelectorAll("input").forEach((input) => {
      if (input.disabled) {
        input.checked = false;
        return;
      }
      input.checked = all;
      if (all) state.models.add(input.value);
    });
    $("#model-count").textContent = `已选 ${state.models.size}`;
    updateEstimate();
  }));
  $$("#modal-tabs .tab").forEach((tab) => tab.addEventListener("click", () => switchTab(tab.dataset.tab)));
  $("#modal-close").addEventListener("click", closeModal);
  $("#modal").addEventListener("click", (event) => { if (event.target.id === "modal") closeModal(); });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (!$("#sample-modal").classList.contains("hidden")) closeSampleModal();
    else closeModal();
  });
  // Server-side pagination: every filter change restarts at page 0 and the
  // typed query is debounced so each keystroke does not hit samples.csv.
  let sampleSearchTimer = null;
  $("#sample-search").addEventListener("input", (event) => {
    state.sampleFilter.q = event.target.value;
    state.sampleFilter.page = 0;
    clearTimeout(sampleSearchTimer);
    sampleSearchTimer = setTimeout(() => renderSamples(), 250);
  });
  ["sample-model", "sample-dataset", "sample-status"].forEach((id) =>
    $(`#${id}`).addEventListener("change", (event) => {
      const key = id.replace("sample-", "");
      state.sampleFilter[key] = event.target.value;
      state.sampleFilter.page = 0;
      renderSamples();
    }));
  $("#sample-prev").addEventListener("click", () => {
    if (!state.samplesPage) return;
    state.sampleFilter.page = Math.max(0, state.sampleFilter.page - 1);
    renderSamples();
  });
  $("#sample-next").addEventListener("click", () => {
    if (!state.samplesPage) return;
    state.sampleFilter.page += 1;
    renderSamples();
  });

  $("#sample-modal-close").addEventListener("click", closeSampleModal);
  $("#sample-modal-done").addEventListener("click", closeSampleModal);
  $("#sample-modal").addEventListener("click", (event) => { if (event.target.id === "sample-modal") closeSampleModal(); });
  $("#sample-modal-search").addEventListener("input", (event) => {
    if (state.sampleModal) state.sampleModal.filter = event.target.value;
    renderSampleModal();
  });
  $("#sample-modal-all").addEventListener("click", () => {
    if (!state.sampleModal) return;
    const st = state.scenarios.get(state.sampleModal.scenario);
    if (!st) return;
    st.manual = true;
    st.selectedIds = new Set((st.samples || []).map((sample) => sample.sample_id));
    updateManualBadge(state.sampleModal.scenario, st);
    renderSampleModal();
  });
  $("#sample-modal-none").addEventListener("click", () => {
    if (!state.sampleModal) return;
    const st = state.scenarios.get(state.sampleModal.scenario);
    if (!st) return;
    st.manual = false;
    st.selectedIds = new Set();
    updateManualBadge(state.sampleModal.scenario, st);
    renderSampleModal();
  });
  $("#sample-modal-checkall").addEventListener("change", (event) => {
    if (!state.sampleModal) return;
    const st = state.scenarios.get(state.sampleModal.scenario);
    if (!st) return;
    const samples = filteredModalSamples(st);
    if (event.target.checked) {
      samples.forEach((sample) => st.selectedIds.add(sample.sample_id));
      st.manual = st.selectedIds.size > 0;
    } else {
      samples.forEach((sample) => st.selectedIds.delete(sample.sample_id));
      st.manual = st.selectedIds.size > 0;
    }
    updateManualBadge(state.sampleModal.scenario, st);
    renderSampleModal();
  });

  loadOverview().catch((err) => toast(`加载失败：${err.message}`, true));
  restoreActiveJob();
  // Heartbeat keeps the optional --idle-timeout watchdog aware that a page is
  // still open. It is a no-op unless the server was started with that flag.
  setInterval(() => {
    fetch("/api/health", { cache: "no-store" }).catch(() => {});
  }, 30000);
});

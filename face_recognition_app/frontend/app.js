// Frontend de la app de reconocimiento facial. Solo consume la API /api/faces.
"use strict";

const API = "/api/faces";
const PAGE = 10;
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  config: null,
  health: null,
  query: null,            // {blob, url} compartida entre Reconocer y Comparar
  rec: null,              // respuesta de /recognize
  recFace: 0,
  recTab: "persons",
  recPage: 0,
  cmp: null,              // respuesta de /compare
  cmpFace: 0,
  cmpPage: 0,
  sbsPage: 0,
  regPhotos: [],
  verPhotos: [],
  persons: { page: 0, perPage: 24 },
};

// ================================================================ utilidades

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const nf = (x, d = 1) => Number(x).toLocaleString("es-PE", { minimumFractionDigits: d, maximumFractionDigits: d });
const ms = (x) => (x >= 100 ? nf(x, 0) : x >= 10 ? nf(x, 1) : nf(x, 2)) + " ms";
const pct = (x, d = 1) => nf(x, d) + " %";
const frac = (x, d = 1) => (x == null ? "–" : pct(x * 100, d));
const imgUrl = (photoId) => `${API}/images/${photoId}`;

function label(kind, id) {
  const item = state.config && state.config[kind].find((x) => x.id === id);
  return item ? item.label : id;
}
const shortIndex = (id) => ({ exact: "Exacto", hnsw: "HNSW", ivf: "IVF" }[id] || id);
const shortMetric = (id) => ({ coseno: "Coseno", euclidiana: "Euclidiana", producto_punto: "Prod. punto" }[id] || id);
const configName = (r) => `${label("models", r.model)} · ${shortIndex(r.index)} · ${shortMetric(r.metric)}`;

function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.className = "toast" + (isError ? " error" : "");
  el.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { el.hidden = true; }, isError ? 6000 : 3000);
}

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(API + path, options);
  } catch {
    throw new Error("No hay conexión con el servidor.");
  }
  let body = null;
  try { body = await res.json(); } catch { /* sin JSON */ }
  if (!res.ok) {
    let detail = body && body.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    throw new Error(detail || `Error ${res.status}`);
  }
  return body;
}

function busy(button, on, text = "Procesando…") {
  if (on) {
    button.dataset.label = button.textContent;
    button.innerHTML = `<span class="spinner"></span>${esc(text)}`;
    button.disabled = true;
    button.dataset.busy = "1";
  } else {
    button.textContent = button.dataset.label || button.textContent;
    delete button.dataset.busy;
    refreshEnabled();
  }
}

// ---------------------------------------------------------------- componentes

function segmented(el, items, value, onChange) {
  el.innerHTML = items.map((i) => `<button type="button" data-value="${esc(i.id)}">${esc(i.label)}</button>`).join("");
  const set = (v) => $$("button", el).forEach((b) => b.classList.toggle("on", b.dataset.value === v));
  set(value);
  $$("button", el).forEach((b) => b.addEventListener("click", () => { set(b.dataset.value); onChange && onChange(b.dataset.value); }));
  el.value = () => ($("button.on", el) || {}).dataset.value;
  el.set = set;
}

function chips(el, items, selected, onChange) {
  el.innerHTML = items.map((i) => `<button type="button" class="chip-toggle" data-value="${esc(i.id)}">${esc(i.label)}</button>`).join("");
  const set = (values) => $$("button", el).forEach((b) => b.classList.toggle("on", values.includes(b.dataset.value)));
  set(selected);
  $$("button", el).forEach((b) => b.addEventListener("click", () => { b.classList.toggle("on"); onChange && onChange(); }));
  el.values = () => $$("button.on", el).map((b) => b.dataset.value);
  el.set = set;
}

function pager(el, total, page, perPage, onPage, noun = "resultados") {
  const pages = Math.ceil(total / perPage);
  if (pages <= 1) {
    el.innerHTML = total ? `<span class="info">${total} ${noun}</span>` : "";
    return;
  }
  const from = page * perPage + 1;
  const to = Math.min(total, (page + 1) * perPage);
  const nums = [];
  for (let i = 0; i < pages; i++) {
    if (i === 0 || i === pages - 1 || Math.abs(i - page) <= 1) nums.push(i);
    else if (nums[nums.length - 1] !== "…") nums.push("…");
  }
  el.innerHTML = `<span class="info">${from}–${to} de ${total} ${noun}</span>
    <button data-p="${page - 1}" ${page === 0 ? "disabled" : ""}>‹</button>
    ${nums.map((n) => (n === "…" ? `<span class="muted">…</span>` : `<button data-p="${n}" class="${n === page ? "on" : ""}">${n + 1}</button>`)).join("")}
    <button data-p="${page + 1}" ${page >= pages - 1 ? "disabled" : ""}>›</button>`;
  $$("button[data-p]", el).forEach((b) => b.addEventListener("click", () => onPage(Number(b.dataset.p))));
}

function hbars(rows, { max, format, alt = false } = {}) {
  const top = max ?? Math.max(...rows.map((r) => r.value), 1e-9);
  return rows.map((r) => `
    <div class="hbar"><span class="lbl" title="${esc(r.label)}">${esc(r.label)}</span>
      <span class="track"><i class="${alt || r.alt ? "alt" : ""}" style="width:${Math.max(0, Math.min(100, (r.value / top) * 100))}%"></i></span>
      <span class="val">${format ? format(r.value) : r.value}</span></div>`).join("");
}

function stage(el, response, selected, onSelect) {
  if (!state.query) {
    el.innerHTML = `<div class="placeholder">Sube o toma una foto para empezar.</div>`;
    return;
  }
  let boxes = "";
  if (response) {
    const { width, height } = response.image;
    boxes = response.faces.map((f, i) => {
      const r = f.result || f.configs[0];
      const d = r.decision;
      const text = d.status === "known" ? `${d.name} · ${pct(d.similarity_pct)}` : `Desconocido · ${pct(d.similarity_pct)}`;
      const [x1, y1, x2, y2] = f.bbox;
      return `<div class="box ${d.status} ${response.faces.length > 1 && i !== selected ? "dim" : ""}" data-i="${i}"
        style="left:${(x1 / width) * 100}%;top:${(y1 / height) * 100}%;width:${((x2 - x1) / width) * 100}%;height:${((y2 - y1) / height) * 100}%">
        <span>${response.faces.length > 1 ? `${i + 1}. ` : ""}${esc(text)}</span></div>`;
    }).join("");
  }
  el.innerHTML = `<div class="frame"><img src="${state.query.url}" alt="Foto de consulta">${boxes}</div>`;
  $$(".box", el).forEach((b) => b.addEventListener("click", () => onSelect(Number(b.dataset.i))));
}

function facePicker(response, selected, onSelect) {
  if (response.faces.length < 2) return "";
  return `<div class="face-picker">${response.faces.map((_, i) =>
    `<button class="btn small ${i === selected ? "primary" : ""}" data-face="${i}">Rostro ${i + 1}</button>`).join("")}</div>`;
}
function bindFacePicker(root, onSelect) {
  $$("[data-face]", root).forEach((b) => b.addEventListener("click", () => onSelect(Number(b.dataset.face))));
}

// ================================================================ estado del sistema

function engineReady() {
  // "idle" = modelos aún no cargados pero se cargan en la primera consulta (precarga desactivada)
  return Boolean(state.health) && ["ready", "idle"].includes(state.health.engine);
}

function refreshEnabled() {
  const ready = engineReady();
  $$(".needs-engine").forEach((b) => {
    if (b.dataset.busy) return;
    let ok = ready;
    if (b.id === "rec-run" || b.id === "cmp-run") ok = ok && Boolean(state.query);
    if (b.id === "cmp-run") ok = ok && compareCount() > 0;
    if (b.id === "reg-submit") ok = ok && state.regPhotos.length > 0 && $("#reg-name").value.trim().length > 0;
    if (b.id === "ver-run") ok = ok && state.verPhotos.length >= 2 && state.verPhotos.length <= maxVerify();
    b.disabled = !ok;
  });
  $("#rec-to-compare").disabled = !state.query;
}

async function pollHealth() {
  let next = 15000;
  const status = $("#status");
  const banner = $("#banner");
  try {
    const h = await api("/health");
    state.health = h;
    const tasks = h.tasks || [];
    if (h.engine === "error") {
      status.className = "status down";
      $("#status-text").textContent = "Error en los modelos";
      showBanner("No se pudieron cargar los modelos de IA", h.engine_error || "", true);
    } else if (h.engine === "loading") {
      status.className = "status loading";
      $("#status-text").textContent = "Cargando…";
      showBanner("Preparando el sistema", "Cargando los modelos de IA. Reconocer, comparar y registrar estarán disponibles en unos segundos.");
      next = 2000;
    } else if (tasks.length) {
      status.className = "status loading";
      $("#status-text").textContent = "Trabajando…";
      showBanner("Tarea en curso", tasks.map((t) => `${t.label} (${nf(t.elapsed_s, 0)} s)`).join(" · ") + ". Las consultas pueden demorar hasta que termine.");
      next = 2000;
    } else {
      status.className = "status ready";
      $("#status-text").textContent = "Listo";
      banner.hidden = true;
    }
  } catch (e) {
    state.health = null;
    status.className = "status down";
    $("#status-text").textContent = "Sin conexión";
    showBanner("Sin conexión con el servidor", "Reintentando automáticamente…", true);
    next = 3000;
  }
  refreshEnabled();
  clearTimeout(pollHealth.t);
  pollHealth.t = setTimeout(pollHealth, next);
}

function showBanner(title, detail, down = false) {
  const b = $("#banner");
  $("#banner-title").textContent = title;
  $("#banner-detail").textContent = detail;
  b.classList.toggle("down", down);
  $(".spinner", b).hidden = down;
  b.hidden = false;
}

// ================================================================ pestañas

function openTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "persons") loadPersons();
  if (name === "performance") loadPerformance();
  if (name === "data") loadData();
  if (name === "compare") stage($("#cmp-stage"), state.cmp, state.cmpFace, selectCmpFace);
  if (name === "recognize") stage($("#rec-stage"), state.rec, state.recFace, selectRecFace);
}
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  history.replaceState(null, "", "#" + t.dataset.tab);
  openTab(t.dataset.tab);
}));

// ================================================================ cámara

const camera = {
  stream: null,
  cb: null,
  async open(cb) {
    this.cb = cb;
    $("#camera-error").hidden = true;
    $("#camera-dialog").showModal();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      $("#camera-video").srcObject = this.stream;
    } catch (e) {
      $("#camera-error").textContent = "No se pudo abrir la cámara: " + e.message;
      $("#camera-error").hidden = false;
    }
  },
  close() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    $("#camera-dialog").close();
  },
  shoot() {
    const v = $("#camera-video");
    if (!v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob((blob) => { const cb = this.cb; this.close(); cb && cb(blob, `camara_${Date.now()}.jpg`); }, "image/jpeg", 0.92);
  },
};
$("#camera-shot").addEventListener("click", () => camera.shoot());
$("#camera-close").addEventListener("click", () => camera.close());
$("#camera-dialog").addEventListener("cancel", () => camera.close());

// ================================================================ consulta compartida

function setQuery(blob, autorun) {
  if (state.query) URL.revokeObjectURL(state.query.url);
  state.query = { blob, url: URL.createObjectURL(blob), name: blob.name || "consulta.jpg" };
  state.rec = null;
  state.cmp = null;
  $("#rec-results").hidden = true;
  $("#cmp-results").hidden = true;
  stage($("#rec-stage"), null);
  stage($("#cmp-stage"), null);
  refreshEnabled();
  if (autorun && engineReady()) autorun();
}

$$(".query-file").forEach((input) => input.addEventListener("change", (e) => {
  if (e.target.files[0]) setQuery(e.target.files[0]);
  e.target.value = "";
}));
$$(".query-camera").forEach((btn) => btn.addEventListener("click", () => {
  const inCompare = Boolean(btn.closest("#tab-compare"));
  camera.open((blob) => setQuery(blob, inCompare ? runCompare : recognize));
}));

// ================================================================ RECONOCER

function threshold() { return Number($("#p-threshold").value) / 100; }

function updateParams() {
  const index = $("#p-index").value();
  $("#wrap-ef").hidden = index !== "hnsw";
  $("#wrap-nprobe").hidden = index !== "ivf";
  $("#p-threshold-val").textContent = pct(threshold() * 100, 0);
  const cal = state.config.thresholds[$("#p-model").value()];
  const isCal = Math.round(cal * 100) === Math.round(threshold() * 100);
  $("#p-threshold-reset").textContent = isCal ? "(calibrado)" : `restaurar calibrado (${pct(cal * 100, 0)})`;
}

function resetThreshold() {
  $("#p-threshold").value = Math.round(state.config.thresholds[$("#p-model").value()] * 100);
  updateParams();
}

function recParams() {
  return {
    model: $("#p-model").value(),
    index: $("#p-index").value(),
    metric: $("#p-metric").value(),
    k: Number($("#p-k").value) || 5,
    threshold: threshold(),
    ef_search: Number($("#p-ef").value) || 50,
    nprobe: Number($("#p-nprobe").value) || 8,
  };
}

async function recognize() {
  if (!state.query) return;
  const btn = $("#rec-run");
  busy(btn, true, "Reconociendo…");
  try {
    const form = new FormData();
    form.append("image", state.query.blob, state.query.name);
    Object.entries(recParams()).forEach(([k, v]) => form.append(k, String(v)));
    state.rec = await api("/recognize", { method: "POST", body: form });
    state.recFace = 0;
    state.recPage = 0;
    stage($("#rec-stage"), state.rec, 0, selectRecFace);
    renderRecognition();
  } catch (e) {
    toast(e.message, true);
  } finally {
    busy(btn, false);
  }
}

function selectRecFace(i) {
  state.recFace = i;
  state.recPage = 0;
  stage($("#rec-stage"), state.rec, i, selectRecFace);
  renderRecognition();
}

function timingLine(resp, r) {
  const embed = Object.values(resp.timings.embed_ms).reduce((a, b) => a + b, 0);
  return `<div class="metrics-line">
      <div class="main"><span>Búsqueda en chutaDB</span><b>${ms(r.search_ms)}</b></div>
      <div><span>Detección del rostro</span><b>${ms(resp.timings.detect_ms)}</b></div>
      <div><span>Embedding</span><b>${ms(embed)}</b></div>
      <div><span>Tiempo total</span><b>${ms(resp.timings.total_ms)}</b></div>
      <div><span>Rostros evaluados</span><b>${r.candidates.toLocaleString("es-PE")} de ${r.gallery_size.toLocaleString("es-PE")}</b></div>
    </div>`;
}

function renderRecognition() {
  const resp = state.rec;
  const box = $("#rec-results");
  box.hidden = false;
  if (!resp.faces.length) {
    box.innerHTML = `<div class="verdict unknown"><div class="who"><div class="kicker">Sin rostros</div>
      <div class="name">No se detectó ningún rostro</div><div class="sub">Prueba con una foto más cercana, de frente y bien iluminada.</div></div></div>`;
    return;
  }
  const r = resp.faces[state.recFace].result;
  const d = r.decision;
  const best = r.persons[0];
  const score = `<div class="score"><b>${pct(d.similarity_pct)}</b><span>similitud · umbral ${pct(d.threshold_pct, 0)}</span>
      ${best ? `<span>distancia ${esc(shortMetric(r.metric).toLowerCase())}: ${nf(best.distance, 4)}</span>` : ""}</div>`;
  const verdict = d.status === "known"
    ? `<div class="verdict known"><img src="${imgUrl(best.best_photo_id)}" alt="">
        <div class="who"><div class="kicker">Persona reconocida</div><div class="name">${esc(d.name)}</div>
          <div class="sub">${best.votes} de sus fotos entre los ${r.k} vecinos más cercanos</div></div>${score}</div>`
    : `<div class="verdict unknown">${best ? `<img src="${imgUrl(best.best_photo_id)}" alt="">` : ""}
        <div class="who"><div class="kicker">No reconocido</div><div class="name">Persona desconocida</div>
          <div class="sub">${best ? `La más parecida es ${esc(best.name)}, pero no alcanza el umbral` : "Galería vacía"}</div></div>${score}</div>`;

  const params = r.index === "hnsw" ? ` · ef_search ${r.params.ef_search}` : r.index === "ivf" ? ` · nprobe ${r.params.nprobe}` : "";
  box.innerHTML = `
    ${facePicker(resp, state.recFace)}
    ${verdict}
    ${timingLine(resp, r)}
    <div class="config-line">${esc(label("models", r.model))} · ${esc(label("indexes", r.index))} · ${esc(label("metrics", r.metric))} · k = ${r.k}${params}</div>
    <div class="subtabs">
      <button class="subtab ${state.recTab === "persons" ? "on" : ""}" data-sub="persons">Personas (${r.persons.length})</button>
      <button class="subtab ${state.recTab === "photos" ? "on" : ""}" data-sub="photos">Fotos más parecidas (${r.matches.length})</button>
      <i class="tip push" data-tip="Similitud (%) = similitud coseno × 100, igual para las tres métricas para poder compararlas. Distancia = valor de la métrica elegida, el que usa el índice para ordenar (menor = más parecido). Como los embeddings están normalizados, las tres métricas ordenan igual: al cambiarla cambia la distancia, no el %. Con HNSW/IVF puede variar algún vecino porque cada métrica tiene su propio índice aproximado.">i</i>
    </div>
    <div id="rec-list"></div>
    <div id="rec-pager" class="pager"></div>`;
  bindFacePicker(box, selectRecFace);
  $$(".subtab", box).forEach((b) => b.addEventListener("click", () => { state.recTab = b.dataset.sub; state.recPage = 0; renderRecognition(); }));
  renderRecList(r);
}

function renderRecList(r) {
  const list = $("#rec-list");
  const items = state.recTab === "persons" ? r.persons : r.matches;
  const slice = items.slice(state.recPage * PAGE, (state.recPage + 1) * PAGE);
  const th = r.decision.threshold * 100;
  if (state.recTab === "persons") {
    list.innerHTML = `<div class="list">
      <div class="row head"><span></span><span></span><span>Persona</span><span>Similitud</span><span></span><span style="text-align:right">Distancia (${esc(shortMetric(r.metric).toLowerCase())})</span></div>
      ${slice.map((p) => `<div class="row">
        <span class="rank">${p.rank}</span><img src="${imgUrl(p.best_photo_id)}" alt="" loading="lazy">
        <div><div class="name">${esc(p.name)}</div><div class="sub">${p.votes} ${p.votes === 1 ? "foto" : "fotos"} en el top-${r.k}</div></div>
        <div class="bar ${p.similarity_pct < th ? "low" : ""}"><i style="width:${p.similarity_pct}%"></i></div>
        <span class="pct">${pct(p.similarity_pct)}</span><span class="dist">${nf(p.distance, 4)}</span></div>`).join("")}</div>`;
  } else {
    list.innerHTML = `<div class="photo-grid">${slice.map((m) => `
      <div class="photo"><img src="${imgUrl(m.photo_id)}" alt="" loading="lazy">
        <div class="meta"><div class="n">${m.rank}. ${esc(m.name)}</div>
          <div class="s"><b>${pct(m.similarity_pct)}</b><span>d = ${nf(m.distance, 3)}</span></div></div></div>`).join("")}</div>`;
  }
  pager($("#rec-pager"), items.length, state.recPage, PAGE, (p) => { state.recPage = p; renderRecList(r); },
    state.recTab === "persons" ? "personas" : "fotos");
}

$("#rec-run").addEventListener("click", recognize);
$("#rec-to-compare").addEventListener("click", () => {
  // misma foto: ambos modelos × los tres índices, con la métrica elegida en Reconocer; se ejecuta directo
  applyRecParamsToCompare();
  $("#c-models").set(state.config.models.map((m) => m.id));
  $("#c-indexes").set(state.config.indexes.map((i) => i.id));
  $("#c-metrics").set([$("#p-metric").value()]);
  updateCompareCount();
  history.replaceState(null, "", "#compare");
  openTab("compare");
  setCompareMode("configs");
  runCompare();
});
$("#p-threshold").addEventListener("input", updateParams);
$("#p-threshold-reset").addEventListener("click", (e) => { e.preventDefault(); resetThreshold(); });

// ================================================================ COMPARAR

function setCompareMode(mode) {
  $("#cmp-mode").set(mode);
  $("#cmp-configs").hidden = mode !== "configs";
  $("#cmp-pairs").hidden = mode !== "pairs";
}

function compareCount() {
  if (!$("#c-models").values) return 0;
  return $("#c-models").values().length * $("#c-indexes").values().length * $("#c-metrics").values().length;
}

function updateCompareCount() {
  const n = compareCount();
  const sizes = [$("#c-models"), $("#c-indexes"), $("#c-metrics")].map((el) => el.values().length);
  $("#c-count").textContent = n
    ? `Se compararán ${n} configuraciones (${sizes.join(" × ")}).`
    : "Elige al menos un modelo, un índice y una métrica.";
  refreshEnabled();
}

function applyRecParamsToCompare() {
  const p = recParams();
  $("#c-k").value = p.k;
  $("#c-ef").value = p.ef_search;
  $("#c-nprobe").value = p.nprobe;
}

function quickCompare(kind) {
  const p = recParams();
  const all = (k) => state.config[k].map((x) => x.id);
  $("#c-models").set(kind === "models" || kind === "all" ? all("models") : [p.model]);
  $("#c-indexes").set(kind === "indexes" || kind === "all" ? all("indexes") : [p.index]);
  $("#c-metrics").set(kind === "metrics" || kind === "all" ? all("metrics") : [p.metric]);
  updateCompareCount();
}

async function runCompare() {
  if (!state.query || !compareCount()) return;
  const btn = $("#cmp-run");
  busy(btn, true, "Comparando…");
  try {
    const p = recParams();
    const form = new FormData();
    form.append("image", state.query.blob, state.query.name);
    form.append("models", $("#c-models").values().join(","));
    form.append("indexes", $("#c-indexes").values().join(","));
    form.append("metrics", $("#c-metrics").values().join(","));
    form.append("k", String(Number($("#c-k").value) || 5));
    form.append("ef_search", String(Number($("#c-ef").value) || 50));
    form.append("nprobe", String(Number($("#c-nprobe").value) || 8));
    form.append("threshold", String(p.threshold));
    form.append("threshold_model", p.model);
    state.cmp = await api("/compare", { method: "POST", body: form });
    state.cmpFace = 0;
    state.cmpPage = 0;
    state.sbsPage = 0;
    stage($("#cmp-stage"), state.cmp, 0, selectCmpFace);
    renderCompare();
  } catch (e) {
    toast(e.message, true);
  } finally {
    busy(btn, false);
  }
}

function selectCmpFace(i) {
  state.cmpFace = i;
  state.cmpPage = 0;
  state.sbsPage = 0;
  stage($("#cmp-stage"), state.cmp, i, selectCmpFace);
  renderCompare();
}

function renderCompare() {
  const resp = state.cmp;
  const box = $("#cmp-results");
  box.hidden = false;
  if (!resp.faces.length) {
    box.innerHTML = `<div class="card"><div class="empty-state">No se detectó ningún rostro en la foto.</div></div>`;
    return;
  }
  const rows = resp.faces[state.cmpFace].configs;
  const k = resp.params.k;
  const embed = Object.entries(resp.timings.embed_ms).map(([m, v]) => `${label("models", m)} ${ms(v)}`).join(" · ");

  box.innerHTML = `
    <div class="card">
      ${facePicker(resp, state.cmpFace)}
      <div class="card-title"><h2>Resumen</h2>
        <span class="muted small">Detección ${ms(resp.timings.detect_ms)} · Embedding ${embed} · k = ${k}</span></div>
      <div class="table-wrap"><table>
        <thead><tr><th>Configuración</th><th>Persona más parecida</th><th>Similitud</th><th>Decisión</th>
          <th class="r">Igual al exacto <i class="tip" data-tip="Recall@${k}: de las ${k} fotos que devuelve la búsqueda exacta (que revisa toda la galería), cuántas devolvió también este índice. No dice si la persona es la correcta (en una foto cualquiera no lo sabemos): mide cuánto se pierde por usar un índice aproximado. Que la persona sea la correcta se mide en Rendimiento, con fotos de identidad conocida.">i</i></th>
          <th class="r">Búsqueda</th>
          <th class="r">Distancias calculadas <i class="tip" data-tip="Cuántos rostros de la galería comparó el índice con la consulta. Exacto compara con todos; HNSW recorre solo parte del grafo e IVF solo las celdas más cercanas (nprobe). Es el trabajo que se ahorra cada índice.">i</i></th></tr></thead>
        <tbody id="cmp-tbody"></tbody></table></div>
      <div id="cmp-pager" class="pager"></div>
    </div>
    <div class="card">
      <div class="card-title"><h2>Comparación visual</h2></div>
      <div class="charts">
        <div class="chart"><h3>Tiempo de búsqueda</h3><div class="note">Milisegundos dentro de chutaDB. Menor es mejor.</div>
          ${hbars(rows.map((r) => ({ label: configName(r), value: r.search_ms })), { format: ms })}</div>
        <div class="chart"><h3>Igual a la búsqueda exacta (Recall@${k})</h3><div class="note">Cuántas de las ${k} fotos del resultado exacto recupera cada índice.</div>
          ${hbars(rows.map((r) => ({ label: configName(r), value: (r.recall_vs_exact ?? 0) * 100 })), { max: 100, format: (v) => pct(v, 0) })}</div>
        <div class="chart"><h3>Similitud de la persona más parecida</h3><div class="note">Si es igual en todas las filas de un modelo, todos los índices encontraron la misma foto más parecida. Los modelos tienen escalas distintas: compárala con su umbral.</div>
          ${hbars(rows.map((r) => ({ label: configName(r), value: r.persons[0] ? r.persons[0].similarity_pct : 0, alt: r.decision.status !== "known" })), { max: 100, format: (v) => pct(v) })}</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title"><h2>Vecinos lado a lado</h2>
        <div class="segmented" id="sbs-mode" style="width:auto"></div></div>
      <div class="table-wrap"><table class="sbs" id="sbs-table"></table></div>
      <div class="legend" id="sbs-legend"></div>
      <div id="sbs-pager" class="pager"></div>
    </div>`;
  segmented($("#sbs-mode"), [{ id: "photos", label: "Por foto" }, { id: "persons", label: "Por persona" }], state.sbsMode || "photos",
    (v) => { state.sbsMode = v; state.sbsPage = 0; renderSideBySide(rows, k); });
  bindFacePicker(box, selectCmpFace);
  renderCompareTable(rows, k);
  renderSideBySide(rows, k);
}

function renderCompareTable(rows, k) {
  const slice = rows.slice(state.cmpPage * PAGE, (state.cmpPage + 1) * PAGE);
  $("#cmp-tbody").innerHTML = slice.map((r) => {
    const top = r.persons[0];
    const d = r.decision;
    return `<tr><td>${esc(configName(r))}</td>
      <td>${top ? esc(top.name) : "–"}</td>
      <td><span class="bar ${d.status !== "known" ? "low" : ""}"><i style="width:${top ? top.similarity_pct : 0}%"></i></span>${top ? pct(top.similarity_pct) : "–"}</td>
      <td><span class="tag ${d.status}">${d.status === "known" ? "Reconocido" : "Desconocido"}</span> <span class="muted small">umbral ${pct(d.threshold_pct, 0)}</span></td>
      <td class="r">${frac(r.recall_vs_exact, 0)}</td>
      <td class="r">${ms(r.search_ms)}</td>
      <td class="r">${r.candidates.toLocaleString("es-PE")}</td></tr>`;
  }).join("");
  pager($("#cmp-pager"), rows.length, state.cmpPage, PAGE, (p) => { state.cmpPage = p; renderCompareTable(rows, k); }, "configuraciones");
}

function renderSideBySide(rows, k) {
  // Por foto: los k vecinos tal cual (una persona con varias fotos aparece varias veces).
  // Por persona: agrupados, cada persona una vez con su foto más parecida.
  const byPerson = state.sbsMode === "persons";
  const list = (r) => (byPerson ? r.persons : r.matches);
  const inExact = (r, i) => (byPerson ? (r.in_exact_person ? r.in_exact_person[i] : true) : (r.in_exact ? r.in_exact[i] : true));
  const ranks = Math.max(...rows.map((r) => list(r).length));
  const from = state.sbsPage * PAGE;
  const to = Math.min(ranks, from + PAGE);
  let body = "";
  for (let i = from; i < to; i++) {
    body += `<tr><td class="muted r">${i + 1}</td>${rows.map((r) => {
      const m = list(r)[i];
      if (!m) return "<td></td>";
      const extra = byPerson ? ` · ${m.votes} ${m.votes === 1 ? "foto" : "fotos"}` : "";
      return `<td class="${inExact(r, i) ? "" : "miss"}"><div class="cell-n">${esc(m.name)}</div><div class="cell-s">${pct(m.similarity_pct)}${extra}</div></td>`;
    }).join("")}</tr>`;
  }
  $("#sbs-table").innerHTML = `<thead><tr><th class="r">#</th>${rows.map((r) => `<th>${esc(configName(r))}</th>`).join("")}</tr></thead><tbody>${body}</tbody>`;
  $("#sbs-legend").innerHTML = byPerson
    ? `<span>Cada persona aparece una vez (su foto más parecida) · similitud · cuántas de sus fotos están entre los ${k} vecinos</span><span class="warn">En naranja: persona que no aparece en el resultado exacto</span>`
    : `<span>Los ${k} vecinos son fotos: una persona con varias fotos parecidas aparece varias veces</span><span class="warn">En naranja: foto que no está en el top-${k} de la búsqueda exacta</span>`;
  pager($("#sbs-pager"), ranks, state.sbsPage, PAGE, (p) => { state.sbsPage = p; renderSideBySide(rows, k); }, "posiciones");
}

$("#cmp-run").addEventListener("click", runCompare);
$$("[data-quick]").forEach((b) => b.addEventListener("click", () => {
  quickCompare(b.dataset.quick);
  if (state.query && engineReady()) runCompare();
}));
["#c-k", "#c-ef", "#c-nprobe"].forEach((s) => $(s).addEventListener("input", refreshEnabled));

// ---------------------------------------------------------------- foto vs foto (1:1)

const maxVerify = () => (state.config ? state.config.max_verify_images : 6);

function thumbs(container, photos, onRemove, badges = []) {
  container.innerHTML = photos.map((p, i) => `
    <div class="thumb"><img src="${p.url}" alt="">
      <button class="x" title="Quitar" data-i="${i}">×</button>
      ${badges[i] ? `<span class="badge ${badges[i].cls || ""}">${esc(badges[i].text)}</span>` : ""}</div>`).join("");
  $$(".x", container).forEach((b) => b.addEventListener("click", () => onRemove(Number(b.dataset.i))));
}

function addFiles(list, files) {
  for (const f of files) list.push({ blob: f, url: URL.createObjectURL(f), name: f.name || "foto.jpg" });
}

function renderVerifyThumbs() {
  thumbs($("#ver-thumbs"), state.verPhotos, (i) => {
    URL.revokeObjectURL(state.verPhotos[i].url);
    state.verPhotos.splice(i, 1);
    renderVerifyThumbs();
  }, state.verPhotos.map((_, i) => ({ text: `#${i + 1}` })));
  refreshEnabled();
}

function cellStyle(v, same) {
  if (v == null) return "";
  return same ? `background:rgba(47,191,113,${0.15 + 0.5 * (v / 100)})` : `background:rgba(240,82,82,${Math.max(0.1, 0.35 - v / 300)})`;
}

function matrixHtml(model, r, photos) {
  const head = `<tr><th></th>${photos.map((p, i) => `<th><img src="${p.url}" alt="">#${i + 1}</th>`).join("")}</tr>`;
  const body = r.similarity_pct.map((row, i) => `<tr><th><img src="${photos[i].url}" alt="">#${i + 1}</th>${row.map((v, j) =>
    `<td class="cell" style="${i === j ? "" : cellStyle(v, r.same_person[i][j])}">${v == null ? "–" : i === j ? "" : pct(v, 0) + (r.same_person[i][j] ? " ✓" : "")}</td>`).join("")}</tr>`).join("");
  return `<div class="card"><div class="card-title"><h2>${esc(label("models", model))}</h2>
    <span class="muted small">✓ misma persona (≥ ${pct(r.threshold_pct, 0)})</span></div>
    <div class="table-wrap"><table class="matrix">${head}${body}</table></div></div>`;
}

$("#ver-files").addEventListener("change", (e) => { addFiles(state.verPhotos, e.target.files); e.target.value = ""; renderVerifyThumbs(); });
$("#ver-camera").addEventListener("click", () => camera.open((blob, name) => { state.verPhotos.push({ blob, url: URL.createObjectURL(blob), name }); renderVerifyThumbs(); }));
$("#ver-run").addEventListener("click", async () => {
  const btn = $("#ver-run");
  busy(btn, true, "Comparando…");
  try {
    const form = new FormData();
    state.verPhotos.forEach((p) => form.append("images", p.blob, p.name));
    const res = await api("/verify", { method: "POST", body: form });
    const issues = res.images.map((img, i) => (img.status !== "ok" ? `<li><span class="bad">✕</span> #${i + 1} ${esc(img.filename)}: ${esc(img.error)}</li>` : "")).join("");
    $("#ver-result").innerHTML = (issues ? `<div class="card"><ul class="status-list">${issues}</ul></div>` : "")
      + `<div class="pair-grid">${Object.entries(res.models).map(([m, r]) => matrixHtml(m, r, state.verPhotos)).join("")}</div>`;
  } catch (e) {
    toast(e.message, true);
  } finally {
    busy(btn, false);
  }
});

// ================================================================ REGISTRAR

let regCheckTimer = null;
let regBadges = [];

function renderRegister() {
  thumbs($("#reg-thumbs"), state.regPhotos, (i) => {
    URL.revokeObjectURL(state.regPhotos[i].url);
    state.regPhotos.splice(i, 1);
    regBadges = [];
    renderRegister();
    scheduleRegCheck();
  }, regBadges);
  refreshEnabled();
}

function scheduleRegCheck() {
  clearTimeout(regCheckTimer);
  const n = state.regPhotos.length;
  if (n < 2) { $("#reg-check").textContent = ""; return; }
  if (n > maxVerify()) { $("#reg-check").textContent = `Revisión de consistencia disponible hasta ${maxVerify()} fotos.`; return; }
  if (!engineReady()) return;
  $("#reg-check").innerHTML = `<span class="muted">Revisando que las fotos sean de la misma persona…</span>`;
  regCheckTimer = setTimeout(checkRegisterPhotos, 400);
}

async function checkRegisterPhotos() {
  const photos = [...state.regPhotos];
  try {
    const form = new FormData();
    photos.forEach((p) => form.append("images", p.blob, p.name));
    const res = await api("/verify", { method: "POST", body: form });
    if (photos.length !== state.regPhotos.length) return;          // cambió mientras tanto
    const model = state.config.defaults.model;
    const r = res.models[model];
    regBadges = res.images.map((img, i) => {
      if (img.status !== "ok") return { text: "sin rostro", cls: "bad" };
      if (r.outliers.includes(i)) return { text: "distinta", cls: "bad" };
      return { text: "✓", cls: "good" };
    });
    const bad = regBadges.filter((b) => b.cls === "bad").length;
    $("#reg-check").innerHTML = bad
      ? `<span class="warn">${bad} ${bad === 1 ? "foto no parece" : "fotos no parecen"} de la misma persona que las demás (o no tienen rostro). Revísalas antes de registrar.</span>`
      : `<span class="ok">Las ${photos.length} fotos parecen de la misma persona (${esc(label("models", model))}, umbral ${pct(r.threshold_pct, 0)}).</span>`;
    renderRegister();
  } catch (e) {
    $("#reg-check").textContent = "";
  }
}

function photoReport(photos) {
  return `<ul class="status-list">${photos.map((p) => (p.status === "ok"
    ? `<li><span class="ok">✓</span><span>${esc(p.filename)}${p.warning ? ` <span class="warn">— ${esc(p.warning)}</span>` : ""}</span></li>`
    : `<li><span class="bad">✕</span><span>${esc(p.filename)} — ${esc(p.error)}</span></li>`)).join("")}</ul>`;
}

$("#reg-files").addEventListener("change", (e) => { addFiles(state.regPhotos, e.target.files); e.target.value = ""; regBadges = []; renderRegister(); scheduleRegCheck(); });
$("#reg-camera").addEventListener("click", () => camera.open((blob, name) => {
  state.regPhotos.push({ blob, url: URL.createObjectURL(blob), name });
  regBadges = [];
  renderRegister();
  scheduleRegCheck();
}));
$("#reg-name").addEventListener("input", refreshEnabled);
$("#reg-submit").addEventListener("click", async () => {
  const btn = $("#reg-submit");
  busy(btn, true, "Procesando fotos…");
  try {
    const form = new FormData();
    form.append("name", $("#reg-name").value.trim());
    state.regPhotos.forEach((p) => form.append("photos", p.blob, p.name));
    const res = await api("/persons", { method: "POST", body: form });
    const first = res.photos.find((p) => p.photo_id);
    $("#reg-result").innerHTML = `<div class="card-title"><h2>Resultado</h2></div>
      <div class="verdict known">${first ? `<img src="${imgUrl(first.photo_id)}" alt="">` : ""}
        <div class="who"><div class="kicker">Persona registrada</div><div class="name">${esc(res.person.name)}</div>
        <div class="sub">${res.person.photo_count} ${res.person.photo_count === 1 ? "foto guardada" : "fotos guardadas"} · id ${res.person.person_id}</div></div></div>
      ${photoReport(res.photos)}`;
    state.regPhotos.forEach((p) => URL.revokeObjectURL(p.url));
    state.regPhotos = [];
    regBadges = [];
    $("#reg-name").value = "";
    $("#reg-check").textContent = "";
    toast("Persona registrada");
  } catch (e) {
    toast(e.message, true);
  } finally {
    busy(btn, false);
    renderRegister();
  }
});

// ================================================================ PERSONAS

let personsTimer = null;
$("#per-q").addEventListener("input", () => { clearTimeout(personsTimer); personsTimer = setTimeout(() => { state.persons.page = 0; loadPersons(); }, 250); });
$("#per-source").addEventListener("change", () => { state.persons.page = 0; loadPersons(); });
$("#per-sort").addEventListener("change", () => { state.persons.page = 0; loadPersons(); });

async function loadPersons() {
  const { page, perPage } = state.persons;
  try {
    const params = new URLSearchParams({ q: $("#per-q").value, limit: perPage, offset: page * perPage, sort: $("#per-sort").value });
    if ($("#per-source").value) params.set("source", $("#per-source").value);
    const res = await api("/persons?" + params);
    $("#per-count").textContent = `${res.total.toLocaleString("es-PE")} personas`;
    $("#per-list").innerHTML = res.persons.length ? res.persons.map((p) => `
      <button class="person-card" data-id="${p.person_id}">
        ${p.photo_ids.length ? `<img src="${imgUrl(p.photo_ids[0])}" alt="" loading="lazy">` : ""}
        <div class="n">${esc(p.name)}</div>
        <div class="s">${p.photo_count} ${p.photo_count === 1 ? "foto" : "fotos"} · ${p.source === "own" ? "registrada aquí" : "LFW"}</div>
      </button>`).join("") : `<div class="empty-state">No hay personas que coincidan.</div>`;
    $$(".person-card").forEach((c) => c.addEventListener("click", () => showPerson(Number(c.dataset.id))));
    pager($("#per-pager"), res.total, page, perPage, (p) => { state.persons.page = p; loadPersons(); }, "personas");
  } catch (e) {
    toast(e.message, true);
  }
}

async function showPerson(id) {
  const dlg = $("#person-dialog");
  const body = $("#person-body");
  try {
    const p = await api(`/persons/${id}`);
    body.innerHTML = `
      <div class="card-title"><div><h2>${esc(p.name)}</h2>
        <span class="muted small">id ${p.person_id} · ${p.source === "own" ? "registrada aquí" : "dataset LFW"} · ${p.photos.length} fotos</span></div>
        <button class="btn" id="pd-close">Cerrar</button></div>
      <div class="thumbs">${p.photos.map((ph) => `<div class="thumb"><img src="${imgUrl(ph.photo_id)}" alt="" loading="lazy"></div>`).join("")}</div>
      <div id="pd-msg"></div>
      <div class="actions">
        <label class="btn needs-engine-label">Agregar fotos<input id="pd-files" type="file" accept="image/*" multiple hidden></label>
        <button class="btn" id="pd-camera">Tomar foto</button>
        <button class="btn danger push" id="pd-delete">Eliminar persona</button>
      </div>`;
    if (!dlg.open) dlg.showModal();
    $("#pd-close").addEventListener("click", () => dlg.close());
    const upload = async (files) => {
      const form = new FormData();
      files.forEach(([b, n]) => form.append("photos", b, n));
      $("#pd-msg").innerHTML = `<span class="muted small"><span class="spinner"></span> Procesando…</span>`;
      try {
        const res = await api(`/persons/${id}/photos`, { method: "POST", body: form });
        await showPerson(id);
        $("#pd-msg").innerHTML = photoReport(res.photos);
        loadPersons();
      } catch (e) { $("#pd-msg").innerHTML = ""; toast(e.message, true); }
    };
    $("#pd-files").addEventListener("change", (e) => upload([...e.target.files].map((f) => [f, f.name])));
    $("#pd-camera").addEventListener("click", () => { dlg.close(); camera.open((b, n) => { showPerson(id).then(() => upload([[b, n]])); }); });
    const del = $("#pd-delete");
    del.addEventListener("click", async () => {
      if (!del.dataset.confirm) {
        del.dataset.confirm = "1";
        del.textContent = "Confirmar eliminación";
        setTimeout(() => { delete del.dataset.confirm; del.textContent = "Eliminar persona"; }, 4000);
        return;
      }
      try {
        await api(`/persons/${id}`, { method: "DELETE" });
        dlg.close();
        toast("Persona eliminada");
        loadPersons();
      } catch (e) { toast(e.message, true); }
    });
  } catch (e) {
    toast(e.message, true);
  }
}

// ================================================================ RENDIMIENTO

let evalTimer = null;

async function loadPerformance() {
  await refreshEvalStatus();
  const box = $("#perf-content");
  let ev;
  try {
    ev = await api("/evaluation/last");
  } catch {
    box.innerHTML = `<div class="card"><div class="empty-state">Todavía no hay resultados. Ejecuta una evaluación.</div></div>`;
    return;
  }
  const rows = ev.rows;
  const models = [...new Set(rows.map((r) => r.model))];
  const find = (m, i, met) => rows.find((r) => r.model === m && r.index === i && r.metric === met);
  const metric = "coseno";
  const k = ev.k;

  const quality = models.map((m) => {
    const r = find(m, "exact", metric);
    return `<div class="hgroup"><div class="hgroup-title">${esc(label("models", m))}</div>${hbars([
      { label: "Acierto (Top-1)", value: r.top1 * 100 },
      { label: "Decisión correcta", value: r.decision_accuracy * 100 },
      { label: "Desconocido aceptado (FAR)", value: r.far * 100, alt: true },
      { label: "Registrado rechazado (FRR)", value: r.frr * 100, alt: true },
    ], { max: 100, format: (v) => pct(v) })}</div>`;
  }).join("");

  const recall = models.map((m) => `<div class="hgroup"><div class="hgroup-title">${esc(label("models", m))}</div>${hbars(
    ["exact", "hnsw", "ivf"].filter((i) => find(m, i, metric)).map((i) => ({ label: label("indexes", i), value: find(m, i, metric).recall_at_k * 100 })),
    { max: 100, format: (v) => pct(v) })}</div>`).join("");

  const allLat = rows.filter((r) => r.metric === metric).map((r) => r.latency_mean_ms);
  const latency = models.map((m) => `<div class="hgroup"><div class="hgroup-title">${esc(label("models", m))}</div>${hbars(
    ["exact", "hnsw", "ivf"].filter((i) => find(m, i, metric)).map((i) => ({ label: label("indexes", i), value: find(m, i, metric).latency_mean_ms })),
    { max: Math.max(...allLat), format: ms })}</div>`).join("");

  const metricsRows = rows.filter((r) => r.index === "exact");
  const metricLat = models.map((m) => `<div class="hgroup"><div class="hgroup-title">${esc(label("models", m))}</div>${hbars(
    metricsRows.filter((r) => r.model === m).map((r) => ({ label: label("metrics", r.metric), value: r.latency_mean_ms })),
    { max: Math.max(...metricsRows.map((r) => r.latency_mean_ms)), format: ms })}</div>`).join("");
  const sameTop1 = models.every((m) => new Set(metricsRows.filter((r) => r.model === m).map((r) => r.top1)).size === 1);

  box.innerHTML = `
    <div class="card">
      <div class="card-title"><h2>Resultados</h2>
        <span class="muted small">${ev.n_queries ?? ev.n_known + ev.n_unknown} consultas (${ev.n_known} de personas registradas, ${ev.n_unknown} de no registradas) · galería de ${ev.gallery_size.toLocaleString("es-PE")} rostros · k = ${k} · ${esc(ev.created_at)}</span></div>
      <div class="charts">
        <div class="chart"><h3>Calidad del reconocimiento</h3>
          <div class="note">Búsqueda exacta, coseno, umbral calibrado de cada modelo. Barras naranjas: errores (menor es mejor).</div>${quality}</div>
        <div class="chart"><h3>Recall@${k} de cada índice</h3>
          <div class="note">Qué fracción del top-${k} exacto recupera cada índice (coseno).</div>${recall}</div>
        <div class="chart"><h3>Tiempo medio de búsqueda por índice</h3>
          <div class="note">Milisegundos por consulta dentro de chutaDB (coseno). Con una galería de este tamaño la búsqueda
            exacta (una sola multiplicación de matrices) ya es muy rápida; HNSW e IVF comparan con menos rostros y su ventaja crece
            a medida que crece la galería.</div>${latency}</div>
        <div class="chart"><h3>Tiempo por métrica (búsqueda exacta)</h3>
          <div class="note">${sameTop1 ? "Las tres métricas dan el mismo acierto: los embeddings están normalizados, así que ordenan igual; solo cambia el costo de calcularlas." : "Costo de calcular cada métrica."}</div>${metricLat}</div>
      </div>
      <details class="more"><summary>Ver tabla completa (${rows.length} configuraciones)</summary>
        <div class="table-wrap"><table><thead><tr><th>Configuración</th><th class="r">Top-1</th><th class="r">Top-5</th><th class="r">Decisión correcta</th>
          <th class="r">FAR</th><th class="r">FRR</th><th class="r">Recall@${k}</th><th class="r">Búsqueda media</th><th class="r">p95</th><th class="r">Rostros evaluados</th></tr></thead>
          <tbody>${rows.map((r) => `<tr><td>${esc(configName(r))}</td><td class="r">${frac(r.top1)}</td><td class="r">${frac(r.top5)}</td>
            <td class="r">${frac(r.decision_accuracy)}</td><td class="r">${frac(r.far)}</td><td class="r">${frac(r.frr)}</td><td class="r">${frac(r.recall_at_k)}</td>
            <td class="r">${ms(r.latency_mean_ms)}</td><td class="r">${ms(r.latency_p95_ms)}</td><td class="r">${nf(r.candidates_mean, 0)}</td></tr>`).join("")}</tbody></table></div>
      </details>
    </div>`;
}

async function refreshEvalStatus() {
  try {
    const s = await api("/evaluation/status");
    const el = $("#ev-status");
    const btn = $("#ev-run");
    clearTimeout(evalTimer);
    if (s.state === "running") {
      el.innerHTML = `<span class="spinner"></span> Evaluando ${s.n_queries} consultas… ${s.done} de ${s.total} configuraciones`;
      btn.disabled = true;
      evalTimer = setTimeout(async () => {
        const now = await api("/evaluation/status").catch(() => null);
        if (now && now.state !== "running") loadPerformance();   // terminó: recargar gráficos
        else refreshEvalStatus();
      }, 1500);
    } else {
      btn.disabled = false;
      el.innerHTML = s.state === "error" ? `<span class="bad">La evaluación falló: ${esc(s.error)}</span>`
        : s.state === "done" ? `<span class="ok">Evaluación terminada en ${nf(s.elapsed_s, 0)} s.</span>` : "";
    }
  } catch { /* sin estado */ }
}

$("#ev-run").addEventListener("click", async () => {
  try {
    const limit = $("#ev-limit").value;
    await api("/evaluation/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit: limit ? Number(limit) : null, k: 10 }),
    });
    refreshEvalStatus();
  } catch (e) {
    toast(e.message, true);
  }
});

// ================================================================ CARGA DE DATOS

async function loadData() {
  const box = $("#data-content");
  try {
    const s = await api("/stats");
    const lfw = s.lfw;
    const idxRows = s.indexes.map((i) => `<tr>
      <td>${esc(label("models", i.model))}</td><td>${esc(label("indexes", i.index_type))}</td><td>${esc(label("metrics", i.metric))}</td>
      <td>${i.built ? `<span class="tag known">construido</span>` : `<span class="tag neutral">se construye al usarse</span>`}</td>
      <td class="r">${i.built ? (i.size ?? i.built_size).toLocaleString("es-PE") : "–"}</td>
      <td>${i.built ? (i.index_type === "hnsw" ? `M ${i.M} · ef_construction ${i.ef_construction}` : `nlist ${i.nlist}`) : "–"}</td>
      <td class="r">${i.build_ms != null ? nf(i.build_ms / 1000, 1) + " s" : "–"}</td>
      <td class="r">${i.disk_bytes != null ? nf(i.disk_bytes / 1048576, 1) + " MB" : "–"}</td></tr>`).join("");
    box.innerHTML = `
      <div class="stats">
        <div class="stat"><span>Personas registradas</span><b>${s.persons.total.toLocaleString("es-PE")}</b>
          <small>${s.persons.lfw.toLocaleString("es-PE")} de LFW · ${s.persons.own} registradas aquí</small></div>
        <div class="stat"><span>Fotos en la galería</span><b>${s.photos.toLocaleString("es-PE")}</b><small>guardadas en disco; ruta en FaceImage.image_path</small></div>
        <div class="stat"><span>Filas en FaceImage</span><b>${s.face_rows.toLocaleString("es-PE")}</b><small>una por foto y modelo, con su embedding</small></div>
        ${s.models.map((m) => `<div class="stat"><span>Embeddings ${esc(m.label)}</span><b>${m.embeddings.toLocaleString("es-PE")}</b>
          <small>${m.dim ? `${m.dim} dimensiones` : "sin datos"} · umbral ${m.threshold != null ? pct(m.threshold * 100, 0) : "–"}</small></div>`).join("")}
      </div>
      <div class="card">
        <div class="card-title"><h2>Dataset</h2></div>
        ${lfw ? `<dl class="kv">
          <dt>Galería LFW</dt><dd>${lfw.gallery_people} personas · ${lfw.gallery_photos} fotos (${lfw.params.gallery_per_person ?? "?"} por persona)</dd>
          <dt>Consultas de prueba</dt><dd>${lfw.known_queries} de personas registradas · ${lfw.unknown_queries} de ${lfw.unknown_people} personas no registradas</dd>
          <dt>Umbral calibrado</dt><dd>${s.calibration ? Object.entries(s.calibration).map(([m, c]) => `${esc(label("models", m))} ${pct(c.threshold * 100, 0)} (exactitud ${frac(c.val_accuracy)} en ${c.n_val} consultas de validación)`).join(" · ") : "sin calibrar (valores por defecto)"}</dd>
        </dl>` : `<p class="muted">No se cargó LFW (ver README).</p>`}
      </div>
      <div class="card">
        <div class="card-title"><h2>Índices vectoriales</h2>
          <span class="muted small">Uno por modelo, tipo y métrica; se construyen desde FaceImage.embedding</span></div>
        <div class="table-wrap"><table><thead><tr><th>Modelo</th><th>Índice</th><th>Métrica</th><th>Estado</th><th class="r">Vectores</th>
          <th>Parámetros</th><th class="r">Construcción</th><th class="r">Disco</th></tr></thead><tbody>${idxRows}</tbody></table></div>
        <p class="muted small">Para cambiar los parámetros de un índice, ve a Rendimiento → Ajustar un índice.</p>
      </div>`;
  } catch (e) {
    box.innerHTML = `<div class="card"><p class="error">${esc(e.message)}</p></div>`;
  }
}

async function rebuildIndex() {
  const btn = $("#rb-run");
  busy(btn, true, "Construyendo…");
  try {
    const body = { model: $("#rb-model").value, index: $("#rb-index").value, metric: $("#rb-metric").value };
    if (body.index === "hnsw") {
      body.M = Number($("#rb-M").value) || null;
      body.ef_construction = Number($("#rb-efc").value) || null;
    } else {
      body.nlist = Number($("#rb-nlist").value) || null;
    }
    const info = await api("/indexes/rebuild", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const params = info.index_type === "hnsw" ? `M ${info.M}, ef_construction ${info.ef_construction}` : `nlist ${info.nlist}`;
    $("#rb-msg").textContent = `Listo: ${label("models", info.model)} · ${shortIndex(info.index_type)} · ${shortMetric(info.metric)} `
      + `(${params}) reconstruido en ${nf(info.build_ms / 1000, 1)} s. Ejecuta la evaluación para ver el efecto.`;
  } catch (e) {
    toast(e.message, true);
  } finally {
    busy(btn, false);
  }
}

function initRebuildForm() {
  const c = state.config;
  $("#rb-model").innerHTML = c.models.map((m) => `<option value="${m.id}">${esc(m.label)}</option>`).join("");
  $("#rb-metric").innerHTML = c.metrics.map((m) => `<option value="${m.id}">${esc(m.label)}</option>`).join("");
  $("#rb-M").value = c.defaults.M;
  $("#rb-efc").value = c.defaults.ef_construction;
  $("#rb-index").addEventListener("change", () => {
    const hnsw = $("#rb-index").value === "hnsw";
    $$(".rb-hnsw").forEach((e) => { e.hidden = !hnsw; });
    $$(".rb-ivf").forEach((e) => { e.hidden = hnsw; });
  });
  $("#rb-run").addEventListener("click", rebuildIndex);
}

// ================================================================ INICIO

async function init() {
  stage($("#rec-stage"), null);
  stage($("#cmp-stage"), null);
  pollHealth();
  try {
    state.config = await api("/config");
  } catch {
    setTimeout(init, 3000);
    return;
  }
  const c = state.config;
  segmented($("#p-model"), c.models, c.defaults.model, resetThreshold);
  segmented($("#p-index"), c.indexes.map((i) => ({ id: i.id, label: shortIndex(i.id) })), c.defaults.index, updateParams);
  segmented($("#p-metric"), c.metrics.map((m) => ({ id: m.id, label: shortMetric(m.id) })), c.defaults.metric, updateParams);
  $("#p-k").max = c.defaults.max_k;
  $("#p-k").value = c.defaults.k;
  $("#p-ef").value = c.defaults.ef_search;
  $("#p-nprobe").value = c.defaults.nprobe;
  resetThreshold();

  segmented($("#cmp-mode"), [{ id: "configs", label: "Configuraciones sobre una consulta" }, { id: "pairs", label: "Foto vs foto (1:1)" }], "configs", setCompareMode);
  chips($("#c-models"), c.models, c.models.map((m) => m.id), updateCompareCount);
  chips($("#c-indexes"), c.indexes.map((i) => ({ id: i.id, label: shortIndex(i.id) })), c.indexes.map((i) => i.id), updateCompareCount);
  chips($("#c-metrics"), c.metrics, [c.defaults.metric], updateCompareCount);
  $("#c-k").max = c.defaults.max_k;
  applyRecParamsToCompare();
  updateCompareCount();
  renderRegister();
  renderVerifyThumbs();
  initRebuildForm();
  $("#reg-max").textContent = c.max_photos_per_person ?? 10;
  const tab = location.hash.slice(1);
  if ($(`.tab[data-tab="${CSS.escape(tab)}"]`)) openTab(tab);
}

window.addEventListener("hashchange", () => {
  const tab = location.hash.slice(1);
  if ($(`.tab[data-tab="${CSS.escape(tab)}"]`)) openTab(tab);
});

init();

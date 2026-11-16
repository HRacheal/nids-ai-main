/* DeepShield AI dashboard client. All server values are escaped before insertion. */
"use strict";
const CLASSES = ["Normal","Generic","Exploits","Fuzzers","DoS","Reconnaissance","Analysis","Backdoor","Shellcode","Worms"];
const SERIES = ["#3987e5","#d95926","#199e70","#c98500","#d55181","#008300","#9085e9","#e66767"];
const SEV = {CRITICAL:"#d03b3b", HIGH:"#ec835a", MEDIUM:"#fab219", LOW:"#0ca30c"};
const INK2 = "#b4bfd0", GRID = "#22304d";
const $ = s => document.querySelector(s);
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = n => (n ?? 0).toLocaleString();
const pct = (v, d=1) => v == null ? "–" : (v*100).toFixed(d) + "%";
const f3 = v => v == null ? "–" : Number(v).toFixed(3);
const utc = s => s ? new Date(s*1000).toISOString().replace("T"," ").slice(0,19) : "–";
const sevBadge = s => `<span class="badge sev-${esc(s)}">${esc(s)}</span>`;
const catCell = (c, att) => `<span class="${att ? "cat-attack" : "cat-normal"}">${esc(c)}</span>`;
async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({error: `HTTP ${r.status}`}));
  if (!r.ok) throw Object.assign(new Error(j.error || `HTTP ${r.status}`), {body: j});
  return j;
}
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.style.display = "block"; clearTimeout(t._h); t._h = setTimeout(() => t.style.display = "none", 4000); }

Chart.defaults.color = INK2; Chart.defaults.borderColor = GRID;
Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif';
const charts = {};
function chart(id, cfg) { charts[id]?.destroy(); charts[id] = new Chart($(id), cfg); return charts[id]; }
const barOpts = (horizontal, extra={}) => ({responsive:true, maintainAspectRatio:false, indexAxis: horizontal ? "y" : "x",
  plugins:{legend:{display:false}}, scales:{x:{grid:{display:!horizontal}}, y:{grid:{display:horizontal ? false : true}}},
  datasets:{bar:{borderRadius:4, borderSkipped:"start", maxBarThickness:28}}, ...extra});

/* ── tabs ─────────────────────────────────────────────── */
const loaders = {};
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll("#tabs button").forEach(x => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach(s => s.classList.toggle("active", s.id === b.dataset.tab));
  loaders[b.dataset.tab]?.();
});

/* ── status ───────────────────────────────────────────── */
let STATUS = null;
async function loadStatus() {
  STATUS = await api("/api/status");
  const opt = (m, sel) => `<option value="${esc(m.id)}" ${m.id === sel ? "selected" : ""}>${esc(m.name)}</option>`;
  $("#dt-model").innerHTML = STATUS.track_a_models.map(m => opt(m, "rf_multiclass")).join("");
  $("#rp-model").innerHTML = STATUS.replay_models.map(m => opt(m, "lstm")).join("");
  const errs = Object.entries(STATUS.errors).map(([k, v]) => `${k}: ${v}`).join("\n");
  $("#ov-models").innerHTML =
    `<p>${STATUS.track_a_models.length} Track A models loaded: ${STATUS.track_a_models.map(m => esc(m.name)).join(", ") || "none"}</p>` +
    `<p>${STATUS.replay_models.length} replay models loaded: ${STATUS.replay_models.map(m => esc(m.name)).join(", ") || "none"}` +
    (STATUS.lstm_sequence_length ? ` (LSTM window ${STATUS.lstm_sequence_length} flows)` : "") + `</p>` +
    `<p class="muted">Replay source: ${fmt(STATUS.replay_records)} test-block flows · ATT&amp;CK v${esc(STATUS.attack_version)}</p>` +
    (errs ? `<p class="msg err">${esc(errs)}</p>` : "");
  updateReplayButtons(STATUS.replay);
}

/* ── overview ─────────────────────────────────────────── */
async function loadOverview() {
  const ov = await api("/api/overview");
  $("#ov-total").textContent = fmt(ov.total); $("#ov-normal").textContent = fmt(ov.normal);
  $("#ov-mal").textContent = fmt(ov.malicious); $("#ov-share").textContent = ov.total ? pct(ov.attack_share) + " of records" : "";
  $("#ov-crit").textContent = `${fmt(ov.severity.CRITICAL)} / ${fmt(ov.severity.HIGH)}`;
  $("#ov-empty").hidden = ov.total > 0;
  const dist = Object.entries(ov.attack_distribution);
  chart("#ch-dist", {type:"bar", data:{labels:dist.map(d => d[0]), datasets:[{data:dist.map(d => d[1]), backgroundColor:SERIES[1]}]},
    options:barOpts(true)});
  const sev = Object.entries(ov.severity);
  chart("#ch-sev", {type:"bar", data:{labels:sev.map(s => s[0]), datasets:[{data:sev.map(s => s[1]), backgroundColor:sev.map(s => SEV[s[0]])}]},
    options:barOpts(false)});
  $("#ov-alerts tbody").innerHTML = ov.recent_alerts.map(a => `<tr><td>${utc(a.flow_time || a.created)}</td><td>${sevBadge(a.severity)}</td>
    <td>${esc(a.category)}</td><td>${esc(a.srcip || "upload")}</td><td>${esc(a.model)}</td></tr>`).join("") ||
    `<tr><td colspan="5" class="muted">No alerts yet.</td></tr>`;
  $("#ov-sources").innerHTML = ov.top_sources.map(s => `<div>${esc(s.srcip)} <span class="muted">${fmt(s.n)} alerts</span></div>`).join("") ||
    `<p class="muted">–</p>`;
}
loaders.overview = loadOverview;

/* ── replay ───────────────────────────────────────────── */
const socket = io();
const live = {labels: [], normal: [], flagged: []};
function updateReplayButtons(st) {
  $("#rp-start").disabled = !!st?.running; $("#rp-stop").disabled = !st?.running;
  $("#rp-status").textContent = st?.running ? `running: ${st.model} · ${fmt(st.processed)} / ${fmt(st.total)}` :
    (st?.processed ? `stopped at ${fmt(st.processed)} / ${fmt(st.total)}` : "");
}
$("#rp-start").onclick = () => socket.emit("replay_start",
  {model: $("#rp-model").value, speed: +$("#rp-speed").value, restart: $("#rp-restart").checked},
  r => { if (r?.error) toast(r.error); else { updateReplayButtons(r.status); $("#rp-restart").checked = false; } });
$("#rp-stop").onclick = () => socket.emit("replay_stop", {});
socket.on("replay_status", updateReplayButtons);
socket.on("replay_error", e => toast("Replay error: " + e.error));
socket.on("replay_batch", d => {
  updateReplayButtons(d.status);
  const ev = d.evaluation;
  $("#rp-n").textContent = fmt(ev.evaluated); $("#rp-acc").textContent = pct(ev.accuracy);
  $("#rp-dr").textContent = pct(ev.detection_rate); $("#rp-fpr").textContent = pct(ev.false_positive_rate, 2);
  if (d.flows.length) $("#rp-time").textContent = "flow time " + utc(d.flows[d.flows.length - 1].time) + " UTC";
  const tb = $("#rp-table tbody");
  const rows = d.flows.slice().reverse().map(f => `<tr><td>${utc(f.time)}</td><td>${esc(f.src)}</td><td>${esc(f.dst)}</td>
    <td>${esc(f.proto)}</td><td>${esc(f.service)}</td><td>${catCell(f.category, f.is_attack)}</td><td class="num">${pct(f.confidence)}</td></tr>`).join("");
  tb.insertAdjacentHTML("afterbegin", rows);
  while (tb.rows.length > 200) tb.deleteRow(-1);
  const flagged = d.flows.filter(f => f.is_attack).length;
  live.labels.push(""); live.normal.push(d.flows.length - flagged); live.flagged.push(flagged);
  if (live.labels.length > 120) { live.labels.shift(); live.normal.shift(); live.flagged.shift(); }
  const c = charts["#ch-live"];
  if (c) { c.update("none"); }
  if (d.alerts.length && $("#overview").classList.contains("active")) loadOverview();
});
socket.on("alert", () => { if ($("#overview").classList.contains("active")) loadOverview(); });
chart("#ch-live", {type:"bar", data:{labels:live.labels, datasets:[
  {label:"normal", data:live.normal, backgroundColor:SERIES[0]}, {label:"flagged as attack", data:live.flagged, backgroundColor:SERIES[1]}]},
  options:{...barOpts(false), animation:false, plugins:{legend:{display:true, position:"top", align:"end"}},
    scales:{x:{stacked:true, grid:{display:false}}, y:{stacked:true}}}});
loaders.monitor = loadStatus;

/* ── threat detection ─────────────────────────────────── */
let lastRecords = [], lastTrue = [], lastModel = null;
function showMsg(text, isErr) { const m = $("#dt-msg"); m.textContent = text; m.classList.toggle("err", !!isErr); }
function renderDetect(res) {
  lastModel = res.model;
  $("#dt-summary").textContent = `${fmt(res.n)} records · ${Object.entries(res.summary).map(([k, v]) => `${k} ${fmt(v)}`).join(" · ")} · ${res.alerts_created} alerts`;
  $("#dt-table tbody").innerHTML = res.predictions.map((p, i) => {
    const conf = p.confidence != null ? pct(p.confidence) : (p.anomaly_score != null ? `score ${p.anomaly_score.toPrecision(3)} (thr ${p.threshold.toPrecision(3)})` : "–");
    const truth = lastTrue[i] ?? "";
    const can = ["rf_multiclass","mlp","cnn","autoencoder"].includes(res.model) && lastRecords[i];
    return `<tr><td>${i + 1}</td><td>${esc(p.proto)}</td><td>${esc(p.service)}</td><td>${catCell(p.category, p.is_attack)}</td>
      <td class="num">${conf}</td><td>${esc(truth)}</td><td>${can ? `<button data-i="${i}" class="explain">Explain</button>` : ""}</td></tr>`;
  }).join("");
  document.querySelectorAll("#dt-table .explain").forEach(b => b.onclick = () => explainRecord(+b.dataset.i));
  const probs = (res.problems || []).map(p => p.rows_rejected ? `${p.rows_rejected} rows rejected` : `row ${p.row}: ${p.column}=${p.value} (${p.problem})`);
  showMsg(probs.length ? "Some records were rejected:\n" + probs.slice(0, 8).join("\n") : "", probs.length > 0);
}
$("#dt-upload").onclick = async () => {
  const f = $("#dt-file").files[0]; if (!f) return showMsg("Choose a CSV file first.", true);
  const fd = new FormData(); fd.append("file", f); fd.append("model", $("#dt-model").value);
  showMsg("Analysing…");
  try {
    const text = await f.text();
    lastRecords = []; lastTrue = [];
    const res = await api("/api/analyze", {method:"POST", body:fd});
    renderDetect(res);
    if (text.length < 5e6) {   // keep parsed rows for explanations (small files only)
      const lines = text.trim().split(/\r?\n/), head = lines[0].split(",").map(h => h.trim().toLowerCase());
      lastRecords = lines.slice(1, 501).map(l => Object.fromEntries(l.split(",").map((v, i) => [head[i], v])));
      lastTrue = lastRecords.map(r => r.attack_cat || "");
      renderDetect(res);
    }
  } catch (e) { showMsg(e.message + (e.body?.problems ? "\n" + JSON.stringify(e.body.problems.slice(0, 5)) : ""), true); }
};
$("#dt-sample").onclick = async () => {
  showMsg("Loading records from the official test set…");
  try {
    const s = await api(`/api/sample_records?n=${+$("#dt-n").value}&category=${encodeURIComponent($("#dt-cat").value)}`);
    lastRecords = s.records; lastTrue = s.true_labels;
    renderDetect(await api("/api/analyze", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({model: $("#dt-model").value, records: s.records})}));
  } catch (e) { showMsg(e.message, true); }
};
async function explainRecord(i) {
  $("#dt-explain-card").hidden = false; $("#dt-explain-title").textContent = "computing…";
  try {
    const ex = await api("/api/explain", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({model: lastModel, record: lastRecords[i]})});
    renderLocal(ex, "#ch-local");
    $("#dt-explain-title").textContent = `record ${i + 1}: ${ex.prediction.category} · ${ex.method}`;
    $("#dt-explain-note").textContent = ex.note + (ex.base_value != null ? ` Base value ${ex.base_value.toFixed(3)} (${ex.unit}).` : ` Units: ${ex.unit}.`);
  } catch (e) { $("#dt-explain-title").textContent = e.message; }
}
function renderLocal(ex, id) {
  const c = ex.contributions;
  chart(id, {type:"bar", data:{labels:c.map(x => `${x.feature} = ${x.value ?? ""}`), datasets:[{data:c.map(x => x.contribution),
    backgroundColor:c.map(x => x.contribution >= 0 ? SERIES[1] : SERIES[0])}]},
    options:{...barOpts(true), plugins:{legend:{display:false}, tooltip:{callbacks:{label: t => `${t.raw >= 0 ? "+" : ""}${t.raw.toPrecision(3)} ${ex.unit}`}}}}});
}
loaders.detect = loadStatus;

/* ── alerts & history ─────────────────────────────────── */
let alPage = 0; const AL_N = 50;
const alView = () => document.querySelector("input[name=al-view]:checked").value;
async function loadAlerts() {
  const q = new URLSearchParams({search: $("#al-search").value, category: $("#al-cat").value, severity: $("#al-sev").value,
    limit: AL_N, offset: alPage * AL_N});
  const view = alView();
  const res = await api((view === "alerts" ? "/api/alerts?" : "/api/predictions?") + q);
  const head = view === "alerts"
    ? "<tr><th>Flow time</th><th>Severity</th><th>Category</th><th>Source → destination</th><th>Model</th><th class=num>Conf.</th><th>Origin</th></tr>"
    : "<tr><th>Flow time</th><th>Prediction</th><th>Severity</th><th>Source → destination</th><th>Proto</th><th>Model</th><th class=num>Conf.</th><th>Origin</th></tr>";
  $("#al-table thead").innerHTML = head;
  $("#al-table tbody").innerHTML = res.rows.map(r => view === "alerts"
    ? `<tr class="click" data-id="${esc(r.id)}"><td>${utc(r.flow_time || r.created)}</td><td>${sevBadge(r.severity)}</td><td>${esc(r.category)}</td>
       <td>${esc(r.srcip || "–")} → ${esc(r.dstip || "–")}</td><td>${esc(r.model)}</td><td class="num">${pct(r.confidence)}</td><td>${esc(r.source)}</td></tr>`
    : `<tr><td>${utc(r.flow_time || r.created)}</td><td>${catCell(r.category, r.is_attack)}</td><td>${sevBadge(r.severity)}</td>
       <td>${esc(r.srcip || "–")} → ${esc(r.dstip || "–")}</td><td>${esc(r.proto)}</td><td>${esc(r.model)}</td><td class="num">${r.confidence != null ? pct(r.confidence) : "–"}</td><td>${esc(r.source)}</td></tr>`).join("")
    || `<tr><td colspan="8" class="muted">Nothing recorded.</td></tr>`;
  $("#al-page").textContent = `${fmt(res.total ? alPage * AL_N + 1 : 0)}–${fmt(Math.min((alPage + 1) * AL_N, res.total))} of ${fmt(res.total)}`;
  $("#al-prev").disabled = alPage === 0; $("#al-next").disabled = (alPage + 1) * AL_N >= res.total;
  document.querySelectorAll("#al-table tr.click").forEach(tr => tr.onclick = () => {
    document.querySelectorAll("#al-table tr.sel").forEach(x => x.classList.remove("sel")); tr.classList.add("sel"); alertDetail(tr.dataset.id); });
}
async function alertDetail(id) {
  const a = await api("/api/alerts/" + encodeURIComponent(id));
  const m = a.mitre || {techniques: []};
  $("#al-detail").innerHTML = `<h2>Alert ${esc(a.id)} ${sevBadge(a.severity)}</h2>
    <dl class="kv"><dt>Category</dt><dd>${esc(a.category)}</dd><dt>Model</dt><dd>${esc(a.model)} (${esc(a.source)})</dd>
    <dt>Confidence</dt><dd>${pct(a.confidence)}</dd><dt>Flow time</dt><dd>${utc(a.flow_time)} UTC</dd>
    <dt>Source</dt><dd>${esc(a.srcip || "–")}${a.sport != null ? ":" + esc(a.sport) : ""}</dd><dt>Destination</dt><dd>${esc(a.dstip || "–")}${a.dsport != null ? ":" + esc(a.dsport) : ""}</dd>
    <dt>Protocol / service</dt><dd>${esc(a.proto)} / ${esc(a.service)}</dd><dt>Recorded</dt><dd>${utc(a.created)} UTC</dd></dl>
    <h2>Candidate ATT&amp;CK techniques <span class="muted">confidence: ${esc(m.confidence || "none")}</span></h2>
    ${(m.techniques || []).map(t => `<div class="tech"><a href="${esc(t.url)}" target="_blank" rel="noopener">${esc(t.id)} ${esc(t.name)}</a> <span class="muted">${esc((t.tactics || []).join(", "))}</span></div>`).join("") || `<p class="muted">No defensible technique-level mapping for this category.</p>`}
    <p class="muted">${esc(m.basis || "")} ${esc(m.disclaimer || "")}</p>
    ${a.record ? `<button id="al-explain">Explain this prediction</button><div class="chart short" style="margin-top:8px"><canvas id="ch-alert"></canvas></div><p class="muted" id="al-ex-note"></p>` : ""}`;
  const b = $("#al-explain");
  if (b) b.onclick = async () => {
    b.disabled = true; b.textContent = "Computing…";
    try { const ex = await api(`/api/alerts/${encodeURIComponent(id)}/explain`); renderLocal(ex, "#ch-alert"); $("#al-ex-note").textContent = `${ex.method}. ${ex.note}`; b.remove(); }
    catch (e) { b.textContent = e.message; }
  };
}
["#al-search", "#al-cat", "#al-sev"].forEach(s => $(s).addEventListener("change", () => { alPage = 0; loadAlerts(); }));
document.querySelectorAll("input[name=al-view]").forEach(r => r.onchange = () => { alPage = 0; loadAlerts(); });
$("#al-refresh").onclick = loadAlerts;
$("#al-prev").onclick = () => { alPage--; loadAlerts(); }; $("#al-next").onclick = () => { alPage++; loadAlerts(); };
$("#al-clear").onclick = async () => { if (!confirm("Delete all stored predictions and alerts?")) return;
  try { await api("/api/history/clear", {method:"POST"}); toast("History cleared"); loadAlerts(); } catch (e) { toast(e.message); } };
loaders.alerts = loadAlerts;

/* ── model comparison ─────────────────────────────────── */
let MODELS = null;
async function loadModels() {
  MODELS = await api("/api/models");
  const multi = [], bin = [];
  for (const [track, ms] of Object.entries(MODELS.models)) for (const [id, m] of Object.entries(ms))
    (m.macro_f1 != null ? multi : bin).push({track: track.toUpperCase(), id, ...m});
  const sd = m => m.seed_summary?.macro_f1_std != null ? ` ±${m.seed_summary.macro_f1_std.toFixed(3)}` : "";
  const ci = b => b ? ` <span class="muted">[${b.low.toFixed(3)}, ${b.high.toFixed(3)}]</span>` : "";
  const time = m => `${m.fit_seconds != null ? m.fit_seconds.toFixed(0) + " s" : "–"} / ${m.predict_us_per_record != null ? m.predict_us_per_record.toFixed(1) + " µs" : "–"}`;
  $("#md-multi").innerHTML = `<tr><th>Track</th><th>Model</th><th class=num>Accuracy</th><th class=num>Macro precision</th><th class=num>Macro recall</th>
    <th class=num>Macro-F1</th><th class=num>Weighted F1</th><th class=num>ROC-AUC (OvR)</th><th class=num>PR-AUC</th><th class=num>Attack-vs-normal F1</th><th class=num>Train / predict per record</th></tr>` +
    multi.map(m => `<tr><td>${m.track}</td><td>${esc(m.name)}</td><td class=num>${f3(m.accuracy)}</td><td class=num>${f3(m.macro_precision)}</td>
    <td class=num>${f3(m.macro_recall)}</td><td class=num><b>${f3(m.macro_f1)}</b>${sd(m)}${ci(m.bootstrap)}</td><td class=num>${f3(m.weighted_f1)}</td>
    <td class=num>${f3(m.roc_auc_ovr_macro)}</td><td class=num>${f3(m.pr_auc_macro)}</td><td class=num>${f3(m.binary?.f1)}</td><td class=num>${time(m)}</td></tr>`).join("");
  $("#md-binary").innerHTML = `<tr><th>Track</th><th>Model</th><th>Task</th><th class=num>Accuracy</th><th class=num>Precision</th><th class=num>Detection rate</th>
    <th class=num>F1</th><th class=num>ROC-AUC</th><th class=num>PR-AUC</th><th class=num>False-positive rate</th><th class=num>Train / predict per record</th></tr>` +
    bin.map(m => `<tr><td>${m.track}</td><td>${esc(m.name)}</td><td>${esc(m.task)}</td><td class=num>${f3(m.accuracy)}</td><td class=num>${f3(m.precision)}</td>
    <td class=num>${f3(m.recall_detection_rate)}</td><td class=num>${f3(m.f1)}</td><td class=num>${f3(m.roc_auc)}</td><td class=num>${f3(m.pr_auc)}</td>
    <td class=num>${pct(m.false_positive_rate, 2)}</td><td class=num>${time(m)}</td></tr>`).join("");
  const a = multi.filter(m => m.track === "A");
  chart("#ch-f1", {type:"bar", data:{labels:a.map(m => m.name), datasets:[{data:a.map(m => m.macro_f1), backgroundColor:SERIES[0]}]},
    options:{...barOpts(true), scales:{x:{min:0, max:1}, y:{grid:{display:false}}}}});
  $("#md-cm-model").innerHTML = multi.map((m, i) => `<option value="${i}">${m.track}: ${esc(m.name)}</option>`).join("");
  $("#md-cm-model").onchange = () => renderCM(multi[+$("#md-cm-model").value]);
  if (multi.length) renderCM(multi[0]);
  const L = MODELS.leave_one_class_out;
  $("#md-loco").innerHTML = L ? `<p class="muted">Each attack class was removed from training and validation; the table shows what the closed-set models did with that class's test records. ${esc(L.rejection_rule)}.</p>
    <div class="scroll"><table class="tbl"><tr><th>Held-out class</th><th>Model</th><th class=num>Test records</th><th class=num>Classified Normal (missed)</th>
    <th class=num>Flagged as another attack</th><th>Most common wrong attack</th><th class=num>Rejected as unknown</th><th class=num>Known classes rejected</th><th class=num>Autoencoder detection</th></tr>` +
    L.results.map(r => `<tr><td>${esc(r.held_out)}</td><td>${esc(r.model.toUpperCase())}</td><td class=num>${fmt(r.held_out_test_support)}</td>
    <td class=num>${pct(r.held_out_predicted_as_normal)}</td><td class=num>${pct(r.held_out_predicted_as_other_attack)}</td><td>${esc(r.most_common_wrong_attack)}</td>
    <td class=num>${pct(r.held_out_rejected_as_unknown)}</td><td class=num>${pct(r.known_classes_rejected_as_unknown)}</td>
    <td class=num>${pct(L.anomaly_detector_detection_rate_by_category.autoencoder[r.held_out])}</td></tr>`).join("") + "</table></div>" : `<p class="muted">Not run yet.</p>`;
  $("#md-figs").innerHTML = MODELS.figures.map(f => `<figure><a href="/figures/${esc(f)}" target="_blank"><img loading="lazy" src="/figures/${esc(f)}" alt="${esc(f)}"></a><figcaption>${esc(f.split("/").pop())}</figcaption></figure>`).join("");
}
function renderCM(m) {
  const cm = m.confusion_matrix, cls = m.classes || CLASSES;
  const rows = cm.map((r, i) => { const s = r.reduce((a, b) => a + b, 0);
    return `<tr><th class="rowh">${esc(cls[i])}</th>` + r.map((v, j) => { const p = s ? v / s : 0;
      return `<td title="${esc(cls[i])} → ${esc(cls[j])}: ${fmt(v)}" style="background:rgba(57,135,229,${(0.08 + 0.92 * p).toFixed(2)});color:${p > .45 ? "#fff" : INK2}">${s ? (p * 100).toFixed(0) : ""}</td>`; }).join("") + "</tr>"; }).join("");
  $("#md-cm").innerHTML = `<div class="scroll"><table class="cm"><tr><th></th>${cls.map(c => `<th>${esc(c.slice(0, 5))}</th>`).join("")}</tr>${rows}</table></div>
    <p class="muted">Rows: true class; columns: predicted; cells: % of the true class (row-normalised).</p>`;
}
loaders.models = loadModels;

/* ── explainability ───────────────────────────────────── */
let XAI = null;
async function loadXAI() {
  XAI = await api("/api/explanations");
  const g = XAI.global;
  if (!g) { $("#xa-checks").textContent = "Explanations not computed yet (scripts/explain_models.py)."; return; }
  const methods = {"RF - SHAP": g.rf_shap.global_mean_abs, "MLP - Integrated Gradients": g.mlp_ig.global_mean_abs,
    "1D-CNN - Integrated Gradients": g.cnn_ig.global_mean_abs, "Autoencoder - error share": g.autoencoder_error_share_attacks};
  $("#xa-method").innerHTML = Object.keys(methods).map(m => `<option>${esc(m)}</option>`).join("");
  const draw = () => { const imp = Object.entries(methods[$("#xa-method").value]).sort((a, b) => b[1] - a[1]).slice(0, 15);
    chart("#ch-global", {type:"bar", data:{labels:imp.map(i => i[0]), datasets:[{data:imp.map(i => i[1]), backgroundColor:SERIES[0]}]}, options:barOpts(true)}); };
  $("#xa-method").onchange = draw; draw();
  $("#xa-agree").innerHTML = "<tr><th>Methods</th><th class=num>Spearman ρ (42 features)</th><th class=num>Shared top-10</th></tr>" +
    Object.entries(g.agreement_spearman).map(([k, v]) => `<tr><td>${esc(k)}</td><td class=num>${v.toFixed(2)}</td><td class=num>${g.top10_overlap[k]}</td></tr>`).join("");
  $("#xa-checks").textContent = `Checks: SHAP additivity max error ${g.rf_shap.additivity_max_error.toExponential(1)}; Integrated Gradients completeness median relative error MLP ${(g.mlp_ig.completeness_relative_error_median * 100).toFixed(2)}%, CNN ${(g.cnn_ig.completeness_relative_error_median * 100).toFixed(2)}%.`;
  $("#xa-figs").innerHTML = XAI.figures.map(f => `<figure><a href="/figures/${esc(f)}" target="_blank"><img loading="lazy" src="/figures/${esc(f)}" alt="${esc(f)}"></a><figcaption>${esc(f.split("/").pop())}</figcaption></figure>`).join("");
}
loaders.xai = loadXAI;

/* ── reports / MITRE ──────────────────────────────────── */
async function loadReports() {
  const [ov, m] = await Promise.all([api("/api/overview"), api("/api/mitre")]);
  $("#rep-info").textContent = `Current history: ${fmt(ov.total)} predictions, ${fmt(Object.values(ov.severity).reduce((a, b) => a + b, 0))} alerts.`;
  $("#mitre-version").textContent = `validated against ATT&CK v${m.attack_version}`;
  $("#mitre-table").innerHTML = "<tr><th>Category</th><th>Confidence</th><th>Candidate techniques</th><th>Basis</th></tr>" +
    Object.entries(m.categories).filter(([c]) => c !== "Normal").map(([c, e]) => `<tr><td>${esc(c)}</td><td>${esc(e.confidence)}</td>
    <td>${e.techniques.map(t => `<a href="${esc(t.url)}" target="_blank" rel="noopener" style="color:#86b6ef">${esc(t.id)}</a> ${esc(t.name)}`).join("<br>") || "–"}</td>
    <td style="white-space:normal">${esc(e.basis)}</td></tr>`).join("");
}
loaders.reports = loadReports;

/* ── init ─────────────────────────────────────────────── */
for (const sel of ["#dt-cat", "#al-cat"]) $(sel).insertAdjacentHTML("beforeend", CLASSES.map(c => `<option>${c}</option>`).join(""));
loadStatus().then(loadOverview).catch(e => toast(e.message));
setInterval(() => { if ($("#overview").classList.contains("active")) loadOverview().catch(() => {}); }, 5000);

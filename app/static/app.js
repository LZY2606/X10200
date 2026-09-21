/* 晶格候选室前端：纯原生 JS，所有交互回到 REST API。 */
"use strict";

const state = {
  analysisId: null,
  data: null,
  candidate: null,
  selectedCandidateId: null,
};

const $ = (sel) => document.querySelector(sel);

function toast(msg, isError) {
  const el = $("#toast");
  el.textContent = msg;
  el.style.borderColor = isError ? "#ef6f6f" : "#6ea8fe";
  el.style.display = "block";
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.style.display = "none"; }, 4000);
}

async function api(method, url, body) {
  const opt = { method, headers: { "Content-Type": "application/json" } };
  if (body) opt.body = JSON.stringify(body);
  const resp = await fetch(url, opt);
  const text = await resp.text();
  const json = text ? JSON.parse(text) : {};
  if (!resp.ok) {
    throw new Error(json.detail || `${resp.status}`);
  }
  return json;
}

function fmt(v, digits) {
  if (v === null || v === undefined) return "—";
  if (!isFinite(v)) return "∞";
  return Number(v).toFixed(digits === undefined ? 4 : digits);
}

async function loadDemo() {
  try {
    const res = await api("POST", "/api/demo");
    state.analysisId = res.analysis_id;
    await refresh();
    toast(`已创建 ${res.candidates_created} 个候选（真值偏移 ${
      res.true_offset.map(v => v.toFixed(3))}）`);
  } catch (e) { toast(e.message, true); }
}

async function refresh() {
  if (!state.analysisId) return;
  state.data = await api("GET", `/api/analyses/${state.analysisId}`);
  const a = state.data.analysis;
  $("#meta").textContent =
    `${a.name}  rev=${a.revision} 状态=${a.status} ` +
    `校准版本=${a.calibration_version_id} 容差=${a.index_tol} ` +
    `离群预算=${a.outlier_budget}`;
  $("#cal-x").value = a.active_offset[0];
  $("#cal-y").value = a.active_offset[1];
  $("#cal-z").value = a.active_offset[2];

  const actives = state.data.candidates.filter(c => c.status === "active");
  if (!actives.find(c => c.id === state.selectedCandidateId)) {
    state.selectedCandidateId = actives.length ? actives[0].id : null;
  }
  renderCandidates();
  if (state.selectedCandidateId) {
    await loadCandidate(state.selectedCandidateId);
  } else {
    state.candidate = null;
    $("#refl-table tbody").innerHTML = "";
    $("#chain-json").textContent = "—";
  }
  drawSpace();
}

function renderCandidates() {
  const tbody = $("#cand-table tbody");
  tbody.innerHTML = "";
  const palette = ["#6ea8fe", "#69c792", "#f0a35e", "#c792ea",
                   "#5ec8d8", "#ef6f6f"];
  for (const c of state.data.candidates) {
    const tr = document.createElement("tr");
    tr.className = c.status;
    if (c.id === state.selectedCandidateId) tr.classList.add("sel");
    const s = c.scores;
    const dom = c.domain_label === null ? "—" : c.domain_label;
    const over = s.outlier_budget.exceeded
      ? `<span class="tag out">超预算</span> ` : "";
    tr.innerHTML = `
      <td>${c.seq}</td>
      <td><span class="tag fam">${(c.family_id || "").slice(-6)}</span></td>
      <td><span class="tag dom" style="color:${
          palette[(c.domain_label||0) % palette.length]}">D${dom}</span></td>
      <td>${c.status === "active" ? "有效" :
           c.status === "expired" ? "过期" : "被取代"}</td>
      <td>${fmt(s.indexed_fraction, 3)} (${c.indexed_count}/${
        c.eligible_count})</td>
      <td>${fmt(s.weighted_residual)}</td>
      <td>${fmt(s.complexity, 3)}</td>
      <td>${over}${s.outlier_budget.used}/${s.outlier_budget.budget}</td>
      <td>${fmt(c.basis_condition, 2)}</td>`;
    tr.onclick = () => loadCandidate(c.id);
    tbody.appendChild(tr);
  }
}

async function loadCandidate(cid) {
  try {
    state.candidate = await api("GET", `/api/candidates/${cid}`);
    state.selectedCandidateId = cid;
    renderCandidates();
    renderReflections();
    $("#chain-json").textContent =
      JSON.stringify(state.candidate.transform_chain, null, 1);
    drawSpace();
    drawHistogram();
  } catch (e) { toast(e.message, true); }
}

function renderReflections() {
  const tbody = $("#refl-table tbody");
  tbody.innerHTML = "";
  const obsById = Object.fromEntries(
    state.data.observations.map(o => [o.id, o]));
  for (const r of state.candidate.reflections) {
    const obs = obsById[r.observation_id];
    if (!obs) continue;
    const tr = document.createElement("tr");
    const roleText = {indexed: "已索引", outlier: "离群",
      locked_outlier: "锁定离群", unindexed: "未索引",
      excluded: "排除"}[r.role] || r.role;
    const markBtn = obs.mark === "excluded"
      ? `<button data-act="normal">恢复</button>`
      : obs.mark === "locked"
        ? `<button data-act="normal">解锁</button>`
        : `<button data-act="locked">锁定</button>
           <button data-act="excluded">排除</button>`;
    tr.innerHTML = `
      <td>${obs.peak_index}</td>
      <td><button class="hkl-btn">${
        r.hkl ? `(${r.hkl.join(",")})` : "—"}</button></td>
      <td>${fmt(r.residual_norm)}</td>
      <td>${fmt(r.mahalanobis)}</td>
      <td>${roleText}</td>
      <td>${obs.is_overlap ? '<span class="tag out">重叠</span> ' : ""}${
        obs.mark === "locked" ? '<span class="tag lock">锁定</span>' :
        obs.mark === "excluded" ? '<span class="tag">排除</span>' : ""}</td>
      <td>${markBtn}</td>`;
    tr.querySelector(".hkl-btn").onclick = () => showObservation(obs, r);
    tr.querySelectorAll("button[data-act]").forEach(b => {
      b.onclick = async () => {
        try {
          await api("POST",
            `/api/observations/${obs.id}/mark?analysis_id=${
              state.analysisId}`, { mark: b.dataset.act });
          await refresh();
        } catch (e) { toast(e.message, true); }
      };
    });
    tbody.appendChild(tr);
  }
}

function showObservation(obs, ref) {
  $("#obs-title").textContent = `峰 #${obs.peak_index}  hkl=(${
    (ref.hkl || []).join(",")})`;
  $("#obs-json").textContent = JSON.stringify({
    raw_point: obs.raw_point,
    calibrated_point: obs.calibrated_point,
    intensity: obs.intensity,
    measurement_covariance: obs.covariance,
    calibration_version_id: obs.calibration_version_id,
    is_overlap: obs.is_overlap,
    manual_mark: obs.mark,
    predicted_point: ref.predicted,
    residual: ref.residual,
    mahalanobis: ref.mahalanobis,
  }, null, 1);
}

function projectSetup(points, canvas) {
  const margin = 34;
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY) || 1;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const scale = (canvas.width - 2 * margin) / span;
  return {
    xy: (p) => [
      canvas.width / 2 + (p[0] - cx) * scale,
      canvas.height / 2 - (p[1] - cy) * scale,
    ],
  };
}

function drawSpace() {
  const canvas = $("#space");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.data) return;
  const obsList = state.data.observations;
  const setup = projectSetup(obsList.map(o => o.calibrated_point), canvas);

  // 当前候选的预测十字
  if (state.candidate) {
    ctx.strokeStyle = "#f0a35e";
    ctx.lineWidth = 1;
    for (const r of state.candidate.reflections) {
      if (r.role === "unindexed" || r.role === "excluded") continue;
      const [x, y] = setup.xy(r.predicted);
      ctx.beginPath();
      ctx.moveTo(x - 3, y); ctx.lineTo(x + 3, y);
      ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 3);
      ctx.stroke();
    }
  }

  const roleOfObs = {};
  if (state.candidate) {
    for (const r of state.candidate.reflections) roleOfObs[r.observation_id] = r.role;
  }
  const colors = {indexed: "#6ea8fe", outlier: "#ef6f6f",
                  locked_outlier: "#f0a35e", unindexed: "#8b96ad",
                  excluded: "#4a5468", undefined: "#8b96ad"};
  for (const o of obsList) {
    const role = roleOfObs[o.id];
    const [x, y] = setup.xy(o.calibrated_point);
    ctx.fillStyle = colors[role];
    ctx.beginPath();
    ctx.arc(x, y, o.is_overlap ? 6 : 3.5, 0, Math.PI * 2);
    ctx.fill();
    if (o.mark === "excluded") {
      ctx.strokeStyle = "#ef6f6f";
      ctx.beginPath();
      ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5);
      ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5);
      ctx.stroke();
    }
    if (o.mark === "locked") {
      ctx.strokeStyle = "#f0a35e";
      ctx.strokeRect(x - 6, y - 6, 12, 12);
    }
  }
}

function drawHistogram() {
  const canvas = $("#hist");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.candidate) return;
  const vals = state.candidate.reflections
    .filter(r => r.role === "indexed").map(r => r.mahalanobis)
    .sort((a, b) => a - b);
  if (!vals.length) return;
  const bins = 20;
  const max = vals[vals.length - 1] || 1;
  const counts = new Array(bins).fill(0);
  for (const v of vals) counts[Math.min(bins - 1, Math.floor(v / max * bins))]++;
  const maxCount = Math.max(...counts);
  const w = canvas.width / bins;
  ctx.fillStyle = "#6ea8fe";
  counts.forEach((c, i) => {
    const h = (c / maxCount) * (canvas.height - 40);
    ctx.fillRect(i * w + 1, canvas.height - 24 - h, w - 2, h);
  });
  ctx.fillStyle = "#8b96ad";
  ctx.fillText(`0`, 4, canvas.height - 6);
  ctx.fillText(`max ${fmt(max, 2)}`, canvas.width - 70, canvas.height - 6);
}

$("#btn-demo").onclick = loadDemo;
$("#btn-regen").onclick = async () => {
  try {
    await api("POST", `/api/analyses/${state.analysisId}/generate`,
              { max_candidates: 120 });
    await refresh();
    toast("已重新生成候选（旧候选保留为“被取代”）");
  } catch (e) { toast(e.message, true); }
};
$("#btn-cal").onclick = async () => {
  try {
    const offset = [+$("#cal-x").value, +$("#cal-y").value, +$("#cal-z").value];
    await api("POST", `/api/analyses/${state.analysisId}/calibration`,
              { offset });
    await refresh();
    toast("校准已更新：依赖它的候选全部过期");
  } catch (e) { toast(e.message, true); }
};
$("#btn-refine").onclick = async () => {
  try {
    const res = await api("POST",
      `/api/analyses/${state.analysisId}/calibration/refine`,
      { candidate_id: state.selectedCandidateId });
    await refresh();
    toast(`加权重拟合偏移：${res.active_offset.map(v=>v.toFixed(4))}`);
  } catch (e) { toast(e.message, true); }
};
$("#btn-fork").onclick = async () => {
  try {
    const res = await api("POST",
      `/api/analyses/${state.analysisId}/fork`, {});
    state.analysisId = res.analysis_id;
    state.selectedCandidateId = null;
    await refresh();
    toast("已分叉到新草案（原始观测已复制，候选不复制）");
  } catch (e) { toast(e.message, true); }
};
$("#btn-freeze").onclick = async () => {
  try {
    const rev = state.data.analysis.revision;
    await api("POST", `/api/analyses/${state.analysisId}/freeze`,
              { expected_revision: rev });
    await refresh();
    toast("解释已冻结：校准、容差、人工标记均已钉住");
  } catch (e) { toast(`冻结冲突：${e.message}`, true); }
};

// trainview 화면 — 학습·재생·비교 탭 (TRAIN_VIEWER.md 6절). 순수 ES 모듈, 빌드 단계 없음.
// DOM 은 한 번 짓고 바뀐 글자만 바꾼다. 캔버스는 내용 서명이 바뀔 때만 다시 그린다. 보이지 않는 탭은 그리지 않는다.
import { Plot, ema, fmt, color, esc, hist, cssv } from "./charts.js";
import { Replay } from "./replay.js";

const $ = id => document.getElementById(id);
const enc = encodeURIComponent;
async function api(path, q = {}) {
  const u = path + "?" + Object.entries(q).filter(([, v]) => v !== undefined && v !== null).map(([k, v]) => k + "=" + enc(v)).join("&");
  const r = await fetch(u);
  return r.json();
}
function setText(el, t) { if (typeof el === "string") el = $(el); if (el && el.textContent !== t) el.textContent = t; }
function setHTML(el, h) { if (typeof el === "string") el = $(el); if (el && el._h !== h) { el.innerHTML = h; el._h = h; } }

const S = {
  runs: [], byId: {}, latest: null, sel: null, meta: null, d: null, keys: [], tab: "train",
  xmode: "env_steps", ema: 0.6, cursorPos: 1000, es: null, poll: null, retry: null,
  eps: 0, rep: 0, plots: [], built: "", epsRows: null, cmpSel: new Set(), cmpData: {}, cmpKeys: {},
};
const KIND_ROW = { teacher: "teacher", bc: "student", dagger: "student", rlft: "student", eval: "student", behavior: "behavior" };
let showSyn = false, showArch = false; try { showSyn = localStorage.getItem("tv_show_syn") === "1"; showArch = localStorage.getItem("tv_show_arch") === "1"; } catch (e) {}
const STOPS = [10, 20, 50, 100, 200, 500, 0];
// 학습 상태 불(서버 runs::status): 목록·파이프라인·머리줄, SSE "status" 로 바뀜
S.status = {};
const ST_KO = { training: "학습 중", stalled: "멈춤?(프로세스는 살아 있는데 기록 없음)", finished: "끝남(정상)", crashed: "죽음(끝 표시 없이 프로세스 없음)", unknown: "모름(pid 없는 옛 실행)" };
function stTitle(st) {
  if (!st) return "";
  const lw = st.last_write ? new Date(st.last_write * 1000).toLocaleTimeString() : "—", age = st.age != null ? (st.age < 120 ? st.age.toFixed(0) + " s" : (st.age / 60).toFixed(0) + " min") + " ago" : "";
  return `${st.state} — ${ST_KO[st.state] || ""}\nlast write ${lw} (${age})\niteration ${st.iter ?? "—"}${st.recording ? "\nrecording replay: " + st.recording : ""}`;
}
function stDot(id) { const st = S.status[id] || {}; return `<span class="stl ${st.state || "unknown"}" data-st="${esc(id)}" title="${esc(stTitle(st))}"></span>`; }
function recBadge(id) { const st = S.status[id] || {}; return `<span class="recb" data-rec="${esc(id)}" title="recording replay (record_replay, background)"${st.recording ? "" : " hidden"}>REC</span>`; }
function applyStatus(list) {
  for (const st of list) S.status[st.id] = st;
  document.querySelectorAll("[data-st]").forEach(el => { const st = S.status[el.dataset.st]; if (!st) return; el.className = "stl " + st.state; el.title = stTitle(st); });
  document.querySelectorAll("[data-stl]").forEach(el => { const st = S.status[el.dataset.stl]; if (st) { el.textContent = st.state; el.parentElement.title = stTitle(st); } });
  document.querySelectorAll("[data-rec]").forEach(el => { const st = S.status[el.dataset.rec]; el.hidden = !(st && st.recording); if (st && st.recording) el.title = "recording replay: " + st.recording; });
}
const replay = new Replay({ api, getCursorIter: () => cursorIter() });

// ---------------------------------------------------------------- 실행 목록
function shortName(r) { return r.id.split("/").slice(1).join("/") || r.id; }
async function loadRuns() {
  let j;
  try { j = await api("/api/runs"); } catch (e) { setText("conn", "server offline"); return; }
  S.runs = j.runs; S.byId = Object.fromEntries(j.runs.map(r => [r.id, r])); S.latest = j.latest;
  for (const r of j.runs) if (r.status) S.status[r.id] = r.status;
  // 줄: pipelines(학생 ← 교사), teachers(학생 없는 교사), BEHAVIOR, labs, archive(옛 v2 이전, 끔), test data(가짜, 끔)
  $("show_syn").checked = showSyn; $("show_arch").checked = showArch;
  const pc = v => v == null ? "—" : (v * 100).toFixed(0) + "%";
  const sr = r => { const h = r.health || {}; return h["rollout/success_rate"] ?? Object.entries(h).find(([k]) => k.startsWith("rollout/success_rate/"))?.[1]; };
  const teacherHealth = r => r ? `SR ${pc(sr(r))} · coll ${pc((r.health || {})["rollout/collision_rate"])} · len ${(r.health || {})["rollout/ep_len_mean"] != null ? r.health["rollout/ep_len_mean"].toFixed(0) : "—"}` : "";
  const chip = (r, extra = "") => `<span class="chip${r.id === S.sel ? " on" : ""}" data-id="${esc(r.id)}" title="${esc(r.id)}\n${esc(r.dir)}${r.imported_from ? "\n(imported from " + esc(r.imported_from) + ")" : ""}">${stDot(r.id)}${esc(shortName(r))}<span class="k">${esc(r.kind === "lab" ? "test" : r.kind)}${r.synthetic ? " · syn" : ""}${r.imported_from ? " · csv" : ""}</span>${recBadge(r.id)}${extra}</span>`;
  const isStudent = r => ["bc", "dagger", "rlft", "eval"].includes(r.kind);
  // 시험 자료 = synthetic 또는 옛 "lab" 실행(labs/ 밑, kind lab) — 한 무리, 기본으로 숨김
  for (const r of j.runs) r.test = !!(r.synthetic || r.kind === "lab" || (r.lab && r.kind !== "behavior"));
  const live = j.runs.filter(r => !r.test && !r.archive);
  const students = live.filter(isStudent), used = new Set(students.map(r => r.teacher_run).filter(Boolean));
  setHTML("pick_pipelines", students.map(r => {
    const t = S.byId[r.teacher_run], h = r.health || {};
    const ev = h["eval/success_rate"], tsr = h["eval_teacher/success_rate"];
    return `<span class="pipe${r.id === S.sel || r.teacher_run === S.sel ? " on" : ""}"><span class="stu chip-like" data-id="${esc(r.id)}" title="student / VLA — ${esc(r.id)}">${stDot(r.id)}${esc(shortName(r))} <span class="m">${esc(r.kind)} · eval SR ${pc(ev)}${tsr != null ? " (teacher " + pc(tsr) + ")" : ""}</span>${recBadge(r.id)}</span><span class="arrow">←</span>` +
      (t ? `<span class="tea" data-id="${esc(t.id)}" title="teacher (RL, privileged) — ${esc(t.id)}\ncheckpoint ${esc(r.teacher_ckpt || "")}">${stDot(t.id)}teacher ${esc(shortName(t))} · ${esc(r.teacher_ckpt || "")} · ${teacherHealth(t)}${recBadge(t.id)}</span>` : `<span class="tea" title="${esc(r.teacher || "")}">teacher ${esc(String(r.teacher || "?").split("/").slice(-3).join("/"))} (not under roots)</span>`) + `</span>`;
  }).join(" ") || '<span class="muted small">no student run yet — teachers below</span>');
  setHTML("pick_teachers", live.filter(r => r.kind === "teacher" && !used.has(r.id)).map(r => chip(r, `<span class="h">${teacherHealth(r)}</span>`)).join("") || '<span class="muted small">none</span>');
  setHTML("pick_behavior", live.filter(r => r.kind === "behavior").map(r => chip(r)).join("") || '<span class="muted small">none</span>');
  const arch = j.runs.filter(r => r.archive && !r.test);
  setHTML("pick_archive", showArch ? arch.map(r => chip(r, isStudent(r) ? "" : `<span class="h">${teacherHealth(r)}</span>`)).join("") : `<span class="muted small">${arch.length} hidden (pre-v2 / csv imports)</span>`);
  const syn = j.runs.filter(r => r.test);
  setHTML("pick_synthetic", showSyn ? syn.map(r => chip(r)).join("") : (syn.length ? `<span class="muted small">${syn.length} hidden</span>` : ""));
  if (!S.sel) {
    const want = new URLSearchParams(location.hash.slice(1)).get("run");
    const ok = id => id && S.byId[id] && (showSyn || !S.byId[id].test);
    // 기본: 가장 최근 학생(결과물), 없으면 최근 본실행
    const stu = live.filter(isStudent).sort((a, b) => (a.age ?? 1e12) - (b.age ?? 1e12))[0];
    selectRun(ok(want) ? want : (stu && stu.id) || j.latest || (live[0] || {}).id);
  } else updateStatus();
  buildCompareList();
}
$("show_arch").onchange = e => { showArch = e.target.checked; try { localStorage.setItem("tv_show_arch", showArch ? "1" : "0"); } catch (x) {} loadRuns(); };
$("show_syn").onchange = e => { showSyn = e.target.checked; try { localStorage.setItem("tv_show_syn", showSyn ? "1" : "0"); } catch (x) {} loadRuns(); };
document.addEventListener("click", e => {
  const c = e.target.closest(".chip, .stu, .tea"); if (c && c.dataset.id) selectRun(c.dataset.id);
});

function updateStatus() {
  const r = S.byId[S.sel]; if (!r) return;
  setText("runname", r.id);
  let b = "";
  const st = S.status[r.id] || {};
  b += `<span class="badge stb" title="${esc(stTitle(st))}">${stDot(r.id)}<span data-stl="${esc(r.id)}">${esc(st.state || "unknown")}</span></span>${recBadge(r.id)}`;
  b += `<span class="badge">${esc(r.kind)}</span>`;
  if (r.synthetic) b += '<span class="badge syn" title="fake_run 이 만든 가짜 실행(synthetic data)">synthetic</span>';
  if (r.imported_from) b += `<span class="badge syn" title="${esc(r.imported_from)}">imported csv</span>`;
  if (r.archive) b += `<span class="badge" title="신경망·관측 v2 이전 코드 또는 csv 로 옮긴 옛 실행">archive</span>`;
  if (r.teacher) b += `<span class="badge" title="${esc(r.teacher)}">trained from teacher ${r.teacher_run ? `<a href="#run=${encodeURIComponent(r.teacher_run)}" data-id="${esc(r.teacher_run)}" class="tea">${esc(shortName(S.byId[r.teacher_run] || { id: r.teacher_run }))}</a> · ` : ""}${esc(r.teacher_ckpt || String(r.teacher).split("/").pop())}</span>`;
  setHTML("badges", b);
  $("syn_banner").hidden = !r.test;
  const d = S.d;
  const last = d && d.total ? lastAt("time/iterations", d.total - 1) : null;
  const age = r.age != null ? (r.age < 120 ? r.age.toFixed(0) + " s ago" : r.age < 7200 ? (r.age / 60).toFixed(0) + " min ago" : (r.age / 3600).toFixed(1) + " h ago") : "—";
  setText("status", `iterations ${fmt(last)} · rows ${d ? d.total : 0}${d && d.rewound ? ` (${d.rewound} rewound rows dropped)` : ""} · logged episodes ${r.eps} · last write ${age}`);
}

async function selectRun(id) {
  if (!id) return;
  S.sel = id; S.d = null; S.meta = null; S.built = ""; S.epsRows = null;
  const h = new URLSearchParams(location.hash.slice(1)); h.set("run", id); history.replaceState(null, "", "#" + h.toString());
  document.querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c.dataset.id === id));
  S.built = ""; loadRuns();
  stopLive();
  setHTML("groups", ""); setHTML("cards", ""); setHTML("checks", ""); setHTML("table", ""); setHTML("evals", "");
  const [meta] = await Promise.all([api("/api/meta", { run: id })]);
  if (S.sel !== id) return;
  S.meta = meta.why ? null : meta;
  S.xmode = S.meta && /^time\/iter/.test(S.meta.x_default || "") ? "iter" : "env_steps"; $("xmode").value = S.xmode;
  setText("metajson", JSON.stringify(meta, null, 1));
  const r = S.byId[id] || {};
  S.eps = r.eps || 0; S.rep = r.replays || 0;
  replay.setRun(id, S.meta, r.streams || ["main"]);
  fillTableControls(r);
  await reloadProgress();
  loadEvals(); loadEvents(); loadTable(); loadSample();
  startLive();
}

// ---------------------------------------------------------------- progress
async function reloadProgress() {
  const id = S.sel;
  const d = await api("/api/progress", { run: id, keys: "*", max_points: 4000 });
  if (S.sel !== id) return;
  S.d = d.total ? d : { total: 0, sig: d.sig || 0, x: {}, cols: {}, why: d.why };
  S.keys = Object.keys(S.d.cols || {}).sort();
  S.built = "";
  render();
}
function merge(d) {
  const D = S.d;
  if (!D || d.sig !== D.sig || d.start > D.total) { reloadProgress(); return; }
  const skip = D.total - d.start; if (d.total <= D.total) return;
  for (const k of Object.keys(d.x)) { D.x[k] = (D.x[k] || []).concat(d.x[k].slice(skip)); }
  let newKey = false;
  for (const k of Object.keys(d.cols)) {
    if (!D.cols[k]) { D.cols[k] = new Array(D.total).fill(null); newKey = true; }
    D.cols[k] = D.cols[k].concat(d.cols[k].slice(skip));
    if (D.lo && D.lo[k]) { D.lo[k] = D.lo[k].concat(new Array(d.total - D.total).fill(null)); D.hi[k] = D.hi[k].concat(new Array(d.total - D.total).fill(null)); }
  }
  const n = d.total;
  for (const k of Object.keys(D.cols)) if (D.cols[k].length < n) D.cols[k] = D.cols[k].concat(new Array(n - D.cols[k].length).fill(null));
  D.total = n; D.rewound = d.rewound; D.bad = d.bad;
  if (newKey) { S.keys = Object.keys(D.cols).sort(); S.built = ""; }
  render();
}

// ---------------------------------------------------------------- 실시간: SSE, 끊기면 3 초 폴링
function stopLive() {
  if (S.es) { S.es.close(); S.es = null; }
  clearInterval(S.poll); S.poll = null; clearTimeout(S.retry);
}
function startLive() {
  stopLive();
  const id = S.sel, D = S.d || { total: 0, sig: 0 };
  const es = new EventSource(`/api/live?runs=${enc(id)}&from=${D.total}&sig=${D.sig}&eps=${S.eps}&rep=${S.rep}`);
  S.es = es;
  es.onopen = () => { setText("conn", "live (SSE)"); $("conn").classList.add("ok"); clearInterval(S.poll); S.poll = null; };
  es.addEventListener("progress", e => { const m = JSON.parse(e.data); if (m.run === S.sel) merge(m.data); });
  es.addEventListener("reset", e => { const m = JSON.parse(e.data); if (m.run === S.sel) reloadProgress(); });
  es.addEventListener("episodes", e => { const m = JSON.parse(e.data); if (m.run === S.sel && m.total !== S.eps) { S.eps = m.total; debounce("eps", () => { loadTable(); loadSample(); }, 1500); } });
  es.addEventListener("replay", e => { const m = JSON.parse(e.data); if (m.run === S.sel) { S.rep = m.n; replay.refreshList(); } });
  es.addEventListener("runs", () => loadRuns());
  es.addEventListener("status", e => applyStatus(JSON.parse(e.data)));
  es.onerror = () => {
    es.close(); if (S.es === es) S.es = null;
    setText("conn", "polling 3 s"); $("conn").classList.remove("ok");
    if (!S.poll) S.poll = setInterval(pollOnce, 3000);
    S.retry = setTimeout(() => { if (S.sel === id) startLive(); }, 8000);
  };
}
async function pollOnce() {
  const id = S.sel, D = S.d; if (!D) return;
  const d = await api("/api/progress", { run: id, keys: "*", from: D.total, sig: D.sig });
  if (S.sel !== id) return;
  if (d.sig !== D.sig) { reloadProgress(); return; }
  if (d.total > D.total) merge(d);
}
const timers = {};
function debounce(k, f, ms) { clearTimeout(timers[k]); timers[k] = setTimeout(f, ms); }
setInterval(loadRuns, 10000);

// ---------------------------------------------------------------- 값 꺼내기
function col(k) { return S.d && S.d.cols ? S.d.cols[k] : null; }
function lastAt(k, i) { const c = col(k); if (!c) return null; for (let j = Math.min(i, c.length - 1); j >= 0; j--) if (c[j] != null && isFinite(c[j])) return c[j]; return null; }
function firstVal(k) { const c = col(k); if (!c) return null; for (const v of c) if (v != null && isFinite(v)) return v; return null; }
function cursorRow() { const n = S.d ? S.d.total : 0; if (!n) return -1; return S.cursorPos >= 1000 ? n - 1 : Math.round(S.cursorPos / 1000 * (n - 1)); }
function cursorIter() { const i = cursorRow(); return i < 0 || S.cursorPos >= 1000 ? null : S.d.x.iterations[i]; }
function xs() { if (!S.d || !S.d.x) return []; return S.xmode === "wall" ? S.d.x.time_elapsed : S.xmode === "iter" ? S.d.x.iterations : S.d.x.total_timesteps; }
function expand(pats) {
  const out = [];
  for (const p of pats) {
    if (!p.includes("*")) { if (col(p)) out.push(p); continue; }
    const re = new RegExp("^" + p.split("*").map(s => s.replace(/[.+?^${}()|[\]\\/]/g, "\\$&")).join("([^]*)") + "$");
    for (const k of S.keys) if (re.test(k) && !out.includes(k)) out.push(k);
  }
  return out;
}
function metaNum(path) { let v = S.meta; for (const p of path.split(".")) { if (v == null) return null; v = v[p]; } return typeof v === "number" ? v : null; }
function ref(key, label) { const v = S.meta && S.meta.refs ? S.meta.refs[key] : null; return v != null ? [{ y: v, label: label || key + " " + fmt(v) }] : []; }

// ---------------------------------------------------------------- 곡선 묶음 (CHARTS 한 곳이 화면을 정한다). 키 이름 = SB3 꼴 표준(trainfmt keys.rs)
function chartsSpec() {
  const skills = (S.meta && S.meta.skills) || [...new Set(S.keys.filter(k => k.startsWith("reward/")).map(k => k.split("/")[1]))];
  const budget = sk => { const b = metaNum(`budget_s.${sk}`), hz = metaNum("ctrl_hz"); return b != null && hz != null ? [{ y: b * hz, label: `budget ${sk} ${b} s` }] : []; };
  return [
    { g: "Rollout (all episodes)", ko: "학습 판 전체(모집단)", c: [
      { t: "Success rate", ko: "성공률", k: ["rollout/success_rate", "rollout/success_rate/*"], pct: 1 },
      { t: "Eval success rate", ko: "평가 성공률(평가 판)", k: ["eval/success_rate", "eval/success_rate/*", "eval_teacher/success_rate", "eval_teacher/success_rate/*"], pct: 1 },
      { t: "Episode length (steps) — rollout/ep_len_mean", ko: "판 길이", k: ["rollout/ep_len_mean", "rollout/ep_len_mean/*", "eval/mean_ep_length"], refs: () => skills.flatMap(budget) },
      { t: "Timeout · collision · drop rate", ko: "시간 초과·충돌·떨어뜨림 비율", k: ["rollout/timeout_rate", "rollout/collision_rate", "rollout/drop_rate", "eval/collision_rate", "eval/timeout_rate"], pct: 1 },
      { t: "Teacher-driven episodes (separate split)", ko: "교사가 몬 판 — 학생 집계와 따로", k: ["rollout_teacher/success_rate", "rollout_teacher/success_rate/*", "rollout_teacher/collision_rate", "rollout_teacher/timeout_rate"], pct: 1 },
      { t: "Episodes per row", ko: "줄마다 끝난 판 수", k: ["rollout/n_episodes", "rollout_teacher/n_episodes"] },
    ]},
    { g: "Reward", ko: "보상", c: [
      { t: "Episode return — rollout/ep_rew_mean", ko: "판 보상 합(리턴) 평균", k: ["rollout/ep_rew_mean"] },
      { t: "Reward per step", ko: "스텝당 보상", k: ["rollout/step_reward_mean"] },
      ...skills.map(sk => ({ t: `Reward terms — ${sk} (per-episode sum)`, ko: "보상 항목별 판 합 평균", k: [`reward/${sk}/*`] })),
    ]},
    { g: "Safety", ko: "안전", c: [
      { t: "Contacts per episode", ko: "판당 접촉", k: ["rollout/contacts_per_ep"] },
      { t: "Steps near joint limits", ko: "관절 한계 근처 스텝", k: ["rollout/joint_limit_steps"] },
      { t: "Safety-filter interventions", ko: "거르개 개입", k: ["rollout/filter_rate"], pct: 1 },
      { t: "Min clearance (m)", ko: "최소 여유 거리", k: ["rollout/min_clear_m"] },
    ]},
    { g: "Policy optimization (train/)", ko: "학습 건강", c: [
      { t: "Entropy loss — train/entropy_loss", ko: "엔트로피 손실(= −엔트로피, SB3)", k: ["train/entropy_loss"] },
      { t: "Policy log std — train/log_std", ko: "정책 log σ (탐색)", k: ["train/log_std"] },
      { t: "Policy std per action — train/std", ko: "행동별 σ", k: ["train/std", "train/std/*"] },
      { t: "Approx KL — train/approx_kl", ko: "근사 KL", k: ["train/approx_kl"], refs: () => ref("train/approx_kl", "target_kl") },
      { t: "Clip fraction — train/clip_fraction", ko: "잘린 비율", k: ["train/clip_fraction"], pct: 1 },
      { t: "Explained variance — train/explained_variance", ko: "설명된 분산", k: ["train/explained_variance"] },
      { t: "Value loss — train/value_loss", ko: "가치 손실", k: ["train/value_loss"] },
      { t: "Policy gradient loss — train/policy_gradient_loss", ko: "정책 손실", k: ["train/policy_gradient_loss"] },
      { t: "Gradient norm — train/grad_norm", ko: "기울기 노름", k: ["train/grad_norm"], refs: () => ref("train/grad_norm", "max_grad_norm") },
      { t: "Learning rate — train/learning_rate", ko: "학습률", k: ["train/learning_rate"] },
      { t: "Advantage std · value mean", ko: "이득 표준편차 · 가치 평균", k: ["train/advantage_std", "train/value_mean"] },
    ]},
    { g: "Imitation (BC · DAgger · RL fine-tuning)", ko: "학생", c: [
      { t: "Training loss — train/loss", ko: "BC 손실", k: ["train/loss", "train/flow_loss", "val/flow_loss"] },
      { t: "Validation action MSE", ko: "검증 행동 MSE", k: ["val/action_mse"] },
      { t: "End-signal BCE", ko: "끝 신호 BCE", k: ["train/end_bce"] },
      { t: "Student success by start map", ko: "학생이 몬 판 성공률, 처음 지도별", k: ["rollout/success_rate_by_start_map/*"], pct: 1 },
      { t: "Teacher success by start map", ko: "교사 기록 판 성공률", k: ["rollout_teacher/success_rate_by_start_map/*"], pct: 1 },
      { t: "Eval success by map completeness", ko: "평가 — 처음 지도 완성도 칸별", k: ["eval/success_rate_by_completion/*"], pct: 1 },
      { t: "Student–teacher action MSE — dagger/action_mse", ko: "학생–교사 행동 차", k: ["dagger/action_mse"] },
      { t: "DAgger β · round", ko: "DAgger β · 바퀴", k: ["dagger/beta", "dagger/round"] },
      { t: "Dataset size — dagger/dataset_size", ko: "자료 표본 수", k: ["dagger/dataset_size", "dagger/new_samples"] },
      { t: "RL fine-tuning", ko: "RL 다듬기", k: ["rlft/*"] },
    ]},
    { g: "Curriculum · map", ko: "커리큘럼 · 지도", c: [
      { t: "Success by start map (C0 full · C1 partial · C2 empty)", ko: "처음 지도별 성공률", k: ["curriculum/success_rate/*"], pct: 1 },
      { t: "Collision rate by start map", ko: "처음 지도별 충돌률", k: ["curriculum/collision_rate/*"], pct: 1 },
      { t: "Start-map mix", ko: "처음 지도 섞임(판 비율)", k: ["curriculum/start_map_fraction/*"], pct: 1 },
      { t: "Curriculum stage", ko: "커리큘럼 단계 번호", k: ["curriculum/stage", "curriculum/env_level"] },
      { t: "Start map completeness (mean)", ko: "시작 완성도 평균", k: ["curriculum/start_completion_mean"], pct: 1 },
      { t: "Goal known · task object confirmed", ko: "목표를 앎 · 지도에 컵 확정", k: ["curriculum/goal_known_rate", "map/task_confirmed"], pct: 1 },
    ]},
    { g: "Throughput · resources (time/)", ko: "속도 · 자원", c: [
      { t: "FPS (env steps/s) — time/fps", ko: "초당 환경 스텝", k: ["time/fps", "time/fps_gpu"] },
      { t: "Iteration time (ms, GPU events)", ko: "이터 시간", k: ["time/rollout_ms", "time/update_ms", "time/iter_ms"] },
      { t: "BC phase GPU time (ms)", ko: "BC 단계 GPU 시간", k: ["time/update_gpu_ms", "time/rollout_gpu_ms", "time/eval_rollout_gpu_ms"] },
      { t: "GPU memory (MB)", ko: "GPU 메모리", k: ["gpu/mem_used_mb", "gpu/mem_reserved_mb"] },
      { t: "Dropped log records (should be 0)", ko: "로그 버림", k: ["log/*"] },
      { t: "Iterations merged per row", ko: "한 줄에 합친 업데이트 수", k: ["time/iterations_merged"] },
    ]},
    { g: "FP8", ko: "FP8 감시", c: [
      { t: "amax per layer", ko: "층별 amax", k: ["fp8/amax/*"] },
      { t: "Overflow rate", ko: "넘침 비율", k: ["fp8/overflow/*"], pct: 1 },
      { t: "Underflow rate", ko: "밑넘침 비율", k: ["fp8/underflow/*"], pct: 1 },
      { t: "Gradient cosine vs BF16", ko: "BF16 기울기와 코사인", k: ["fp8/grad_cos"], refs: () => [{ y: (S.meta && S.meta.refs && S.meta.refs["fp8/grad_cos"]) || 0.99, label: "0.99 (GPU_TRAINING 9.1)" }] },
    ]},
  ];
}
const WANT = { "Rollout (all episodes)": ["rollout/success_rate", "rollout/ep_len_mean", "rollout/timeout_rate"], "Safety": ["rollout/contacts_per_ep", "rollout/min_clear_m"], "Policy optimization (train/)": ["train/explained_variance"], "FP8": ["fp8/grad_cos"], "Imitation (BC · DAgger · RL fine-tuning)": ["train/loss", "dagger/action_mse"], "Reward": ["rollout/ep_rew_mean", "reward/<skill>/<term>"] };

function buildCharts() {
  const sig = S.sel + "|" + S.keys.join(",");
  if (S.built === sig) return;
  S.built = sig; S.plots = [];
  const host = $("groups"); host.innerHTML = "";
  for (const grp of chartsSpec()) {
    const have = grp.c.map(c => ({ c, keys: expand(c.k) })).filter(x => x.keys.length);
    const missing = grp.c.filter(c => !expand(c.k).length).flatMap(c => c.k);
    const div = document.createElement("div"); div.className = "group";
    const nl = have.length ? "" : `not logged: ${(WANT[grp.g] || missing.slice(0, 4)).map(k => "<code>" + esc(k) + "</code>").join(" ")}`;
    div.innerHTML = `<h3 title="${esc(grp.ko)}">${esc(grp.g)} <span class="nl">${have.length ? `${have.length} charts` : nl}</span></h3>`;
    if (!have.length) { host.appendChild(div); continue; }
    const box = document.createElement("div"); box.className = "charts"; div.appendChild(box);
    for (const { c, keys } of have) {
      const el = document.createElement("div"); el.className = "chart";
      el.innerHTML = `<div class="ct"><span title="${esc(c.ko || "")}">${esc(c.t)}</span><span class="u mono"></span></div><canvas></canvas><div class="lg"></div><div class="cn"></div>`;
      box.appendChild(el);
      const p = new Plot(el.querySelector("canvas"));
      S.plots.push({ c, keys, el, p, sig: "" });
    }
    host.appendChild(div);
  }
  // 판 기록 ⚠ 표본: 모집단에 없는 칸만(분포)
  const div = document.createElement("div"); div.className = "group"; div.id = "sample_group";
  div.innerHTML = `<h3 title="판 기록(표본)">Logged episodes <span class="warn-s">⚠ sample</span> <span class="nl" id="sample_note"></span></h3><div class="charts" id="sample_charts"></div>`;
  host.appendChild(div);
  if (S.eps) debounce("smp", loadSample, 100);
}

function label(keys, k) {
  if (keys.length === 1) return k;
  const parts = keys.map(x => x.split("/"));
  let i = 0; while (parts.every(p => p.length > i + 1 && p[i] === parts[0][i])) i++;
  return k.split("/").slice(i).join("/");
}

function drawCharts() {
  if (!S.d || S.tab !== "train") return;
  const x = xs(), n = S.d.total, crow = cursorRow();
  const cx = S.cursorPos < 1000 && crow >= 0 ? x[crow] : null;
  const xl = { iter: "time/iterations", env_steps: "time/total_timesteps", wall: "time/time_elapsed (s)" }[S.xmode];
  const theme = matchMedia("(prefers-color-scheme: dark)").matches;
  for (const P of S.plots) {
    const w = P.el.clientWidth;
    const sig = [n, S.xmode, S.ema, cx, w, theme].join("|");
    if (P.sig === sig) continue;
    P.sig = sig;
    const series = P.keys.slice(0, 8).map((k, i) => {
      const y = S.d.cols[k];
      return { name: label(P.keys, k), x, y: ema(y, S.ema), lo: S.d.lo && S.ema === 0 ? S.d.lo[k] : null, hi: S.d.hi && S.ema === 0 ? S.d.hi[k] : null, color: color(i) };
    });
    const refs = P.c.refs ? P.c.refs() : [];
    P.p.draw({ series, refs, pct: P.c.pct, cursor: cx, xlabel: xl, onPick: xv => setCursorX(xv) });
    const last = series.map(s => { for (let i = s.y.length - 1; i >= 0; i--) if (s.y[i] != null) return s.y[i]; return null; });
    setText(P.el.querySelector(".u"), series.length === 1 ? fmt(last[0], P.c.pct) : "");
    setHTML(P.el.querySelector(".lg"), series.length > 1 ? series.map(s => `<span style="--c:${s.color}">${esc(s.name)}</span>`).join("") : "");
    const notes = [];
    if (P.keys.length > 8) notes.push(`${P.keys.length - 8} more keys (8 series max)`);
    if (P.c.refs && !refs.length) notes.push("no reference value in run.json refs — line not drawn");
    if (S.d.agg) notes.push("downsampled: min–max band per bucket (smoothing 0) + last value");
    setText(P.el.querySelector(".cn"), notes.join(" · "));
  }
}
function setCursorX(xv) {
  const x = xs(); if (!x.length) return;
  let bi = 0, bd = Infinity; for (let i = 0; i < x.length; i++) { const d = Math.abs(x[i] - xv); if (d < bd) { bd = d; bi = i; } }
  S.cursorPos = x.length > 1 ? Math.round(bi / (x.length - 1) * 999) : 1000; $("cursor").value = S.cursorPos; render(); loadTable();
}

// ---------------------------------------------------------------- 요약 카드 (summary cards)
function card(t, v, s, bad, ko) { return `<div class="card${bad ? " bad" : ""}"${ko ? ` title="${esc(ko)}"` : ""}><div class="t">${t}</div><div class="v">${v}</div><div class="s">${s || ""}</div></div>`; }
function renderCards() {
  if (!S.d) return;
  const i = cursorRow(), m = S.meta || {};
  if (i < 0) { setHTML("cards", card("progress", "—", esc(S.d.why || "no rows"))); return; }
  const g = k => lastAt(k, i);
  const out = [];
  out.push(card("Iterations · timesteps", fmt(g("time/iterations")), `time/total_timesteps ${fmt(g("time/total_timesteps"))}`, false, "이터 · 누적 환경 스텝"));
  const wall = g("time/time_elapsed");
  out.push(card("FPS · time elapsed", fmt(g("time/fps") ?? g("time/fps_gpu")), `${wall != null ? (wall >= 3600 ? (wall / 3600).toFixed(2) + " h" : (wall / 60).toFixed(1) + " min") : "—"}${g("time/fps") == null && g("time/fps_gpu") != null ? " · GPU-event based" : ""}`, false, "초당 환경 스텝 · 경과"));
  const stages = m.curriculum && m.curriculum.stages;
  const si = g("curriculum/stage");
  const stName = stages && si != null && stages[si] ? (stages[si].name || stages[si]) : m.stage || "—";
  const mix = ["C0", "C1", "C2"].map(c => g("curriculum/start_map_fraction/" + c)).some(v => v != null) ? ["C0", "C1", "C2"].map(c => `${c} ${fmt(g("curriculum/start_map_fraction/" + c), 1)}`).join(" · ") : "";
  out.push(card("Run type · curriculum stage", esc(`${m.kind || "?"} · ${stName}`), mix, false, "실행 종류 · 커리큘럼 단계 · 처음 지도 섞임"));
  const sk = expand(["rollout/success_rate", "rollout/success_rate/*"]);
  if (sk.length) {
    const bars = sk.map(k => { const v = g(k); const n = k === "rollout/success_rate" ? "all" : k.split("/").pop(); return `<div class="minibar"><span>${esc(n)}</span><span class="b"><i style="width:${v == null ? 0 : (v * 100).toFixed(1)}%"></i></span><span>${v == null ? "—" : (v * 100).toFixed(1)}</span></div>`; }).join("");
    out.push(`<div class="card" title="학습 판 전체의 성공률(모집단)"><div class="t">rollout/success_rate (%)</div>${bars}</div>`);
  } else out.push(card("Success rate", "—", "not logged: <code>rollout/success_rate</code>"));
  out.push(card("Episode return · length", fmt(g("rollout/ep_rew_mean")), `rollout/ep_len_mean ${fmt(g("rollout/ep_len_mean"))}`, false, "판 보상 합(리턴) 평균 · 판 길이"));
  const ne = col("rollout/n_episodes");
  let pop = null; if (ne) { pop = 0; for (let j = 0; j <= i; j++) if (ne[j] != null) pop += ne[j]; }
  out.push(card("Episodes", fmt(pop), `all (rollout) / logged sample ${fmt(S.eps)}`, false, "끝난 판 수: 모집단 / 기록된 표본"));
  const ls = g("train/log_std"), ls0 = firstVal("train/log_std");
  out.push(card("Policy log std", fmt(ls), ls != null && ls0 != null ? `std ${fmt(Math.exp(ls))} · ${fmt(Math.exp(ls - ls0), 1)} of initial` : "not logged: <code>train/log_std</code>", false, "정책 log σ (탐색)"));
  const kl = g("train/approx_kl"), tk = m.target_kl ?? (m.refs && m.refs["train/approx_kl"]);
  out.push(card("Approx KL · explained var.", fmt(kl), `${tk != null ? "target_kl " + fmt(tk) : "no target_kl"} · EV ${g("train/explained_variance") != null ? fmt(g("train/explained_variance")) : "not logged"}`, kl != null && tk != null && kl > 1.5 * tk, "근사 KL · 설명된 분산"));
  const mem = g("gpu/mem_used_mb");
  out.push(card("GPU memory", mem != null ? fmt(mem / 1024) + " GB" : "—", mem != null ? `/ 16 GB (${fmt(mem / 16384, 1)})` : "not logged: <code>gpu/mem_used_mb</code>", false, "GPU 메모리"));
  const dk = expand(["log/*"]);
  let drop = null; for (const k of dk) { const c = col(k); for (let j = 0; j <= i; j++) if (c[j] > 0) drop = (drop || 0) + c[j]; }
  out.push(card("Dropped log records", dk.length ? fmt(drop || 0) : "—", dk.length ? dk.map(k => k.split("/").pop()).join(", ") : "not logged: <code>log/*_dropped</code>", drop > 0, "로그 버림"));
  if (S.sr) out.push(card("SR · SPL (logged sample)", fmt(S.sr.success, 1), `SPL ${fmt(S.sr.spl, 1)} · collision ${fmt(S.sr.collision, 1)} · n ${S.sr.n}`, false, "성공률 · SPL (표본)"));
  const es = g("eval/success_rate") ?? expand(["eval/success_rate/*"]).map(g).find(v => v != null);
  const et = g("eval_teacher/success_rate") ?? expand(["eval_teacher/success_rate/*"]).map(g).find(v => v != null);
  if (es != null || et != null) out.push(card("eval/success_rate (student · teacher)", `${fmt(es, 1)}`, `teacher ${fmt(et, 1)}${es != null && et ? " · ratio " + fmt(es / et, 1) : ""}`, false, "평가 성공률 학생 / 교사"));
  if (col("train/loss")) out.push(card("train/loss (BC)", fmt(g("train/loss")), `dagger/action_mse ${fmt(g("dagger/action_mse"))} · round ${fmt(g("dagger/round"))}`, false, "BC 손실 · 학생–교사 행동 차"));
  setHTML("cards", out.join(""));
}

// ---------------------------------------------------------------- 학습 진단(training health checks). 못 잰 검사는 이유와 함께 따로
function slope(k, a, b) {
  const c = col(k), x = S.d.x.total_timesteps; if (!c) return null;
  let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = a; i <= b; i++) { const v = c[i]; if (v == null || !isFinite(v)) continue; const xx = x[i]; n++; sx += xx; sy += v; sxx += xx * xx; sxy += xx * v; }
  if (n < 5) return null;
  const d = n * sxx - sx * sx; return d ? (n * sxy - sx * sy) / d : null;
}
function meanOf(k, a, b) { const c = col(k); if (!c) return null; let s = 0, n = 0; for (let i = a; i <= b; i++) if (c[i] != null && isFinite(c[i])) { s += c[i]; n++; } return n ? s / n : null; }
function renderChecks() {
  if (!S.d || !S.d.total) { setHTML("checks", '<span class="muted">no progress rows — nothing to check</span>'); setText("checks_sum", ""); return; }
  const b = cursorRow(), a = Math.max(0, b - Math.max(20, Math.floor((b + 1) * 0.2))), m = S.meta || {}, th = m.checks || {};
  const R = [];
  const add = (st, name, msg, ko) => R.push({ st, name, msg, ko });
  const sks = expand(["rollout/success_rate", "rollout/success_rate/*"]);
  if (!col("rollout/ep_rew_mean") || !sks.length) add("na", "Reward hacking", "not measurable: needs rollout/ep_rew_mean and rollout/success_rate", "꼼수 의심");
  else {
    const sr = slope("rollout/ep_rew_mean", a, b);
    const bad = sks.filter(k => { const s = slope(k, a, b); return sr != null && s != null && sr > 0 && s <= 0 && meanOf(k, a, b) < 0.98; });
    if (sr == null) add("na", "Reward hacking", "not measurable: < 5 values in window", "꼼수 의심");
    else add(bad.length ? "warn" : "ok", "Reward hacking", bad.length ? `return rises while success is flat/falling: ${bad.map(k => k.split("/").pop()).join(", ")}` : "return and success trend together", "꼼수 의심: 리턴은 오르는데 성공률은 그대로");
  }
  const ls = lastAt("train/log_std", b), ls0 = firstVal("train/log_std");
  if (ls == null) add("na", "Entropy collapse", "not measurable: no train/log_std", "탐색 죽음");
  else { const lim = th.explore_drop ?? 2.3; add(ls0 - ls > lim ? "warn" : "ok", "Entropy collapse", `log std ${fmt(ls0)} → ${fmt(ls)} (threshold: drop > ${lim}, assumed)`, "탐색 죽음"); }
  const tk = m.target_kl ?? (m.refs && m.refs["train/approx_kl"]), kl = meanOf("train/approx_kl", a, b);
  if (kl == null || tk == null) add("na", "KL too high", `not measurable: no ${kl == null ? "train/approx_kl" : "target_kl in run.json"}`, "KL 넘침");
  else add(kl > tk * 1.5 ? "warn" : "ok", "KL too high", `window mean ${fmt(kl)} vs target_kl ${fmt(tk)} × 1.5`, "KL 넘침");
  const ev = meanOf("train/explained_variance", a, b);
  if (ev == null || kl == null) add("na", "Value saturation", `not measurable: no ${ev == null ? "train/explained_variance" : "train/approx_kl"}`, "가치 머리 포화");
  else add(ev >= 0.99 && kl < 0.001 ? "warn" : "ok", "Value saturation", `EV ${fmt(ev)}, KL ${fmt(kl)}`, "가치 머리 포화");
  const to = meanOf("rollout/timeout_rate", a, b);
  if (to == null) add("na", "Agent not moving", "not measurable: no rollout/timeout_rate", "안 움직임");
  else add(to >= 0.99 ? "warn" : "ok", "Agent not moving", `timeout rate ${fmt(to, 1)}`, "안 움직임");
  const cs = slope("rollout/contacts_per_ep", a, b);
  if (cs == null) add("na", "Contacts rising", "not measurable: no rollout/contacts_per_ep", "접촉 늘어남");
  else add(cs > 0 && meanOf("rollout/contacts_per_ep", a, b) > 0.05 ? "warn" : "ok", "Contacts rising", `slope ${cs > 0 ? "+" : ""}${fmt(cs * 1e9)} per 1e9 steps`, "접촉 늘어남");
  const dk = expand(["log/*"]);
  if (!dk.length) add("na", "Dropped logs", "not measurable: log/*_dropped not logged", "로그 버림");
  else { let s = 0; for (const k of dk) { const c = col(k); for (let i = 0; i <= b; i++) if (c[i] > 0) s += c[i]; } add(s > 0 ? "warn" : "ok", "Dropped logs", `total ${fmt(s)}`, "로그 버림"); }
  const ov = expand(["fp8/overflow/*"]), gc = meanOf("fp8/grad_cos", a, b);
  const fp8on = m.precision && m.precision.fp8;
  if (!ov.length && gc == null) add("na", "FP8 numerics", fp8on ? "not measurable: fp8 on but no fp8/* monitors logged" : "FP8 off (or no monitors)", "FP8 위험");
  else { const o = Math.max(...ov.map(k => meanOf(k, a, b) || 0)); const lim = th.fp8_overflow ?? 1e-3; add(o > lim || (gc != null && gc < 0.99) ? "warn" : "ok", "FP8 numerics", `max overflow ${fmt(o)} (threshold ${lim}), cosine ${fmt(gc)}`, "FP8 위험"); }
  const ic = { ok: "✓", warn: "!", na: "?" };
  setHTML("checks", R.map(r => `<div class="chk ${r.st}" title="${esc(r.ko)}"><span class="ic">${ic[r.st]}</span><b>${r.name}</b> <span>${esc(r.msg)}</span></div>`).join(""));
  const ok = R.filter(r => r.st !== "na").length, warn = R.filter(r => r.st === "warn").length;
  setText("checks_sum", `checked ${ok} (flagged ${warn}) · not measurable ${R.length - ok} · window = last ${b - a + 1} rows before cursor`);
}

function renderFirstNote() {
  const n = $("note_first"), m = S.meta || {};
  const ne = col("rollout/n_episodes") || col("rollout_teacher/n_episodes");
  const any = ne && ne.some(v => v > 0);
  if (!S.d || !S.d.total || any || !ne) { n.hidden = true; return; }
  const tm = m.t_max ? Math.max(...Object.values(m.t_max)) : null;
  n.hidden = false;
  n.textContent = tm && m.ctrl_hz && m.rollout_T ? `No episode finished yet — the first one ends around iteration ${Math.ceil(tm * m.ctrl_hz / m.rollout_T)} (t_max × ctrl_hz / rollout_T). Empty chart ≠ broken.` : "No episode finished yet (run.json has no t_max to estimate when)";
}

function render() {
  if (!S.d) return;
  updateStatus();
  const i = cursorRow();
  setText("cursor_v", S.cursorPos >= 1000 ? "latest (live)" : `iteration ${fmt(S.d.x.iterations[i])}`);
  const m = S.meta || {};
  const git = m.git && m.git.commit ? `git ${m.git.commit.slice(0, 7)}${m.git.dirty ? "+" : ""}` : "";
  setHTML("metaline", [m.trainer, m.group ? `group <b>${esc(m.group)}</b>` : "", m.seed != null ? `seed ${m.seed}` : "", m.precision ? `fp8 ${m.precision.fp8}` : "", m.teacher ? `teacher ${esc(String(m.teacher).split("/").slice(-3).join("/"))}` : "", m.n_envs ? `N ${m.n_envs} × T ${m.rollout_T}` : "", git, m.logged && m.logged.why && !m.logged.episodes ? `<span title="${esc(m.logged.why)}">no per-episode log (?)</span>` : ""].filter(Boolean).join(" · ") || esc(S.d.why || ""));
  if (S.tab === "train") {
    buildCharts(); drawCharts(); renderCards(); renderChecks(); renderFirstNote();
  }
}

// ---------------------------------------------------------------- 성공 표 (표본)
function fillTableControls(r) {
  setHTML("t_stream", (r.streams || ["main"]).map(s => `<option>${esc(s)}</option>`).join(""));
}
async function loadTable() {
  const id = S.sel; if (!id) return;
  const last = STOPS[+$("t_last").value];
  setText("t_last_v", last ? String(last) : "all");
  const q = { run: id, stream: $("t_stream").value || "main", rows: $("t_rows").value, cols: $("t_cols").value, last, home: $("t_home").value, stage: $("t_stage").value };
  const ci = cursorIter(); if (ci != null) q.upto = S.d.x.total_timesteps[cursorRow()];
  const [t, all] = await Promise.all([api("/api/table", q), api("/api/table", { run: id, stream: "main", rows: "all", cols: "all", last: 0, upto: q.upto })]);
  if (S.sel !== id) return;
  S.sr = all.cells && all.cells[0] ? { ...all.cells[0][0] } : null;
  if (S.sr && !S.sr.n) S.sr = null;
  renderCards();
  if (t.why || !t.cells || !t.cells.length) { setHTML("table", `<span class="muted">${esc(t.why || "no table")} — ${S.meta && S.meta.logged && S.meta.logged.why ? esc(S.meta.logged.why) : "needs episodes.jsonl"}</span>`); setText("t_note", ""); return; }
  for (const [sel, dim] of [["t_home", "home"], ["t_stage", "stage"]]) {
    const cur = $(sel).value, opts = ['<option value="">all</option>'].concat((t.dims[dim] || []).map(n => `<option${n === cur ? " selected" : ""}>${esc(n)}</option>`)).join("");
    setHTML(sel, opts);
  }
  const met = $("t_metric").value, pct = ["success", "spl", "collision", "timeout"].includes(met);
  const hue = met === "collision" || met === "timeout" ? "--s2" : "--s1";
  const m = S.meta || {};
  setText("t_note", `⚠ sample: every episode of ${m.log_envs ?? "?"} logged envs (${t.total} episodes). Last ${last || "all"} episodes per cell${ci != null ? `, up to iteration ${fmt(ci)}` : ""}. n < 20 is faded (small sample). Compare with rollout/success_rate (all episodes) above.`);
  let h = `<table class="t"><tr><th></th>${t.cols.map(c => `<th>${esc(c)}</th>`).join("")}</tr>`;
  t.rows.forEach((rn, ri) => {
    h += `<tr><th>${esc(rn)}</th>`;
    t.cols.forEach((cn, ci2) => {
      const c = t.cells[ri][ci2];
      if (!c.n) { h += `<td class="cell none">—</td>`; return; }
      const v = c[met];
      const a = pct && v != null ? Math.round(8 + 55 * Math.min(1, Math.max(0, v))) : 0;
      h += `<td class="cell${c.n < 20 ? " low" : ""}" style="${a ? `background:color-mix(in srgb, var(${hue}) ${a}%, transparent)` : ""}" title="n ${c.n}${c.n < 20 ? " — small sample" : ""}\nsuccess rate ${fmt(c.success, 1)} · SPL ${fmt(c.spl, 1)} · collision ${fmt(c.collision, 1)} · timeout ${fmt(c.timeout, 1)}\nmean length ${fmt(c.t)} s · contacts ${fmt(c.contacts)} · return ${fmt(c.ret)}"><b>${fmt(v, pct)}</b><span class="n">n ${c.n} · t ${fmt(c.t)} s</span></td>`;
    });
    h += "</tr>";
  });
  setHTML("table", h + "</table>");
}
["t_stream", "t_rows", "t_cols", "t_home", "t_stage", "t_metric"].forEach(id => $(id).addEventListener("change", loadTable));
$("t_last").addEventListener("input", loadTable);

async function loadSample() {
  const id = S.sel; if (!id) return;
  const host = $("sample_charts"); if (!host) return;
  if (!S.eps) { setText("sample_note", "no episodes.jsonl — " + (S.meta && S.meta.logged && S.meta.logged.why || "")); host.innerHTML = ""; return; }
  const from = Math.max(0, S.eps - 20000);
  const e = await api("/api/episodes", { run: id, from });
  if (S.sel !== id || !e.rows) return;
  setText("sample_note", `last ${e.rows.length} episodes (of ${e.total}${e.truncated ? ", truncated at cap " + e.cap : ""}${e.ret_mismatch ? `, ${e.ret_mismatch} rows with ret ≠ Σr` : ""}${e.bad ? `, ${e.bad} unparsable rows` : ""})`);
  const specs = [["Episode length t (s)", r => r.t, "--s1"], ["Min clearance (m)", r => r.min_clear_m, "--s3"], ["Final distance (successes, m)", r => r.success ? r.final_dist : null, "--s7"], ["Final aim error (successes, °)", r => r.success ? r.final_aim_deg : null, "--s4"]];
  if (!host.children.length) host.innerHTML = specs.map(s => `<div class="chart"><div class="ct">${esc(s[0])}</div><canvas></canvas></div>`).join("");
  specs.forEach((s, i) => hist(host.children[i].querySelector("canvas"), e.rows.map(s[1]), 30, "episodes", cssv(s[2])));
}

// ---------------------------------------------------------------- 평가 표 · 사건
async function loadEvals() {
  const id = S.sel, j = await api("/api/evals", { run: id }); if (S.sel !== id) return;
  if (!j.evals.length) { setHTML("evals", `<span class="muted">${esc(j.why)}</span>`); return; }
  setHTML("evals", j.evals.slice(-12).reverse().map(({ file, eval: e }) => {
    const rows = e.rows || [];
    const hasRef = rows.some(r => r.ref != null);
    return `<div class="small" style="margin-top:8px"><b class="mono">${esc(file)}</b> ${e.iter != null ? "iter " + fmt(e.iter) : ""} ${e.driver ? "· " + esc(e.driver) : ""} ${e.ckpt ? '· <span class="mono muted">' + esc(String(e.ckpt).split("/").pop()) + "</span>" : ""}</div>
      <table class="t"><tr><th>eval</th><th>split</th><th>n</th><th>success_rate</th><th>collision_rate</th><th>timeout_rate</th>${hasRef ? "<th>reference</th>" : ""}</tr>${rows.map(r => `<tr><td class="l">${esc(r.eval)}</td><td class="l">${esc(r.split)}</td><td>${fmt(r.n)}</td><td class="${r.pass === true ? "pass" : r.pass === false ? "fail" : ""}">${fmt(r.success, 1)}${r.pass === true ? " ✓" : r.pass === false ? " ✗" : ""}</td><td>${fmt(r.collision, 1)}</td><td>${fmt(r.timeout, 1)}</td>${hasRef ? `<td>${fmt(r.ref, 1)}</td>` : ""}</tr>`).join("")}</table>`;
  }).join(""));
}
async function loadEvents() {
  const id = S.sel, j = await api("/api/events", { run: id }); if (S.sel !== id) return;
  setText("events", j.lines.length ? j.lines.join("\n") : "no events.txt");
}

// ---------------------------------------------------------------- 탭 · 조작
function setTab(t) {
  S.tab = t;
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === t));
  document.querySelectorAll(".tab").forEach(s => s.classList.toggle("on", s.id === "tab_" + t));
  const h = new URLSearchParams(location.hash.slice(1)); h.set("tab", t); history.replaceState(null, "", "#" + h.toString());
  replay.show(t === "replay"); document.body.classList.toggle("rp", t === "replay");
  if (t === "train") { S.plots.forEach(p => p.sig = ""); render(); }
  if (t === "compare") { buildCompareList(); drawCompare(); }
}
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => setTab(b.dataset.tab));
$("xmode").onchange = e => { S.xmode = e.target.value; render(); };
$("ema").oninput = e => { S.ema = +e.target.value; setText("ema_v", S.ema.toFixed(2)); render(); };
setText("ema_v", S.ema.toFixed(2));
$("cursor").oninput = e => { S.cursorPos = +e.target.value; render(); debounce("tbl", loadTable, 250); replay.refreshList(); };
$("cursor_off").onclick = () => { S.cursorPos = 1000; $("cursor").value = 1000; render(); loadTable(); replay.refreshList(); };
addEventListener("resize", () => debounce("rs", () => { S.plots.forEach(p => p.sig = ""); render(); drawCompare(); }, 150));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { S.plots.forEach(p => p.sig = ""); render(); drawCompare(); });

// ---------------------------------------------------------------- 비교 탭
const cmpCharts = [];
function buildCompareList() {
  const host = $("cmp_runs");
  const groups = {};
  for (const r of S.runs) (groups[r.group || r.id] ||= []).push(r);
  const sig = S.runs.map(r => r.id).join(",") + [...S.cmpSel].join(",");
  if (host._sig === sig) return; host._sig = sig;
  if (!S.cmpSel.size && S.sel) S.cmpSel.add(S.sel);
  const order = [...S.cmpSel];
  host.innerHTML = Object.entries(groups).map(([g, rs]) => `<div class="grp">${esc(g)} <span class="muted">(${rs.length})</span></div>` + rs.map(r => {
    const on = S.cmpSel.has(r.id), ci = order.indexOf(r.id);
    return `<label class="run"><input type="checkbox" data-id="${esc(r.id)}"${on ? " checked" : ""}><span class="sw" style="background:${on ? color(ci) : "transparent"}"></span>${esc(shortName(r))}<span class="muted small">${esc(r.kind)}</span></label>`;
  }).join("")).join("");
  host.querySelectorAll("input").forEach(i => i.onchange = () => { if (i.checked) S.cmpSel.add(i.dataset.id); else S.cmpSel.delete(i.dataset.id); host._sig = ""; buildCompareList(); loadCompare(); });
  loadCompare();
}
async function loadCompare() {
  const ids = [...S.cmpSel];
  await Promise.all(ids.filter(id => !S.cmpKeys[id]).map(async id => { const j = await api("/api/progress", { run: id }); S.cmpKeys[id] = (j.keys || []).map(k => k.k); }));
  const union = [...new Set(ids.flatMap(id => S.cmpKeys[id] || []))].sort();
  for (const [sel, pref] of [["cmp_a", ["rollout/success_rate", "eval/success_rate", "rollout/success_rate/pick"]], ["cmp_b", ["rollout/collision_rate", "train/loss", "train/approx_kl"]]]) {
    const cur = $(sel).value || pref.find(p => union.includes(p)) || union[0] || "";
    setHTML(sel, union.map(k => `<option${k === cur ? " selected" : ""}>${esc(k)}</option>`).join(""));
  }
  const keys = [$("cmp_a").value, $("cmp_b").value].filter(Boolean);
  const tableKeys = union.filter(k => /^(rollout\/success_rate|curriculum\/success_rate\/|eval\/success_rate|eval_teacher\/success_rate|rollout\/collision_rate$|rollout_teacher\/success_rate)/.test(k) && !/by_/.test(k));
  const need = [...new Set([...keys, ...tableKeys, "time/time_elapsed"])];
  await Promise.all(ids.map(async id => { S.cmpData[id] = await api("/api/progress", { run: id, keys: need.join(","), max_points: 1500 }); }));
  S.cmpTableKeys = tableKeys;
  drawCompare();
}
["cmp_a", "cmp_b"].forEach(id => $(id).onchange = loadCompare);
["cmp_x", "cmp_group"].forEach(id => $(id).onchange = drawCompare);
$("cmp_ema").oninput = drawCompare;

function cmpX(d, mode) {
  if (mode === "ts") return d.x.ts;
  if (mode === "wall") return d.cols["time/time_elapsed"] || d.x.time_elapsed;
  return d.x[{iter:'iterations',env_steps:'total_timesteps'}[mode] || mode];
}
function drawCompare() {
  if (S.tab !== "compare") return;
  const ids = [...S.cmpSel].filter(id => S.cmpData[id] && S.cmpData[id].total);
  const keys = [$("cmp_a").value, $("cmp_b").value].filter(Boolean);
  const host = $("cmp_charts");
  while (cmpCharts.length < keys.length) {
    const el = document.createElement("div"); el.className = "chart";
    el.innerHTML = `<div class="ct"><span></span></div><canvas></canvas><div class="lg"></div><div class="cn"></div>`;
    host.appendChild(el); cmpCharts.push({ el, p: new Plot(el.querySelector("canvas")) });
  }
  const mode = $("cmp_x").value, a = +$("cmp_ema").value, grp = $("cmp_group").checked;
  const order = [...S.cmpSel];
  keys.forEach((k, ki) => {
    const C = cmpCharts[ki]; C.el.hidden = false;
    setText(C.el.querySelector(".ct span"), k);
    const pct = /success|_rate|fraction|timeout|collision/.test(k);
    const series = [], missing = [];
    const byGroup = {};
    for (const id of ids) {
      const d = S.cmpData[id], y = d.cols[k];
      if (!y || !y.some(v => v != null)) { missing.push(id); continue; }
      const g = (S.byId[id] && S.byId[id].group) || id;
      (byGroup[g] ||= []).push(id);
    }
    const gi = Object.keys(byGroup);
    for (const [g, members] of Object.entries(byGroup)) {
      const dash = gi.indexOf(g) % 2 === 1;
      if (grp && members.length > 1) {
        // 평균 ± 표준편차 띠: 겹치는 x 구간을 200 칸으로 나눠 선형 보간
        const curves = members.map(id => { const d = S.cmpData[id]; const x = cmpX(d, mode), y = ema(d.cols[k], a); const pts = []; for (let i = 0; i < x.length; i++) if (y[i] != null && x[i] != null) pts.push([x[i], y[i]]); return pts; }).filter(p => p.length > 1);
        const lo = Math.max(...curves.map(p => p[0][0])), hi = Math.min(...curves.map(p => p[p.length - 1][0]));
        if (curves.length > 1 && hi > lo) {
          const X = [], M = [], L = [], H = [];
          for (let j = 0; j <= 200; j++) {
            const xv = lo + (hi - lo) * j / 200;
            const vs = curves.map(p => { let i = p.findIndex(q => q[0] >= xv); if (i <= 0) return p[Math.max(0, i)][1]; const [x0, y0] = p[i - 1], [x1, y1] = p[i]; return y0 + (y1 - y0) * (xv - x0) / (x1 - x0 || 1); });
            const m = vs.reduce((s, v) => s + v, 0) / vs.length, sd = Math.sqrt(vs.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, vs.length - 1));
            X.push(xv); M.push(m); L.push(pct ? Math.max(0, m - sd) : m - sd); H.push(pct ? Math.min(1, m + sd) : m + sd);
          }
          series.push({ name: `${g} mean ± std (${curves.length} seeds)`, x: X, y: M, lo: L, hi: H, color: color(order.indexOf(members[0])), dash });
          continue;
        }
      }
      for (const id of members) {
        const d = S.cmpData[id];
        series.push({ name: shortName(S.byId[id] || { id }), x: cmpX(d, mode), y: ema(d.cols[k], a), color: color(order.indexOf(id)), dash });
      }
    }
    C.p.draw({ series, pct, xlabel: mode, xfmt: mode === "ts" ? (v => new Date(v * 1000).toLocaleTimeString()) : undefined, empty: "selected runs do not log this key" });
    setHTML(C.el.querySelector(".lg"), series.map(s => `<span class="${s.dash ? "dash" : ""}" style="--c:${s.color}">${esc(s.name)}</span>`).join(""));
    setText(C.el.querySelector(".cn"), missing.length ? `not logged in: ${missing.map(id => shortName(S.byId[id] || { id })).join(", ")}` : "");
  });
  for (let i = keys.length; i < cmpCharts.length; i++) cmpCharts[i].el.hidden = true;
  // 비교 표
  const tk = S.cmpTableKeys || [];
  let h = `<table class="t"><tr><th>run</th><th>group</th><th>total_timesteps</th>${tk.map(k => `<th title="${esc(k)}">${esc(k)}</th>`).join("")}</tr>`;
  for (const id of [...S.cmpSel]) {
    const d = S.cmpData[id]; if (!d) continue;
    const n = d.total || 0, a0 = Math.floor(n * 0.9);
    h += `<tr><td class="l">${esc(shortName(S.byId[id] || { id }))}</td><td class="l">${esc((S.byId[id] || {}).group || "")}</td><td>${fmt(d.x && d.x.total_timesteps ? d.x.total_timesteps[d.x.total_timesteps.length - 1] : null)}</td>`;
    for (const k of tk) {
      const c = d.cols && d.cols[k];
      if (!c) { h += `<td class="none">not logged</td>`; continue; }
      let s = 0, m = 0; for (let i = Math.min(a0, c.length - 1); i < c.length; i++) if (c[i] != null) { s += c[i]; m++; }
      h += `<td>${m ? fmt(s / m, 1) : "—"}</td>`;
    }
    h += "</tr>";
  }
  setHTML("cmp_table", h + "</table>");
}

// ---------------------------------------------------------------- 시작
const h0 = new URLSearchParams(location.hash.slice(1));
loadRuns().then(() => { if (h0.get("tab")) setTab(h0.get("tab")); });
window.__trainview = S;

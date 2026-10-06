// 재생 탭 "Inputs" 패널: 정책이 그 스텝에 받은 입력(record_bc beh_rec.h 의 "inputs" 섹션)을 재생 시각에 맞춰 보인다.
// 묶음: 과제·목표 칸(VLA_INPUT 2.1) / 몸 상태(G1 obs 80) / 지도 물체 칸(3절, 지도 토큰 그대로 m·rad) / 벽·방·안 본 곳 방향·경유 지점(4절) /
// 행동 대 교사 라벨(DAgger) / 특권 상태(교사만) / 과제 조건(특권·환경 — 학생은 못 봄) / X0(신경망이 받은 정규화 값 그대로).
import { esc } from "./charts.js";
const nf = (v, d = 2) => v == null || !isFinite(v) ? "—" : (+v).toFixed(d);
import { parseTrp } from "./replay.js";

const f16 = h => { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * m * 5.960464477539063e-8 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * Math.pow(2, e - 15); };
const bf16 = (() => { const b = new ArrayBuffer(4), u = new Uint32Array(b), f = new Float32Array(b); return h => { u[0] = h << 16; return f[0]; }; })();
// 지도 토큰 v3 앞 1360 B(map_tok.h MapTok)
const TOK = { slot: 0, name_id: 1056, app_id: 1088, wall: 1120, room: 1232, comp: 1252, n_slot: 1260, flags: 1262, front: 1264, way: 1280, instr1: 1288, bkind: 1290, goal: 1296 };
const APP = ["cup", "item", "chair", "table", "cabinet", "bin", "ghost"];
const STATE4 = ["seen", "gone", "moved", "held"];
const SL_MODE = ["none", "fail", "nav", "approach0", "rotate", "arm_ready", "drive", "fine", "wait", "arm_grasp", "close", "reopen", "lift", "fold", "back", "arm_place", "open", "retreat", "done", "explore", "fail_arm", "backup"];
const ACT = ["vx", "wz", "q1", "q2", "q3", "q4", "q5", "grip"];
const ROOMT = ["kitchen", "bathroom", "bedroom", "living", "office", "unknown"];
const OBS_GROUPS = [["arm q (5) + gripper", 0, 6], ["arm qd", 6, 12], ["joint xyz (5×3)", 12, 27], ["fingertip xyz", 27, 30], ["fingertip R 6D", 30, 36], ["fingertip v·w", 36, 42],
  ["base v, 0, w", 42, 45], ["previous action", 45, 53], ["fingertip→target (student: 0)", 53, 56], ["wall rays 16 (/4 m)", 56, 72], ["target block (student: 0)", 72, 80]];

export class InputsPanel {
  constructor(host) { this.host = host; this.tr = null; this.recs = null; this.last = -1; this.cond = ""; }
  clear(msg) { this.tr = null; this.recs = null; this.last = -1; this.host.innerHTML = msg ? `<div class="muted small">${esc(msg)}</div>` : ""; }
  // buf = .trp 바이트(.trp 판 그대로, .sg 판은 안의 episode.trp)
  load(buf) {
    let tr;
    try { tr = parseTrp(buf); } catch (e) { this.clear("inputs: " + e.message); return null; }
    this.tr = tr;
    const H = tr.head, I = H.inputs || {};
    this.lay = I.layout || null;
    this.recs = (tr.extra.inputs || []).slice().sort((a, b) => a.frame - b.frame);
    this.names = I.names || {};
    this.cond = condHtml(H.scene || {}, I);
    if (!this.lay || !this.recs.length) { this.host.innerHTML = `<div class="muted small">This episode has no recorded policy inputs (recorded before the inputs record — re-record the checkpoint).</div><details class="card in_cond" open><summary>Task conditions</summary>${this.cond}</details>`; return tr; }
    this.last = -1;
    this.show(0);
    return tr;
  }
  frameAt(t) { const dt = this.tr ? this.tr.head.dt || 0.1 : 0.1; return Math.max(0, Math.round(t / dt)); }
  show(f) {
    if (!this.recs || !this.recs.length) return;
    let k = 0; for (let i = 0; i < this.recs.length; i++) { if (this.recs[i].frame <= f) k = i; else break; }
    if (k === this.last) return;
    this.last = k;
    const keepOpen = {}; this.host.querySelectorAll("details[data-k]").forEach(d => keepOpen[d.dataset.k] = d.open);
    const sc = this.host.scrollTop;
    this.host.innerHTML = this.render(this.recs[k]);
    this.host.querySelectorAll("details[data-k]").forEach(d => { if (d.dataset.k in keepOpen) d.open = keepOpen[d.dataset.k]; });
    this.host.scrollTop = sc;
    const more = this.host.querySelector("[data-more]"); if (more) more.onclick = () => { this.showAll = !this.showAll; this.last = -1; this.show(f); };
  }
  // 시간 막대용: 교사 단계(priv[5]) 묶음 구간 + 사건(목표 처음 봄·잡힘·놓음·충돌·성공)
  timeline() {
    if (!this.recs || !this.recs.length || !this.lay || !this.tr) return null;
    const L = this.lay, dt = this.tr.head.dt || 0.1, tr = this.tr;
    const G = m => ["nav", "explore", "none", "backup"].includes(SL_MODE[m]) ? "nav" : ["approach0", "rotate", "arm_ready", "drive", "fine", "wait"].includes(SL_MODE[m]) ? "approach"
      : ["arm_grasp", "close", "reopen"].includes(SL_MODE[m]) ? "grasp" : ["lift", "fold", "back"].includes(SL_MODE[m]) ? "carry" : ["arm_place", "open", "retreat"].includes(SL_MODE[m]) ? "place"
      : SL_MODE[m] === "done" ? "done" : "fail";
    const phases = [], events = [];
    let held = false, wasHeld = false;
    for (const rec of this.recs) {
      const dv = new DataView(rec.bytes.buffer, rec.bytes.byteOffset, rec.bytes.byteLength), t = rec.frame * dt;
      const pv = i => dv.getFloat32(L.priv + 4 * i, true), g = G(pv(5) | 0);
      const last = phases[phases.length - 1];
      if (last && last.g === g) last.t1 = t + dt; else phases.push({ g, t0: t, t1: t + dt });
      held = (pv(4) | 0) === 2;
      if (held && !wasHeld) events.push({ t, name: "grasp" });
      if (!held && wasHeld) events.push({ t, name: "release" });
      wasHeld = held || wasHeld && false;
    }
    const nf = tr.nf;
    let seen = false, col = false, ok = false;
    for (let f = 0; f < nf; f++) {
      if (!seen && tr.ci.goal_known != null && tr.v(f, "goal_known") > 0.5) { seen = true; events.push({ t: f * dt, name: "target seen" }); }
      const ev = tr.ci.ev != null ? tr.v(f, "ev") : 0;
      if (!col && (ev & 1)) { col = true; events.push({ t: f * dt, name: "collision" }); }
      if (!ok && (ev & 8)) { ok = true; events.push({ t: f * dt, name: "success" }); }
    }
    return { phases, events };
  }
  render(rec) {
    const L = this.lay, b = rec.bytes, dv = new DataView(b.buffer, b.byteOffset, b.byteLength), H = this.tr.head, I = H.inputs || {};
    const u16 = o => dv.getUint16(o, true), i16 = o => dv.getInt16(o, true), f32 = o => dv.getFloat32(o, true);
    const th = (o, n) => Array.from({ length: n }, (_, i) => f16(u16(o + 2 * i)));
    const priv = Array.from({ length: L.n_priv }, (_, i) => f32(L.priv + 4 * i));
    const act = k => Array.from({ length: 8 }, (_, i) => f32(L[k] + 4 * i));
    const obs = Array.from({ length: 80 }, (_, i) => f32(L.obs + 4 * i));
    const W = H.scene && H.scene.world || {};
    const nm = r => r < 0 ? "" : this.names[r] ?? `row ${r}`;
    const actor = I.actor || "?", deg = r => nf(r * 57.2958, 0) + "°";
    const card = (k, title, sub, body, open = true) => `<details class="card" ${open ? "open" : ""} data-k="${k}"><summary>${title}${sub ? ` <span class="muted">${sub}</span>` : ""}</summary>${body}</details>`;
    let h = `<div class="in_head"><b>Policy input</b> · step ${rec.frame} · <span class="pill2 ${actor}">${actor === "student" ? "student" : "teacher (scripted)"}</span></div>`;
    // 1 Goal
    const instr = u16(L.tok + TOK.instr1);
    const goal = e => th(L.tok + TOK.goal + 32 * e, 16);
    const gRow = (lab, g) => g[0] > 0.5 ? `<tr><td><b>${lab}</b></td><td>${g[1] > 0.5 ? "object" : g[2] > 0.5 ? "point" : "yes"}${g[3] > 0.5 ? "" : " (unknown)"}${g[4] > 0.5 ? " · lost" : ""}</td><td class="mono">${nf(g[8], 2)} m</td><td class="mono">${nf(Math.atan2(g[9], g[10]) * 57.3, 0)}°</td></tr>` : `<tr><td><b>${lab}</b></td><td colspan="3" class="muted">not given</td></tr>`;
    h += card("goal", "Goal", "(slots, m · base_link)", `<div class="in_instr">${esc(I.instr_text || W.instr_text || (instr ? "row " + (instr - 1) : "—"))}</div>
      <div class="small">pick <b>${esc(W.pick ? W.pick.name || W.pick.cat : "?")}</b> → place <b>${esc(W.place ? W.place.name || W.place.cat : "?")}</b> · stage B${u16(L.tok + TOK.bkind) || (W.kind ?? "?")}</div>
      <table class="in_t"><tr><th></th><th>given</th><th>distance</th><th>bearing</th></tr>${gRow("PICK", goal(0))}${gRow("PLACE", goal(1))}</table>`);
    // 2 Body
    const q = obs.slice(0, 6);
    h += card("body", "Body", "(arm q, gripper, base)", `<table class="in_t"><tr><td>arm q1–q5 (rad)</td><td class="mono">${q.slice(0, 5).map(x => nf(x, 2)).join("  ")}</td></tr>
      <tr><td>gripper</td><td class="mono">${nf(q[5], 2)}</td></tr><tr><td>base v (m/s) · w (rad/s)</td><td class="mono">${nf(obs[42], 2)} · ${nf(obs[44], 2)}</td></tr>
      <tr><td>fingertip x y z (m)</td><td class="mono">${obs.slice(27, 30).map(x => nf(x, 2)).join("  ")}</td></tr></table>
      <details class="raw"><summary class="muted">all 80 obs values</summary><table class="in_t">` + OBS_GROUPS.map(([n, a, z]) => `<tr><td>${esc(n)}</td><td class="mono">${obs.slice(a, z).map(x => nf(x, 2)).join(" ")}</td></tr>`).join("") + `</table></details>`);
    // 3 Map slots: target first, then by distance; top N, "show all"
    const ns = i16(L.tok + TOK.n_slot), rows = [];
    for (let s = 0; s < 16; s++) {
      const id = i16(L.tok + TOK.name_id + 2 * s); if (id < 0) continue;
      const v = th(L.tok + TOK.slot + 66 * s, 33);
      rows.push({ s, name: nm(id), x: v[0], y: v[1], dist: v[6], st: [20, 21, 22, 23].reduce((a, k, i) => v[k] > 0.5 ? STATE4[i] : a, "?"), tgt: v[30] > 0.5, reach: v[12] });
    }
    rows.sort((a, b) => (b.tgt - a.tgt) || (a.dist - b.dist));
    const TOPN = 6, all = this.showAll, vis = all ? rows : rows.slice(0, TOPN);
    h += card("slots", "Map slots", `(${ns} of 16 filled · m)`, `<table class="in_t"><tr><th>name</th><th>x, y</th><th>dist</th><th>state</th></tr>` +
      (vis.map(r => `<tr class="${r.tgt ? "in_tgt" : ""}"><td>${r.tgt ? "◎ " : ""}<b>${esc(r.name)}</b></td><td class="mono">${nf(r.x, 2)}, ${nf(r.y, 2)}</td><td class="mono">${nf(r.dist, 2)}</td><td>${r.st}</td></tr>`).join("") || '<tr><td colspan="4" class="muted">no objects in the map token</td></tr>') + `</table>` +
      (rows.length > TOPN ? `<button class="in_more" data-more="1">${all ? "show top " + TOPN : "show all " + rows.length}</button>` : ""));
    // 4 Sectors: compass of unseen-area sectors (8 × 45°, 0 = ahead, counter-clockwise) + wall rays 16
    const wall = th(L.tok + TOK.wall, 56), room = th(L.tok + TOK.room, 10), comp = th(L.tok + TOK.comp, 4), front = th(L.tok + TOK.front, 8), way = th(L.tok + TOK.way, 4);
    const rt = room.slice(0, 6).reduce((a, x, i) => x > room[a] ? i : a, 0);
    const wedge = (i, v, R, r0, col) => { const a0 = (-90 - (i + 0.5) * 45) * Math.PI / 180, a1 = (-90 - (i - 0.5) * 45) * Math.PI / 180, p = (a, r) => `${(60 + r * Math.cos(a)).toFixed(1)},${(60 + r * Math.sin(a)).toFixed(1)}`;
      return `<path d="M${p(a0, r0)} L${p(a0, R)} A${R},${R} 0 0 1 ${p(a1, R)} L${p(a1, r0)} A${r0},${r0} 0 0 0 ${p(a0, r0)}Z" fill="${col}" fill-opacity="${(0.12 + 0.88 * Math.max(0, Math.min(1, v))).toFixed(2)}" stroke="var(--line)" stroke-width="0.5"><title>${(i * 45)}° ${nf(v, 2)}</title></path>`; };
    const rays = wall.slice(0, 16), rw = (i, v) => { const a = (-90 - i * 22.5) * Math.PI / 180, r = 14 + 14 * Math.min(1, v / 4); return `<line x1="${60 + 14 * Math.cos(a)}" y1="${60 + 14 * Math.sin(a)}" x2="${60 + r * Math.cos(a)}" y2="${60 + r * Math.sin(a)}" stroke="var(--ink2)" stroke-width="1.5"><title>${nf(v, 2)} m</title></line>`; };
    h += card("env", "Sectors", "(unseen area · walls)", `<div class="in_comp"><svg viewBox="0 0 120 120" width="120" height="120">${front.map((v, i) => wedge(i, v, 58, 32, "#eda100")).join("")}${rays.map((v, i) => rw(i, v)).join("")}
      <polygon points="60,50 55,64 65,64" fill="var(--accent)"><title>robot front</title></polygon><text x="60" y="8" text-anchor="middle" font-size="8" fill="var(--muted)">front</text></svg>
      <div class="small"><div>orange = unseen-area strength per 45° sector</div><div>grey spokes = wall distance (16 rays, 0–4 m)</div><div>room <b>${ROOMT[rt]}</b> ${nf(room[rt], 2)}${room[9] > 0.5 ? ` · door ${nf(room[8], 1)} m` : ""}</div>
      <div>map complete <span class="mono">${comp.map(x => nf(x, 2)).join(" ")}</span></div><div>waypoint ${way[3] > 0.5 ? `${nf(way[2], 1)} m ahead` : "—"}</div></div></div>`);
    // 5 Action vs teacher label
    const a = act("act"), lb = act("label"), ex = act("exec");
    const gap = a.reduce((s, x, i) => s + (isFinite(lb[i]) ? (x - lb[i]) ** 2 : 0), 0);
    const bar = (x, col) => { const w = Math.min(50, Math.abs(x) * 50); return `<span class="in_bar" style="${x >= 0 ? "left:50%" : "left:" + (50 - w) + "%"};width:${w}%;background:${col}"></span>`; };
    const lab5 = { vx: "forward v", wz: "turn w", q1: "joint 1", q2: "joint 2", q3: "joint 3", q4: "joint 4", q5: "joint 5", grip: "gripper" };
    h += card("act", "Action vs teacher", `<span class="${gap > 0.5 ? "in_no" : ""}">gap Σ(μ−label)² = ${nf(gap, 3)}</span>`, `<table class="in_t in_act"><tr><th></th><th>${actor === "student" ? "μ" : "teacher"}</th><th>label</th><th style="width:44%"><span style="color:#2a78d6">■</span> action <span style="color:#9a9a9a">■</span> teacher label</th></tr>` +
      ACT.map((n, i) => { const big = isFinite(lb[i]) && Math.abs(a[i] - lb[i]) > 0.25; return `<tr class="${big ? "in_gap" : ""}"><td>${lab5[n]}</td><td class="mono">${nf(a[i], 2)}</td><td class="mono">${nf(lb[i], 2)}</td><td><div class="in_bars">${bar(lb[i], "#9a9a9a")}${bar(a[i], "#2a78d6")}</div></td></tr>`; }).join("") + `</table><div class="small muted">row tinted red = |μ − label| &gt; 0.25 (the DAgger loss pulls these together)</div>`);
    // 6 Privileged + task conditions (collapsed): student can't see
    const mode = priv[5] | 0;
    h += `<details class="card in_cond" data-k="priv"><summary>Privileged &amp; task conditions <span class="muted">— student can't see</span></summary><table class="in_t">
      <tr><td>object (window)</td><td class="mono">${nf(priv[0], 2)}, ${nf(priv[1], 2)}, ${nf(priv[2], 2)} yaw ${nf(priv[3] * 57.3, 0)}° · ${["none", "rest", "held"][priv[4] | 0] ?? priv[4]}</td></tr>
      <tr><td>teacher phase</td><td><b>${esc(SL_MODE[mode] ?? String(mode))}</b></td></tr>
      <tr><td>planned stance</td><td class="mono">${priv[10] > 0.5 ? `${nf(priv[6], 2)}, ${nf(priv[7], 2)} yaw ${nf(priv[8] * 57.3, 0)}° back ${nf(priv[9], 2)}${priv[11] > 0.5 ? "" : " (no plan)"}` : "—"}</td></tr>
      <tr><td>place point</td><td class="mono">${nf(priv[12], 2)}, ${nf(priv[13], 2)}, ${nf(priv[14], 2)} (goal mode ${priv[15] | 0})</td></tr>
      <tr><td>held offset a n b</td><td class="mono">${nf(priv[17], 3)} ${nf(priv[18], 3)} ${nf(priv[19], 3)} · drops ${priv[20] | 0}</td></tr></table>${this.cond}
      <details class="raw"><summary class="muted">network input X0 (normalized)</summary>${(() => { const x0 = Array.from({ length: L.n_x0 }, (_, i) => bf16(u16(L.x0 + 2 * i)));
        return `<div class="mono small in_x0">` + [["G1 obs 80", 0, 80], ["wall 56", 80, 136], ["room 10", 136, 146], ["completeness 4", 146, 150], ["bias", 150, 151], ["unseen rays 8", 151, 159], ["waypoint 4", 159, 163], ["skill 3", 163, 166], ["(0)", 166, 176], ["goal 2×16", 176, 208]]
          .map(([n, a0, z]) => `<div><b>${n}</b> ${x0.slice(a0, z).map(x => nf(x, 2)).join(" ")}</div>`).join("") + `</div>`; })()}</details>
      <details class="raw"><summary class="muted">wall segments · executed action</summary><div class="mono small">near wall segments 8×5: ${wall.slice(16).map(x => nf(x, 1)).join(" ")}<br>executed: ${ex.map(x => nf(x, 2)).join(" ")}</div></details></details>`;
    return h;
  }
}

// 과제 조건(특권·환경): 잡기 모형·질량·면 높이·잡기 가능 표·서는 자리
function condHtml(SC, I) {
  const P = SC.pnp; if (!P) return "";
  const ok = b => b ? '<span class="in_ok">✓</span>' : '<span class="in_no">✗</span>';
  const g = P.grasp || {}, m = P.mass || {}, s = P.surface || {}, fe = P.feas || {}, st = P.stance || {};
  const ch = I.chosen_stance || [];
  return `<div class="small muted" style="margin-top:6px">Task conditions</div><table class="in_t">
    <tr><td>gripper aperture</td><td>${ok(g.width_ok)} object width <b>${nf(g.obj_width, 3)} m</b> ≤ max ${nf(g.aperture_max, 2)} m (depth ${nf(g.obj_depth, 3)})</td></tr>
    <tr><td>object height</td><td>${ok(g.height_ok)} ${nf(g.obj_height, 3)} m ≥ ${nf(g.min_height, 4)} m</td></tr>
    <tr><td>mass vs payload</td><td>${ok(m.ok)} <b>${nf(m.kg, 3)} kg</b>${m.known ? "" : " (unknown → assumed)"} ≤ ${nf(m.payload_max, 2)} kg (grip side ${m.grip_side}, top ${m.grip_top}; arm ${m.arm_near} kg at r ≤ ${m.arm_r_near} m → ${m.arm_far} kg at ${m.arm_r_far} m)</td></tr>
    <tr><td>source surface</td><td>${ok(s.src_ok)} top ${nf(s.src_top, 2)} m (pick height limit ${s.pick_z ? s.pick_z.join(" / ") : "?"})</td></tr>
    <tr><td>place surface</td><td>${ok(s.dst_ok)} top ${nf(s.dst_top, 2)} m (limit ${s.place_top ? s.place_top.join(" / ") : "?"}); reach band: low surface ${nf(s.reach_low, 2)} m, high ${nf(s.reach_high, 2)} m + edge ${nf(s.edge_dist, 2)}</td></tr>
    <tr><td>static grasp model</td><td>${esc(P.static)}</td></tr>
    <tr><td>feasibility table</td><td>B4 ${ok(fe.B4)} ${esc(fe.grasp_why || "")} · B5 ${ok(fe.B5)} ${esc(fe.place5_why || "")} · B6 ${ok(fe.B6)} ${esc(fe.place6_why || "")}</td></tr>
    <tr><td>stance candidates</td><td>${st.n_cands ?? 0} (pnp_stance_cands) · table stance gst4 ${(st.gst4 || []).slice(0, 2).map(x => nf(x, 2)).join(", ")}${ch.length && isFinite(ch[0]) ? ` · <b>teacher chose</b> ${nf(ch[0], 2)}, ${nf(ch[1], 2)} yaw ${nf(ch[2] * 57.3, 0)}°` : ""}</td></tr>
  </table>`;
}

// 3D 덧그림(sgview iframe 의 THREE 또는 3D classic 의 THREE): 집을 물체·놓을 곳 표시, 서는 자리 후보(LIMO 발자국 + 화살표, 교사가 고른 것 굵게),
// 팔 닿는 띠(목표 둘레 r 0.30·0.38 m 고리 — 가반 하중 보통·뻗음), 잡기 가능 색칠(창 안 집을 후보: 초록 = 잡기 모형 됨, 빨강 = 안 됨 + 까닭).
// off = 창 → 그리는 틀(OG 판은 창 가운데 wx·wy, .trp 판은 0). 돌려줌 { cond, env, feas } 묶음(켜고 끔)
export function buildOverlays(T, SC, I, off, label) {
  const ox = off[0] || 0, oy = off[1] || 0, out = { cond: new T.Group(), env: new T.Group(), feas: new T.Group(), goal: new T.Group(), hover: [] };
  const P = SC.pnp || {}, st = P.stance || {}, boxes = SC.boxes || [];
  const tgt = boxes.find(b => b.kind === "target"), place = boxes.find(b => b.kind === "place");
  const foot = (x, y, yaw, col, op, w) => {
    const g = new T.Group(); g.position.set(x + ox, y + oy, 0.02); g.rotation.z = yaw;
    const r = new T.Mesh(new T.PlaneGeometry(0.32, 0.22), new T.MeshBasicMaterial({ color: col, transparent: true, opacity: op * 0.35, depthWrite: false, side: T.DoubleSide }));
    const e = new T.LineSegments(new T.EdgesGeometry(new T.PlaneGeometry(0.32, 0.22)), new T.LineBasicMaterial({ color: col, transparent: true, opacity: op }));
    const a = new T.ArrowHelper(new T.Vector3(1, 0, 0), new T.Vector3(0, 0, 0.01), 0.28, col, 0.07, 0.05);
    g.add(r, e, a); g.renderOrder = 6; return g;
  };
  // 목표 표시(과제 지시 그대로 — 정책이 받은 목표): "go here" 면 지도 위 점, 아니면 목표 물체. 기둥 + 이름표
  {
    const pole = (x, y, z0, col, text) => {
      const c = new T.Mesh(new T.CylinderGeometry(0.035, 0.035, 1.2, 16), new T.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.9 }));
      c.rotation.x = Math.PI / 2; c.position.set(x + ox, y + oy, z0 + 0.6); c.renderOrder = 7; out.goal.add(c);
      const d = new T.Mesh(new T.RingGeometry(0.18, 0.26, 48), new T.MeshBasicMaterial({ color: col, side: T.DoubleSide, transparent: true, opacity: 0.8, depthWrite: false }));
      d.position.set(x + ox, y + oy, Math.max(0.02, z0)); d.renderOrder = 7; out.goal.add(d);
      if (label) out.goal.add(label(text, x + ox, y + oy, z0 + 1.4, 0.3, "rgba(255,255,255,0.92)"));
    };
    const it = ((SC.world && SC.world.instr_text) || "").toLowerCase();
    const b1 = (SC.world || {}).kind === 1 && P.goal_pt;   // B1 지점 가기: 목표 = 바닥 점(goal_pt) — 지시문 문장과 상관없이
    if (b1 || it.startsWith("go here")) {   // 지점으로 가기: 목표 = 지도 위 점(B1 은 goal_pt, B3 는 놓을 점, 없으면 목표 상자 가운데)
      const g = b1 || P.place_pt || (tgt && tgt.c);
      if (g) pole(g[0], g[1], 0, 0x0ca30c, "목표 지점");
    } else if (tgt) pole(tgt.c[0], tgt.c[1], Math.max(0, tgt.c[2] - tgt.h[2]), 0xe34948, `목표 물체: ${tgt.name}`);
  }
  for (const c of st.cands || []) out.cond.add(foot(c[0], c[1], c[2], 0x3987e5, 0.6));
  if (st.gst4 && (st.gst4[0] || st.gst4[1])) out.cond.add(foot(st.gst4[0], st.gst4[1], st.gst4[2], 0x4a3aa7, 0.8));
  const ch = I && I.chosen_stance;
  if (ch && isFinite(ch[0])) { const f = foot(ch[0], ch[1], ch[2], 0xe34948, 1); f.scale.set(1.05, 1.05, 1); out.cond.add(f); if (label) out.cond.add(label("teacher stance", ch[0] + ox, ch[1] + oy, 0.45, 0.09, "rgba(255,225,225,0.9)")); }
  if (tgt) {
    const [x, y, z] = tgt.c, m = P.mass || {};
    for (const [r, col] of [[m.arm_r_near || 0.3, 0x1baf7a], [m.arm_r_far || 0.38, 0xeda100]]) {
      const ring = new T.Mesh(new T.RingGeometry(r - 0.006, r + 0.006, 64), new T.MeshBasicMaterial({ color: col, side: T.DoubleSide, transparent: true, opacity: 0.85, depthWrite: false }));
      ring.position.set(x + ox, y + oy, Math.max(0.02, z - tgt.h[2])); ring.renderOrder = 6; out.cond.add(ring);
    }
    const s = new T.Mesh(new T.BoxGeometry(1, 1, 1), new T.MeshBasicMaterial({ color: 0xe34948, transparent: true, opacity: 0.85 }));
    s.scale.set(2 * tgt.h[0], 2 * tgt.h[1], 2 * tgt.h[2]); s.position.set(x + ox, y + oy, z); s.rotation.z = tgt.yaw || 0; out.cond.add(s);
    if (label) out.cond.add(label(`pick: ${tgt.name} (${(P.grasp && P.grasp.obj_width || 0).toFixed(3)} m, ${(P.mass && P.mass.kg || 0).toFixed(2)} kg)`, x + ox, y + oy, z + 0.3, 0.09, "rgba(255,225,225,0.92)"));
  }
  if (place) {
    const [x, y, z] = place.c;
    const s = new T.Mesh(new T.BoxGeometry(1, 1, 1), new T.MeshBasicMaterial({ color: 0x0ca30c, transparent: true, opacity: 0.25, depthWrite: false }));
    s.scale.set(2 * place.h[0], 2 * place.h[1], Math.max(0.01, 2 * place.h[2])); s.position.set(x + ox, y + oy, z); out.cond.add(s);
    const pp = P.place_pt;
    if (pp) { const d = new T.Mesh(new T.SphereGeometry(0.04, 16, 12), new T.MeshBasicMaterial({ color: 0x0ca30c })); d.position.set(pp[0] + ox, pp[1] + oy, pp[2] + 0.04); out.cond.add(d); }
    if (label) out.cond.add(label(place.name, x + ox, y + oy, z + place.h[2] + 0.25, 0.09, "rgba(220,245,220,0.92)"));
  }
  // 환경 상자(GPU 환경이 쓰는 근사): 벽·가구·창 테두리만
  const KC = { wall: 0x6b6b70, furniture: 0xb07a2a, window: 0x3a8fd0, door: 0x9c6b3c };
  const eg = new T.EdgesGeometry(new T.BoxGeometry(1, 1, 1));
  for (const b of boxes) {
    if (b.kind === "target" || b.kind === "place") continue;
    const l = new T.LineSegments(eg, new T.LineBasicMaterial({ color: KC[b.kind] || 0xb07a2a, transparent: true, opacity: b.kind === "door" ? 0.4 : 0.9 }));
    l.scale.set(2 * b.h[0], 2 * b.h[1], Math.max(0.01, 2 * b.h[2])); l.position.set(b.c[0] + ox, b.c[1] + oy, b.c[2]); l.rotation.z = b.yaw || 0;
    out.env.add(l);
    if (label && b.kind === "furniture" && b.h[0] * b.h[1] > 0.02) out.env.add(label(b.name, b.c[0] + ox, b.c[1] + oy, b.c[2] + b.h[2] + 0.1, 0.08, "rgba(255,240,215,0.85)"));
  }
  // 잡기 가능 색칠
  for (const p of SC.picks || []) {
    const m = new T.Mesh(new T.SphereGeometry(Math.max(0.035, p.w / 2), 14, 10), new T.MeshBasicMaterial({ color: p.ok ? 0x1baf7a : 0xd03b3b }));
    m.position.set(p.c[0] + ox, p.c[1] + oy, p.c[2]); m.userData.tip = `${p.name}: ${p.ok ? "graspable" : "not graspable — " + p.why} (w ${p.w.toFixed(3)} m, h ${p.h.toFixed(3)} m, ${p.kg.toFixed(2)} kg)`;
    out.feas.add(m); out.hover.push(m);
    if (label) out.feas.add(label(`${p.name}${p.ok ? "" : " ✗ " + p.why}`, p.c[0] + ox, p.c[1] + oy, p.c[2] + 0.18, 0.11, p.ok ? "rgba(215,245,230,0.9)" : "rgba(255,215,215,0.9)"));
  }
  return out;
}

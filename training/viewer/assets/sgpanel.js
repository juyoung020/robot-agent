// 재생 탭 sgview 화면: sgview 페이지(src/scene_graph/sgview/assets/index.html)를 고치지 않고 iframe 으로 띄운다.
// 서버(src/sg.rs)가 sgview 의 실시간 경로(/stream SSE 등)를 판 재생으로 흉내 내고, 이 파일은
//   ① 시간 막대·재생·속도(/api/sg/ctl), ② 같은 출처 iframe 안 장면에 덧그림: 참(GT) 궤적, BEHAVIOR 집 바탕 층(방 바닥 색·벽·가구·놓을 곳·집을 것·과제 물체),
//   ③ LIMO 관절 순서(robot.json 의 joint_order 가 비어 있어 sgview 가 관절 스트림을 안 씀 → proprio 순서를 넣음), ④ 카메라 그림 을 맡는다.
import { fmt, esc } from "./charts.js";
import { buildOverlays } from "./inputs.js";

const ROOM_COL = { kitchen: 0xeda100, bathroom: 0x1baf7a, bedroom: 0x9085e9, living_room: 0x3987e5, dining_room: 0xeb6834, corridor: 0x9a9a9a, childs_room: 0xe87ba4,
  private_office: 0x4a3aa7, storage_room: 0x8c6d46, utility_room: 0x6b6b70, garage: 0x555555, entryway: 0xb0b0b0, closet: 0x8c6d46, empty_room: 0xc8c8c8, playroom: 0xe87ba4,
  television_room: 0x3987e5, home_office: 0x4a3aa7, lobby: 0xb0b0b0, break_room: 0xeb6834, meeting_room: 0x4a3aa7, shared_office: 0x4a3aa7, room: 0x6da7ec };
const hashCol = s => { let h = 0; for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) >>> 0; const p = [0xc8a878, 0xb59a7a, 0xa9b48c, 0x9fb3c8, 0xc9a3a3, 0xb3a3c9, 0xa3c9c0, 0xd2b48c]; return p[h % p.length]; };

export class SgPanel {
  constructor({ api, host, getSel, onTime }) {
    this.api = api; this.host = host; this.onTime = onTime; this.trHead = null; this.ov = null; this.ovOn = { cond: false, env: false, feas: false };
    this.sess = (() => { try { const k = sessionStorage.getItem("tv_sgsess"); if (k) return k; } catch (e) {} const k = Math.random().toString(36).slice(2, 12); try { sessionStorage.setItem("tv_sgsess", k); } catch (e) {} return k; })();
    document.cookie = `sgsess=${this.sess}; path=/; SameSite=Lax`;
    this.state = null; this.info = null; this.timer = null; this.win = null; this.ul = null; this.gt = null;
    const $ = id => document.getElementById(id);
    this.$ = $;
  }
  async open(run, stream, id) {
    // 판을 고르면 처음부터 바로 재생(끝나면 startPoll 이 처음부터 다시 — 반복)
    const st = await (await fetch(`/api/sg/ctl?sess=${this.sess}&run=${encodeURIComponent(run)}&stream=${encodeURIComponent(stream)}&id=${encodeURIComponent(id)}&t=0&playing=1`)).json();
    this.userPaused = false;
    if (st.error) { this.$("rp_empty").textContent = st.error; this.$("rp_empty").hidden = false; return false; }
    this.state = st;
    this.info = (await (await fetch(`/api/sg/info?sess=${this.sess}`)).json()).info || {};
    this.$("rp_time").max = Math.round(st.duration * 10); this.$("rp_time").value = 0;
    // 같은 페이지를 다시 쓰면 sgview 가 상태를 이어 붙이므로 새로 띄움(옮김은 SSE reset 으로).
    // 사람이 맞춘 시점·켜고 끔은 같은 실행의 다른 판으로 옮겨도 이어 감(저장 → 새 iframe 에 되돌림)
    this.saveView(run);
    this.host.innerHTML = ""; this.$("rp_cams").innerHTML = "";
    const f = document.createElement("iframe");
    f.src = "/sg/?k=" + Math.random().toString(36).slice(2);
    f.style.cssText = "width:100%;height:100%;border:0;display:block";
    this.host.appendChild(f);
    this.frame = f; this.win = null; this.ul = null; this.gt = null; this.pm = null; this._pmT = null; this.ov = null; this.trHead = null; this.mesh = null; this.ulBoxes = [];
    f.onload = () => this.attach();
    this.run = run; this.stream = stream; this.id = id;
    this.startPoll();
    return true;
  }
  close() { this.saveView(this.run); clearInterval(this.timer); this.timer = null; this.host.innerHTML = ""; this.frame = null; this.win = null; }
  async ctl(q) {
    const st = await (await fetch(`/api/sg/ctl?sess=${this.sess}&${q}`)).json();
    if (!st.error) { this.state = st; this.update(); }
    return st;
  }
  seek(t) { return this.ctl(`t=${Math.max(0, t).toFixed(3)}`); }
  toggle() { this.userPaused = !!(this.state && this.state.playing); return this.ctl(`playing=${this.userPaused ? 0 : 1}`); }
  speed(v) { return this.ctl(`speed=${v}`); }
  startPoll() {
    clearInterval(this.timer);
    this.timer = setInterval(async () => {
      if (!this.state) return;
      if (this.state.playing) await this.ctl("");
      // 끝까지 갔고 사람이 멈춘 게 아니면 처음부터 다시(서버는 끝에서 playing=1 을 받으면 0 초로 되감음)
      const st = this.state;
      if (st && !st.playing && !this.userPaused && this.loop !== false && st.t >= st.duration - 1e-3) await this.ctl("playing=1");
    }, 200);
  }
  update() {
    const st = this.state, i = this.info || {}, m = i.meta || {};
    if (!st) return;
    this.$("rp_time").value = Math.round(st.t * 10);
    this.$("rp_tlabel").textContent = `${st.t.toFixed(1)} / ${st.duration.toFixed(1)} s`;
    this.$("rp_play").textContent = st.playing ? "❚❚" : "▶";
    const ul = i.underlay || {};
    this.$("rp_hud").textContent = `${m.skill || "?"} · ${m.task || ""} · ${ul.scene || m.home || ""}\n` +
      `${m.skill === "layout" ? "BEHAVIOR house layout (RASC v3) — no episode, robot at the task start pose" : i.has_policy_map ? "GPU env episode → real scenemap (" + (m.driver || "") + ")" + (this.pmOn ? " · orange = policy map (G2)" : "") : i.kind === "sg" ? "OmniGibson LIMO explore (real sim record → scenemap replay)" : "GPU env episode (G2 map, " + (m.driver || "") + ")"}\n` +
      `t ${st.t.toFixed(1)} s / ${st.duration.toFixed(1)} s` + (m.gt_cov != null ? `   final coverage ${fmt(m.gt_cov, 1)}` : "") + (m.path_m != null ? `   path ${fmt(m.path_m)} m` : "") +
      (m.skill === "layout" ? "" : `\nGT path green · SLAM path blue (sgview)`) + (this.$("rp_hud").dataset.pipe ? `\n${this.$("rp_hud").dataset.pipe}` : "");
    this.drawGt(st.t);
    if (this.pmOn) this.drawPolicyMap(st.t);
    if (this.onTime) this.onTime(st.t);
    // 카메라 그림 두 장(RecallVLA 입력 RGB 둘): LIMO 몸통(eyes) · 손목(wrist_eye) — 2 Hz 기록, 시간 막대와 같은 시각. 끄고 켜기·크기 조절(오른쪽 아래 모서리 끌기)
    this.updatePips(st.t);
  }
  pipOn(k) { const el = this.$("rp_cam_" + k); return !el || el.checked; }
  setPip(k, on) { this.pipShow = this.pipShow || {}; this.pipShow[k] = on; this.updatePips(this.state ? this.state.t : 0); }
  updatePips(t) {
    const i = this.info || {}, host = this.$("rp_cams");
    // 이름표는 판의 로봇에서(meta.robot) — LIMO 가 아닌 판(예: BEHAVIOR 사람 시연 r1pro)을 LIMO 로 적지 않게
    const rob = i.robot || "limo_omx", isLimo = /limo/i.test(rob);
    const bodyT = isLimo ? "LIMO RGB" : `${rob} head RGB (not LIMO)`, wristT = isLimo ? "wrist RGB" : `${rob} wrist RGB (not LIMO)`;
    for (const [k, title, list] of [["body", bodyT, i.cams || []], ["wrist", wristT, i.wcams || []]]) {
      let el = host.querySelector(`.pip[data-k="${k}"]`);
      if (!list.length || !this.pipOn(k)) { if (el) el.remove(); continue; }
      if (!el) {
        el = document.createElement("div"); el.className = "pip"; el.dataset.k = k; el.title = "click to enlarge"; el.onclick = () => el.classList.toggle("big");
        el.innerHTML = `<div class="pt">${title} <span class="pts"></span></div><img>`;
        host.appendChild(el);
      }
      let best = list[0]; for (const c of list) { if (c.t <= t) best = c; else break; }
      const url = `/api/sg/cam?run=${encodeURIComponent(this.run)}&stream=${encodeURIComponent(this.stream)}&id=${encodeURIComponent(this.id)}&file=${encodeURIComponent(best.file)}`;
      const img = el.querySelector("img");
      if (img.dataset.f !== best.file) { img.dataset.f = best.file; img.src = url; }
      el.querySelector(".pts").textContent = `t ${best.t.toFixed(1)} s`;
    }
  }

  // ---------------------------------------------------------------- iframe 안 덧그림
  attach() {
    const w = this.frame && this.frame.contentWindow;
    if (!w) return;
    const tryIt = () => {
      let ok = false;
      try { ok = !!w.THREE && w.eval("typeof scene !== 'undefined' && typeof robotModel !== 'undefined'"); } catch (e) { ok = false; }
      if (!ok) return setTimeout(tryIt, 150);
      this.win = w;
      const order = JSON.stringify(this.info.joint_order || []);
      // 관절: robot.json joint_order 가 비어 있어 sgview 가 관절 스트림을 안 씀 → LIMO proprio 순서를 넣고, 그리퍼 둘째 손가락 = −첫째
      w.eval(`(function(){ const ord = ${order};
        const setOrder = () => { if (!robotModel) return setTimeout(setOrder, 200); robotModel.order = ord; const c = document.getElementById("j_stream"); if (c) c.checked = true; applyJointStream(); };
        setOrder();
        const base = applyJointStream;
        applyJointStream = function () { base(); if (robotModel && jointsCur && robotModel.order.length) { const i = robotModel.order.indexOf("omx_gripper_joint_1"); if (i >= 0 && i < jointsCur.length) setJoints({ omx_gripper_joint_2: -jointsCur[i] }); } };
      })()`);
      // 반복 재생(끝 → 0 초: 서버가 epoch 를 올려 SSE reset)에서 sgview 는 autoFit 을 다시 켜 시점을 처음으로 돌린다 →
      // 사람이 한 번이라도 시점을 움직였으면 맞춤(fitViewTo·fitView)을 건너뛴다. "Fit"·"Top view" 단추는 그대로 됨(누르면 다시 따라감)
      w.eval(`(function(){ window.__tvKeep = false;
        controls.addEventListener("start", () => { window.__tvKeep = true; });
        const fvt = fitViewTo; fitViewTo = function (g, inst) { if (window.__tvKeep) return; return fvt(g, inst); };
        const fv = fitView; fitView = function (top) { if (window.__tvKeep && !window.__tvBtn) return; return fv(top); };
        for (const id of ["b_top", "b_fit"]) { const b = document.getElementById(id); if (!b) continue; const h = b.onclick; b.onclick = e => { window.__tvKeep = false; window.__tvBtn = true; try { h && h(e); } finally { window.__tvBtn = false; } }; }
      })()`);
      this.restoreView();
      this.buildGt(); this.buildUnderlay(); this.buildOv(); this.buildMesh(); this.update();
    };
    tryIt();
  }
  scene() { return this.win.eval("scene"); }
  // 시점(카메라 자리·바라보는 점·줌)과 sgview 패널의 켜고 끔·고르기 — 판을 바꿔도 같은 실행이면 이어 감
  // OG 판(kind sg)은 지도 틀 = 첫 자세가 원점이라 판이 달라도 "처음 자리 기준" 시점이 이어 감. .trp 판은 창 좌표라 같은 창끼리만
  frameKey(run, info) { return run + "|" + (info && info.kind) + "|" + (info && info.kind === "sg" ? "" : JSON.stringify((info && info.window_origin) || null)); }
  saveView(nextRun) {
    const w = this.win;
    if (!w) { if (this.saved && this.saved.run !== nextRun) this.saved = null; return; }
    try {
      const key = this.frameKey(this.run, this.info), old = this.saved || {};
      this.saved = { run: this.run, cams: Object.assign({}, old.run === this.run ? old.cams : {}) };
      if (w.eval("window.__tvKeep")) this.saved.cams[key] = { p: w.eval("camera.position.toArray()"), t: w.eval("controls.target.toArray()"), z: w.eval("camera.zoom") };
      const ui = {};
      w.document.querySelectorAll("input[id], select[id]").forEach(el => { ui[el.id] = el.type === "checkbox" ? el.checked : el.value; });
      this.saved.ui = ui;
    } catch (e) { this.saved = null; }
    if (this.saved && this.saved.run !== nextRun) this.saved = null;
  }
  restoreView() {
    const w = this.win, S = this.saved;
    if (!w || !S || S.run !== this.run) return;
    try {
      const C = (S.cams || {})[this.frameKey(this.run, this.info)];   // 같은 좌표 틀(OG 세계 / 창)의 판끼리만 시점을 이어 감
      if (C) w.eval(`camera.position.fromArray(${JSON.stringify(C.p)}); controls.target.fromArray(${JSON.stringify(C.t)}); camera.zoom = ${C.z}; camera.updateProjectionMatrix(); controls.update(); window.__tvKeep = true;`);
      else w.eval("window.__tvKeep = false;");
      for (const [id, v] of Object.entries(S.ui || {})) {
        const el = w.document.getElementById(id); if (!el || el.type === "range" || el.type === "text" || el.type === "number") continue;
        if (el.type === "checkbox") { if (el.checked !== v) { el.checked = v; el.dispatchEvent(new w.Event("change")); } }
        else if (el.tagName === "SELECT" && el.value !== v && [...el.options].some(o => o.value === v)) { el.value = v; el.dispatchEvent(new w.Event("change")); }
      }
      this.invalidate();
    } catch (e) {}
  }
  // 정책 기록 머리(.trp / .sg 안 episode.trp) — 집기·놓기 조건·환경 상자·잡기 가능 색칠 덧그림
  setTrp(head) { this.trHead = head; this.buildOv(); this.buildMesh(); }
  // BEHAVIOR 집 진짜 메시(줄인 것, /api/scene_mesh) — 바탕 층(house layout)에서 상자 대신. 없으면 상자 그대로(대신 씀)
  async buildMesh() {
    const W = (this.info && this.info.world) || (this.trHead && this.trHead.scene && this.trHead.scene.world);
    if (!this.win || !W || !W.scene || this.mesh || this._meshLoading) return;
    this._meshLoading = true;
    const win = this.win, r = await fetch(`/api/scene_mesh?scene=${encodeURIComponent(W.scene)}`);
    this._meshLoading = false;
    if (!r.ok || win !== this.win) return;
    const buf = await r.arrayBuffer(), dv = new DataView(buf), hl = dv.getUint32(4, true);
    const H = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)));
    // 배열은 iframe 쪽 생성자로(다른 창의 Float32Array 는 three 가 형을 못 알아봐 GL_INVALID_ENUM)
    const T = win.THREE, o0 = 8 + hl, V = new win.Float32Array(new Float32Array(buf, o0, H.n_v * 3)), F = new Uint32Array(buf, o0 + 12 * H.n_v, H.n_f * 3), I = new Uint16Array(buf, o0 + 12 * H.n_v + 12 * H.n_f, H.n_v);
    // 창 좌표 판(.trp)은 세계 − (wx, wy), OG 판은 세계 그대로
    const wo = this.info && this.info.window_origin ? [0, 0] : [-(W.wx || 0), -(W.wy || 0)];
    const KC = { wall: 0xbdb7ad, floor: 0xd9d0c1, door: 0x9c6b3c, window: 0x8fc5e8 };
    const mk = (want, op) => {
      const keep = new Uint8Array(H.n_v); let nf = 0;
      for (let f = 0; f < H.n_f; f++) { const k = H.objects[I[F[3 * f]]].kind; if (want(k)) nf++; }
      const idx = new win.Uint32Array(nf * 3); let j = 0;
      for (let f = 0; f < H.n_f; f++) { const k = H.objects[I[F[3 * f]]].kind; if (!want(k)) continue; idx[j++] = F[3 * f]; idx[j++] = F[3 * f + 1]; idx[j++] = F[3 * f + 2]; }
      const col = new win.Float32Array(H.n_v * 3), c = new T.Color();
      for (let v = 0; v < H.n_v; v++) { const o = H.objects[I[v]]; c.setHex(KC[o.kind] ?? hashCol(o.cat)); const d = o.kind === "furniture" ? 0.5 : 0.62; col.set([c.r * d, c.g * d, c.b * d], 3 * v); }   // sgview 빛이 세서 어둡게
      const g = new T.BufferGeometry(); g.setAttribute("position", new T.BufferAttribute(V, 3)); g.setAttribute("color", new T.BufferAttribute(col, 3)); g.setIndex(new T.BufferAttribute(idx, 1));
      g.computeVertexNormals();
      const m = new T.Mesh(g, new T.MeshLambertMaterial({ vertexColors: true, transparent: op < 1, opacity: op, side: T.DoubleSide, depthWrite: op >= 0.7 }));
      m.renderOrder = op < 1 ? 3 : 1; return m;
    };
    const G = new T.Group(); G.name = "house_mesh"; G.position.set(wo[0], wo[1], 0);
    G.add(mk(k => k === "floor", 0.5), mk(k => k === "wall" || k === "window" || k === "door", 0.3), mk(k => k === "furniture", 0.8));
    // 가구 이름(바닥 자국이 큰 것만)
    const bb = new Map();
    for (let v = 0; v < H.n_v; v++) { const k = I[v]; let b = bb.get(k); if (!b) bb.set(k, b = [1e9, 1e9, 1e9, -1e9, -1e9, -1e9]); for (let a = 0; a < 3; a++) { b[a] = Math.min(b[a], V[3 * v + a]); b[a + 3] = Math.max(b[a + 3], V[3 * v + a]); } }
    for (const [k, b] of bb) {
      const o = H.objects[k];
      if (o.kind !== "furniture" || (b[3] - b[0]) * (b[4] - b[1]) < 0.12 || b[2] > 2.0) continue;
      G.add(this.mkLabel(o.cat.replace(/_/g, " "), (b[0] + b[3]) / 2, (b[1] + b[4]) / 2, b[5] + 0.1, 0.1, "rgba(255,248,235,0.85)"));
    }
    this.mesh = G; this.meshInfo = H;
    if (this.ul) { this.ul.add(G); for (const m of this.ulBoxes || []) m.visible = this.meshOn === false; }
    else { const w = this.mfw(G); w.visible = this.ulOn === true; this.scene().add(w); }
    G.visible = this.meshOn !== false;
    this.invalidate();
  }
  setMesh(on) { this.meshOn = on; if (this.mesh) this.mesh.visible = on; for (const m of this.ulBoxes || []) m.visible = !on || !this.mesh; this.invalidate(); }
  setOv(k, on) { this.ovOn[k] = on; if (this.ov && this.ov[k]) { this.ov[k].visible = on; this.invalidate(); } }
  buildOv() {
    if (!this.win || !this.trHead || this.ov || !(this.trHead.scene && this.trHead.scene.boxes)) return;
    const T = this.win.THREE, off = (this.info && this.info.window_origin) || [0, 0];
    this.ov = buildOverlays(T, this.trHead.scene, this.trHead.inputs || {}, off, (t, x, y, z, h, bg) => this.mkLabel(t, x, y, z, h, bg));
    for (const k of ["cond", "env", "feas"]) { this.ov[k].visible = !!this.ovOn[k]; this.scene().add(this.mfw(this.ov[k])); }
    if (this.hoverables) this.hoverables.push(...this.ov.hover);
    this.invalidate();
  }
  // 세계 좌표 묶음 → 장면 틀(map = R·world + t, info.map_from_world)
  mfw(g) { const m = (this.info && this.info.map_from_world) || [1, 0, 0, 0]; const w = new this.win.THREE.Group(); w.rotation.z = Math.atan2(m[1], m[0]); w.position.set(m[2], m[3], 0); w.add(g); g.userData.wrap = w; return w; }
  mkLabel(text, x, y, z, h = 0.3, bg = "rgba(255,255,255,0.85)") {
    const T = this.win.THREE, cv = document.createElement("canvas"), g = cv.getContext("2d"), fs = 30;
    g.font = `600 ${fs}px system-ui, sans-serif`; cv.width = Math.ceil(g.measureText(text).width) + 16; cv.height = fs + 14;
    g.font = `600 ${fs}px system-ui, sans-serif`; g.fillStyle = bg; g.fillRect(0, 0, cv.width, cv.height); g.fillStyle = "#222"; g.fillText(text, 8, fs + 2);
    const sp = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(cv), depthTest: false, transparent: true }));
    sp.scale.set(h * cv.width / cv.height, h, 1); sp.position.set(x, y, z); sp.renderOrder = 11; return sp;
  }
  invalidate() { try { this.win.eval("invalidate()"); } catch (e) {} }
  buildGt() {
    const T = this.win.THREE, pts = this.info.gt_path || [];
    if (!pts.length) return;
    const a = new this.win.Float32Array(pts.length * 3);
    pts.forEach((p, i) => a.set([p[1], p[2], 0.08], 3 * i));
    const g = new T.BufferGeometry(); g.setAttribute("position", new T.BufferAttribute(a, 3)); g.setDrawRange(0, 0);
    this.gt = new T.Line(g, new T.LineBasicMaterial({ color: 0x0ca30c })); this.gt.frustumCulled = false; this.gt.renderOrder = 5;
    this.scene().add(this.gt);
  }
  drawGt(t) {
    if (!this.gt) return;
    const pts = this.info.gt_path || [];
    let n = 0; while (n < pts.length && pts[n][0] <= t) n++;
    this.gt.geometry.setDrawRange(0, Math.max(n, 1)); this.invalidate();
  }
  // 정책이 본 지도(G2 근사판) — 진짜 scenemap 지도 위 반투명 주황(점유 진하게, 빈칸 옅게, 모름 투명). 세계 좌표 → map_from_world
  setPolicyMap(on) { this.pmOn = on; if (this.pm) this.pm.visible = on; if (on && this.state) this.drawPolicyMap(this.state.t, true); this.invalidate(); }
  async drawPolicyMap(t, force) {
    if (!this.win || !this.info || !this.info.has_policy_map) return;
    const k = Math.round(t * 2) / 2;
    if (!force && this._pmT === k) return;
    this._pmT = k;
    const g = await (await fetch(`/api/sg/policymap?sess=${this.sess}&t=${k}`)).json();
    if (g.why) return;
    const T = this.win.THREE, px = Uint8Array.from(atob(g.b64), c => c.charCodeAt(0)), rgba = new this.win.Uint8Array(g.w * g.h * 4);
    for (let i = 0; i < g.w * g.h; i++) { const v = px[i]; if (v === 205) continue; const occ = 255 - v; rgba[4 * i] = 235; rgba[4 * i + 1] = 104; rgba[4 * i + 2] = 52; rgba[4 * i + 3] = 26 + occ * 0.42; }
    if (!this.pm) {
      const mfw = this.info.map_from_world || [1, 0, 0, 0];
      this.pm = new T.Group(); this.pm.rotation.z = Math.atan2(mfw[1], mfw[0]); this.pm.position.set(mfw[2], mfw[3], 0);
      this.scene().add(this.pm);
    }
    this.pm.clear();
    const tex = new T.DataTexture(rgba, g.w, g.h, T.RGBAFormat); tex.needsUpdate = true; tex.magFilter = T.NearestFilter;
    const mesh = new T.Mesh(new T.PlaneGeometry(g.w * g.res, g.h * g.res), new T.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
    mesh.position.set(g.ox + g.w * g.res / 2, g.oy + g.h * g.res / 2, 0.03); mesh.renderOrder = 4;
    this.pm.add(mesh); this.pm.visible = !!this.pmOn; this.invalidate();
  }
  setUnderlay(on) { this.ulOn = on; if (this.ul) { this.ul.visible = on; this.invalidate(); } else if (this.mesh && this.mesh.userData.wrap) { this.mesh.userData.wrap.visible = on; this.invalidate(); } }
  buildUnderlay() {
    const U = this.info.underlay, T = this.win.THREE;
    if (!U) return;
    const G = new T.Group(); G.name = "behavior_underlay";
    const mfw = this.info.map_from_world || [1, 0, 0, 0];   // map = R·world + t
    G.rotation.z = Math.atan2(mfw[1], mfw[0]); G.position.set(mfw[2], mfw[3], 0);
    const label = (text, x, y, z, h = 0.45, bg = "rgba(255,255,255,0.85)") => {
      const cv = document.createElement("canvas"), g = cv.getContext("2d"), fs = 30;
      g.font = `600 ${fs}px system-ui, sans-serif`; cv.width = Math.ceil(g.measureText(text).width) + 16; cv.height = fs + 14;
      g.font = `600 ${fs}px system-ui, sans-serif`; g.fillStyle = bg; g.fillRect(0, 0, cv.width, cv.height); g.fillStyle = "#222"; g.fillText(text, 8, fs + 2);
      const sp = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(cv), depthTest: false, transparent: true }));
      sp.scale.set(h * cv.width / cv.height, h, 1); sp.position.set(x, y, z); sp.renderOrder = 10; return sp;
    };
    // 방 바닥(방 종류 색) + 벽 칸 세우기(문 자리는 비어 있음)
    if (U.grid && U.room_grid_b64) {
      const { w, h, res, ox, oy } = U.grid;
      const rg = Uint8Array.from(atob(U.room_grid_b64), c => c.charCodeAt(0)); this._rg = rg;
      const rooms = U.rooms || [], col = rooms.map(r => new T.Color(ROOM_COL[r.type] ?? hashCol(r.type)));
      const rgba = new this.win.Uint8Array(w * h * 4);
      for (let i = 0; i < w * h; i++) { const k = rg[i]; if (!k) continue; const c = col[k - 1] || new T.Color(0x999999); rgba[4 * i] = c.r * 255; rgba[4 * i + 1] = c.g * 255; rgba[4 * i + 2] = c.b * 255; rgba[4 * i + 3] = 95; }
      const tex = new T.DataTexture(rgba, w, h, T.RGBAFormat); tex.needsUpdate = true; tex.magFilter = T.NearestFilter;
      const floor = new T.Mesh(new T.PlaneGeometry(w * res, h * res), new T.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
      floor.position.set(ox + w * res / 2, oy + h * res / 2, 0.012); floor.renderOrder = 2; G.add(floor);
      for (const r of rooms) G.add(label(r.name.replace(/_\d+$/, "").replace(/_/g, " "), r.centroid[0], r.centroid[1], 1.2));
      if (U.wall_grid_b64) {
        const wg = Uint8Array.from(atob(U.wall_grid_b64), c => c.charCodeAt(0));
        const runs = [];
        for (let y = 0; y < h; y++) { let x = 0; while (x < w) { if (!wg[y * w + x]) { x++; continue; } const x0 = x; while (x < w && wg[y * w + x]) x++; runs.push([x0, x - 1, y]); } }
        const WH = 1.0, geo = new T.BoxGeometry(1, 1, 1);
        const im = new T.InstancedMesh(geo, new T.MeshStandardMaterial({ color: 0x7d7f87, transparent: true, opacity: 0.55 }), runs.length);
        const m4 = new T.Matrix4();
        runs.forEach(([a, b, y], i) => { m4.makeScale((b - a + 1) * res, res, WH); m4.setPosition(ox + (a + b + 1) / 2 * res, oy + (y + 0.5) * res, WH / 2); im.setMatrixAt(i, m4); });
        G.add(im);
      }
    }
    // 가구·문(yaw 상자) — 칠한 상자, 마우스를 올리면 종류
    this.hoverables = [];
    // 방 안(방 칸 위)에 있고, 천장 붙이·아주 큰 구조물이 아닌 것만 — 집 밖 덤불·나무, 천장 등은 그리지 않는다
    const inRoom = (x, y) => { if (!U.grid || !this._rg) return true; const { w, h, res, ox, oy } = U.grid; const c = Math.floor((x - ox) / res), r = Math.floor((y - oy) / res); return c >= 0 && r >= 0 && c < w && r < h && this._rg[r * w + c] > 0; };
    for (const b of U.boxes || []) {
      if (b.kind === "carpet" || b.kind === "window") continue;
      if (!inRoom(b.c[0], b.c[1]) || b.c[2] - b.h[2] > 1.6 || 2 * b.h[2] > 2.4 || Math.max(b.h[0], b.h[1]) > 3) continue;
      const isDoor = b.kind === "door";
      const m = new T.Mesh(new T.BoxGeometry(1, 1, 1), new T.MeshStandardMaterial({ color: isDoor ? 0x9c6b3c : hashCol(b.cat), transparent: true, opacity: isDoor ? 0.3 : 0.4, depthWrite: false }));
      m.scale.set(2 * b.h[0], 2 * b.h[1], Math.max(2 * b.h[2], 0.02)); m.position.set(b.c[0], b.c[1], b.c[2]); m.rotation.z = b.yaw || 0;
      m.add(new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(1, 1, 1)), new T.LineBasicMaterial({ color: 0x4a4038, transparent: true, opacity: 0.25 })));
      m.userData.tip = `${b.kind}: ${b.cat}${b.name ? " (" + b.name + ")" : ""}`;
      G.add(m); this.hoverables.push(m); this.ulBoxes.push(m);
    }
    // 놓을 곳(받침 윗면, RASC v3 PLACES) · 집을 것(PICKS) · 이 과제의 물체
    for (const p of U.places || []) {
      const m = new T.Mesh(new T.BoxGeometry(1, 1, 1), new T.MeshBasicMaterial({ color: p.inner ? 0x0ca30c : 0x7bd389, transparent: true, opacity: 0.45, depthWrite: false }));
      m.scale.set(2 * p.half[0], 2 * p.half[1], 0.02); m.position.set(p.c[0], p.c[1], p.top + 0.012); m.rotation.z = p.yaw || 0;
      m.userData.tip = `place ${p.kind === 2 ? "inside" : "ontop"}: ${p.cat}${p.inner ? " (strict limits)" : ""}`; G.add(m); this.hoverables.push(m);
    }
    for (const p of U.picks || []) {
      const m = new T.Mesh(new T.SphereGeometry(Math.max(0.05, p.w / 2), 16, 12), new T.MeshBasicMaterial({ color: 0xeb6834 }));
      m.position.set(p.c[0], p.c[1], p.c[2]); m.userData.tip = `pick candidate (w ${fmt(p.w)} m)`; G.add(m); this.hoverables.push(m);
    }
    for (const o of U.task_objects || []) {
      const m = new T.Mesh(new T.BoxGeometry(1, 1, 1), new T.MeshStandardMaterial({ color: 0xe0359a, transparent: true, opacity: 0.8 }));
      m.scale.set(2 * o.h[0], 2 * o.h[1], 2 * o.h[2]); m.position.set(o.c[0], o.c[1], o.c[2]); m.rotation.z = o.yaw || 0;
      m.userData.tip = `task object: ${o.name}`; G.add(m); this.hoverables.push(m);
      G.add(label(o.cat || o.name, o.c[0], o.c[1], o.c[2] + 0.6, 0.3, "rgba(255,220,240,0.9)"));
    }
    this.ul = G; G.visible = this.ulOn === true;
    this.scene().add(G); this.invalidate();
    // 마우스 올림 → 종류(부모 화면의 풍선)
    const cv = this.win.eval("renderer.domElement"), cam = () => this.win.eval("camera"), tip = document.getElementById("tip");
    const ray = new T.Raycaster(), v = new T.Vector2();
    cv.addEventListener("mousemove", e => {
      if (!this.ul || !this.ul.visible) { tip.hidden = true; return; }
      const r = cv.getBoundingClientRect(); v.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(v, cam());
      const hit = ray.intersectObjects(this.hoverables, false)[0];
      if (!hit) { tip.hidden = true; return; }
      const fr = this.frame.getBoundingClientRect();
      tip.innerHTML = esc(hit.object.userData.tip); tip.hidden = false;
      tip.style.left = (fr.left + e.clientX + 14) + "px"; tip.style.top = (fr.top + e.clientY + 12) + "px";
    });
    cv.addEventListener("mouseleave", () => { tip.hidden = true; });
  }
}

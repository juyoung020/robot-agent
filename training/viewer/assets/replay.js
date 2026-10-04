// 재생 탭 — .trp 판 하나를 3D 로 (TRAIN_VIEWER.md 4.4·6.3). three.js·GLTFLoader·OrbitControls·로봇 GLB 는 sgview 것(바이너리에 같이 넣음).
// 좌표는 전부 map 프레임(x 앞, y 왼쪽, z 위) — sgview 장면이 z 위라 바꿀 것이 없다. 격자 행 0 = 최소 y (sgview onMapEvent 와 같음).
// 실행이 바뀌면 화면부터 비운다(전투기 뷰어 교훈 17).
import { fmt, esc, cssv } from "./charts.js";
import { SgPanel } from "./sgpanel.js";

const STATE_COL = [0x2ea043, 0x8c8c8c, 0xf58c14, 0x286ee6];   // 보임 · 사라짐 · 옮겨짐 · 들고 있음 (sgview STATE_COL)
const STATE_NAME = ["seen", "gone", "moved", "held"];
const EV = { 1: "contact", 2: "grasp", 4: "drop", 8: "success", 16: "filter", 32: "reset" };
const SNAP = 50;   // 지도 사본 간격(프레임)

function cellGrey(v) { return v < 0 ? 205 : v >= 65 ? 0 : v <= 25 ? 254 : Math.round(254 - (v * 254) / 100); }   // sgview cell_grey

export function parseTrp(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== "TRP1") throw new Error("not a TRP1 file");
  const hl = dv.getUint32(4, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)));
  const sec = Object.fromEntries((head.sections || []).map(s => [s.name, s]));
  const nc = head.cols.length, nf = head.n_frames;
  const frames = new Float32Array(buf, sec.frames.off, nf * nc);
  let slots = null;
  const ns = head.n_slots || 16, nsc = (head.slot_cols || []).length;
  if (sec.slots && nsc) {
    const u = new Uint16Array(buf, sec.slots.off, sec.slots.len / 2);
    slots = new Float32Array(u.length);
    for (let i = 0; i < u.length; i++) {
      const h = u[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
      slots[i] = e === 0 ? s * m * 5.960464477539063e-8 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * Math.pow(2, e - 15);
    }
  }
  const maps = [];
  if (sec.map) {
    let p = sec.map.off; const end = p + sec.map.len;
    while (p + 52 <= end) {
      const r = { frame: dv.getUint32(p, true), w: dv.getInt32(p + 4, true), h: dv.getInt32(p + 8, true), res: dv.getFloat64(p + 12, true), ox: dv.getFloat64(p + 20, true), oy: dv.getFloat64(p + 28, true), x0: dv.getInt32(p + 36, true), y0: dv.getInt32(p + 40, true), x1: dv.getInt32(p + 44, true), y1: dv.getInt32(p + 48, true) };
      const n = (r.x1 - r.x0 + 1) * (r.y1 - r.y0 + 1);
      r.cells = new Int8Array(buf, p + 52, n); p += 52 + n; maps.push(r);
    }
  }
  const imgs = [];
  if (sec.img) {
    let p = sec.img.off; const end = p + sec.img.len;
    while (p + 9 <= end) { const frame = dv.getUint32(p, true), cam = dv.getUint8(p + 4), len = dv.getUint32(p + 5, true); imgs.push({ frame, cam, bytes: new Uint8Array(buf, p + 9, len), url: null }); p += 9 + len; }
  }
  // 덧붙인 섹션(trainfmt TrpWriter::record): [u32 frame][u32 len][len 바이트] 이어짐. 예: segs = 벽 선분 n × (x0, y0, x1, y1) f32
  const extra = {};
  for (const s of head.sections || []) {
    if (["frames", "slots", "map", "img"].includes(s.name)) continue;
    const recs = []; let p = s.off; const end = s.off + s.len;
    while (p + 8 <= end) { const frame = dv.getUint32(p, true), len = dv.getUint32(p + 4, true); recs.push({ frame, bytes: new Uint8Array(buf.slice(p + 8, p + 8 + len)) }); p += 8 + len; }
    extra[s.name] = recs;
  }
  const ci = {}; head.cols.forEach((c, i) => ci[c] = i);
  const sci = {}; (head.slot_cols || []).forEach((c, i) => sci[c] = i);
  return { head, nc, nf, frames, slots, ns, nsc, maps, imgs, extra, ci, sci, v: (f, c) => ci[c] == null ? NaN : frames[f * nc + ci[c]], sv: (f, s, c) => sci[c] == null ? NaN : slots[(f * ns + s) * nsc + sci[c]] };
}

export class Replay {
  constructor({ api, getCursorIter }) {
    this.api = api; this.getCursorIter = getCursorIter;
    this.run = null; this.meta = null; this.rows = []; this.tr = null; this.f = 0; this.t = 0; this.playing = false; this.visible = false; this.ready = false;
    const $ = id => document.getElementById(id);
    this.$ = $;
    this.mode = "sg";
    // 예전 3D 그리기("3D classic")는 디버그로만(?debug=1): sgview 화면이 기본·전부이고, 3D classic 은 판마다 보상 띠와 .trp 안 카메라 JPEG 를 보는 데만 남김
    if (/[?&]debug=1/.test(location.search)) $("rp_mode").hidden = false;
    this.sg = new SgPanel({ api, host: $("rp_sg") });
    $("rp_mode").onchange = e => { this.mode = e.target.value; const f = this.file; if (f) this.open(f); };
    $("rp_ul").onchange = e => this.sg.setUnderlay(e.target.checked);
    $("rp_pmap").onchange = e => this.sg.setPolicyMap(e.target.checked);
    $("rp_ckpt").onchange = () => { const i = this.ckList.indexOf($("rp_ckpt").value); if (i >= 0) $("rp_ckslide").value = i; this.renderList(); };
    $("rp_ckslide").oninput = e => { $("rp_ckpt").value = this.ckList[+e.target.value] || ""; this.renderList(); };
    for (const id of ["rp_stream", "rp_skill", "rp_outcome", "rp_home", "rp_near"]) $(id).addEventListener("change", () => id === "rp_stream" ? this.refreshList() : this.renderList());
    $("rp_play").onclick = () => this.sgOn ? this.sg.toggle() : this.toggle();
    $("rp_prev").onclick = () => this.sgOn ? this.sg.seek((this.sg.state ? this.sg.state.t : 0) - 0.5) : this.step(-1);
    $("rp_next").onclick = () => this.sgOn ? this.sg.seek((this.sg.state ? this.sg.state.t : 0) + 0.5) : this.step(1);
    $("rp_time").oninput = e => { if (this.sgOn) this.sg.seek(+e.target.value / 10); else this.seek(+e.target.value); };
    $("rp_speed").onchange = e => { if (this.sgOn) this.sg.speed(+e.target.value); };
    $("rp_camsel").onchange = e => this.setCam(+e.target.value);
    $("rp_ee").onchange = () => this.apply();
    $("rp_slam").onchange = () => this.apply();
    $("rp_strip").addEventListener("click", e => { if (!this.tr) return; const r = e.target.getBoundingClientRect(); this.seek(Math.round((e.clientX - r.left) / r.width * (this.tr.nf - 1))); });
    addEventListener("keydown", e => {
      if (!this.visible || e.target.tagName === "INPUT" && e.target.type !== "range" || e.target.tagName === "SELECT") return;
      if (e.key >= "1" && e.key <= "4") { $("rp_camsel").value = e.key; this.setCam(+e.key); }
      else if (e.key === " ") { e.preventDefault(); $("rp_play").onclick(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); $("rp_prev").onclick(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); $("rp_next").onclick(); }
    });
    addEventListener("resize", () => this.resize());
  }

  // ---------------------------------------------------------------- 장면 (처음 볼 때 한 번)
  init() {
    if (this.ready) return; this.ready = true;
    const host = this.$("rp_view");
    const R = this.renderer = new THREE.WebGLRenderer({ antialias: true });
    R.setPixelRatio(Math.min(devicePixelRatio, 2));
    host.prepend(R.domElement);
    THREE.Object3D.DefaultUp.set(0, 0, 1);
    const sc = this.scene = new THREE.Scene();
    sc.background = new THREE.Color(cssv("--surface") || "#fcfcfb");
    const cam = this.camera = new THREE.PerspectiveCamera(50, 1, 0.02, 200);
    cam.up.set(0, 0, 1); cam.position.set(-4, -4, 5);
    this.controls = new THREE.OrbitControls(cam, R.domElement); this.controls.target.set(0, 0, 0); this.controls.update();
    sc.add(new THREE.HemisphereLight(0xffffff, 0x8890a8, 0.95));
    const dl = new THREE.DirectionalLight(0xffffff, 0.75); dl.position.set(-3, -4, 8); sc.add(dl);
    sc.add(new THREE.AmbientLight(0xffffff, 0.35));
    this.gEp = new THREE.Group(); sc.add(this.gEp);
    this.robot = new THREE.Group(); sc.add(this.robot);
    const fb = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.22, 0.15), new THREE.MeshStandardMaterial({ color: 0x9aa3b5 }));
    fb.position.z = 0.1; fb.userData.fallback = true; this.robot.add(fb);
    this.camMode = 4;
    this.loadRobotModel();
    this.resize();
    const loop = () => { requestAnimationFrame(loop); this.tick(); };
    this.last = performance.now(); loop();
  }
  async loadRobotModel() {
    // sgview loadRobotModel 을 옮김: robot.json = 링크·관절 나무, GLB = 메시
    let spec; try { const r = await fetch("/robot/robot.json"); if (!r.ok) return; spec = await r.json(); } catch (e) { return; }
    if (!spec.links) return;
    const loader = new THREE.GLTFLoader(), gltf = {};
    const files = [...new Set(spec.links.flatMap(l => l.visuals.map(v => v.mesh)))];
    await Promise.all(files.map(f => new Promise(res => loader.load("/robot/" + f, g => { gltf[f] = g.scene; res(); }, undefined, () => res()))));
    const rpy = r => new THREE.Euler(r[0], r[1], r[2], "ZYX");
    const groups = {};
    for (const l of spec.links) {
      const g = new THREE.Group(); g.name = l.name; groups[l.name] = g;
      for (const v of l.visuals) { if (!gltf[v.mesh]) continue; const m = gltf[v.mesh].clone(true); m.traverse(o => { if (o.isMesh) o.material.side = THREE.DoubleSide; }); m.position.set(...v.xyz); m.rotation.copy(rpy(v.rpy)); g.add(m); }
    }
    const joints = {};
    for (const j of spec.joints) {
      const jg = new THREE.Group(), motion = new THREE.Group();
      jg.position.set(...j.xyz); jg.rotation.copy(rpy(j.rpy)); jg.add(motion); motion.add(groups[j.child]); groups[j.parent].add(jg);
      if (j.type !== "fixed") joints[j.name] = { motion, axis: new THREE.Vector3(...j.axis).normalize() };
    }
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x15171d });
    groups[spec.root].traverse(o => { if (!o.isMesh) return; const c = o.material.color; if (c.r > 0.9 && c.g > 0.9 && c.b > 0.9) c.set(0xc3c9d6); o.add(new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry, 38), edgeMat)); });
    this.model = { root: groups[spec.root], joints, home: spec.home || {}, ee: groups["omx_end_effector_link"] || null, wrist: groups["wrist_cam_link"] || groups["omx_end_effector_link"] || null };
    this.robot.remove(...this.robot.children.filter(c => c.userData.fallback));
    this.robot.add(this.model.root);
    this.setJoints(this.model.home);
    this.apply();
  }
  setJoints(vals) { if (!this.model) return; for (const [n, v] of Object.entries(vals)) { const j = this.model.joints[n]; if (j && isFinite(v)) j.motion.quaternion.setFromAxisAngle(j.axis, v); } }
  resize() {
    if (!this.ready) return;
    const host = this.$("rp_view"), w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); this.dirty = true;
  }

  // ---------------------------------------------------------------- 실행 · 목록
  setRun(id, meta, streams) {
    this.run = id; this.meta = meta; this.rows = [];
    this.clearEpisode();
    this.$("rp_run").textContent = id;
    this.$("rp_stream").innerHTML = streams.map(s => `<option>${esc(s)}</option>`).join("");
    this.$("rp_list").innerHTML = "";
    this.$("rp_note").textContent = "";
    if (this.visible) this.refreshList(); else this.stale = true;
  }
  show(v) {
    this.visible = v;
    if (v) { this.init(); this.resize(); if (this.stale) { this.stale = false; this.refreshList(); } this.dirty = true; }
  }
  async refreshList() {
    if (!this.run) return;
    if (!this.visible) { this.stale = true; return; }
    const id = this.run, st = this.$("rp_stream").value || "main";
    const j = await this.api("/api/replays", { run: id, stream: st });
    if (id !== this.run) return;
    this.rows = j.rows || [];
    const why = j.why || "";
    const m = this.meta || {};
    this.$("rp_note").textContent = this.rows.length ? `${this.rows.length} episodes${j.bad ? ` (${j.bad} unreadable headers)` : ""}` : `${why}${m.logged && m.logged.why ? " — " + m.logged.why : ""}`;
    for (const [sel, key] of [["rp_skill", "skill"], ["rp_outcome", "outcome"], ["rp_home", "home"]]) {
      const cur = this.$(sel).value, vals = [...new Set(this.rows.map(r => r.meta && r.meta[key]).filter(Boolean))].sort();
      this.$(sel).innerHTML = '<option value="">all</option>' + vals.map(v => `<option${v === cur ? " selected" : ""}>${esc(v)}</option>`).join("");
    }
    // 체크포인트(학습 중 자동 기록): 이터 순으로
    const ck = new Map(); for (const r of this.rows) { const m = r.meta || {}; if (m.ckpt) ck.set(m.ckpt, m.ckpt_iter ?? 0); }
    this.ckList = [...ck.entries()].sort((a, b) => a[1] - b[1]).map(e => e[0]);
    this.$("rp_ckbar").hidden = !this.ckList.length;
    const curCk = this.$("rp_ckpt").value;
    this.$("rp_ckpt").innerHTML = '<option value="">all checkpoints</option>' + this.ckList.map(c => `<option${c === curCk ? " selected" : ""}>${esc(c)}</option>`).join("");
    this.$("rp_ckslide").max = Math.max(0, this.ckList.length - 1);
    this.renderList();
  }
  renderList() {
    const f = { skill: this.$("rp_skill").value, outcome: this.$("rp_outcome").value, home: this.$("rp_home").value };
    const ci = this.$("rp_near").checked ? this.getCursorIter() : null;
    const fck = this.$("rp_ckpt").value;
    let rows = this.rows.filter(r => { const m = r.meta || {}; return (!f.skill || m.skill === f.skill) && (!f.outcome || m.outcome === f.outcome) && (!f.home || m.home === f.home) && (!fck || m.ckpt === fck); });
    rows.sort((a, b) => ((b.meta || {}).ckpt_iter ?? -1) - ((a.meta || {}).ckpt_iter ?? -1));
    if (ci != null) { const tol = Math.max(50, Math.abs(ci) * 0.05); rows = rows.filter(r => r.meta && Math.abs(r.meta.iter - ci) <= tol); }
    else if (this.$("rp_near").checked) this.$("rp_note").textContent = "cursor is at latest — near-step filter off";
    this.$("rp_list").innerHTML = rows.map(r => {
      const m = r.meta || {};
      return `<div class="item${this.file === r.file ? " on" : ""}" data-f="${esc(r.file)}"><div class="r1"><span><b>${esc(m.skill || "?")}</b> <span class="oc ${esc(m.outcome || "")}">${esc(m.outcome || (m.success ? "success" : "?"))}${this.meta && this.meta.synthetic ? " (synthetic)" : ""}</span>${r.pin ? " 📌" : ""}</span><span class="mono muted">ep ${m.ep ?? "?"}</span></div>
        <div class="muted small">${esc(m.home || "")} · ${esc(m.stage || "")} · map ${fmt(m.completion0 ?? m.map_completeness, 1)} · ${fmt(m.t)} s · return ${fmt(m.ret)}${m.iter != null ? " · iter " + fmt(m.iter) : ""}${m.driver ? " · " + esc(m.driver) : ""}${m.ckpt ? ` · <b>${esc(m.ckpt)}</b>` : ""}</div></div>`;
    }).join("") || '<div class="muted small">no episode matches the filters</div>';
    this.$("rp_list").querySelectorAll(".item").forEach(el => el.onclick = () => this.open(el.dataset.f));
  }

  // ---------------------------------------------------------------- 판 하나 열기
  clearEpisode() {
    this.tr = null; this.file = null; this.playing = false;
    if (this.sgOn) this.setSg(false);
    if (this.ready) {
      this.gEp.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); } });
      this.gEp.clear();
      this.robot.visible = false;
    }
    this.$("rp_hud").textContent = ""; this.$("rp_cams").innerHTML = ""; this.$("rp_empty").hidden = false;
    this.$("rp_time").max = 0; this.$("rp_tlabel").textContent = ""; this.$("rp_play").textContent = "▶";
    const cv = this.$("rp_strip"); cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
    this.$("rp_legend").innerHTML = "";
    this.dirty = true;
  }
  setSg(on) {
    this.sgOn = on;
    this.$("rp_sg").hidden = !on;
    this.$("rp_view").classList.toggle("sgmode", on);
    if (this.renderer) this.renderer.domElement.style.display = on ? "none" : "";
    for (const id of ["rp_strip", "rp_legend"]) this.$(id).style.display = on ? "none" : "";
    // 3D classic 전용 조작은 sgview 화면에서 숨김(sgview 자체 패널에 시점·궤적 켜고 끔이 있음)
    for (const id of ["rp_camsel", "rp_ee", "rp_slam"]) { const el = this.$(id); (el.closest("label") || el).style.display = on ? "none" : ""; }
    for (const id of ["rp_ul", "rp_pmap"]) { const el = this.$(id); el.closest("label").style.display = on ? "" : "none"; }
    if (!on) this.sg.close();
  }
  async open(file) {
    const id = this.run, st = this.$("rp_stream").value || "main";
    this.clearEpisode(); this.file = file; this.renderList();
    // sgview 화면(장면 그래프 뷰어 그대로) — .sg(OmniGibson) 는 늘, .trp 는 고른 방식대로
    if (this.mode === "sg" || file.endsWith(".sg")) {
      this.setSg(true);
      this.$("rp_empty").hidden = true;
      await this.sg.open(id, st, file);
      return;
    }
    this.setSg(false);
    this.$("rp_empty").textContent = "loading…"; this.$("rp_empty").hidden = false;
    const t0 = performance.now();
    const r = await fetch(`/api/replay?run=${encodeURIComponent(id)}&stream=${encodeURIComponent(st)}&id=${encodeURIComponent(file)}`);
    if (id !== this.run || this.file !== file) return;
    let tr;
    try { tr = parseTrp(await r.arrayBuffer()); } catch (e) { this.$("rp_empty").textContent = "read failed: " + e.message; return; }
    this.tr = tr; this.loadMs = performance.now() - t0;
    this.$("rp_empty").hidden = true;
    this.build();
    this.$("rp_time").max = tr.nf - 1;
    this.t = 0; this.seek(0);
    this.setCam(+this.$("rp_camsel").value, true);
  }
  build() {
    const tr = this.tr, H = tr.head, G = this.gEp, v = tr.v;
    // 자라는 지도: DataTexture + 사본(SNAP 프레임마다)
    const g0 = tr.maps[0] || (H.grid ? { w: H.grid.w, h: H.grid.h, res: H.grid.res, ox: H.grid.ox, oy: H.grid.oy } : null);
    if (g0) {
      const w = g0.w, h = g0.h, rgba = new Uint8Array(w * h * 4);
      for (let i = 0; i < w * h; i++) { rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = 205; rgba[4 * i + 3] = 255; }
      const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat); tex.magFilter = tex.minFilter = THREE.NearestFilter;
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w * g0.res, h * g0.res), new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide }));
      mesh.position.set(g0.ox + w * g0.res / 2, g0.oy + h * g0.res / 2, -0.002); G.add(mesh);
      this.map = { w, h, rgba, tex, snaps: [], at: -1, ri: 0, grid: g0 };
      const apply = r => { if (r.w !== w || r.h !== h) return; const rw = r.x1 - r.x0 + 1; for (let y = r.y0; y <= r.y1; y++) for (let x = 0; x < rw; x++) { const g = cellGrey(r.cells[(y - r.y0) * rw + x]), o = (y * w + r.x0 + x) * 4; rgba[o] = rgba[o + 1] = rgba[o + 2] = g; } };
      this.map.applyRec = apply;
      // 사본: snaps[k] = 프레임 ≤ k·SNAP 의 기록을 모두 적용한 상태
      let ri = 0;
      for (let k = 0; k * SNAP < tr.nf; k++) {
        while (ri < tr.maps.length && tr.maps[ri].frame <= k * SNAP) apply(tr.maps[ri++]);
        this.map.snaps.push({ rgba: rgba.slice(), ri });
      }
    } else this.map = null;
    // 참 장면(머리 scene): 벽·문·가구·물체·목표 상자(yaw), 방 범위·이름
    const SC = H.scene;
    this.sceneInfo = SC ? `${SC.kind === "behavior" ? "BEHAVIOR " : ""}${SC.name || ""}${SC.task ? " · task " + SC.task : ""}` : "";
    if (SC && SC.boxes) {
      const KCOL = { wall: [0x6b6b70, 0.95], door: [0x9c6b3c, 0.8], window: [0x8fc5e8, 0.5], furniture: [0xc8a878, 0.45], object: [0xeb6834, 0.8], target: [0xe34948, 0.9], carpet: [0xb8b0a0, 0.25] };
      const geo = new THREE.BoxGeometry(1, 1, 1), eg = new THREE.EdgesGeometry(geo);
      for (const b of SC.boxes) {
        const [col, op] = KCOL[b.kind] || KCOL.furniture;
        const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: col, transparent: op < 1, opacity: op, depthWrite: op > 0.6 }));
        m.position.set(b.c[0], b.c[1], b.c[2]); m.scale.set(2 * b.h[0], 2 * b.h[1], Math.max(2 * b.h[2], 0.01)); m.rotation.z = b.yaw || 0;
        m.add(new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x3a3a3a, transparent: true, opacity: 0.5 })));
        G.add(m);
      }
    }
    if (SC && SC.rooms) {
      for (const r of SC.rooms) {
        const pts = [[r.bmin[0], r.bmin[1]], [r.bmax[0], r.bmin[1]], [r.bmax[0], r.bmax[1]], [r.bmin[0], r.bmax[1]], [r.bmin[0], r.bmin[1]]].map(p => new THREE.Vector3(p[0], p[1], 0.01));
        const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: 0x4a3aa7, dashSize: 0.15, gapSize: 0.1 }));
        l.computeLineDistances(); G.add(l);
        if (SC.rooms.length > 1) G.add(this.label(r.name, (r.bmin[0] + r.bmax[0]) / 2, (r.bmin[1] + r.bmax[1]) / 2, 0.05, H.grid ? Math.max(0.35, H.grid.w * H.grid.res / 60) : 0.35));
      }
    }
    // 지도가 뽑은 벽 선분(segs 섹션) — 프레임 따라 바꿈
    this.segs = (tr.extra.segs || []).map(r => ({ frame: r.frame, v: new Float32Array(r.bytes.buffer, r.bytes.byteOffset, r.bytes.byteLength / 4) }));
    if (this.segs.length) {
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(2 * 3 * 512), 3)); g.setDrawRange(0, 0);
      this.segLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xd03b3b })); this.segLines.frustumCulled = false; G.add(this.segLines); this.segAt = -2;
    } else this.segLines = null;
    // 궤적: 참(파랑) · slam(주황) · 손끝
    const line = (xs, ys, zs, col, op) => {
      const pts = new Float32Array(tr.nf * 3);
      for (let f = 0; f < tr.nf; f++) { pts[3 * f] = v(f, xs); pts[3 * f + 1] = v(f, ys); pts[3 * f + 2] = typeof zs === "number" ? zs : v(f, zs); }
      const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(pts, 3));
      const l = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: col, transparent: op < 1, opacity: op })); l.frustumCulled = false; G.add(l); return l;
    };
    const has = c => tr.ci[c] != null;
    this.lines = {
      trueAll: line("x", "y", 0.01, 0x2a78d6, 0.3), trueNow: line("x", "y", 0.012, 0x2a78d6, 1),
      slamAll: has("sx") ? line("sx", "sy", 0.008, 0xeb6834, 0.3) : null, slamNow: has("sx") ? line("sx", "sy", 0.01, 0xeb6834, 1) : null,
      ee: has("ee_x") ? line("ee_x", "ee_y", "ee_z", 0x1baf7a, 1) : null,
    };
    // 목표 표시 + 손끝에서 잇는 선
    this.tgt = new THREE.Mesh(new THREE.SphereGeometry(0.05, 16, 12), new THREE.MeshBasicMaterial({ color: 0xe34948 })); G.add(this.tgt);
    const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    this.tgtLine = new THREE.Line(lg, new THREE.LineDashedMaterial({ color: 0xe34948, dashSize: 0.05, gapSize: 0.04 })); this.tgtLine.frustumCulled = false; G.add(this.tgtLine);
    // 물체 칸
    this.slotObjs = [];
    if (tr.slots) {
      for (let s = 0; s < tr.ns; s++) {
        const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x2ea043, transparent: true, opacity: 0.8 }));
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: 0x15171d }));
        const tgtEdge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1.15, 1.15, 1.15)), new THREE.LineBasicMaterial({ color: 0xe34948 }));
        const ghost = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x8c8c8c, transparent: true, opacity: 0.18, depthWrite: false }));
        const gl = new THREE.BufferGeometry(); gl.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
        const link = new THREE.Line(gl, new THREE.LineBasicMaterial({ color: 0x8c8c8c })); link.frustumCulled = false;
        const ring = new THREE.Mesh(new THREE.RingGeometry(0.96, 1, 48), new THREE.MeshBasicMaterial({ color: 0xeda100, side: THREE.DoubleSide, transparent: true, opacity: 0.7 }));
        const grp = new THREE.Group(); grp.add(box, edges, tgtEdge, ghost, link, ring); G.add(grp);
        this.slotObjs.push({ grp, box, edges, tgtEdge, ghost, link, ring });
      }
    }
    // 바퀴 각: vx 적분(보기용)
    this.wheel = new Float32Array(tr.nf); const dt = H.dt || 0.1;
    for (let f = 1; f < tr.nf; f++) this.wheel[f] = this.wheel[f - 1] + (v(f, "vx") || 0) * dt / 0.045;
    this.robot.visible = true;
    // 보상 띠 범례
    const rcols = H.cols.filter(c => c.startsWith("r_"));
    this.rcols = rcols;
    this.$("rp_legend").innerHTML = rcols.map((c, i) => `<span><i style="background:${cssv("--s" + (1 + i % 8))}"></i>${esc(c.slice(2))}</span>`).join("") + `<span><i style="background:${cssv("--ink")}"></i>value</span><span><i style="background:${cssv("--muted")}"></i>end_p</span><span>ticks: events (contact · grasp · drop · success)</span>`;
    const objs = H.objects || [];
    this.objName = s => { const id = tr.sv(this.f, s, "id"); const o = objs.find(o => o.id === id || o.slot === s); return o ? o.name : `slot ${s}`; };
    this.drawStrip();
  }
  label(text, x, y, z, h = 0.35) {
    const cv = document.createElement("canvas"), g = cv.getContext("2d"), fs = 28;
    g.font = `${fs}px system-ui, sans-serif`; cv.width = Math.ceil(g.measureText(text).width) + 12; cv.height = fs + 12;
    g.font = `${fs}px system-ui, sans-serif`; g.fillStyle = "rgba(255,255,255,0.8)"; g.fillRect(0, 0, cv.width, cv.height); g.fillStyle = "#2a2a2a"; g.fillText(text, 6, fs);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), depthTest: false }));
    sp.scale.set(h * cv.width / cv.height, h, 1); sp.position.set(x, y, z + 0.3); return sp;
  }
  setSegFrame(f) {
    if (!this.segLines) return;
    let k = -1; for (let i = 0; i < this.segs.length; i++) if (this.segs[i].frame <= f) k = i; else break;
    if (k === this.segAt) return;
    this.segAt = k;
    const a = this.segLines.geometry.attributes.position.array, v = k >= 0 ? this.segs[k].v : new Float32Array(0), n = Math.min(512, v.length / 4);
    for (let i = 0; i < n; i++) { a.set([v[4 * i], v[4 * i + 1], 0.03, v[4 * i + 2], v[4 * i + 3], 0.03], 6 * i); }
    this.segLines.geometry.setDrawRange(0, 2 * n); this.segLines.geometry.attributes.position.needsUpdate = true;
  }
  setMapFrame(f) {
    const M = this.map, tr = this.tr; if (!M) return;
    if (M.at === f) return;
    const k = Math.floor(f / SNAP);
    if (!(M.at >= 0 && f > M.at && Math.floor(M.at / SNAP) === k)) { M.rgba.set(M.snaps[k].rgba); M.ri = M.snaps[k].ri; }
    while (M.ri < tr.maps.length && tr.maps[M.ri].frame <= f) M.applyRec(tr.maps[M.ri++]);
    M.at = f; M.tex.needsUpdate = true;
  }

  // ---------------------------------------------------------------- 프레임
  seek(f) {
    if (!this.tr) return;
    this.f = Math.max(0, Math.min(this.tr.nf - 1, f | 0));
    this.t = this.f * (this.tr.head.dt || 0.1);
    this.apply();
  }
  step(d) { this.playing = false; this.$("rp_play").textContent = "▶"; this.seek(this.f + d); }
  toggle() { if (!this.tr) return; if (this.f >= this.tr.nf - 1) this.seek(0); this.playing = !this.playing; this.$("rp_play").textContent = this.playing ? "❚❚" : "▶"; }
  apply() {
    const tr = this.tr; if (!tr) return;
    const f = this.f, v = c => tr.v(f, c), H = tr.head;
    this.$("rp_time").value = f;
    this.$("rp_tlabel").textContent = `${fmt(v("t"))} s · ${f + 1}/${tr.nf}`;
    this.setMapFrame(f);
    this.setSegFrame(f);
    // 로봇
    this.robot.position.set(v("x"), v("y"), 0); this.robot.rotation.set(0, 0, v("yaw"));
    const jm = H.joint_map || (this.meta && this.meta.joint_map) || {};
    const vals = {};
    for (const [colName, targets] of Object.entries(jm)) { const x = v(colName); if (!isFinite(x)) continue; for (const [jn, sc, off] of targets) vals[jn] = x * (sc ?? 1) + (off ?? 0); }
    const wl = tr.ci.wl != null ? v("wl") : this.wheel[f], wr = tr.ci.wr != null ? v("wr") : this.wheel[f];   // 환경이 바퀴 각을 내면 그것
    Object.assign(vals, { front_left_wheel: wl, rear_left_wheel: wl, front_right_wheel: wr, rear_right_wheel: wr });
    this.setJoints(vals);
    // 궤적
    const L = this.lines;
    L.trueNow.geometry.setDrawRange(0, f + 1);
    if (L.slamAll) { L.slamAll.visible = L.slamNow.visible = this.$("rp_slam").checked; L.slamNow.geometry.setDrawRange(0, f + 1); }
    if (L.ee) { L.ee.visible = this.$("rp_ee").checked; L.ee.geometry.setDrawRange(0, f + 1); }
    // 목표
    const tx = v("tgt_x"), ty = v("tgt_y"), tz = v("tgt_z");
    const hasT = isFinite(tx) && isFinite(ty);
    this.tgt.visible = this.tgtLine.visible = hasT;
    let dist = NaN;
    if (hasT) {
      this.tgt.position.set(tx, ty, isFinite(tz) ? tz : 0);
      const ex = isFinite(v("ee_x")) ? v("ee_x") : v("x"), ey = isFinite(v("ee_y")) ? v("ee_y") : v("y"), ez = isFinite(v("ee_z")) ? v("ee_z") : 0.2;
      const a = this.tgtLine.geometry.attributes.position.array; a.set([ex, ey, ez, tx, ty, isFinite(tz) ? tz : 0]); this.tgtLine.geometry.attributes.position.needsUpdate = true; this.tgtLine.computeLineDistances();
      dist = Math.hypot(tx - ex, ty - ey, (isFinite(tz) ? tz : 0) - ez);
    }
    // 물체 칸: 믿는 위치 상자(보는 중 = 꽉 참, 기억 = 반투명), 불확실도 원, 참 위치 유령 + 잇는 선, 목표 칸 굵은 테
    let held = false;
    this.slotObjs.forEach((o, s) => {
      const sv = c => tr.sv(f, s, c), src = sv("src");
      if (!(src > 0)) { o.grp.visible = false; return; }
      o.grp.visible = true;
      const ex = Math.max(sv("ex"), 0.03), ey = Math.max(sv("ey"), 0.03), ez = Math.max(sv("ez"), 0.03);
      const ctr = H.slot_z === "center";   // 머리 slot_z: "center" 면 bz·pz 가 상자 가운데(아니면 바닥)
      const bx = sv("bx"), by = sv("by"), bz = sv("bz") + (ctr ? 0 : ez / 2), st = Math.max(0, Math.min(3, Math.round(sv("state")) || 0));
      if (st === 3) held = true;
      for (const m of [o.box, o.edges, o.tgtEdge]) { m.position.set(bx, by, bz); m.scale.set(ex, ey, ez); }
      o.box.material.color.setHex(STATE_COL[st]);
      const conf = tr.sci.confirmed == null || sv("confirmed") > 0.5;   // 확정 안 된 칸(검출 1 번 등)은 아주 흐리게
      o.box.material.opacity = !conf ? 0.08 : src === 1 ? 0.85 : 0.18; o.box.material.depthWrite = conf && src === 1;
      o.tgtEdge.visible = sv("is_tgt") > 0.5;
      const px = sv("px"), py = sv("py"), pz = sv("pz") + (ctr ? 0 : ez / 2);
      const off = Math.hypot(px - bx, py - by, pz - bz) > 0.05 && isFinite(px);
      o.ghost.visible = o.link.visible = off;
      if (off) { o.ghost.position.set(px, py, pz); o.ghost.scale.set(ex, ey, ez); const a = o.link.geometry.attributes.position.array; a.set([bx, by, bz, px, py, pz]); o.link.geometry.attributes.position.needsUpdate = true; }
      const u = sv("unc"); o.ring.visible = u > 0.01; if (o.ring.visible) { o.ring.position.set(bx, by, 0.004); o.ring.scale.set(u, u, 1); }
    });
    // 카메라 영상(가장 가까운 프레임)
    if (tr.imgs.length) {
      const html = [];
      for (const cam of [...new Set(tr.imgs.map(i => i.cam))]) {
        let best = null; for (const im of tr.imgs) if (im.cam === cam && im.frame <= f && (!best || im.frame > best.frame)) best = im;
        if (!best) continue; if (!best.url) best.url = URL.createObjectURL(new Blob([best.bytes], { type: "image/jpeg" }));
        html.push(`<img src="${best.url}" title="cam ${cam} · frame ${best.frame}">`);
      }
      const h = html.join(""); if (this.$("rp_cams")._h !== h) { this.$("rp_cams").innerHTML = h; this.$("rp_cams")._h = h; }
    }
    // HUD
    const m = H.meta || {};
    let contacts = 0, evs = [];
    const ev = tr.ci.ev != null;
    if (ev) for (let k = 0; k <= f; k++) { const b = tr.v(k, "ev"); if (b & 1) contacts++; }
    if (ev) { const b = v("ev"); for (const [bit, n] of Object.entries(EV)) if (b & bit) evs.push(n); }
    const rs = this.rcols.map(c => `${c.slice(2)} ${fmt(v(c))}`).filter((_, i) => true);
    const stepR = this.rcols.reduce((s, c) => s + (v(c) || 0), 0);
    const tslot = this.slotObjs.findIndex((o, s) => tr.sv(f, s, "is_tgt") > 0.5 && tr.sv(f, s, "src") > 0);
    this.$("rp_hud").textContent = (H.synthetic ? "⚠ 가짜 시험 자료(synthetic) — 실제 학습 결과 아님\n" : "") +
      `${m.skill || "?"} · ${m.home || ""} · ${m.outcome || ""}${H.synthetic ? "  (synthetic)" : ""}${H.source ? "  [" + H.source.kind + "]" : ""}\n` +
      (this.sceneInfo ? this.sceneInfo + "\n" : "") +
      `t ${fmt(v("t"))} s   frame ${f + 1}/${tr.nf}\n` +
      `reward ${fmt(stepR)}   return ${fmt(v("ret_cum"))}\n  ${rs.join("  ")}\n` +
      `value ${fmt(v("value"))}   end prob ${fmt(v("end_p"))}\n` +
      `gripper ${held ? "holding" : "-"}   contacts ${contacts}   to goal ${fmt(dist)} m${tslot >= 0 ? " (" + this.objName(tslot) + ")" : ""}\n` +
      `map: objects ${fmt(v("completion"), 1)}${tr.ci.map_seen != null ? "  area seen " + fmt(v("map_seen"), 1) : ""}   ${evs.join(" ")}` + (this.loadMs != null ? `\nload ${this.loadMs.toFixed(0)} ms` : "");
    this.drawStrip();
    this.dirty = true;
  }
  drawStrip() {
    const tr = this.tr, cv = this.$("rp_strip"); if (!tr) return;
    const dpr = Math.min(devicePixelRatio || 1, 2), W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const n = tr.nf, bw = W / n, mid = H * 0.55;
    // 쌓은 막대: + 위, − 아래. 한 스텝의 큰 값(성공 +20)은 띠를 납작하게 하므로 칸 높이를 98 백분위로 자른다
    const sums = []; for (let f = 0; f < n; f++) { let p = 0, q = 0; for (const c of this.rcols) { const x = tr.v(f, c); if (x > 0) p += x; else q -= x; } sums.push(Math.max(p, q)); }
    const sorted = sums.slice().sort((a, b) => a - b), sc = (sorted[Math.floor(n * 0.98)] || 1e-6) || 1e-6;
    const k = (mid - 4) / sc;
    for (let f = 0; f < n; f++) {
      let up = 0, dn = 0;
      this.rcols.forEach((c, i) => {
        const x = tr.v(f, c); if (!x) return;
        g.fillStyle = cssv("--s" + (1 + i % 8));
        const h = Math.min(Math.abs(x) * k, mid - 2);
        if (x > 0) { g.fillRect(f * bw, mid - up - h, Math.max(bw - 0.5, 0.6), h); up += h; } else { g.fillRect(f * bw, mid + dn, Math.max(bw - 0.5, 0.6), h); dn += h; }
      });
    }
    g.strokeStyle = cssv("--grid"); g.beginPath(); g.moveTo(0, mid + 0.5); g.lineTo(W, mid + 0.5); g.stroke();
    // 가치·끝 신호 선
    const lineOf = (c, col, lo, hi) => { if (tr.ci[c] == null) return; g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath(); for (let f = 0; f < n; f++) { const y = H - 4 - (tr.v(f, c) - lo) / (hi - lo || 1) * (H - 8); f ? g.lineTo((f + 0.5) * bw, y) : g.moveTo((f + 0.5) * bw, y); } g.stroke(); };
    if (tr.ci.value != null) { let lo = Infinity, hi = -Infinity; for (let f = 0; f < n; f++) { const x = tr.v(f, "value"); lo = Math.min(lo, x); hi = Math.max(hi, x); } lineOf("value", cssv("--ink"), lo, hi); }
    lineOf("end_p", cssv("--muted"), 0, 1);
    // 사건 눈금
    if (tr.ci.ev != null) for (let f = 0; f < n; f++) { const b = tr.v(f, "ev"); if (!(b & 15)) continue; g.fillStyle = b & 8 ? cssv("--good") : b & 1 ? cssv("--critical") : cssv("--ink2"); g.fillRect(f * bw, 0, Math.max(2, bw), 6); }
    // 커서
    g.fillStyle = cssv("--accent"); g.fillRect((this.f + 0.5) * bw - 1, 0, 2, H);
  }

  // ---------------------------------------------------------------- 시점: 1 지도 위 · 2 따라감 · 3 손목 · 4 자유
  setCam(m, refit) {
    this.camMode = m; if (!this.ready) return;
    const tr = this.tr;
    this.controls.enabled = m === 4;
    if ((m === 4 && refit) || m === 1) {
      const g = this.map ? this.map.grid : null;
      const cx = g ? g.ox + g.w * g.res / 2 : 0, cy = g ? g.oy + g.h * g.res / 2 : 0, R = g ? Math.max(g.w, g.h) * g.res : 8;
      if (m === 1) { this.camera.position.set(cx, cy - 0.001, R * 1.1); this.camera.lookAt(cx, cy, 0); this.controls.target.set(cx, cy, 0); }
      else { this.controls.target.set(cx, cy, 0); this.camera.position.set(cx - R * 0.45, cy - R * 0.6, R * 0.6); this.controls.update(); }
    }
    this.dirty = true;
  }
  followCam() {
    const m = this.camMode; if (!this.tr || m === 1 || m === 4) return;
    const v = c => this.tr.v(this.f, c), x = v("x"), y = v("y"), yaw = v("yaw");
    if (m === 2) { this.camera.position.set(x - 1.6 * Math.cos(yaw), y - 1.6 * Math.sin(yaw), 1.2); this.camera.lookAt(x + 0.5 * Math.cos(yaw), y + 0.5 * Math.sin(yaw), 0.2); }
    else if (m === 3) {
      const p = new THREE.Vector3(), d = new THREE.Vector3(1, 0, 0);
      if (this.model && this.model.wrist) { this.model.root.updateWorldMatrix(true, true); this.model.wrist.getWorldPosition(p); const q = new THREE.Quaternion(); this.model.wrist.getWorldQuaternion(q); d.applyQuaternion(q); }
      else { p.set(v("ee_x"), v("ee_y"), v("ee_z")); d.set(Math.cos(yaw), Math.sin(yaw), -0.3); }
      this.camera.position.copy(p).addScaledVector(d, -0.08).add(new THREE.Vector3(0, 0, 0.05)); this.camera.lookAt(p.clone().addScaledVector(d, 1));
    }
  }
  tick() {
    const now = performance.now(), dt = (now - this.last) / 1000; this.last = now;
    if (!this.visible || this.sgOn) return;   // 보이지 않는 탭·sgview 화면일 때는 그리지 않는다
    if (this.playing && this.tr) {
      this.t += dt * +this.$("rp_speed").value;
      const f = Math.floor(this.t / (this.tr.head.dt || 0.1));
      if (f >= this.tr.nf - 1) { this.playing = false; this.$("rp_play").textContent = "▶"; this.seek(this.tr.nf - 1); }
      else if (f !== this.f) { this.f = f; const t = this.t; this.apply(); this.t = t; }
    }
    if (this.camMode === 2 || this.camMode === 3) { this.followCam(); this.dirty = true; }
    if (this.dirty || this.camMode === 4) { this.renderer.render(this.scene, this.camera); this.dirty = false; }
  }
}

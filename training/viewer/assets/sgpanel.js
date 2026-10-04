// 재생 탭 sgview 화면: sgview 페이지(src/scene_graph/sgview/assets/index.html)를 고치지 않고 iframe 으로 띄운다.
// 서버(src/sg.rs)가 sgview 의 실시간 경로(/stream SSE 등)를 판 재생으로 흉내 내고, 이 파일은
//   ① 시간 막대·재생·속도(/api/sg/ctl), ② 같은 출처 iframe 안 장면에 덧그림: 참(GT) 궤적, BEHAVIOR 집 바탕 층(방 바닥 색·벽·가구·놓을 곳·집을 것·과제 물체),
//   ③ LIMO 관절 순서(robot.json 의 joint_order 가 비어 있어 sgview 가 관절 스트림을 안 씀 → proprio 순서를 넣음), ④ 카메라 그림 을 맡는다.
import { fmt, esc } from "./charts.js";

const ROOM_COL = { kitchen: 0xeda100, bathroom: 0x1baf7a, bedroom: 0x9085e9, living_room: 0x3987e5, dining_room: 0xeb6834, corridor: 0x9a9a9a, childs_room: 0xe87ba4,
  private_office: 0x4a3aa7, storage_room: 0x8c6d46, utility_room: 0x6b6b70, garage: 0x555555, entryway: 0xb0b0b0, closet: 0x8c6d46, empty_room: 0xc8c8c8, playroom: 0xe87ba4,
  television_room: 0x3987e5, home_office: 0x4a3aa7, lobby: 0xb0b0b0, break_room: 0xeb6834, meeting_room: 0x4a3aa7, shared_office: 0x4a3aa7, room: 0x6da7ec };
const hashCol = s => { let h = 0; for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) >>> 0; const p = [0xc8a878, 0xb59a7a, 0xa9b48c, 0x9fb3c8, 0xc9a3a3, 0xb3a3c9, 0xa3c9c0, 0xd2b48c]; return p[h % p.length]; };

export class SgPanel {
  constructor({ api, host, getSel }) {
    this.api = api; this.host = host;
    this.sess = (() => { try { const k = sessionStorage.getItem("tv_sgsess"); if (k) return k; } catch (e) {} const k = Math.random().toString(36).slice(2, 12); try { sessionStorage.setItem("tv_sgsess", k); } catch (e) {} return k; })();
    document.cookie = `sgsess=${this.sess}; path=/; SameSite=Lax`;
    this.state = null; this.info = null; this.timer = null; this.win = null; this.ul = null; this.gt = null;
    const $ = id => document.getElementById(id);
    this.$ = $;
  }
  async open(run, stream, id) {
    const st = await (await fetch(`/api/sg/ctl?sess=${this.sess}&run=${encodeURIComponent(run)}&stream=${encodeURIComponent(stream)}&id=${encodeURIComponent(id)}&t=0&playing=0`)).json();
    if (st.error) { this.$("rp_empty").textContent = st.error; this.$("rp_empty").hidden = false; return false; }
    this.state = st;
    this.info = (await (await fetch(`/api/sg/info?sess=${this.sess}`)).json()).info || {};
    this.$("rp_time").max = Math.round(st.duration * 10); this.$("rp_time").value = 0;
    // 같은 페이지를 다시 쓰면 sgview 가 상태를 이어 붙이므로 새로 띄움(옮김은 SSE reset 으로)
    this.host.innerHTML = "";
    const f = document.createElement("iframe");
    f.src = "/sg/?k=" + Math.random().toString(36).slice(2);
    f.style.cssText = "width:100%;height:100%;border:0;display:block";
    this.host.appendChild(f);
    this.frame = f; this.win = null; this.ul = null; this.gt = null; this.pm = null; this._pmT = null;
    f.onload = () => this.attach();
    this.run = run; this.stream = stream; this.id = id;
    this.startPoll();
    return true;
  }
  close() { clearInterval(this.timer); this.timer = null; this.host.innerHTML = ""; this.frame = null; this.win = null; }
  async ctl(q) {
    const st = await (await fetch(`/api/sg/ctl?sess=${this.sess}&${q}`)).json();
    if (!st.error) { this.state = st; this.update(); }
    return st;
  }
  seek(t) { return this.ctl(`t=${Math.max(0, t).toFixed(3)}`); }
  toggle() { return this.ctl(`playing=${this.state && this.state.playing ? 0 : 1}`); }
  speed(v) { return this.ctl(`speed=${v}`); }
  startPoll() {
    clearInterval(this.timer);
    this.timer = setInterval(async () => { if (!this.state) return; if (this.state.playing) await this.ctl(""); }, 200);
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
      (m.skill === "layout" ? "" : `\nGT path green · SLAM path blue (sgview)`);
    this.drawGt(st.t);
    if (this.pmOn) this.drawPolicyMap(st.t);
    // 카메라 그림(몸통 카메라, 2 Hz)
    const cams = i.cams || [];
    if (cams.length) {
      let best = cams[0]; for (const c of cams) { if (c.t <= st.t) best = c; else break; }
      const url = `/api/sg/cam?run=${encodeURIComponent(this.run)}&stream=${encodeURIComponent(this.stream)}&id=${encodeURIComponent(this.id)}&file=${encodeURIComponent(best.file)}`;
      const h = `<img src="${url}" title="body camera t ${best.t.toFixed(1)} s">`;
      if (this.$("rp_cams")._h !== h) { this.$("rp_cams").innerHTML = h; this.$("rp_cams")._h = h; }
    } else if (this.$("rp_cams")._h) { this.$("rp_cams").innerHTML = ""; this.$("rp_cams")._h = ""; }
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
      this.buildGt(); this.buildUnderlay(); this.update();
    };
    tryIt();
  }
  scene() { return this.win.eval("scene"); }
  invalidate() { try { this.win.eval("invalidate()"); } catch (e) {} }
  buildGt() {
    const T = this.win.THREE, pts = this.info.gt_path || [];
    if (!pts.length) return;
    const a = new Float32Array(pts.length * 3);
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
    const T = this.win.THREE, px = Uint8Array.from(atob(g.b64), c => c.charCodeAt(0)), rgba = new Uint8Array(g.w * g.h * 4);
    for (let i = 0; i < g.w * g.h; i++) { const v = px[i]; if (v === 205) continue; const occ = 255 - v; rgba[4 * i] = 235; rgba[4 * i + 1] = 104; rgba[4 * i + 2] = 52; rgba[4 * i + 3] = 40 + occ * 0.7; }
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
  setUnderlay(on) { this.ulOn = on; if (this.ul) { this.ul.visible = on; this.invalidate(); } }
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
      const rgba = new Uint8Array(w * h * 4);
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
      G.add(m); this.hoverables.push(m);
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
    this.ul = G; G.visible = this.ulOn !== false;
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

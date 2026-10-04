// 재생 탭의 sgview 화면 — sgview(장면 그래프 실시간 뷰어, src/scene_graph/sgview)의 index.html 을 고치지 않고 그대로 iframe 에 띄우고,
// 이 서버가 sgview 의 실시간 경로(/api/mode → live, /stream SSE, /file/, /api/robot)를 판 하나의 시각에 맞춰 흉내 낸다.
//
//   판 = OmniGibson 기록에서 og2sg 가 만든 <이름>.sg/ (scenemap 이 보낸 sgview 스트림 프레임 + 시뮬 시각) 또는
//        GPU 환경 판 .trp (지도 MAP_RECT · 자세 · 관절 · 물체 칸 → 같은 프레임으로 바꿈 — 점구름 없는 물체는 상자, sgview 와 같음).
//   재생 = 브라우저마다 세션(쿠키 sgsess): 시각 t·속도·재생 중. 부모 화면(app.js)이 /api/sg/ctl 로 옮기면 SSE 가 reset + 그 시각 스냅숏을 보낸다.
// SSE 이벤트 모양·벽 계산(scenemap walls.cpp 를 sgview walls_ffi.cpp 로)은 sgview main.rs 와 같게(옮김).
use crate::http::{self, Req};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Write;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

extern "C" {
    fn sgv_wall_segments(cells: *const i8, w: i32, h: i32, res: f64, ox: f64, oy: f64, ignore: *const f64, n_ignore: i32, out: *mut f64, cap: i32, angle_out: *mut f64) -> i32;
    fn sgv_wall_state(cells: *const i8, w: i32, h: i32, res: f64, ox: f64, oy: f64, segs: *const f64, n: i32, pose: *const f64, out: *mut f32);
}
const WALL_STATE_LEN: usize = 56;

pub struct Frame {
    pub t: f64,
    pub ty: u8,
    pub pl: Vec<u8>,
}

pub struct Episode {
    pub dir: PathBuf,        // 파일(/file/) 뿌리: .sg 면 memory/, .trp 면 없음
    pub frames: Vec<Frame>,
    pub duration: f64,
    pub robot: String,
    pub info: Value,         // 부모 화면용: gt_path, underlay, cams, joint_order, map_from_world …
}

// ---------------------------------------------------------------- 읽기

pub fn load_sg(dir: &Path) -> Option<Episode> {
    let b = std::fs::read(dir.join("stream.sgs")).ok()?;
    if b.len() < 4 || &b[..4] != b"SGS1" {
        return None;
    }
    let meta: Value = std::fs::read_to_string(dir.join("meta.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(json!({}));
    let labels: Vec<(String, String)> = meta["labels"].as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()).unwrap_or_default();
    // 끝에 저장한 메모리 폴더의 view.json 에서 물체마다 점구름·RGB 조각 경로(스트림 요약에는 없다 — 실시간에는 파일이 아직 없으므로).
    // 같은 scenemap 문맥이라 물체 번호가 같다
    let files: HashMap<i64, (Value, Value)> = std::fs::read_to_string(dir.join("memory/view.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v["objects"].as_array().cloned()).unwrap_or_default().into_iter()
        .filter_map(|o| Some((o["id"].as_i64()?, (o["rgbd"].clone(), o["points"].clone())))).collect();
    let mut frames = vec![];
    let mut p = 4usize;
    while p + 13 <= b.len() {
        let t = f64::from_le_bytes(b[p..p + 8].try_into().unwrap());
        let len = u32::from_le_bytes(b[p + 8..p + 12].try_into().unwrap()) as usize;
        let ty = b[p + 12];
        if p + 13 + len > b.len() {
            break;
        }
        let mut pl = b[p + 13..p + 13 + len].to_vec();
        if ty == 3 && !labels.is_empty() {
            // 검출 번호 이름(cls<k>) → 정답 실행과 위치로 짝지은 이름(og2sg meta.labels). 따옴표까지 바꿔 cls1 / cls10 이 섞이지 않게
            let mut s = String::from_utf8_lossy(&pl).to_string();
            for (k, v) in &labels {
                if !v.is_empty() {
                    s = s.replace(&format!("\"{}\"", k), &format!("\"{}\"", v));
                }
            }
            pl = s.into_bytes();
        }
        if ty == 3 && !files.is_empty() {
            if let Ok(mut v) = serde_json::from_slice::<Value>(&pl) {
                if let Some(objs) = v["objects"].as_array_mut() {
                    for o in objs {
                        if let Some((rgbd, pts)) = o["id"].as_i64().and_then(|i| files.get(&i)) {
                            if !rgbd.is_null() { o["rgbd"] = rgbd.clone(); }
                            if !pts.is_null() { o["points"] = pts.clone(); }
                        }
                    }
                }
                pl = v.to_string().into_bytes();
            }
        }
        frames.push(Frame { t, ty, pl });
        p += 13 + len;
    }
    let duration = meta["duration"].as_f64().unwrap_or_else(|| frames.last().map(|f| f.t).unwrap_or(0.0));
    let underlay: Value = std::fs::read_to_string(dir.join("underlay.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    let info = json!({
        "kind": "sg", "meta": meta["meta"], "gt_path": meta["gt_path"], "map_from_world": meta["map_from_world"], "cams": meta["cams"],
        "joint_order": meta["joint_order"], "underlay": underlay, "stream": meta["stream"], "og_run": meta["og_run"], "n_objects": meta["n_objects"],
    });
    Some(Episode { dir: dir.join("memory"), frames, duration, robot: meta["robot"].as_str().unwrap_or("limo_omx").to_string(), info })
}

/// GPU 환경 판(.trp) → sgview 프레임. 좌표는 G2 지도 = 세계(map 프레임 같음).
pub fn load_trp(path: &Path) -> Option<Episode> {
    let b = std::fs::read(path).ok()?;
    let h = trainfmt::trp::read_head(&b)?;
    let cols: Vec<String> = h["cols"].as_array()?.iter().map(|c| c.as_str().unwrap_or("").to_string()).collect();
    let scols: Vec<String> = h["slot_cols"].as_array().map(|a| a.iter().map(|c| c.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
    let nf = h["n_frames"].as_u64()? as usize;
    let ns = h["n_slots"].as_u64().unwrap_or(0) as usize;
    let dt = h["dt"].as_f64().unwrap_or(0.1);
    let sec = |n: &str| h["sections"].as_array().and_then(|a| a.iter().find(|s| s["name"] == n)).map(|s| (s["off"].as_u64().unwrap() as usize, s["len"].as_u64().unwrap() as usize));
    let (fo, _) = sec("frames")?;
    let nc = cols.len();
    let ci = |c: &str| cols.iter().position(|x| x == c);
    let fv = |f: usize, c: Option<usize>| -> f64 { c.map(|i| f32::from_le_bytes(b[fo + (f * nc + i) * 4..fo + (f * nc + i) * 4 + 4].try_into().unwrap()) as f64).unwrap_or(f64::NAN) };
    let nsc = scols.len();
    let sl = sec("slots");
    let sci = |c: &str| scols.iter().position(|x| x == c);
    let sv = |f: usize, s: usize, c: Option<usize>| -> f64 {
        match (sl, c) {
            (Some((o, _)), Some(i)) => {
                let k = o + ((f * ns + s) * nsc + i) * 2;
                trainfmt::trp::f16_to_f32(u16::from_le_bytes([b[k], b[k + 1]])) as f64
            }
            _ => f64::NAN,
        }
    };
    let (cx, cy, cyaw) = if ci("sx").is_some() { (ci("sx"), ci("sy"), ci("syaw")) } else { (ci("x"), ci("y"), ci("yaw")) };
    let objects = h["objects"].as_array().cloned().unwrap_or_default();
    let obj_name = |id: f64| objects.iter().find(|o| o["id"].as_f64() == Some(id)).and_then(|o| o["name"].as_str()).unwrap_or("object").to_string();
    let mut frames = vec![];
    // 지도: MAP_RECT 기록을 그대로(같은 바이트: [u32 frame] 뒤 48 B 머리 + 칸)
    let mut maps: Vec<(usize, Vec<u8>)> = vec![];
    if let Some((o, l)) = sec("map") {
        let mut p = o;
        while p + 52 <= o + l {
            let fr = u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize;
            let i32at = |q: usize| i32::from_le_bytes(b[q..q + 4].try_into().unwrap());
            let (x0, y0, x1, y1) = (i32at(p + 36), i32at(p + 40), i32at(p + 44), i32at(p + 48));
            let n = ((x1 - x0 + 1) * (y1 - y0 + 1)) as usize;
            maps.push((fr, b[p + 4..p + 52 + n].to_vec()));
            p += 52 + n;
        }
    }
    let mut mi = 0;
    let mut trail: Vec<Value> = vec![];
    for f in 0..nf {
        let t = f as f64 * dt;
        while mi < maps.len() && maps[mi].0 <= f {
            frames.push(Frame { t, ty: 2, pl: maps[mi].1.clone() });
            mi += 1;
        }
        let (x, y, yaw) = (fv(f, cx), fv(f, cy), fv(f, cyaw));
        let mut pp = vec![];
        for v in [t, x, y, yaw] {
            pp.extend_from_slice(&v.to_le_bytes());
        }
        frames.push(Frame { t, ty: 1, pl: pp });
        // 관절: LIMO proprio 순서(6..10 팔, 11 그리퍼, 12..15 바퀴)
        let mut q = [0f32; 16];
        for (k, c) in ["q1", "q2", "q3", "q4", "q5", "qg"].iter().enumerate() {
            q[6 + k] = fv(f, ci(c)) as f32;
        }
        let (wl, wr) = (fv(f, ci("wl")), fv(f, ci("wr")));
        if wl.is_finite() {
            q[12] = wl as f32; q[13] = wr as f32; q[14] = wl as f32; q[15] = wr as f32;
        }
        let mut jp = t.to_le_bytes().to_vec();
        jp.extend_from_slice(&16i32.to_le_bytes());
        for v in q {
            jp.extend_from_slice(&v.to_le_bytes());
        }
        frames.push(Frame { t, ty: 4, pl: jp });
        if f % 5 == 0 {
            trail.push(json!({"id": format!("a{}", trail.len()), "kind": "agent", "pos": [x, y], "yaw": yaw, "t": t}));
        }
        if f % 2 == 0 || f + 1 == nf {
            // 요약(view): 물체 칸(점구름 없음 → sgview 가 상자로) + 방 하나 + 궤적
            let mut objs = vec![];
            let mut edges: Vec<Value> = vec![];
            for s in 0..ns {
                if !(sv(f, s, sci("src")) > 0.0) {
                    continue;
                }
                let conf = sci("confirmed").map(|_| sv(f, s, sci("confirmed")) > 0.5).unwrap_or(true);
                if !conf {
                    continue;
                }
                let st = ["seen", "gone", "moved", "held"][(sv(f, s, sci("state")).max(0.0) as usize).min(3)];
                let id = s + 1;
                let z_center = h["slot_z"] == "center";
                let ez = sv(f, s, sci("ez"));
                let bz = sv(f, s, sci("bz")) + if z_center { 0.0 } else { ez / 2.0 };
                objs.push(json!({"id": id, "name": obj_name(sv(f, s, sci("id"))), "state": st, "pos": [sv(f, s, sci("bx")), sv(f, s, sci("by")), bz],
                    "extent": [sv(f, s, sci("ex")), sv(f, s, sci("ey")), ez], "first_pos": [sv(f, s, sci("px")), sv(f, s, sci("py")), sv(f, s, sci("pz"))],
                    "n_obs": 1, "last_seen": t - sv(f, s, sci("age")).max(0.0), "score": 1.0, "structural": false, "movable": true}));
                edges.push(json!(["R1", format!("O{}", id), "contains"]));
            }
            for k in 1..trail.len() {
                edges.push(json!([format!("a{}", k - 1), format!("a{}", k), "agent"]));
            }
            let room = h["scene"]["rooms"].as_array().and_then(|a| a.first()).cloned().unwrap_or(json!({"name": "room", "bmin": [-3, -3], "bmax": [3, 3]}));
            let (b0, b1) = (&room["bmin"], &room["bmax"]);
            let g = |a: &Value, i: usize| a[i].as_f64().unwrap_or(0.0);
            let view = json!({
                "stamp": t, "pose": [x, y, yaw], "objects": objs,
                "rooms": [{"id": 1, "value": 1, "name": room["name"], "type": "room", "centroid": [(g(b0, 0) + g(b1, 0)) / 2.0, (g(b0, 1) + g(b1, 1)) / 2.0],
                           "bbox": [g(b0, 0), g(b0, 1), g(b1, 0), g(b1, 1)], "color": [109, 147, 242], "objects": (1..=ns).collect::<Vec<_>>()}],
                "room_doors": [], "graph": {"nodes": trail, "edges": edges}, "events": [],
            });
            frames.push(Frame { t, ty: 3, pl: view.to_string().into_bytes() });
        }
    }
    let duration = (nf.max(1) - 1) as f64 * dt;
    // 부모 화면용: 참 궤적(x, y), 바탕 층(머리 scene 의 상자·방, 세계 = map)
    let gt: Vec<Value> = (0..nf).step_by(1).map(|f| json!([f as f64 * dt, fv(f, ci("x")), fv(f, ci("y")), fv(f, ci("yaw"))])).collect();
    let mut boxes = vec![];
    for bx in h["scene"]["boxes"].as_array().cloned().unwrap_or_default() {
        boxes.push(json!({"kind": bx["kind"], "cat": bx["name"], "c": bx["c"], "h": bx["h"], "yaw": bx["yaw"]}));
    }
    let info = json!({"kind": "trp", "meta": h["meta"], "gt_path": gt, "map_from_world": [1, 0, 0, 0], "cams": [],
        "joint_order": ["", "", "", "", "", "", "omx_joint1", "omx_joint2", "omx_joint3", "omx_joint4", "omx_joint5", "omx_gripper_joint_1", "front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"],
        "underlay": {"scene": h["scene"]["name"], "boxes": boxes, "rooms": h["scene"]["rooms"], "places": [], "picks": [], "task_objects": []}});
    Some(Episode { dir: PathBuf::new(), frames, duration, robot: "limo_omx".into(), info })
}

// ---------------------------------------------------------------- 세션·상태

pub struct Sess {
    pub ep: Arc<Episode>,
    pub key: String,
    pub t0: f64,
    pub wall0: Instant,
    pub speed: f64,
    pub playing: bool,
    pub epoch: u64,
}
impl Sess {
    pub fn t(&self) -> f64 {
        let t = if self.playing { self.t0 + self.wall0.elapsed().as_secs_f64() * self.speed } else { self.t0 };
        t.clamp(0.0, self.ep.duration)
    }
}

#[derive(Default)]
pub struct SgState {
    pub sess: HashMap<String, Sess>,
    pub eps: HashMap<String, Arc<Episode>>,
}

pub fn cookie_sess(req: &Req) -> String {
    req.cookie.split(';').filter_map(|kv| kv.trim().strip_prefix("sgsess=")).next().unwrap_or("").to_string()
}

struct Live {
    w: i32,
    h: i32,
    res: f64,
    ox: f64,
    oy: f64,
    cells: Vec<i8>,
    view: Option<String>,
    pose: Option<[f64; 4]>,
    joints: Option<String>,
    ig: Vec<f64>,
    segs: Vec<f64>,
    theta: f64,
    dirty: bool,
    seg_ver: u64,
    seg_sent: u64,
}

fn sse(ev: &str, data: &str) -> String {
    format!("event: {}\ndata: {}\n\n", ev, data)
}
fn cell_grey(v: i8) -> u8 {
    let v = v as i32;
    if v < 0 { 205 } else if v >= 65 { 0 } else if v <= 25 { 254 } else { (254 - (v * 254) / 100) as u8 }
}
fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut o = String::with_capacity((data.len() + 2) / 3 * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        o.push(T[(n >> 18) as usize & 63] as char);
        o.push(T[(n >> 12) as usize & 63] as char);
        o.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        o.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    o
}
fn map_event(l: &Live, x0: i32, y0: i32, x1: i32, y1: i32) -> String {
    let mut g = Vec::with_capacity(((x1 - x0 + 1) * (y1 - y0 + 1)) as usize);
    for y in y0..=y1 {
        let r = &l.cells[(y * l.w) as usize..((y + 1) * l.w) as usize];
        g.extend(r[x0 as usize..=x1 as usize].iter().map(|&v| cell_grey(v)));
    }
    sse("map", &format!("{{\"w\":{},\"h\":{},\"res\":{},\"ox\":{},\"oy\":{},\"x0\":{},\"y0\":{},\"x1\":{},\"y1\":{},\"b64\":\"{}\"}}", l.w, l.h, l.res, l.ox, l.oy, x0, y0, x1, y1, b64(&g)))
}
fn pose_event(p: &[f64; 4]) -> String {
    sse("pose", &format!("{{\"t\":{:.3},\"x\":{:.4},\"y\":{:.4},\"yaw\":{:.4}}}", p[0], p[1], p[2], p[3]))
}
fn furniture_rects_of(text: &str) -> Vec<f64> {
    // sgview main.rs furniture_rects_of 와 같음: 바닥에 놓인 가구(아래 0.4 m 밑, 변 ≤ 5 m)는 벽으로 치지 않는다
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    let mut out = vec![];
    for o in v["objects"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        if o["state"] == "gone" {
            continue;
        }
        let (p, e) = (&o["pos"], &o["extent"]);
        let g = |a: &Value, i: usize| a[i].as_f64().unwrap_or(0.0);
        let (hx, hy, hz) = (g(e, 0) / 2.0, g(e, 1) / 2.0, g(e, 2) / 2.0);
        if g(p, 2) - hz > 0.4 || hx * 2.0 > 5.0 || hy * 2.0 > 5.0 {
            continue;
        }
        out.extend_from_slice(&[g(p, 0) - hx - 0.1, g(p, 1) - hy - 0.1, g(p, 0) + hx + 0.1, g(p, 1) + hy + 0.1]);
    }
    out
}

impl Live {
    fn new() -> Live {
        Live { w: 0, h: 0, res: 0.05, ox: 0.0, oy: 0.0, cells: vec![], view: None, pose: None, joints: None, ig: vec![], segs: vec![], theta: 0.0, dirty: false, seg_ver: 0, seg_sent: u64::MAX }
    }
    /// 프레임 하나 적용. 내보낼 SSE 를 돌려준다(emit false 면 상태만)
    fn apply(&mut self, f: &Frame, emit: bool) -> Option<String> {
        let pl = &f.pl;
        match f.ty {
            1 if pl.len() == 32 => {
                let g = |i: usize| f64::from_le_bytes(pl[i * 8..i * 8 + 8].try_into().unwrap());
                let p = [g(0), g(1), g(2), g(3)];
                self.pose = Some(p);
                if emit { Some(pose_event(&p)) } else { None }
            }
            2 if pl.len() >= 48 => {
                let i32at = |o: usize| i32::from_le_bytes(pl[o..o + 4].try_into().unwrap());
                let f64at = |o: usize| f64::from_le_bytes(pl[o..o + 8].try_into().unwrap());
                let (w, h, res, ox, oy) = (i32at(0), i32at(4), f64at(8), f64at(16), f64at(24));
                let (x0, y0, x1, y1) = (i32at(32), i32at(36), i32at(40), i32at(44));
                if w <= 0 || h <= 0 || x0 < 0 || y0 < 0 || x1 >= w || y1 >= h || x1 < x0 || y1 < y0 || (x1 - x0 + 1) as usize * (y1 - y0 + 1) as usize + 48 != pl.len() {
                    return None;
                }
                if w != self.w || h != self.h || res != self.res || ox != self.ox || oy != self.oy {
                    self.w = w; self.h = h; self.res = res; self.ox = ox; self.oy = oy;
                    self.cells = vec![-1i8; (w * h) as usize];
                }
                let rw = (x1 - x0 + 1) as usize;
                for y in y0..=y1 {
                    let d = (y * w + x0) as usize;
                    let s = 48 + (y - y0) as usize * rw;
                    for k in 0..rw {
                        self.cells[d + k] = pl[s + k] as i8;
                    }
                }
                self.dirty = true;
                if emit { Some(map_event(self, x0, y0, x1, y1)) } else { None }
            }
            3 => {
                let line = String::from_utf8_lossy(pl).replace('\n', "");
                let ig = furniture_rects_of(&line);
                if ig != self.ig {
                    self.ig = ig;
                    self.dirty = true;
                }
                let e = sse("view", &line);
                self.view = Some(line);
                if emit { Some(e) } else { None }
            }
            4 if pl.len() >= 12 => {
                let t = f64::from_le_bytes(pl[0..8].try_into().unwrap());
                let n = i32::from_le_bytes(pl[8..12].try_into().unwrap()) as usize;
                if n == 0 || pl.len() != 12 + n * 4 {
                    return None;
                }
                let q: Vec<String> = (0..n).map(|i| format!("{:.4}", f32::from_le_bytes(pl[12 + i * 4..16 + i * 4].try_into().unwrap()))).collect();
                let e = sse("joints", &format!("{{\"t\":{:.3},\"q\":[{}]}}", t, q.join(",")));
                self.joints = Some(e.clone());
                if emit { Some(e) } else { None }
            }
            _ => None,
        }
    }
    fn walls(&mut self, force_segs: bool) -> Option<String> {
        if self.w <= 0 {
            return None;
        }
        if self.dirty {
            self.dirty = false;
            let mut buf = vec![0f64; 4 * 512];
            let mut th = 0f64;
            let mut n = unsafe { sgv_wall_segments(self.cells.as_ptr(), self.w, self.h, self.res, self.ox, self.oy, self.ig.as_ptr(), (self.ig.len() / 4) as i32, buf.as_mut_ptr(), 512, &mut th) } as usize;
            if n > 512 {
                buf = vec![0f64; 4 * n];
                n = unsafe { sgv_wall_segments(self.cells.as_ptr(), self.w, self.h, self.res, self.ox, self.oy, self.ig.as_ptr(), (self.ig.len() / 4) as i32, buf.as_mut_ptr(), n as i32, &mut th) } as usize;
            }
            buf.truncate(4 * n);
            if buf != self.segs || th != self.theta {
                self.segs = buf;
                self.theta = th;
                self.seg_ver += 1;
            }
        }
        let p = self.pose?;
        let pose = [p[1], p[2], p[3]];
        let mut out = [0f32; WALL_STATE_LEN];
        unsafe { sgv_wall_state(self.cells.as_ptr(), self.w, self.h, self.res, self.ox, self.oy, self.segs.as_ptr(), (self.segs.len() / 4) as i32, pose.as_ptr(), out.as_mut_ptr()) };
        let with = force_segs || self.seg_ver != self.seg_sent;
        self.seg_sent = self.seg_ver;
        let segs = if with {
            let v: Vec<String> = self.segs.chunks(4).map(|c| format!("[{:.4},{:.4},{:.4},{:.4}]", c[0], c[1], c[2], c[3])).collect();
            format!("[{}],\"theta\":{:.6}", v.join(","), self.theta)
        } else {
            "null".to_string()
        };
        let st: Vec<String> = out.iter().map(|x| format!("{:.5}", x)).collect();
        Some(sse("walls", &format!("{{\"segments\":{},\"state\":[{}],\"pose\":[{:.4},{:.4},{:.4}],\"max_range\":4.0}}", segs, st.join(","), pose[0], pose[1], pose[2])))
    }
    fn snapshot(&mut self) -> String {
        let mut s = String::new();
        if self.w > 0 {
            s.push_str(&map_event(self, 0, 0, self.w - 1, self.h - 1));
        }
        if let Some(v) = &self.view {
            s.push_str(&sse("view", v));
        }
        if let Some(p) = &self.pose {
            s.push_str(&pose_event(p));
        }
        if let Some(j) = &self.joints {
            s.push_str(j);
        }
        if let Some(w) = self.walls(true) {
            s.push_str(&w);
        }
        s
    }
}

/// sgview /stream: 세션의 판을 시각대로 보낸다. 시각이 바뀌면(옮김) reset + 그 시각까지의 스냅숏
pub fn serve_stream(mut s: TcpStream, st: Arc<Mutex<SgState>>, sess: String) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";
    if s.write_all(head.as_bytes()).is_err() {
        return;
    }
    let _ = s.set_nodelay(true);
    let mut cur_epoch = u64::MAX;
    let mut ep: Option<Arc<Episode>> = None;
    let mut live = Live::new();
    let mut idx = 0usize;
    let mut last_walls = Instant::now();
    let mut last_send = Instant::now();
    loop {
        let (e, t, epoch) = {
            let g = st.lock().unwrap();
            match g.sess.get(&sess) {
                Some(x) => (Some(x.ep.clone()), x.t(), x.epoch),
                None => (None, 0.0, 0),
            }
        };
        let mut out = String::new();
        match e {
            None => {}
            Some(e) => {
                if epoch != cur_epoch || ep.as_ref().map(|p| !Arc::ptr_eq(p, &e)).unwrap_or(true) {
                    // 옮김: 화면을 비우고 t 까지 다시 쌓음
                    cur_epoch = epoch;
                    ep = Some(e.clone());
                    live = Live::new();
                    idx = 0;
                    while idx < e.frames.len() && e.frames[idx].t <= t {
                        live.apply(&e.frames[idx], false);
                        idx += 1;
                    }
                    out.push_str(&sse("reset", "{}"));
                    out.push_str(&live.snapshot());
                } else {
                    // 같은 시각 안의 프레임: 자세는 마지막 것만(대역폭), 나머지는 차례대로
                    let mut last_pose: Option<String> = None;
                    while idx < e.frames.len() && e.frames[idx].t <= t {
                        let f = &e.frames[idx];
                        if let Some(ev) = live.apply(f, true) {
                            if f.ty == 1 { last_pose = Some(ev) } else { out.push_str(&ev) }
                        }
                        idx += 1;
                    }
                    if let Some(p) = last_pose {
                        out.push_str(&p);
                    }
                    if last_walls.elapsed() > Duration::from_millis(150) && (live.dirty || !out.is_empty()) {
                        last_walls = Instant::now();
                        if let Some(w) = live.walls(false) {
                            out.push_str(&w);
                        }
                    }
                }
            }
        }
        if out.is_empty() && last_send.elapsed() > Duration::from_secs(10) {
            out.push_str(": keepalive\n\n");
        }
        if !out.is_empty() {
            if s.write_all(out.as_bytes()).is_err() {
                return;
            }
            last_send = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// /api/sg/ctl?sess=&run=&stream=&id=&t=&speed=&playing= — 판 고르기·옮기기·재생. 답: 지금 시각·길이·세대
pub fn ctl(st: &Arc<Mutex<SgState>>, req: &Req, ep_of: impl Fn(&str) -> Option<Arc<Episode>>) -> String {
    let sess = req.get("sess").to_string();
    if sess.is_empty() || !sess.chars().all(|c| c.is_ascii_alphanumeric()) {
        return json!({"error": "sess must be alphanumeric"}).to_string();
    }
    let key = format!("{}|{}|{}", req.get("run"), req.get("stream"), req.get("id"));
    let mut g = st.lock().unwrap();
    let need_new = !req.get("id").is_empty() && g.sess.get(&sess).map(|x| x.key != key).unwrap_or(true);
    if need_new {
        let ep = match g.eps.get(&key) {
            Some(e) => Some(e.clone()),
            None => {
                drop(g);
                let e = ep_of(&key);
                g = st.lock().unwrap();
                if let Some(e) = &e {
                    if g.eps.len() > 6 {
                        g.eps.clear();   // 판 몇 개만 메모리에(한 판 수십 MB)
                    }
                    g.eps.insert(key.clone(), e.clone());
                }
                e
            }
        };
        let Some(ep) = ep else { return json!({"error": "cannot load episode"}).to_string() };
        let epoch = g.sess.get(&sess).map(|x| x.epoch + 1).unwrap_or(1);
        g.sess.insert(sess.clone(), Sess { ep, key: key.clone(), t0: 0.0, wall0: Instant::now(), speed: 1.0, playing: false, epoch });
    }
    let Some(x) = g.sess.get_mut(&sess) else { return json!({"error": "no session — pick an episode"}).to_string() };
    let now_t = x.t();
    if let Some(sp) = req.num("speed") {
        x.t0 = now_t;
        x.wall0 = Instant::now();
        x.speed = sp.clamp(0.05, 64.0);
    }
    if let Some(p) = req.q.get("playing") {
        let t = x.t();
        x.t0 = if t >= x.ep.duration && p == "1" { 0.0 } else { t };
        x.wall0 = Instant::now();
        if t >= x.ep.duration && p == "1" {
            x.epoch += 1;
        }
        x.playing = p == "1";
    }
    if let Some(t) = req.num("t") {
        x.t0 = t.clamp(0.0, x.ep.duration);
        x.wall0 = Instant::now();
        x.epoch += 1;
    }
    let t = x.t();
    if t >= x.ep.duration && x.playing {
        x.playing = false;
        x.t0 = x.ep.duration;
    }
    json!({"t": t, "duration": x.ep.duration, "playing": x.playing, "speed": x.speed, "epoch": x.epoch, "frames": x.ep.frames.len()}).to_string()
}

/// iframe 의 sgview 가 부르는 경로(쿠키 sgsess 로 세션). 처리했으면 true
pub fn handle_sgview(s: &mut TcpStream, req: &Req, st: &Arc<Mutex<SgState>>, sgview_html: &[u8]) -> bool {
    let sess = cookie_sess(req);
    let p = req.path.as_str();
    match p {
        "/sg/" | "/sg/index.html" => http::respond(s, 200, "text/html; charset=utf-8", sgview_html, req.gzip),
        "/api/mode" => http::json(s, "{\"live\":true}", false),
        "/api/robot" => {
            let r = st.lock().unwrap().sess.get(&sess).map(|x| x.ep.robot.clone()).unwrap_or_default();
            http::json(s, &json!({"robot": if r.is_empty() { Value::Null } else { json!(r) }}).to_string(), false)
        }
        "/api/view" => http::json(s, "{\"waiting\":true}", false),
        "/api/walls" => http::respond(s, 404, "text/plain", b"live only", false),
        "/api/map" => http::respond(s, 404, "text/plain", b"live only", false),
        _ if p.starts_with("/file/") => {
            let dir = st.lock().unwrap().sess.get(&sess).map(|x| x.ep.dir.clone());
            let rel = &p["/file/".len()..];
            let ok = !rel.split('/').any(|c| c == ".." || c.is_empty());
            match dir.filter(|d| !d.as_os_str().is_empty() && ok).and_then(|d| std::fs::read(d.join(rel)).ok()) {
                Some(b) => http::respond(s, 200, http::content_type(rel), &b, req.gzip),
                None => http::not_found(s, "no such file in the episode memory dir"),
            }
        }
        _ => return false,
    }
    true
}

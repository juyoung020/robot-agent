// 재생 탭의 sgview 화면 — **진짜 sgview**(src/scene_graph/sgview, 실시간에 쓰는 같은 바이너리·같은 서버 코드)를 쓴다. 흉내 내지 않는다.
//
//   판 = OmniGibson 기록에서 만든 <이름>.sg/ (stream.sgs = scenemap 이 보낸 sgview 스트림 + 시뮬 시각) 또는 GPU 환경 판 .trp(→ 같은 스트림 형식으로 한 번 바꿈).
//   세션(브라우저 쿠키 sgsess)마다: sgview 프로세스 하나(--ingest) + sgs_play 하나(--ctl: 시간 조종은 보내는 쪽이 한다 — seek·pause·rate, src/scene_graph/tools/realbag/sgs_play.cpp).
//   iframe 이 부르는 sgview 경로(/sg/ → /, /stream·/api/*·/file/*)는 그 프로세스로 그대로 넘긴다(역프록시). 이 파일에 sgview 화면 코드는 없다.
//   .trp 처럼 읽는 쪽 일(정책 지도 격자, 부모 화면 정보)만 여기서 한다. 동시에 sgview 는 MAX_PROCS 개, 오래 안 쓰면 끔.
use crate::http::{self, Req};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_PROCS: usize = 2;
const IDLE_S: u64 = 90;

pub struct Frame {
    pub t: f64,
    pub ty: u8,
    pub pl: Vec<u8>,
}

pub struct Episode {
    pub dir: PathBuf,        // sgview 메모리 폴더(/file/ 뿌리): .sg 면 memory/, .trp 면 빈 폴더(세션에서 만듦)
    pub sgs: PathBuf,        // 틀어 줄 스트림 파일(.sg 의 stream.sgs). 아래 sgs_bytes 가 있으면 그걸 쓴다
    pub sgs_bytes: Option<Vec<u8>>,   // 바꾼 스트림(점구름·RGB 조각 경로를 앞 프레임에 넣음 / .trp 에서 만듦)
    pub n_frames: usize,
    pub duration: f64,
    pub robot: String,
    pub info: Value,         // 부모 화면용: gt_path, underlay, cams, joint_order, map_from_world …
    /// 정책이 본 지도(GPU 근사판 G2, .trp 의 MAP_RECT): (시각, 48 B 머리 + 칸) — 진짜 scenemap 과 겹쳐 보기
    pub policy: Vec<(f64, Vec<u8>)>,
}

pub fn to_sgs(frames: &[Frame]) -> Vec<u8> {
    let mut o = b"SGS1".to_vec();
    for f in frames {
        o.extend_from_slice(&f.t.to_le_bytes());
        o.extend_from_slice(&(f.pl.len() as u32).to_le_bytes());
        o.push(f.ty);
        o.extend_from_slice(&f.pl);
    }
    o
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
    let mut changed = false;
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
                    changed = true;
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
                changed = true;
            }
        }
        frames.push(Frame { t, ty, pl });
        p += 13 + len;
    }
    let n_frames = frames.len();
    let sgs_bytes = if changed { Some(to_sgs(&frames)) } else { None };
    let duration = meta["duration"].as_f64().unwrap_or_else(|| frames.last().map(|f| f.t).unwrap_or(0.0));
    let underlay: Value = std::fs::read_to_string(dir.join("underlay.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    let info = json!({
        "kind": "sg", "pipeline": meta["pipeline"], "meta": meta["meta"], "gt_path": meta["gt_path"], "map_from_world": meta["map_from_world"], "cams": meta["cams"], "wcams": meta["wcams"],
        "joint_order": meta["joint_order"], "underlay": underlay, "stream": meta["stream"], "og_run": meta["og_run"], "n_objects": meta["n_objects"],
        "robot": meta["robot"], "window_origin": meta["window_origin"], "world": meta["world"], "policy_trp": meta["policy_trp"], "source": meta["source"],
    });
    let mut policy = vec![];
    if let Some(pt) = meta["policy_trp"].as_str() {
        if let Ok(b) = std::fs::read(dir.join(pt)) {
            if let Some(h) = trainfmt::trp::read_head(&b) {
                let dt = h["dt"].as_f64().unwrap_or(0.1);
                if let Some(s) = h["sections"].as_array().and_then(|a| a.iter().find(|s| s["name"] == "map")) {
                    let (o, l) = (s["off"].as_u64().unwrap_or(0) as usize, s["len"].as_u64().unwrap_or(0) as usize);
                    let mut p = o;
                    while p + 52 <= o + l && o + l <= b.len() {
                        let fr = u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as f64;
                        let i32at = |q: usize| i32::from_le_bytes(b[q..q + 4].try_into().unwrap());
                        let n = ((i32at(p + 44) - i32at(p + 36) + 1) * (i32at(p + 48) - i32at(p + 40) + 1)) as usize;
                        policy.push((fr * dt, b[p + 4..p + 52 + n].to_vec()));
                        p += 52 + n;
                    }
                }
            }
        }
    }
    let mut info = info;
    info["has_policy_map"] = json!(!policy.is_empty());
    Some(Episode { dir: dir.join("memory"), sgs: dir.join("stream.sgs"), sgs_bytes, n_frames, duration, robot: meta["robot"].as_str().unwrap_or("limo_omx").to_string(), info, policy })
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
        boxes.push(json!({"kind": bx["kind"], "cat": bx["name"], "name": bx["obj"], "c": bx["c"], "h": bx["h"], "yaw": bx["yaw"]}));
    }
    let info = json!({"kind": "trp", "meta": h["meta"], "gt_path": gt, "map_from_world": [1, 0, 0, 0], "cams": [],
        "joint_order": ["", "", "", "", "", "", "omx_joint1", "omx_joint2", "omx_joint3", "omx_joint4", "omx_joint5", "omx_gripper_joint_1", "front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"],
        "underlay": {"scene": h["scene"]["name"], "boxes": boxes, "rooms": h["scene"]["rooms"], "places": [], "picks": [], "task_objects": []}});
    let n_frames = frames.len();
    Some(Episode { dir: PathBuf::new(), sgs: PathBuf::new(), sgs_bytes: Some(to_sgs(&frames)), n_frames, duration, robot: "limo_omx".into(), info, policy: vec![] })
}

// ---------------------------------------------------------------- 세션·프로세스

struct PState {
    t: f64,
    playing: bool,
    speed: f64,
    got: Instant,
    t0: f64,   // 자료 시작 시각(sgs_play 의 I 줄)
}

pub struct Sess {
    pub ep: Arc<Episode>,
    pub key: String,
    port: u16,
    sgview: Child,
    play: Child,
    stdin: ChildStdin,
    ps: Arc<Mutex<PState>>,
    last: Instant,
    tmp: Vec<PathBuf>,
}
impl Sess {
    fn send(&mut self, cmd: &str) {
        let _ = writeln!(self.stdin, "{}", cmd);
        let _ = self.stdin.flush();
    }
    fn kill(&mut self) {
        let _ = self.play.kill();
        let _ = self.sgview.kill();
        let _ = self.play.wait();
        let _ = self.sgview.wait();
        for p in &self.tmp {
            if p.is_dir() { let _ = std::fs::remove_dir_all(p); } else { let _ = std::fs::remove_file(p); }
        }
        pids_forget(&[self.play.id(), self.sgview.id()]);
    }
    /// 지금 자료 시각(상태 줄 0.1 s 마다 + 그 뒤 벽시계로 이어 셈)
    fn t(&self) -> f64 {
        let p = self.ps.lock().unwrap();
        let t = if p.playing { p.t + p.got.elapsed().as_secs_f64() * p.speed } else { p.t };
        t.clamp(0.0, self.ep.duration)
    }
}
impl Drop for Sess {
    fn drop(&mut self) {
        self.kill();
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

/// 학습 뷰어 작업 폴더: $RA_TRAINVIEW_WORK, 없으면 <robot-agent>/data/trainview_work (config/paths.env 와 같은 규칙)
pub fn work_root() -> PathBuf {
    if let Ok(d) = std::env::var("RA_TRAINVIEW_WORK") { if !d.is_empty() { return PathBuf::from(d); } }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/trainview_work")
}
fn work_dir() -> PathBuf {
    let d = work_root().join("sgplay");
    let _ = std::fs::create_dir_all(&d);
    d
}
// 바이너리는 build_deps.sh 가 robot-agent 소스에서 만들어 $TRAINVIEW_DEPS(기본 data/trainview_work/deps)에 링크해 둔 것만 쓴다
fn deps() -> PathBuf {
    std::env::var("TRAINVIEW_DEPS").map(PathBuf::from).unwrap_or_else(|_| work_root().join("deps"))
}
fn sgview_bin() -> PathBuf {
    deps().join("sgview")
}
fn sgs_play_bin() -> PathBuf {
    deps().join("sgs_play")
}
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").ok().and_then(|l| l.local_addr().ok()).map(|a| a.port()).unwrap_or(0)
}

// 내가 띄운 프로세스 번호(다음 시작 때 남은 것을 끄려고). 번호만 — 이름 검사로 남의 프로세스는 건드리지 않음
fn pids_file() -> PathBuf {
    work_root().join("run/sgview_pids")
}
fn pids_add(p: &[u32]) {
    let mut t = std::fs::read_to_string(pids_file()).unwrap_or_default();
    for x in p { t.push_str(&format!("{}\n", x)); }
    let _ = std::fs::write(pids_file(), t);
}
fn pids_forget(p: &[u32]) {
    let t = std::fs::read_to_string(pids_file()).unwrap_or_default();
    let keep: Vec<&str> = t.lines().filter(|l| !p.iter().any(|x| l.trim() == x.to_string())).collect();
    let _ = std::fs::write(pids_file(), keep.join("\n") + "\n");
}
/// 서버 시작 때: 지난번에 남은 sgview·sgs_play 끄기(cmdline 에 그 이름이 있는 번호만)
pub fn kill_leftovers() {
    for l in std::fs::read_to_string(pids_file()).unwrap_or_default().lines() {
        let Ok(pid) = l.trim().parse::<u32>() else { continue };
        let cmd = std::fs::read(format!("/proc/{}/cmdline", pid)).unwrap_or_default();
        let c = String::from_utf8_lossy(&cmd);
        if c.contains("sgview") || c.contains("sgs_play") {
            let _ = Command::new("kill").arg(pid.to_string()).status();
        }
    }
    let _ = std::fs::write(pids_file(), "");
}

fn start(sess: &str, key: &str, ep: Arc<Episode>) -> Result<Sess, String> {
    let wd = work_dir();
    let mut tmp = vec![];
    let sgs = match &ep.sgs_bytes {
        Some(b) => {
            let p = wd.join(format!("{}.sgs", sess));
            std::fs::write(&p, b).map_err(|e| e.to_string())?;
            tmp.push(p.clone());
            p
        }
        None => ep.sgs.clone(),
    };
    let mem = if ep.dir.as_os_str().is_empty() || !ep.dir.is_dir() {
        let d = wd.join(format!("{}_mem", sess));
        let _ = std::fs::create_dir_all(&d);
        tmp.push(d.clone());
        d
    } else {
        ep.dir.clone()
    };
    let (port, ing) = (free_port(), free_port());
    if port == 0 || ing == 0 {
        return Err("no free port".into());
    }
    let sgv = Command::new(sgview_bin()).arg(&mem).args(["--port", &port.to_string(), "--ingest", &format!("127.0.0.1:{}", ing)])
        .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().map_err(|e| format!("sgview: {}", e))?;
    let mut sgview = sgv;
    // 포트가 열릴 때까지
    let t0 = Instant::now();
    while TcpStream::connect(("127.0.0.1", ing)).is_err() {
        if t0.elapsed() > Duration::from_secs(8) {
            let _ = sgview.kill();
            return Err("sgview did not start".into());
        }
        std::thread::sleep(Duration::from_millis(30));
    }
    let mut play = Command::new(sgs_play_bin()).arg(&sgs).arg(format!("127.0.0.1:{}", ing)).args(["--ctl", "--rate", "1", "--pose-hz", "60"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().map_err(|e| { let _ = sgview.kill(); format!("sgs_play: {}", e) })?;
    pids_add(&[sgview.id(), play.id()]);
    let stdin = play.stdin.take().unwrap();
    let out = play.stdout.take().unwrap();
    let ps = Arc::new(Mutex::new(PState { t: 0.0, playing: true, speed: 1.0, got: Instant::now(), t0: 0.0 }));
    let ps2 = ps.clone();
    std::thread::spawn(move || {
        for l in BufReader::new(out).lines().flatten() {
            let v: Vec<&str> = l.split_whitespace().collect();
            let mut p = ps2.lock().unwrap();
            if v.len() == 3 && v[0] == "I" {
                p.t0 = v[1].parse().unwrap_or(0.0);
            } else if v.len() == 4 && v[0] == "T" {
                p.t = v[1].parse().unwrap_or(p.t);
                p.playing = v[2] == "1";
                p.speed = v[3].parse().unwrap_or(1.0);
                p.got = Instant::now();
            }
        }
    });
    Ok(Sess { ep, key: key.to_string(), port, sgview, play, stdin, ps, last: Instant::now(), tmp })
}

/// 오래 안 쓴 세션 끄기(서버 스레드가 가끔 부름)
pub fn gc(st: &Arc<Mutex<SgState>>) {
    let mut g = st.lock().unwrap();
    let dead: Vec<String> = g.sess.iter().filter(|(_, s)| s.last.elapsed() > Duration::from_secs(IDLE_S)).map(|(k, _)| k.clone()).collect();
    for k in dead {
        g.sess.remove(&k);
    }
}

/// /api/sg/ctl?sess=&run=&stream=&id=&t=&speed=&playing= — 판 고르기·옮기기·재생. 시간 조종은 sgs_play(--ctl)로 보낸다. 답: 지금 시각·길이
pub fn ctl(st: &Arc<Mutex<SgState>>, req: &Req, ep_of: impl Fn(&str) -> Option<Arc<Episode>>) -> String {
    let sess = req.get("sess").to_string();
    if sess.is_empty() || !sess.chars().all(|c| c.is_ascii_alphanumeric()) {
        return json!({"error": "sess must be alphanumeric"}).to_string();
    }
    let key = format!("{}|{}|{}", req.get("run"), req.get("stream"), req.get("id"));
    let need_new = !req.get("id").is_empty() && st.lock().unwrap().sess.get(&sess).map(|x| x.key != key).unwrap_or(true);
    if need_new {
        let cached = st.lock().unwrap().eps.get(&key).cloned();
        let ep = match cached {
            Some(e) => Some(e),
            None => {
                let e = ep_of(&key);
                let mut g = st.lock().unwrap();
                if let Some(e) = &e {
                    if g.eps.len() > 4 {
                        g.eps.clear();
                    }
                    g.eps.insert(key.clone(), e.clone());
                }
                e
            }
        };
        let Some(ep) = ep else { return json!({"error": "cannot load episode"}).to_string() };
        let mut g = st.lock().unwrap();
        g.sess.remove(&sess);   // 판을 바꾸면 이전 sgview·sgs_play 끔
        while g.sess.len() >= MAX_PROCS {
            let old = g.sess.iter().min_by_key(|(_, s)| s.last).map(|(k, _)| k.clone()).unwrap();
            g.sess.remove(&old);
        }
        match start(&sess, &key, ep) {
            Ok(s) => { g.sess.insert(sess.clone(), s); }
            Err(e) => return json!({"error": e}).to_string(),
        }
    }
    let mut g = st.lock().unwrap();
    let Some(x) = g.sess.get_mut(&sess) else { return json!({"error": "no session — pick an episode"}).to_string() };
    x.last = Instant::now();
    let dur = x.ep.duration;
    let now_t = x.t();
    if let Some(sp) = req.num("speed") {
        let sp = sp.clamp(0.05, 64.0);
        x.send(&format!("rate {}", sp));
        let mut p = x.ps.lock().unwrap();
        p.t = now_t; p.got = Instant::now(); p.speed = sp;
    }
    if let Some(pl) = req.q.get("playing") {
        let on = pl == "1";
        if on && now_t >= dur - 1e-3 {
            x.send("seek 0");
            let mut p = x.ps.lock().unwrap();
            p.t = 0.0; p.got = Instant::now();
        }
        x.send(if on { "play" } else { "pause" });
        let mut p = x.ps.lock().unwrap();
        if !on { p.t = now_t; }
        p.playing = on; p.got = Instant::now();
    }
    if let Some(t) = req.num("t") {
        let t = t.clamp(0.0, dur);
        x.send(&format!("seek {:.4}", t));
        let mut p = x.ps.lock().unwrap();
        p.t = t; p.got = Instant::now();
    }
    let t = x.t();
    let (playing, speed) = { let p = x.ps.lock().unwrap(); (p.playing && t < dur - 1e-3, p.speed) };
    json!({"t": t, "duration": dur, "playing": playing, "speed": speed, "frames": x.ep.n_frames}).to_string()
}

/// iframe 이 부르는 sgview 경로를 그 세션의 sgview 로 그대로 넘긴다(역프록시). 처리했으면 true
pub fn proxy(s: &mut TcpStream, req: &Req, st: &Arc<Mutex<SgState>>) -> bool {
    let p = req.path.as_str();
    let to = if p == "/sg/" || p == "/sg/index.html" { Some("/".to_string()) }
        else if matches!(p, "/stream" | "/api/mode" | "/api/robot" | "/api/view" | "/api/map" | "/api/walls" | "/api/depth") || p.starts_with("/file/") { Some(req.target.clone()) }
        else { None };
    let Some(to) = to else { return false };
    let sess = cookie_sess(req);
    let port = {
        let mut g = st.lock().unwrap();
        match g.sess.get_mut(&sess) {
            Some(x) => { x.last = Instant::now(); x.port }
            None => return false,
        }
    };
    let to = if p == "/sg/" || p == "/sg/index.html" { to } else { to };
    let Ok(mut up) = TcpStream::connect(("127.0.0.1", port)) else { http::not_found(s, "sgview not running"); return true };
    let _ = up.set_nodelay(true);
    if up.write_all(format!("GET {} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\r\n", to, port).as_bytes()).is_err() {
        return true;
    }
    let _ = s.set_nodelay(true);
    let mut buf = [0u8; 65536];
    loop {
        match up.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if s.write_all(&buf[..n]).is_err() {
                    break;
                }
                let _ = s.flush();
            }
        }
    }
    true
}

/// 정책이 본 지도(G2 근사판)의 시각 t 까지 쌓은 격자 — 회색(sgview cell_grey 와 같은 색) base64 + 자리(세계 좌표)
pub fn policy_grid(ep: &Episode, t: f64) -> Value {
    let (mut w, mut h, mut res, mut ox, mut oy) = (0i32, 0i32, 0.05f64, 0f64, 0f64);
    let mut cells: Vec<i8> = vec![];
    for (ft, pl) in &ep.policy {
        if *ft > t + 1e-6 {
            break;
        }
        if pl.len() < 48 {
            continue;
        }
        let i32at = |o: usize| i32::from_le_bytes(pl[o..o + 4].try_into().unwrap());
        let f64at = |o: usize| f64::from_le_bytes(pl[o..o + 8].try_into().unwrap());
        let (nw, nh, nres, nox, noy) = (i32at(0), i32at(4), f64at(8), f64at(16), f64at(24));
        let (x0, y0, x1, y1) = (i32at(32), i32at(36), i32at(40), i32at(44));
        if nw <= 0 || nh <= 0 || x0 < 0 || y0 < 0 || x1 >= nw || y1 >= nh || x1 < x0 || y1 < y0 || (x1 - x0 + 1) as usize * (y1 - y0 + 1) as usize + 48 != pl.len() {
            continue;
        }
        if nw != w || nh != h || nres != res || nox != ox || noy != oy {
            w = nw; h = nh; res = nres; ox = nox; oy = noy;
            cells = vec![-1i8; (w * h) as usize];
        }
        let rw = (x1 - x0 + 1) as usize;
        for y in y0..=y1 {
            let d = (y * w + x0) as usize;
            let s = 48 + (y - y0) as usize * rw;
            for k in 0..rw {
                cells[d + k] = pl[s + k] as i8;
            }
        }
    }
    if w <= 0 {
        return json!({"why": "no policy map"});
    }
    let g: Vec<u8> = cells.iter().map(|&v| { let v = v as i32; if v < 0 { 205 } else if v >= 65 { 0 } else if v <= 25 { 254 } else { (254 - (v * 254) / 100) as u8 } }).collect();
    json!({"w": w, "h": h, "res": res, "ox": ox, "oy": oy, "b64": b64(&g)})
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

// ---------------------------------------------------------------- 파이프라인 최신 여부 · 다시 돌리기

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
/// 지금 인지 소스(robot-agent src/scene_graph 의 scenemap·runtime·ovdet·clip·da·slam_carto 트리 해시) — og_replay_lib.py pipeline_info 와 같은 식. 60 s 캐시
pub fn pipeline_now() -> Value {
    static C: Mutex<Option<(Instant, Value)>> = Mutex::new(None);
    let mut g = C.lock().unwrap();
    if let Some((t, v)) = g.as_ref() {
        if t.elapsed() < Duration::from_secs(60) {
            return v.clone();
        }
    }
    let root = repo_root();
    let git = |a: &str| Command::new("git").arg("-C").arg(&root).args(["rev-parse", a]).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let mut trees = serde_json::Map::new();
    for d in ["scenemap", "runtime", "ovdet", "clip", "da", "slam_carto"] {
        trees.insert(d.into(), json!(git(&format!("HEAD:src/scene_graph/{}", d))));
    }
    let v = json!({"git": git("HEAD"), "trees": trees});
    *g = Some((Instant::now(), v.clone()));
    v
}

/// 이 판을 og_queue 맨 앞에 다시 넣는다(옛 판 폴더는 .prev 로 비켜 둠 — 새 판이 되면 지워도 됨). 답: {ok|error}
pub fn rerun(dir: &Path, id: &str) -> Value {
    let Some(base) = id.strip_suffix("_og.sg") else { return json!({"error": "only *_og.sg episodes can be re-run"}) };
    let trp = dir.join(format!("{}.trp", base));
    if !trp.is_file() {
        return json!({"error": "source .trp is gone"});
    }
    let q = work_root().join("og_queue");
    let sg = dir.join(id);
    let prev = dir.join(format!("{}.prev", id));
    if sg.is_dir() {
        let _ = std::fs::remove_dir_all(&prev);
        if std::fs::rename(&sg, &prev).is_err() {
            return json!({"error": "cannot move the old episode aside"});
        }
    }
    let name = format!("000000{:04}_rerun_{}.job", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() % 10000).unwrap_or(0), base.chars().filter(|c| c.is_ascii_alphanumeric()).take(24).collect::<String>());
    if std::fs::write(q.join(&name), json!({"trp": trp.to_string_lossy(), "force": true}).to_string() + "\n").is_err() {
        return json!({"error": "cannot write the queue job"});
    }
    let w = repo_root().join("training/viewer/tools/og_replay/og_queue.sh");
    let _ = Command::new("setsid").arg("-f").arg(&w).arg(&q).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
    json!({"ok": true, "job": name})
}

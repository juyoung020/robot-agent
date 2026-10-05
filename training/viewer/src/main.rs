// trainview — 학습 뷰어 서버 (docs/map_vla/TRAIN_VIEWER.md 5절). 읽기 전용: 실행 폴더의 파일만 읽고 아무것도 쓰지 않는다.
// 틀은 sgview 와 같다: std TcpListener + 연결마다 스레드, 요청 줄을 직접 읽음, GET 만, Connection: close, SSE 는 /api/live.
//
//   trainview --root training/runs/ppo --root data/trainview_work/student_pnp [--port 7810] [--bind 127.0.0.1]
mod http;
mod live;
mod replay;
mod runs;
mod sg;
mod table;
mod tail;

use http::Req;
use runs::{Root, RunRef};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

include!(concat!(env!("OUT_DIR"), "/assets.rs"));

pub struct App {
    pub roots: Vec<Root>,
    runs: Mutex<(Instant, Arc<Vec<RunRef>>)>,
    prog: Mutex<HashMap<String, Arc<Mutex<tail::Progress>>>>,
    eps: Mutex<HashMap<String, Arc<Mutex<table::Episodes>>>>,
    pub heads: Mutex<replay::HeadCache>,
    lines: Mutex<tail::LineCount>,
    disk: Mutex<runs::DiskCache>,
    pub sg: Arc<Mutex<sg::SgState>>,
    archive: Mutex<runs::Archive>,
}

impl App {
    /// 실행 목록(2 초 캐시)
    pub fn runs(&self) -> Arc<Vec<RunRef>> {
        let mut g = self.runs.lock().unwrap();
        if g.0.elapsed() > Duration::from_secs(2) || g.1.is_empty() {
            *g = (Instant::now(), Arc::new(runs::discover(&self.roots)));
        }
        g.1.clone()
    }
    pub fn find(&self, id: &str) -> Option<RunRef> {
        if let Some(r) = self.runs().iter().find(|r| r.id == id) {
            return Some(r.clone());
        }
        // 막 생긴 실행: 캐시를 버리고 한 번 더
        self.runs.lock().unwrap().0 = Instant::now() - Duration::from_secs(60);
        self.runs().iter().find(|r| r.id == id).cloned()
    }
    pub fn progress(&self, r: &RunRef) -> Arc<Mutex<tail::Progress>> {
        let a = self.prog.lock().unwrap().entry(r.id.clone()).or_insert_with(|| Arc::new(Mutex::new(tail::Progress::new(r.dir.join("progress.jsonl"))))).clone();
        a.lock().unwrap().update();
        a
    }
    pub fn episodes(&self, r: &RunRef, stream: &str) -> Arc<Mutex<table::Episodes>> {
        let key = format!("{}|{}", r.id, stream);
        let a = self
            .eps
            .lock()
            .unwrap()
            .entry(key)
            .or_insert_with(|| {
                let edges: Vec<f64> = runs::read_meta(&r.dir)
                    .and_then(|m| m.get("completion_bins").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|v| v.as_f64()).collect()))
                    .unwrap_or_default();
                Arc::new(Mutex::new(table::Episodes::new(runs::episodes_base(&r.dir, stream), edges)))
            })
            .clone();
        a.lock().unwrap().update();
        a
    }
    pub fn rewound_of(&self, id: &str) -> Option<usize> {
        self.prog.lock().unwrap().get(id).map(|p| p.lock().unwrap().rewound)
    }
    pub fn runs_json(&self) -> String {
        let rs = self.runs();
        let mut lines = self.lines.lock().unwrap();
        let mut disk = self.disk.lock().unwrap();
        let mut rows: Vec<Value> = rs.iter().map(|r| runs::summary(r, &self.roots, &mut lines, &mut disk, self.rewound_of(&r.id))).collect();
        // 보관함(옛 v2 이전 실행)·학생의 교사 실행
        let metas: HashMap<String, Value> = rs.iter().filter_map(|r| Some((r.id.clone(), runs::read_meta(&r.dir)?))).collect();
        let mut ar = self.archive.lock().unwrap();
        for (row, r) in rows.iter_mut().zip(rs.iter()) {
            let m = metas.get(&r.id).cloned().unwrap_or(json!({}));
            row["archive"] = json!(ar.is_old(&m));
            if let Some((tid, file)) = runs::teacher_run(&m, &rs, &metas) {
                row["teacher_run"] = json!(tid);
                row["teacher_ckpt"] = json!(file);
            }
        }
        json!({
            "runs": rows,
            "latest": runs::latest(&self.roots, &rs),
            "roots": self.roots.iter().map(|r| json!({"label": r.label, "path": r.path.display().to_string()})).collect::<Vec<_>>(),
            "now": runs::now(),
            "live_s": runs::LIVE_S,
        })
        .to_string()
    }
    /// 10 분 안 본 실행의 저장소를 내린다(5.5)
    fn evict(&self) {
        let old = |t: Instant| t.elapsed() > Duration::from_secs(600);
        self.prog.lock().unwrap().retain(|_, p| p.try_lock().map(|g| !old(g.last_access)).unwrap_or(true));
        self.eps.lock().unwrap().retain(|_, p| p.try_lock().map(|g| !old(g.last_access)).unwrap_or(true));
    }
}

fn asset(name: &str) -> Option<&'static [u8]> {
    FILES.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
}

fn run_of(app: &App, s: &mut TcpStream, req: &Req) -> Option<RunRef> {
    let id = req.get("run");
    match app.find(id) {
        Some(r) => Some(r),
        None => {
            http::not_found(s, &format!("no run '{}' under the roots", id));
            None
        }
    }
}

fn stream_of(s: &mut TcpStream, req: &Req) -> Option<String> {
    let st = req.get("stream");
    let st = if st.is_empty() { "main" } else { st };
    if !trainfmt::safe_name(st) {
        http::bad(s, "stream name must match [\\w.=-]+");
        return None;
    }
    Some(st.to_string())
}

fn handle(mut s: TcpStream, app: Arc<App>) {
    let Some(req) = http::read_request(&s) else { return };
    let gz = req.gzip;
    let p = req.path.as_str();
    // 재생 탭의 sgview(iframe): 이 세션의 진짜 sgview 프로세스로 그대로 넘김(src/sg.rs)
    if sg::proxy(&mut s, &req, &app.sg) {
        return;
    }
    match p {
        "/" | "/index.html" => http::respond(&mut s, 200, "text/html; charset=utf-8", asset("index.html").unwrap_or(b"no index"), gz),
        "/api/runs" => http::json(&mut s, &app.runs_json(), gz),
        "/api/meta" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            match std::fs::read_to_string(r.dir.join("run.json")) {
                Ok(t) => http::json(&mut s, &t, gz),
                Err(_) => http::json(&mut s, &json!({"why": format!("no run.json in {}", r.dir.display())}).to_string(), gz),
            }
        }
        "/api/progress" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let pa = app.progress(&r);
            let pg = pa.lock().unwrap();
            if pg.total() == 0 {
                let why = if r.dir.join("progress.jsonl").exists() { "progress.jsonl has no complete lines yet" } else { "no progress.jsonl in this run" };
                http::json(&mut s, &json!({"start": 0, "total": 0, "sig": pg.sig, "x": {}, "cols": {}, "keys": [], "why": why}).to_string(), gz);
                return;
            }
            let keys = req.get("keys");
            if keys.is_empty() {
                http::json(&mut s, &pg.keys_json(), gz);
                return;
            }
            let ks: Vec<String> = if keys == "*" { pg.keys.clone() } else { keys.split(',').filter(|k| !k.is_empty()).map(|k| k.to_string()).collect() };
            let mut from = req.num("from").unwrap_or(0.0) as usize;
            if let Some(sig) = req.num("sig") {
                if sig as u64 != pg.sig {
                    from = 0; // 파일이 바뀌었다: 처음부터
                }
            }
            let mp = req.num("max_points").unwrap_or(0.0) as usize;
            http::json(&mut s, &pg.rows_json(Some(&ks), from, mp), gz);
        }
        "/api/episodes" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let Some(st) = stream_of(&mut s, &req) else { return };
            let ea = app.episodes(&r, &st);
            let e = ea.lock().unwrap();
            if e.total() == 0 {
                let why = format!("no episode lines in {}.jsonl", runs::episodes_base(&r.dir, &st).display());
                http::json(&mut s, &json!({"start": 0, "total": 0, "rows": [], "why": why}).to_string(), gz);
                return;
            }
            http::json(&mut s, &e.rows_json(req.num("from").unwrap_or(0.0) as usize), gz);
        }
        "/api/table" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let Some(st) = stream_of(&mut s, &req) else { return };
            let ea = app.episodes(&r, &st);
            let e = ea.lock().unwrap();
            if e.total() == 0 {
                http::json(&mut s, &json!({"rows": [], "cols": [], "cells": [], "why": "no episodes.jsonl (sample) in this stream"}).to_string(), gz);
                return;
            }
            let rows = if req.get("rows").is_empty() { "skill" } else { req.get("rows") };
            let cols = if req.get("cols").is_empty() { "bin" } else { req.get("cols") };
            let filt = [("home", req.get("home")), ("stage", req.get("stage")), ("map_mode", req.get("map_mode")), ("driver", req.get("driver")), ("skill", req.get("skill"))];
            let upto = req.num("upto").unwrap_or(f64::NAN);
            http::json(&mut s, &e.table_json(rows, cols, req.num("last").unwrap_or(0.0) as usize, &filt, upto), gz);
        }
        "/api/replays" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let Some(st) = stream_of(&mut s, &req) else { return };
            let d = runs::stream_dir(&r.dir, &st).join("replays");
            let v = app.heads.lock().unwrap().list(&d);
            http::json(&mut s, &v.to_string(), gz);
        }
        "/api/replay" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let Some(st) = stream_of(&mut s, &req) else { return };
            let id = req.get("id");
            if !trainfmt::safe_name(id) || !(id.ends_with(".trp") || id.ends_with(".sg")) {
                return http::bad(&mut s, "id must be a .trp file or .sg folder name");
            }
            let mut p = runs::stream_dir(&r.dir, &st).join("replays").join(id);
            if id.ends_with(".sg") {
                // sgview 판(OG·진짜 scenemap): 안의 정책 기록(meta.json policy_trp = episode.trp) — 입력 패널·조건 그림용
                let pt = std::fs::read_to_string(p.join("meta.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())
                    .and_then(|m| m["policy_trp"].as_str().map(|x| x.to_string())).unwrap_or_else(|| "episode.trp".into());
                if !trainfmt::safe_name(&pt) {
                    return http::bad(&mut s, "bad policy_trp");
                }
                p = p.join(pt);
                if !p.exists() {
                    // 정책 기록이 없는 판(OmniGibson 탐사 기록 등): 오류가 아니라 "없음" — 204
                    return http::respond(&mut s, 204, "application/octet-stream", b"", false);
                }
            }
            match std::fs::read(&p) {
                Ok(b) => http::respond(&mut s, 200, "application/octet-stream", &b, gz),
                Err(_) => http::not_found(&mut s, &format!("no replay {}", id)),
            }
        }
        "/api/scene_mesh" => {
            // BEHAVIOR 집 진짜 메시(줄인 것, tools/scene_mesh/export_mesh.py 캐시): 재생 탭 바탕 층
            let sc = req.get("scene");
            if !trainfmt::safe_name(sc) || sc.is_empty() {
                return http::bad(&mut s, "scene must be a scene name");
            }
            let dir = std::env::var("TRAINVIEW_SCENE_MESH").unwrap_or_else(|_| crate::sg::work_root().join("scene_mesh").to_string_lossy().into_owned());
            match std::fs::read(std::path::Path::new(&dir).join(format!("{}.smsh", sc))) {
                Ok(b) => http::respond(&mut s, 200, "application/octet-stream", &b, gz),
                Err(_) => http::not_found(&mut s, &format!("no mesh for {}", sc)),
            }
        }
        "/api/evals" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let mut rows: Vec<(String, Value)> = vec![];
            if let Ok(rd) = std::fs::read_dir(r.dir.join("evals")) {
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().to_string();
                    if !n.ends_with(".json") {
                        continue;
                    }
                    if let Some(v) = std::fs::read_to_string(e.path()).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
                        rows.push((n, v));
                    }
                }
            }
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            let why = if rows.is_empty() { json!("no evals/*.json in this run") } else { Value::Null };
            http::json(&mut s, &json!({"evals": rows.into_iter().map(|(n, v)| json!({"file": n, "eval": v})).collect::<Vec<_>>(), "why": why}).to_string(), gz);
        }
        "/api/events" => {
            // 학습기의 사람이 읽는 사건 기록(events.txt, ppo_run): 커리큘럼 넘어가기·체크포인트 — 끝 200 줄
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let t = std::fs::read_to_string(r.dir.join("events.txt")).unwrap_or_default();
            let ls: Vec<&str> = t.lines().collect();
            let k = ls.len().saturating_sub(200);
            http::json(&mut s, &json!({"lines": &ls[k..], "total": ls.len()}).to_string(), gz);
        }
        "/api/live" => live::serve(s, app, &req),
        "/api/sg/ctl" => {
            let a2 = app.clone();
            let body = sg::ctl(&app.sg, &req, move |key| {
                let mut it = key.splitn(3, '|');
                let (run, st, id) = (it.next()?, it.next()?, it.next()?);
                let r = a2.find(run)?;
                if !trainfmt::safe_name(st) || !trainfmt::safe_name(id) {
                    return None;
                }
                let p = runs::stream_dir(&r.dir, st).join("replays").join(id);
                let e = if id.ends_with(".sg") { sg::load_sg(&p) } else { sg::load_trp(&p) };
                e.map(Arc::new)
            });
            http::json(&mut s, &body, gz);
        }
        "/api/sg/pipeline" => http::json(&mut s, &sg::pipeline_now().to_string(), gz),
        "/api/sg/rerun" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let Some(st) = stream_of(&mut s, &req) else { return };
            let id = req.get("id");
            if !trainfmt::safe_name(id) {
                return http::bad(&mut s, "bad id");
            }
            let body = sg::rerun(&runs::stream_dir(&r.dir, &st).join("replays"), id);
            http::json(&mut s, &body.to_string(), false);
        }
        "/api/sg/info" => {
            // 부모 화면용: 참 궤적, 바탕 층(BEHAVIOR 집), 카메라 그림 목록, 관절 순서
            let sess = req.get("sess").to_string();
            let info = app.sg.lock().unwrap().sess.get(&sess).map(|x| json!({"info": x.ep.info, "duration": x.ep.duration, "frames": x.ep.n_frames}));
            http::json(&mut s, &info.unwrap_or(json!({"error": "no session"})).to_string(), gz);
        }
        "/api/sg/policymap" => {
            let sess = req.get("sess").to_string();
            let ep = app.sg.lock().unwrap().sess.get(&sess).map(|x| x.ep.clone());
            let t = req.num("t").unwrap_or(0.0);
            let body = ep.map(|e| sg::policy_grid(&e, t)).unwrap_or(json!({"why": "no session"}));
            http::json(&mut s, &body.to_string(), gz);
        }
        "/api/sg/cam" => {
            let Some(r) = run_of(&app, &mut s, &req) else { return };
            let Some(st) = stream_of(&mut s, &req) else { return };
            let (id, f) = (req.get("id"), req.get("file"));
            let ok = trainfmt::safe_name(id) && id.ends_with(".sg") && f.starts_with("cam/") && trainfmt::safe_name(&f[4..]);
            match if ok { std::fs::read(runs::stream_dir(&r.dir, &st).join("replays").join(id).join(f)).ok() } else { None } {
                Some(b) => http::respond(&mut s, 200, "image/jpeg", &b, false),
                None => http::not_found(&mut s, "no such camera frame"),
            }
        }
        _ => {
            let name = p.trim_start_matches('/');
            match asset(name) {
                Some(b) => http::respond(&mut s, 200, http::content_type(name), b, gz),
                None => http::not_found(&mut s, "no such path"),
            }
        }
    }
}

fn main() {
    let mut roots: Vec<PathBuf> = vec![];
    let (mut port, mut bind) = (7810u16, "127.0.0.1".to_string());
    let mut archive_before: Option<String> = None;   // 이 커밋의 자손이 아닌 실행 = 보관함(기본: "관측·신경망 v2" 커밋)
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--root" => {
                if let Some(r) = it.next() {
                    roots.push(PathBuf::from(r));
                }
            }
            "--port" => port = it.next().and_then(|v| v.parse().ok()).unwrap_or(port),
            "--bind" => bind = it.next().unwrap_or(bind),
            "--archive-before" => archive_before = it.next(),
            "-h" | "--help" => {
                eprintln!("usage: trainview --root DIR [--root DIR …] [--port 7810] [--bind 127.0.0.1]");
                return;
            }
            _ => roots.push(PathBuf::from(a)),
        }
    }
    if roots.is_empty() {
        let h = std::env::var("HOME").unwrap_or_default();
        for d in ["rl_work/runs", "bc_work/runs"] {
            let p = PathBuf::from(&h).join(d);
            if p.is_dir() {
                roots.push(p);
            }
        }
    }
    if roots.is_empty() {
        eprintln!("usage: trainview --root DIR [--root DIR …] [--port 7810] [--bind 127.0.0.1]");
        std::process::exit(2);
    }
    // 뿌리 이름표 = 폴더 이름(겹치면 ~2, ~3)
    let mut rs: Vec<Root> = vec![];
    for p in roots {
        let p = std::fs::canonicalize(&p).unwrap_or(p);
        let base = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or("root".into());
        let mut label = base.clone();
        let mut k = 2;
        while rs.iter().any(|r| r.label == label) {
            label = format!("{}~{}", base, k);
            k += 1;
        }
        rs.push(Root { label, path: p });
    }
    sg::kill_leftovers();
    let app = Arc::new(App {
        roots: rs,
        runs: Mutex::new((Instant::now(), Arc::new(vec![]))),
        prog: Mutex::new(HashMap::new()),
        eps: Mutex::new(HashMap::new()),
        heads: Mutex::new(replay::HeadCache::default()),
        lines: Mutex::new(tail::LineCount::default()),
        disk: Mutex::new(runs::DiskCache(HashMap::new())),
        sg: Arc::new(Mutex::new(sg::SgState::default())),
        archive: Mutex::new(runs::Archive::new(PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")), archive_before)),
    });
    { let st = app.sg.clone(); std::thread::spawn(move || loop { std::thread::sleep(std::time::Duration::from_secs(15)); sg::gc(&st); }); };
    let l = TcpListener::bind((bind.as_str(), port)).unwrap_or_else(|e| {
        eprintln!("cannot listen on {}:{}: {}", bind, port, e);
        std::process::exit(1);
    });
    let n = app.runs().len();
    eprintln!("trainview: http://{}:{}  ({} runs under {})", if bind == "0.0.0.0" { "localhost" } else { &bind }, port, n, app.roots.iter().map(|r| r.path.display().to_string()).collect::<Vec<_>>().join(", "));
    {
        let a = app.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(60));
            a.evict();
        });
    }
    for c in l.incoming().flatten() {
        let _ = c.set_nodelay(true);
        let app = app.clone();
        std::thread::spawn(move || handle(c, app));
    }
}

// 실행 찾기, kind, 줄기, live 판정, latest (TRAIN_VIEWER.md 3·5.7)
use crate::tail;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// pid 가 없는 실행: 마지막 기록 후 이 안이면 살아 있다고 본다(전투기 뷰어 LIVE_S)
pub const LIVE_S: f64 = 180.0;

#[derive(Clone)]
pub struct Root {
    pub label: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct RunRef {
    pub id: String,
    pub root: usize,
    pub dir: PathBuf,
    pub lab: bool,
}

const SKIP: &[&str] = &["replays", "evals", "target", ".git", "node_modules"];

fn is_run(d: &Path) -> bool {
    d.join("run.json").is_file() || d.join("progress.jsonl").is_file()
}

/// 뿌리 밑 깊이 4 까지에서 run.json 또는 progress.jsonl 이 있는 폴더. 실행 폴더 안으로는 내려가지 않는다(줄기·리플레이 폴더).
pub fn discover(roots: &[Root]) -> Vec<RunRef> {
    let mut out = vec![];
    for (ri, r) in roots.iter().enumerate() {
        let mut stack = vec![(r.path.clone(), 0usize)];
        while let Some((d, depth)) = stack.pop() {
            if depth > 0 && is_run(&d) {
                let rel = d.strip_prefix(&r.path).unwrap_or(&d).to_string_lossy().to_string();
                let lab = rel.split('/').any(|c| c == "labs");
                out.push(RunRef { id: format!("{}/{}", r.label, rel), root: ri, dir: d.clone(), lab });
                continue;
            }
            if depth >= 4 {
                continue;
            }
            let Ok(rd) = fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if n.starts_with('.') || n.starts_with("s_") || SKIP.contains(&n.as_str()) {
                    continue;
                }
                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    stack.push((e.path(), depth + 1));
                }
            }
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// 줄기: "main" + 하위 폴더 s_<이름>
pub fn streams(dir: &Path) -> Vec<String> {
    let mut v = vec!["main".to_string()];
    if let Ok(rd) = fs::read_dir(dir) {
        let mut s: Vec<String> = rd
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter_map(|e| e.file_name().to_string_lossy().strip_prefix("s_").map(|x| x.to_string()))
            .filter(|x| trainfmt::safe_name(x))
            .collect();
        s.sort();
        v.extend(s);
    }
    v
}

pub fn stream_dir(dir: &Path, stream: &str) -> PathBuf {
    if stream.is_empty() || stream == "main" { dir.to_path_buf() } else { dir.join(format!("s_{}", stream)) }
}

/// 줄기의 episodes 파일 이름 앞부분(…/episodes 또는 …/s_eval/episodes_eval)
pub fn episodes_base(dir: &Path, stream: &str) -> PathBuf {
    if stream.is_empty() || stream == "main" { dir.join("episodes") } else { dir.join(format!("s_{}", stream)).join(format!("episodes_{}", stream)) }
}

pub fn read_meta(dir: &Path) -> Option<Value> {
    serde_json::from_str(&fs::read_to_string(dir.join("run.json")).ok()?).ok()
}

pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

pub fn mtime(p: &Path) -> Option<f64> {
    fs::metadata(p).ok()?.modified().ok()?.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs_f64())
}

/// 5.7: pid 가 있으면 /proc/<pid> 가 있고 시작 시각이 pid_start 와 같을 때만. 없으면 None(→ 시각 규칙)
pub fn pid_alive(meta: &Value) -> Option<bool> {
    let pid = meta.get("pid")?.as_u64()? as u32;
    if meta.get("ended").is_some() {
        return Some(false);
    }
    let st = trainfmt::pid_start(pid);
    match (st, meta.get("pid_start").and_then(|x| x.as_u64())) {
        (None, _) => Some(false),
        (Some(a), Some(b)) => Some(a == b),
        (Some(_), None) => Some(true),
    }
}

/// 학습 상태 등(초록 불): training = 프로세스 살아 있음(pid·시작 시각 맞음) + STALL_S 안에 기록, stalled = 살아 있는데 기록 없음,
/// finished = run.json 에 ended(정상 끝), crashed = 프로세스가 없는데 ended 없음. pid 를 안 적은 옛 실행은 finished(ended 있음)·unknown.
pub const STALL_S: f64 = 60.0;
pub fn status(r: &RunRef) -> Value {
    let m = read_meta(&r.dir).unwrap_or(json!({}));
    let prog = r.dir.join("progress.jsonl");
    let t_last = [mtime(&prog), mtime(&r.dir.join("episodes.jsonl")), mtime(&r.dir.join("run.json"))].into_iter().flatten().fold(f64::NAN, f64::max);
    let age = now() - t_last;
    let ended = m.get("ended").is_some();
    let state = if m.get("pid").is_some() {
        let alive = !ended && pid_alive(&m).unwrap_or(false);
        if alive {
            if age.is_finite() && age <= STALL_S { "training" } else { "stalled" }
        } else if ended {
            "finished"
        } else {
            "crashed"
        }
    } else if ended || m.get("kind").and_then(|x| x.as_str()) == Some("behavior") {
        "finished"
    } else {
        "unknown"
    };
    // 재생 기록 뒷 프로세스(trainfmt replay_hook 의 s_eval/recording.json, pid 가 살아 있을 때만)
    let rec = fs::read_to_string(r.dir.join("s_eval/recording.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).filter(|v| {
        let pid = v["pid"].as_u64().unwrap_or(0) as u32;
        pid > 0 && trainfmt::pid_start(pid).map(|s| v["pid_start"].as_u64().map(|p| p == s).unwrap_or(true)).unwrap_or(false)
    });
    let last = tail::last_line(&prog);
    let it = last.as_ref().and_then(|l| l.get("time/iterations").or(l.get("time/iter"))).cloned().unwrap_or(Value::Null);
    json!({"id": r.id, "state": state, "age": if age.is_finite() { json!(age) } else { Value::Null }, "last_write": if t_last.is_finite() { json!(t_last) } else { Value::Null },
           "iter": it, "recording": rec.map(|v| v["tag"].clone())})
}

/// 폴더 크기(MB) — 1 분에 한 번만 다시 잰다
pub fn disk_mb(dir: &Path) -> f64 {
    fn walk(d: &Path, depth: usize) -> u64 {
        let Ok(rd) = fs::read_dir(d) else { return 0 };
        let mut s = 0;
        for e in rd.flatten() {
            let Ok(md) = e.metadata() else { continue };
            if md.is_dir() {
                if depth < 3 {
                    s += walk(&e.path(), depth + 1);
                }
            } else {
                s += md.len();
            }
        }
        s
    }
    walk(dir, 0) as f64 / 1e6
}

pub fn count_trp(d: &Path) -> usize {
    fs::read_dir(d).map(|rd| rd.flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".trp")).count()).unwrap_or(0)
}

pub struct DiskCache(pub std::collections::HashMap<PathBuf, (Instant, f64)>);

/// /api/runs 의 한 줄
/// 옛 판(2026-10-04 신경망·관측 v2 이전 코드)인가 — 보관함으로. git 커밋이 v2 커밋의 자손이 아니거나(모르는 커밋 = 옛 것), csv 로 옮긴 실행
pub struct Archive {
    pub v2: Option<String>,
    pub repo: PathBuf,
    pub cache: std::collections::HashMap<String, bool>,
}
impl Archive {
    pub fn new(repo: PathBuf, v2: Option<String>) -> Archive {
        let v2 = v2.or_else(|| {
            let o = std::process::Command::new("git").arg("-C").arg(&repo).args(["log", "--format=%H", "-1", "--grep=관측·신경망 v2"]).output().ok()?;
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() { None } else { Some(s) }
        });
        Archive { v2, repo, cache: Default::default() }
    }
    pub fn is_old(&mut self, m: &Value) -> bool {
        if m.get("imported_from").map(|x| !x.is_null()).unwrap_or(false) {
            return true;
        }
        if m.get("kind").and_then(|x| x.as_str()) == Some("behavior") || m.get("synthetic").and_then(|x| x.as_bool()) == Some(true) {
            return false;
        }
        let Some(c) = m.get("git").and_then(|g| g.get("commit")).and_then(|x| x.as_str()) else { return false };
        let Some(v2) = self.v2.clone() else { return false };
        if let Some(&b) = self.cache.get(c) {
            return b;
        }
        let ok = std::process::Command::new("git").arg("-C").arg(&self.repo).args(["merge-base", "--is-ancestor", &v2, c]).status().map(|s| s.success()).unwrap_or(false);
        self.cache.insert(c.to_string(), !ok);
        !ok
    }
}

/// 학생 run.json 의 teacher(체크포인트 경로) → 그 체크포인트가 든 실행(또는 csv 로 옮긴 실행의 원본 폴더)
pub fn teacher_run(m: &Value, runs: &[RunRef], metas: &std::collections::HashMap<String, Value>) -> Option<(String, String)> {
    let t = m.get("teacher")?.as_str()?;
    let home = std::env::var("HOME").unwrap_or_default();
    let p = PathBuf::from(t.replace('~', &home));
    let p = fs::canonicalize(&p).unwrap_or(p);
    let dir = p.parent()?.to_path_buf();
    let file = p.file_name()?.to_string_lossy().to_string();
    for r in runs {
        let d = fs::canonicalize(&r.dir).unwrap_or(r.dir.clone());
        if d == dir {
            return Some((r.id.clone(), file));
        }
    }
    for r in runs {
        if let Some(src) = metas.get(&r.id).and_then(|x| x.get("imported_from")).and_then(|x| x.as_str()) {
            if fs::canonicalize(src).map(|s| s == dir).unwrap_or(false) {
                return Some((r.id.clone(), file));
            }
        }
    }
    None
}

pub fn summary(r: &RunRef, roots: &[Root], lines: &mut tail::LineCount, disk: &mut DiskCache, rewound: Option<usize>) -> Value {
    let meta = read_meta(&r.dir);
    let m = meta.clone().unwrap_or(json!({}));
    let prog = r.dir.join("progress.jsonl");
    let last = tail::last_line(&prog);
    let first = tail::first_line(&prog);
    let ep_main = r.dir.join("episodes.jsonl");
    let mut eps = lines.count(&ep_main);
    let mut k = 1;
    loop {
        let p = r.dir.join(format!("episodes.{}.jsonl", k));
        if !p.exists() {
            break;
        }
        eps += lines.count(&p);
        k += 1;
    }
    let t_last = [mtime(&prog), mtime(&ep_main), mtime(&r.dir.join("run.json"))].into_iter().flatten().fold(f64::NAN, f64::max);
    let age = now() - t_last;
    let live = match pid_alive(&m) {
        _ if m.get("ended").is_some() => false,
        Some(a) => a,
        None => !r.lab && age.is_finite() && age < LIVE_S,
    };
    let g = |v: &Option<Value>, k: &str| v.as_ref().and_then(|x| x.get(k)).cloned().unwrap_or(Value::Null);
    let first_ts = m
        .get("segments")
        .and_then(|s| s.as_array())
        .and_then(|a| a.first())
        .and_then(|x| x.get("started"))
        .cloned()
        .unwrap_or_else(|| g(&first, "ts"));
    let dm = {
        let e = disk.0.entry(r.dir.clone()).or_insert((Instant::now() - std::time::Duration::from_secs(3600), 0.0));
        if e.0.elapsed().as_secs() >= 60 {
            *e = (Instant::now(), disk_mb(&r.dir));
        }
        e.1
    };
    let kind = m.get("kind").and_then(|x| x.as_str()).map(|s| s.to_string()).unwrap_or_else(|| if r.lab { "lab".into() } else { "?".into() });
    json!({
        "id": r.id,
        "name": m.get("name").cloned().unwrap_or(json!(r.dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default())),
        "root": roots[r.root].path.display().to_string(),
        "dir": r.dir.display().to_string(),
        "kind": kind,
        "lab": r.lab,
        "group": m.get("group"),
        "seed": m.get("seed"),
        "synthetic": m.get("synthetic"),
        "imported_from": m.get("imported_from"),
        "streams": streams(&r.dir),
        "live": live,
        "has_pid": m.get("pid").is_some(),
        "has_meta": meta.is_some(),
        "age": if age.is_finite() { json!(age) } else { Value::Null },
        "iters": g(&last, "time/iterations").as_f64().map(|x| json!(x)).unwrap_or(g(&last, "time/iter")),
        "env_steps": g(&last, "time/total_timesteps").as_f64().map(|x| json!(x)).unwrap_or(g(&last, "time/env_steps")),
        "wall": g(&last, "time/time_elapsed").as_f64().map(|x| json!(x)).unwrap_or(g(&last, "time/wall")),
        "eps": eps,
        "replays": count_trp(&r.dir.join("replays")),
        "evals": fs::read_dir(r.dir.join("evals")).map(|rd| rd.count()).unwrap_or(0),
        "stage": m.get("stage"),
        "first_ts": first_ts,
        "rewound": rewound,
        "status": status(r),
        "health": tail::recent_values(&prog, &["rollout/success_rate", "rollout/success_rate/*", "rollout/collision_rate", "rollout/ep_len_mean", "eval/success_rate", "eval/collision_rate", "eval_teacher/success_rate", "time/total_timesteps", "train/loss"]),
        "teacher": m.get("teacher"),
        "eval_replays": count_trp(&r.dir.join("s_eval").join("replays")),
        "disk_mb": (dm * 10.0).round() / 10.0,
    })
}

/// 기본으로 여는 실행: 뿌리마다 latest.txt — 가장 최근에 바뀐 것. lab 은 후보가 아니다(8절 20번). 없으면 가장 최근에 기록한 본학습.
pub fn latest(roots: &[Root], runs: &[RunRef]) -> Option<String> {
    // 가짜 시험 자료(run.json "synthetic": true)는 기본 후보가 아니다 — 기본 화면은 진짜 실행만
    let syn = |r: &RunRef| read_meta(&r.dir).and_then(|m| m.get("synthetic").and_then(|x| x.as_bool())).unwrap_or(false);
    let runs: Vec<RunRef> = runs.iter().filter(|r| !syn(r)).cloned().collect();
    let runs = &runs[..];
    let mut best: Option<(f64, String)> = None;
    for (ri, r) in roots.iter().enumerate() {
        let p = r.path.join("latest.txt");
        let Ok(t) = fs::read_to_string(&p) else { continue };
        let name = t.trim();
        let Some(rr) = runs.iter().find(|x| x.root == ri && !x.lab && x.dir.strip_prefix(&r.path).map(|s| s.to_string_lossy() == name).unwrap_or(false)) else { continue };
        let mt = mtime(&p).unwrap_or(0.0);
        if best.as_ref().map(|b| mt > b.0).unwrap_or(true) {
            best = Some((mt, rr.id.clone()));
        }
    }
    if best.is_none() {
        for rr in runs.iter().filter(|x| !x.lab) {
            let mt = mtime(&rr.dir.join("progress.jsonl")).or(mtime(&rr.dir.join("run.json"))).unwrap_or(0.0);
            if best.as_ref().map(|b| mt > b.0).unwrap_or(true) {
                best = Some((mt, rr.id.clone()));
            }
        }
    }
    best.map(|b| b.1)
}

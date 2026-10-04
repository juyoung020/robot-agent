//! 학습 실행 폴더 규약 (docs/map_vla/TRAIN_VIEWER.md 4절) — 형식은 이 크레이트 한 곳에만 정의한다(4.6).
//!
//! ```text
//! <root>/latest.txt                      기본으로 열 실행 이름
//! <root>/<이름>/run.json                 설정·상수 (schema 1)
//!               progress.jsonl           업데이트마다(≤ 1 줄/s 로 합침) 한 줄, 모집단
//!               episodes.jsonl           판마다 한 줄, 표본
//!               replays/*.trp            판 하나 궤적(바이너리, trp 모듈)
//!               evals/it<이터>.json       평가 표
//!               s_<줄기>/episodes_<줄기>.jsonl, s_<줄기>/replays/…
//! ```
//!
//! 쓰는 쪽은 **덧붙이기**(progress, episodes)와 **이름 바꾸기**(run.json, evals, replays: `.tmp` 에 쓰고 rename)만 한다.
//! 이 크레이트는 학습 고리와 무관하다: 학습기의 로그 스레드(이미 비동기인 경로)에서만 부른다.

pub mod keys;
pub mod replay_hook;
pub mod trp;

use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SCHEMA: i64 = 1;

/// 유닉스 시각(초)
pub fn now_ts() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// `/proc/<pid>/stat` 의 22 번째 칸(부팅 뒤 시작 시각, 클럭 틱). pid 재사용을 거르는 데 쓴다(5.7).
pub fn pid_start(pid: u32) -> Option<u64> {
    let s = fs::read_to_string(format!("/proc/{}/stat", pid)).ok()?;
    // comm 은 괄호 안에 공백을 품을 수 있다: 마지막 ')' 뒤부터 센다 (3 번째 칸 = state)
    let rest = &s[s.rfind(')')? + 2..];
    rest.split_whitespace().nth(19)?.parse().ok()
}

/// 저장소의 git 커밋·더러움·서브모듈 커밋. 실패하면 빈 객체(키를 뺀다).
pub fn git_info(dir: &Path) -> Value {
    let top = std::process::Command::new("git").arg("-C").arg(dir).args(["rev-parse", "--show-toplevel"]).output().ok().filter(|o| o.status.success());
    let dir: PathBuf = match top {
        Some(o) => PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()),
        None => dir.to_path_buf(),
    };
    let run = |args: &[&str]| -> Option<String> {
        let o = std::process::Command::new("git").arg("-C").arg(&dir).args(args).output().ok()?;
        if !o.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let mut m = Map::new();
    if let Some(h) = run(&["rev-parse", "HEAD"]) {
        m.insert("commit".into(), json!(h));
    }
    if let Some(s) = run(&["status", "--porcelain", "--untracked-files=no"]) {
        m.insert("dirty".into(), json!(!s.is_empty()));
    }
    if let Some(s) = run(&["submodule", "status"]) {
        let mut sm = Map::new();
        for l in s.lines() {
            let t: Vec<&str> = l.trim_start_matches([' ', '+', '-', 'U']).split_whitespace().collect();
            if t.len() >= 2 {
                sm.insert(t[1].to_string(), json!(t[0]));
            }
        }
        if !sm.is_empty() {
            m.insert("submodules".into(), Value::Object(sm));
        }
    }
    Value::Object(m)
}

/// `.tmp` 에 다 쓴 뒤 이름 바꾸기 — 반쪽 파일이 안 보인다.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

// ---------------------------------------------------------------------------------------------
// progress.jsonl 한 줄 모으기

#[derive(Clone, Copy, PartialEq, Debug)]
enum How {
    /// 가중 평균(가중치 = 판 수 등)
    Mean,
    /// 마지막 값
    Last,
    /// 합
    Sum,
}

/// 업데이트 여러 개를 한 줄로 합치는 모음(4.2: 초당 1 번보다 잦으면 합침).
/// 키마다 합치는 법이 다르다: 비율은 판 수 가중 평균, 손실은 평균, 이터·스텝은 마지막, 버린 수는 합.
#[derive(Default, Debug)]
pub struct Agg {
    m: BTreeMap<String, (How, f64, f64)>,
    n: u32,
}

impl Agg {
    fn put(&mut self, k: &str, how: How, v: f64, w: f64) {
        if !v.is_finite() || !w.is_finite() || w <= 0.0 {
            return; // 안 잰 값은 키를 뺀다(8절 1번)
        }
        let e = self.m.entry(k.to_string()).or_insert((how, 0.0, 0.0));
        match how {
            How::Mean => {
                e.1 += v * w;
                e.2 += w;
            }
            How::Last => {
                e.1 = v;
                e.2 = 1.0;
            }
            How::Sum => {
                e.1 += v;
                e.2 = 1.0;
            }
        }
    }
    /// 가중 평균. `w` ≤ 0 이면(그 업데이트에 판이 없으면) 아무것도 넣지 않는다.
    pub fn mean_w(&mut self, k: &str, v: f64, w: f64) {
        self.put(k, How::Mean, v, w)
    }
    pub fn mean(&mut self, k: &str, v: f64) {
        self.put(k, How::Mean, v, 1.0)
    }
    pub fn last(&mut self, k: &str, v: f64) {
        self.put(k, How::Last, v, 1.0)
    }
    pub fn sum(&mut self, k: &str, v: f64) {
        self.put(k, How::Sum, v, 1.0)
    }
    /// 업데이트 하나 끝(합친 수를 센다)
    pub fn tick(&mut self) {
        self.n += 1;
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0 && self.m.is_empty()
    }
    pub fn get(&self, k: &str) -> Option<f64> {
        self.m.get(k).map(|e| if e.0 == How::Mean { e.1 / e.2 } else { e.1 })
    }
    /// 한 줄(JSON 객체)로 만들고 비운다.
    pub fn take_row(&mut self) -> Map<String, Value> {
        let mut o = Map::new();
        for (k, (how, s, w)) in std::mem::take(&mut self.m) {
            let v = if how == How::Mean { s / w } else { s };
            if let Some(n) = serde_json::Number::from_f64(round_sig(v)) {
                o.insert(k, Value::Number(n));
            }
        }
        if self.n > 1 {
            o.insert("time/iterations_merged".into(), json!(self.n));
        }
        self.n = 0;
        o
    }
}

/// 유효 숫자 7 자리로(줄 크기를 줄임, f32 정밀도와 같음). 정수는 그대로.
pub fn round_sig(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() || (v.fract() == 0.0 && v.abs() < 9.0e15) {
        return v;
    }
    let d = 7 - 1 - v.abs().log10().floor() as i32;
    if d <= 0 || d > 300 {
        return v;
    }
    let p = 10f64.powi(d);
    let r = (v * p).round() / p;
    if r.is_finite() { r } else { v }
}

// ---------------------------------------------------------------------------------------------
// 실행 폴더 쓰개

/// 실행 폴더 하나에 규약 파일을 쓰는 쪽. 학습기의 **로그 스레드**가 갖는다.
pub struct RunWriter {
    pub dir: PathBuf,
    pub meta: Value,
    progress: Option<BufWriter<fs::File>>,
    episodes: BTreeMap<String, BufWriter<fs::File>>,
    pub agg: Agg,
    /// 줄을 내보내는 최소 간격(초). 0 이면 업데이트마다.
    pub every_s: f64,
    last_emit: f64,
    rows: u64,
    trim_pending: bool,
}

impl RunWriter {
    /// 실행 폴더를 연다. `meta` 는 run.json 내용(정체·설정 칸). schema·pid·pid_start·started·segments 는 여기서 채운다.
    /// `resume` 이면 progress.jsonl 을 이어 쓰고(첫 줄이 올 때 그 이터 뒤 줄을 지움 — `trim_from`), segments 를 덧붙인다.
    /// 아니면 progress.jsonl 을 새로 만든다. `latest` 면 부모 폴더의 latest.txt 를 이 실행 이름으로 바꾼다.
    pub fn create(dir: &Path, mut meta: Value, resume: bool, latest: bool) -> std::io::Result<RunWriter> {
        fs::create_dir_all(dir)?;
        let pid = std::process::id();
        let started = now_ts();
        let old: Option<Value> = fs::read_to_string(dir.join("run.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
        let mut segs: Vec<Value> = Vec::new();
        if resume {
            if let Some(o) = &old {
                if let Some(a) = o.get("segments").and_then(|x| x.as_array()) {
                    segs = a.clone();
                }
            }
        }
        segs.push(json!({"started": started, "pid": pid}));
        let m = meta.as_object_mut().expect("run meta must be an object");
        m.insert("schema".into(), json!(SCHEMA));
        if !m.contains_key("name") {
            let n = dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            m.insert("name".into(), json!(n));
        }
        m.insert("pid".into(), json!(pid));
        if let Some(s) = pid_start(pid) {
            m.insert("pid_start".into(), json!(s));
        }
        m.insert("started".into(), json!(started));
        m.insert("segments".into(), Value::Array(segs));
        m.remove("ended");
        let path = dir.join("progress.jsonl");
        let f = if resume {
            fs::OpenOptions::new().create(true).append(true).open(&path)?
        } else {
            fs::File::create(&path)?
        };
        let w = RunWriter {
            dir: dir.to_path_buf(),
            meta,
            progress: Some(BufWriter::new(f)),
            episodes: BTreeMap::new(),
            agg: Agg::default(),
            every_s: 1.0,
            last_emit: f64::NEG_INFINITY,
            rows: 0,
            trim_pending: resume,
        };
        w.write_meta()?;
        if latest {
            if let (Some(parent), Some(name)) = (dir.parent(), dir.file_name()) {
                let _ = write_atomic(&parent.join("latest.txt"), format!("{}\n", name.to_string_lossy()).as_bytes());
            }
        }
        Ok(w)
    }

    pub fn write_meta(&self) -> std::io::Result<()> {
        write_atomic(&self.dir.join("run.json"), serde_json::to_string_pretty(&self.meta).unwrap().as_bytes())
    }

    /// run.json 칸 하나 바꾸고 다시 씀(드묾: 단계 바뀜, 끝)
    pub fn set_meta(&mut self, k: &str, v: Value) {
        if let Some(m) = self.meta.as_object_mut() {
            m.insert(k.to_string(), v);
        }
        let _ = self.write_meta();
    }

    /// 재개: `iter` 이상인 줄을 지운다(되감기 — 8절 9번). 첫 줄을 쓰기 전에 한 번 부른다.
    /// 마지막 segment 에 from_iter·from_steps 를 적는다.
    pub fn trim_from(&mut self, iter: f64, env_steps: f64) {
        if !self.trim_pending {
            return;
        }
        self.trim_pending = false;
        if let Some(mut p) = self.progress.take() {
            let _ = p.flush();
        }
        let path = self.dir.join("progress.jsonl");
        let mut kept = String::new();
        let mut cut = 0usize;
        if let Ok(f) = fs::File::open(&path) {
            for l in BufReader::new(f).lines().map_while(Result::ok) {
                let it = serde_json::from_str::<Value>(&l).ok().and_then(|v| v.get("time/iterations").or(v.get("time/iter")).and_then(|x| x.as_f64()));
                match it {
                    Some(i) if i < iter => {
                        kept.push_str(&l);
                        kept.push('\n');
                    }
                    _ => cut += 1, // 되감긴 줄과 반쪽 줄
                }
            }
        }
        let _ = write_atomic(&path, kept.as_bytes());
        if let Ok(f) = fs::OpenOptions::new().append(true).open(&path) {
            self.progress = Some(BufWriter::new(f));
        }
        if let Some(segs) = self.meta.get_mut("segments").and_then(|s| s.as_array_mut()) {
            if let Some(last) = segs.last_mut() {
                last["from_iter"] = json!(iter);
                last["from_steps"] = json!(env_steps);
                last["trimmed_lines"] = json!(cut);
            }
        }
        let _ = self.write_meta();
    }

    /// 이번 `end_update` 에서 줄을 쓰게 되나(파생 값 — 구간 처리량 등 — 을 그 전에 넣으려고)
    pub fn due(&self, wall: f64) -> bool {
        wall - self.last_emit >= self.every_s
    }

    /// 업데이트 하나를 모은 뒤 부른다. `wall` = 실행 시작부터 초. 간격이 지났으면 한 줄 쓴다.
    pub fn end_update(&mut self, wall: f64) {
        self.agg.tick();
        if self.due(wall) {
            self.flush_row(wall);
        }
    }

    /// 모은 것이 있으면 지금 한 줄로 쓴다(단계 경계, 끝).
    pub fn flush_row(&mut self, wall: f64) {
        if self.agg.is_empty() {
            return;
        }
        if self.trim_pending {
            let it = self.agg.get("time/iterations").unwrap_or(0.0);
            let st = self.agg.get("time/total_timesteps").unwrap_or(0.0);
            self.trim_from(it, st);
        }
        let mut row = self.agg.take_row();
        row.insert("ts".into(), json!(round_sig(now_ts())));
        row.entry("time/time_elapsed").or_insert(json!(round_sig(wall)));
        if let Some(p) = &mut self.progress {
            let _ = writeln!(p, "{}", Value::Object(row));
            let _ = p.flush(); // 줄 단위로 내보냄: 뷰어는 줄끝까지 온 줄만 읽는다
        }
        self.last_emit = wall;
        self.rows += 1;
    }

    pub fn rows_written(&self) -> u64 {
        self.rows
    }

    /// 판 한 줄(4.3). `stream` None = 본 줄기(실행 폴더 바로 밑), Some(s) = `s_<s>/episodes_<s>.jsonl`.
    pub fn episode(&mut self, stream: Option<&str>, line: &Value) {
        let key = stream.unwrap_or("").to_string();
        if !self.episodes.contains_key(&key) {
            let p = match stream {
                None => self.dir.join("episodes.jsonl"),
                Some(s) => {
                    let d = self.dir.join(format!("s_{}", s));
                    let _ = fs::create_dir_all(&d);
                    d.join(format!("episodes_{}.jsonl", s))
                }
            };
            match fs::OpenOptions::new().create(true).append(true).open(&p) {
                Ok(f) => {
                    self.episodes.insert(key.clone(), BufWriter::new(f));
                }
                Err(_) => return,
            }
        }
        if let Some(w) = self.episodes.get_mut(&key) {
            let _ = writeln!(w, "{}", line);
            let _ = w.flush();
        }
    }

    /// 평가 표 하나(4.5): `evals/<이름>.json`, 이름 바꾸기로.
    pub fn eval(&self, name: &str, v: &Value) {
        let d = self.dir.join("evals");
        let _ = fs::create_dir_all(&d);
        let _ = write_atomic(&d.join(format!("{}.json", name)), serde_json::to_string_pretty(v).unwrap().as_bytes());
    }

    /// 리플레이 폴더(줄기별). 쓰는 쪽은 `trp::TrpWriter::finish` 로 이름 바꾸기.
    pub fn replay_dir(&self, stream: Option<&str>) -> PathBuf {
        let d = match stream {
            None => self.dir.join("replays"),
            Some(s) => self.dir.join(format!("s_{}", s)).join("replays"),
        };
        let _ = fs::create_dir_all(&d);
        d
    }

    /// 끝: 남은 줄을 쓰고 run.json 에 ended 를 적는다.
    pub fn finish(&mut self, wall: f64, extra: Value) {
        self.flush_row(wall);
        if let Some(p) = &mut self.progress {
            let _ = p.flush();
        }
        for w in self.episodes.values_mut() {
            let _ = w.flush();
        }
        if let Some(m) = self.meta.as_object_mut() {
            m.insert("ended".into(), json!(now_ts()));
            if let Value::Object(e) = extra {
                for (k, v) in e {
                    m.insert(k, v);
                }
            }
        }
        let _ = self.write_meta();
    }
}

/// 실행 폴더 이름·파일 이름으로 받는 글자(전투기 뷰어 `SAFE`): `[\w.=-]+`
pub fn safe_name(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".." && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '=' | '-'))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn agg_rules() {
        let mut a = Agg::default();
        a.mean_w("s", 1.0, 10.0);
        a.mean_w("s", 0.0, 30.0);
        a.mean_w("x", 0.5, 0.0); // 판 없음 → 키 없음
        a.last("it", 1.0);
        a.last("it", 2.0);
        a.sum("d", 1.0);
        a.sum("d", 2.0);
        a.tick();
        a.tick();
        let r = a.take_row();
        assert_eq!(r["s"].as_f64().unwrap(), 0.25);
        assert!(!r.contains_key("x"));
        assert_eq!(r["it"].as_f64().unwrap(), 2.0);
        assert_eq!(r["d"].as_f64().unwrap(), 3.0);
        assert_eq!(r["time/iterations_merged"].as_i64().unwrap(), 2);
    }
    #[test]
    fn sig() {
        assert_eq!(round_sig(0.123456789), 0.1234568);
        assert_eq!(round_sig(12345.0), 12345.0);
    }
}

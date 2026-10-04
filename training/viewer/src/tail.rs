// JSONL 이어 읽기 + progress 열 저장소 (TRAIN_VIEWER.md 5.5)
//
// 파일마다 (ino, 읽은 바이트) 를 들고 늘어난 바이트만 읽는다. 줄끝(\n)까지 온 줄만 받는다 — 마지막 반쪽 줄은 보내지 않고 다음에 다시 읽는다.
// ino 가 바뀌거나 길이가 줄면 새 파일로 보고 처음부터(sig 를 바꿈). progress 는 들어올 때 한 번만 해석해 열로 쌓는다.
use crate::http::{jstr, num32, num64};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// 한 번에 읽는 상한(5.6). 큰 파일은 여러 번에 나눠 읽는다.
pub const READ_CAP: u64 = 64 << 20;

pub struct FileTail {
    pub path: PathBuf,
    ino: u64,
    pub off: u64,
    pub len: u64,
    pub resets: u64,
}

pub enum Got {
    Same,
    /// 새 줄들(이어서). 앞의 bool = 처음부터 다시 읽음(파일이 바뀜)
    Lines(bool, Vec<(u64, String)>),
}

impl FileTail {
    pub fn new(path: PathBuf) -> FileTail {
        FileTail { path, ino: 0, off: 0, len: 0, resets: 0 }
    }
    /// 늘어난 줄. 각 줄의 파일 안 바이트 위치와 함께.
    pub fn poll(&mut self) -> Got {
        let md = match fs::metadata(&self.path) {
            Ok(m) => m,
            Err(_) => {
                if self.off > 0 || self.ino != 0 {
                    self.off = 0;
                    self.ino = 0;
                    self.len = 0;
                    self.resets += 1;
                    return Got::Lines(true, vec![]);
                }
                return Got::Same;
            }
        };
        let mut reset = false;
        if (self.ino != 0 && md.ino() != self.ino) || md.len() < self.off {
            self.off = 0;
            self.resets += 1;
            reset = true;
        }
        self.ino = md.ino();
        self.len = md.len();
        if md.len() == self.off {
            return if reset { Got::Lines(true, vec![]) } else { Got::Same };
        }
        let mut f = match fs::File::open(&self.path) {
            Ok(f) => f,
            Err(_) => return Got::Same,
        };
        let want = (md.len() - self.off).min(READ_CAP);
        let mut buf = Vec::with_capacity(want as usize);
        if f.seek(SeekFrom::Start(self.off)).is_err() || f.take(want).read_to_end(&mut buf).is_err() {
            return Got::Same;
        }
        let end = match buf.iter().rposition(|&b| b == b'\n') {
            Some(i) => i + 1,
            None => return if reset { Got::Lines(true, vec![]) } else { Got::Same }, // 반쪽 줄뿐
        };
        let mut lines = Vec::new();
        let mut s = 0usize;
        for i in 0..end {
            if buf[i] == b'\n' {
                if i > s {
                    lines.push((self.off + s as u64, String::from_utf8_lossy(&buf[s..i]).to_string()));
                }
                s = i + 1;
            }
        }
        self.off += end as u64;
        Got::Lines(reset, lines)
    }
}

/// 마지막 완성된 줄 하나(목록용, 파일 끝 64 KB 만 읽음)
pub fn last_line(p: &Path) -> Option<Value> {
    let mut f = fs::File::open(p).ok()?;
    let len = f.metadata().ok()?.len();
    let start = len.saturating_sub(64 * 1024);
    f.seek(SeekFrom::Start(start)).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    let end = buf.iter().rposition(|&b| b == b'\n')?;
    let s = buf[..end].iter().rposition(|&b| b == b'\n').map(|i| i + 1).unwrap_or(0);
    serde_json::from_slice(&buf[s..end]).ok()
}

/// 첫 줄 하나
pub fn first_line(p: &Path) -> Option<Value> {
    let mut f = fs::File::open(p).ok()?;
    let mut buf = vec![0u8; 64 * 1024];
    let n = f.read(&mut buf).ok()?;
    let end = buf[..n].iter().position(|&b| b == b'\n')?;
    serde_json::from_slice(&buf[..end]).ok()
}

// ---------------------------------------------------------------------------------------------
// progress 열 저장소

pub struct Progress {
    pub tail: FileTail,
    pub keys: Vec<String>,
    kidx: HashMap<String, usize>,
    pub cols: Vec<Vec<f32>>,
    pub iter: Vec<f64>,
    pub steps: Vec<f64>,
    pub wall: Vec<f64>,
    pub ts: Vec<f64>,
    /// 재개로 되감긴 줄(env_steps 가 줄어듦) 걷어낸 수
    pub rewound: usize,
    pub bad: usize,
    /// 내용 서명: 파일이 새로 쓰였거나 되감기로 줄이 빠지면 바뀐다. 클라이언트는 sig 가 바뀌면 처음부터 다시 받는다.
    pub sig: u64,
    pub last_access: Instant,
}

impl Progress {
    pub fn new(path: PathBuf) -> Progress {
        Progress {
            tail: FileTail::new(path),
            keys: vec![],
            kidx: HashMap::new(),
            cols: vec![],
            iter: vec![],
            steps: vec![],
            wall: vec![],
            ts: vec![],
            rewound: 0,
            bad: 0,
            sig: 1,
            last_access: Instant::now(),
        }
    }
    pub fn total(&self) -> usize {
        self.steps.len()
    }
    fn clear(&mut self) {
        self.keys.clear();
        self.kidx.clear();
        self.cols.clear();
        self.iter.clear();
        self.steps.clear();
        self.wall.clear();
        self.ts.clear();
        self.rewound = 0;
        self.bad = 0;
    }
    fn truncate(&mut self, n: usize) {
        for c in &mut self.cols {
            c.truncate(n);
        }
        self.iter.truncate(n);
        self.steps.truncate(n);
        self.wall.truncate(n);
        self.ts.truncate(n);
    }
    /// 파일을 다시 본다. 바뀐 것이 있으면 true.
    pub fn update(&mut self) -> bool {
        self.last_access = Instant::now();
        let mut changed = false;
        loop {
            match self.tail.poll() {
                Got::Same => break,
                Got::Lines(reset, lines) => {
                    changed = true;
                    if reset {
                        self.clear();
                        self.sig += 1;
                    }
                    let more = !lines.is_empty(); // READ_CAP 씩 나눠 읽음: 줄이 왔으면 한 번 더 본다
                    for (_, l) in lines {
                        self.push(&l);
                    }
                    if !more {
                        break;
                    }
                }
            }
        }
        changed
    }
    fn push(&mut self, l: &str) {
        let v: Value = match serde_json::from_str(l) {
            Ok(Value::Object(o)) => Value::Object(o),
            _ => {
                self.bad += 1; // 해석 못 한 줄(죽었다 재개해 중간에 남은 반쪽 줄 등)
                return;
            }
        };
        // 옛 키(2026-10-04 이전)는 표준 이름으로(trainfmt::keys::canon — 되돌림 호환)
        let mut o: Vec<(String, f64)> = Vec::with_capacity(v.as_object().unwrap().len());
        for (k, x) in v.as_object().unwrap() {
            if let Some(x) = x.as_f64() {
                let (nk, sc) = trainfmt::keys::canon(k);
                o.push((nk, x * sc));
            }
        }
        let g = |k: &str| o.iter().find(|e| e.0 == k).map(|e| e.1);
        let n = self.total();
        let steps = g("time/total_timesteps").or(g("time/iterations")).unwrap_or(n as f64);
        // 되감기(8절 9번): 새 줄의 env_steps 보다 큰 앞 줄은 재개 전 미래 → 걷어낸다
        let mut keep = n;
        while keep > 0 && self.steps[keep - 1] > steps {
            keep -= 1;
        }
        if keep < n {
            self.rewound += n - keep;
            self.truncate(keep);
            self.sig += 1;
        }
        let n = self.total();
        self.iter.push(g("time/iterations").unwrap_or(f64::NAN));
        self.steps.push(steps);
        self.wall.push(g("time/time_elapsed").unwrap_or(f64::NAN));
        self.ts.push(g("ts").unwrap_or(f64::NAN));
        for (k, x) in &o {
            let x = *x;
            let i = match self.kidx.get(k) {
                Some(&i) => i,
                None => {
                    self.keys.push(k.clone());
                    self.kidx.insert(k.clone(), self.keys.len() - 1);
                    self.cols.push(vec![f32::NAN; n]);
                    self.keys.len() - 1
                }
            };
            if self.cols[i].len() == n + 1 {
                self.cols[i][n] = x as f32;   // 옛 키 둘이 같은 새 키로(예: dagger/agree·disagree) — 뒤 것
            } else {
                self.cols[i].push(x as f32);
            }
        }
        for c in &mut self.cols {
            if c.len() < n + 1 {
                c.push(f32::NAN); // 그 줄에 키가 없었다
            }
        }
    }
    pub fn col(&self, k: &str) -> Option<&Vec<f32>> {
        self.kidx.get(k).map(|&i| &self.cols[i])
    }
    pub fn last(&self, k: &str) -> Option<f32> {
        self.col(k).and_then(|c| c.iter().rev().find(|v| v.is_finite()).copied())
    }

    /// 키 목록과 키마다 처음·마지막 줄(값이 있는)
    pub fn keys_json(&self) -> String {
        let mut s = String::from("{\"total\":");
        s.push_str(&self.total().to_string());
        s.push_str(&format!(",\"sig\":{},\"rewound\":{},\"bad\":{},\"keys\":[", self.sig, self.rewound, self.bad));
        for (i, k) in self.keys.iter().enumerate() {
            let c = &self.cols[i];
            let first = c.iter().position(|v| v.is_finite());
            let last = c.iter().rposition(|v| v.is_finite());
            let n = c.iter().filter(|v| v.is_finite()).count();
            if i > 0 {
                s.push(',');
            }
            s.push_str("{\"k\":");
            jstr(&mut s, k);
            s.push_str(&format!(",\"first\":{},\"last\":{},\"n\":{}}}", first.map(|x| x as i64).unwrap_or(-1), last.map(|x| x as i64).unwrap_or(-1), n));
        }
        s.push_str("]}");
        s
    }

    /// 고른 키만 열로. `keys` None = 모든 키. `max_points` 가 있고 줄이 더 많으면 칸마다 최소·최대·마지막(그림이 안 거짓말하는 솎기).
    pub fn rows_json(&self, keys: Option<&[String]>, from: usize, max_points: usize) -> String {
        let total = self.total();
        let from = from.min(total);
        let ks: Vec<(String, Option<&Vec<f32>>)> = match keys {
            Some(ks) => ks.iter().map(|k| (k.clone(), self.col(k))).collect(),
            None => self.keys.iter().map(|k| (k.clone(), self.col(k))).collect(),
        };
        let n = total - from;
        let agg = max_points > 0 && n > max_points;
        // 칸: [a, b) 줄 범위
        let bins: Vec<(usize, usize)> = if agg {
            (0..max_points).map(|i| (from + i * n / max_points, from + (i + 1) * n / max_points)).filter(|(a, b)| b > a).collect()
        } else {
            (from..total).map(|i| (i, i + 1)).collect()
        };
        let mut s = String::with_capacity(64 + bins.len() * (ks.len() + 4) * 10);
        s.push_str(&format!("{{\"start\":{},\"total\":{},\"sig\":{},\"rewound\":{},\"bad\":{},\"agg\":{},\"x\":{{", from, total, self.sig, self.rewound, self.bad, agg));
        let xs: [(&str, &Vec<f64>); 4] = [("iterations", &self.iter), ("total_timesteps", &self.steps), ("time_elapsed", &self.wall), ("ts", &self.ts)];
        for (j, (name, v)) in xs.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            s.push_str(&format!("\"{}\":[", name));
            for (i, &(_, b)) in bins.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                num64(&mut s, v[b - 1]);
            }
            s.push(']');
        }
        s.push_str("},\"cols\":{");
        let mut lo = String::new();
        let mut hi = String::new();
        for (j, (k, c)) in ks.iter().enumerate() {
            if j > 0 {
                s.push(',');
                if agg {
                    lo.push(',');
                    hi.push(',');
                }
            }
            jstr(&mut s, k);
            s.push_str(":[");
            if agg {
                jstr(&mut lo, k);
                lo.push_str(":[");
                jstr(&mut hi, k);
                hi.push_str(":[");
            }
            for (i, &(a, b)) in bins.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                    if agg {
                        lo.push(',');
                        hi.push(',');
                    }
                }
                match c {
                    None => {
                        s.push_str("null");
                        if agg {
                            lo.push_str("null");
                            hi.push_str("null");
                        }
                    }
                    Some(c) => {
                        if !agg {
                            num32(&mut s, c[a]);
                        } else {
                            let (mut mn, mut mx, mut last) = (f32::INFINITY, f32::NEG_INFINITY, f32::NAN);
                            for &v in &c[a..b] {
                                if v.is_finite() {
                                    mn = mn.min(v);
                                    mx = mx.max(v);
                                    last = v;
                                }
                            }
                            num32(&mut s, last);
                            num32(&mut lo, mn);
                            num32(&mut hi, mx);
                        }
                    }
                }
            }
            s.push(']');
            if agg {
                lo.push(']');
                hi.push(']');
            }
        }
        s.push('}');
        if agg {
            s.push_str(",\"lo\":{");
            s.push_str(&lo);
            s.push_str("},\"hi\":{");
            s.push_str(&hi);
            s.push('}');
        }
        s.push('}');
        s
    }
}

/// 줄 수 세기(바이트 오프셋 캐시 — 늘어난 바이트만 센다)
#[derive(Default)]
pub struct LineCount {
    m: HashMap<PathBuf, (u64, u64, u64)>, // ino, off, count
}
impl LineCount {
    pub fn count(&mut self, p: &Path) -> u64 {
        let md = match fs::metadata(p) {
            Ok(m) => m,
            Err(_) => return 0,
        };
        let e = self.m.entry(p.to_path_buf()).or_insert((md.ino(), 0, 0));
        if e.0 != md.ino() || md.len() < e.1 {
            *e = (md.ino(), 0, 0);
        }
        if md.len() > e.1 {
            if let Ok(mut f) = fs::File::open(p) {
                if f.seek(SeekFrom::Start(e.1)).is_ok() {
                    let mut buf = vec![0u8; 1 << 20];
                    let mut off = e.1;
                    let mut last_nl = e.1;
                    loop {
                        let n = match f.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => n,
                        };
                        for (i, &b) in buf[..n].iter().enumerate() {
                            if b == b'\n' {
                                e.2 += 1;
                                last_nl = off + i as u64 + 1;
                            }
                        }
                        off += n as u64;
                    }
                    e.1 = last_nl;
                }
            }
        }
        e.2
    }
}

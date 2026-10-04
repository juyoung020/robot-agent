// episodes.jsonl 저장소: 줄 위치(원문은 파일에서 다시 읽음) + 판당 작은 칸(표용). 스킬 × 집 × 시작 지도 완성도 표 (TRAIN_VIEWER.md 4.3·5.5·6.2)
use crate::http::{jstr, num64};
use crate::tail::{FileTail, Got};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::Instant;

/// /api/episodes 한 번에 보내는 줄 상한(잘랐으면 잘랐다고 말한다 — 8절 12번)
pub const MAX_ROWS: usize = 20_000;

#[derive(Clone, Copy)]
pub struct Ep {
    pub skill: u16,
    pub home: u16,
    pub stage: u16,
    pub mode: u16,
    pub driver: u16,
    pub bin: u8,
    pub success: i8,
    pub collided: i8,
    pub timeout: i8,
    pub t: f32,
    pub contacts: f32,
    pub ret: f32,
    pub spl: f32,
    pub env_steps: f64,
}

#[derive(Default)]
pub struct Dict {
    pub names: Vec<String>,
    idx: HashMap<String, u16>,
}
impl Dict {
    fn id(&mut self, s: &str) -> u16 {
        if let Some(&i) = self.idx.get(s) {
            return i;
        }
        self.names.push(s.to_string());
        let i = (self.names.len() - 1) as u16;
        self.idx.insert(s.to_string(), i);
        i
    }
}

pub struct Episodes {
    /// episodes.jsonl, episodes.1.jsonl, … (1 GB 마다 다음 번호)
    pub files: Vec<FileTail>,
    pub lines: Vec<(u16, u64, u32)>,
    pub eps: Vec<Ep>,
    pub skills: Dict,
    pub homes: Dict,
    pub stages: Dict,
    pub modes: Dict,
    pub drivers: Dict,
    pub bins: Vec<f64>,
    pub bin_names: Vec<String>,
    pub bad: usize,
    /// ret ≠ Σ r 인 줄 수(4.3: 서버가 첫 읽기 때 확인)
    pub ret_mismatch: usize,
    pub sig: u64,
    pub last_access: Instant,
    base: PathBuf,
}

pub fn bin_names(edges: &[f64]) -> Vec<String> {
    let pc = |v: f64| format!("{}", (v * 100.0).round());
    let mut n = vec![format!("{} %", pc(edges[0]))];
    for w in edges.windows(2) {
        n.push(format!("{}–{} %", pc(w[0]), pc(w[1])));
    }
    n.push(format!("{} %", pc(*edges.last().unwrap())));
    n
}

/// 칸: 끝점 칸 둘(= 처음 값, = 마지막 값) + 사이 구간들
pub fn bin_of(edges: &[f64], v: f64) -> u8 {
    if !v.is_finite() {
        return 255;
    }
    if v <= edges[0] {
        return 0;
    }
    if v >= *edges.last().unwrap() {
        return edges.len() as u8;
    }
    for (i, w) in edges.windows(2).enumerate() {
        if v > w[0] && v <= w[1] {
            return (i + 1) as u8;
        }
    }
    255
}

impl Episodes {
    /// `base` = 줄기 폴더의 episodes 파일 이름 앞부분(…/episodes 또는 …/s_eval/episodes_eval)
    pub fn new(base: PathBuf, edges: Vec<f64>) -> Episodes {
        let edges = if edges.len() >= 2 { edges } else { vec![0.0, 0.3, 0.7, 1.0] };
        Episodes {
            files: vec![],
            lines: vec![],
            eps: vec![],
            skills: Dict::default(),
            homes: Dict::default(),
            stages: Dict::default(),
            modes: Dict::default(),
            drivers: Dict::default(),
            bin_names: bin_names(&edges),
            bins: edges,
            bad: 0,
            ret_mismatch: 0,
            sig: 1,
            last_access: Instant::now(),
            base,
        }
    }
    fn file_path(&self, i: usize) -> PathBuf {
        let mut s = self.base.as_os_str().to_owned();
        if i > 0 {
            s.push(format!(".{}", i));
        }
        s.push(".jsonl");
        PathBuf::from(s)
    }
    pub fn total(&self) -> usize {
        self.eps.len()
    }
    pub fn update(&mut self) -> bool {
        self.last_access = Instant::now();
        // 다음 번호 파일이 생겼나
        if self.files.is_empty() {
            self.files.push(FileTail::new(self.file_path(0)));
        }
        while self.file_path(self.files.len()).exists() {
            let p = self.file_path(self.files.len());
            self.files.push(FileTail::new(p));
        }
        let mut changed = false;
        for fi in 0..self.files.len() {
            loop {
                match self.files[fi].poll() {
                    Got::Same => break,
                    Got::Lines(reset, lines) => {
                        changed = true;
                        if reset {
                            // 파일이 새로 쓰였다: 전부 다시
                            let base = self.base.clone();
                            let bins = self.bins.clone();
                            let sig = self.sig + 1;
                            *self = Episodes::new(base, bins);
                            self.sig = sig;
                            return true;
                        }
                        let more = !lines.is_empty();
                        for (off, l) in lines {
                            self.push(fi as u16, off, &l);
                        }
                        if !more {
                            break;
                        }
                    }
                }
            }
        }
        changed
    }
    fn push(&mut self, fi: u16, off: u64, l: &str) {
        let v: Value = match serde_json::from_str(l) {
            Ok(v @ Value::Object(_)) => v,
            _ => {
                self.bad += 1;
                return;
            }
        };
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("?").to_string();
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
        let b = |k: &str| match v.get(k) {
            Some(Value::Bool(x)) => *x as i8,
            Some(Value::Number(n)) => (n.as_f64().unwrap_or(0.0) != 0.0) as i8,
            _ => -1,
        };
        if let (Some(ret), Some(r)) = (v.get("ret").and_then(|x| x.as_f64()), v.get("r").and_then(|x| x.as_object())) {
            let sum: f64 = r.values().filter_map(|x| x.as_f64()).sum();
            if (sum - ret).abs() > 1e-3 * ret.abs().max(1.0) {
                self.ret_mismatch += 1;
            }
        }
        let comp = if v.get("completion0").is_some() { f("completion0") } else { f("map_completeness") };
        let stage = v.get("stage").or(v.get("curriculum")).and_then(|x| x.as_str()).unwrap_or("?").to_string();
        let spl = if v.get("spl").is_some() {
            f("spl")
        } else {
            let (l, lo) = (f("path_len"), f("path_len_opt"));
            let sc = b("success");
            if l.is_finite() && lo.is_finite() && sc >= 0 { sc as f64 * lo / l.max(lo).max(1e-9) } else { f64::NAN }
        };
        let to = b("timeout");
        let timeout = if to >= 0 { to } else if v.get("outcome").and_then(|x| x.as_str()) == Some("timeout") { 1 } else { -1 };
        let ep = Ep {
            skill: self.skills.id(&s("skill")),
            home: self.homes.id(&s("home")),
            stage: self.stages.id(&stage),
            mode: self.modes.id(&s("map_mode")),
            driver: self.drivers.id(&s("driver")),
            bin: bin_of(&self.bins, comp),
            success: b("success"),
            collided: if v.get("collided").is_some() { b("collided") } else if v.get("outcome").and_then(|x| x.as_str()) == Some("collision") { 1 } else { -1 },
            timeout,
            t: f("t") as f32,
            contacts: f("contacts") as f32,
            ret: f("ret") as f32,
            spl: spl as f32,
            env_steps: f("env_steps"),
        };
        self.eps.push(ep);
        self.lines.push((fi, off, l.len() as u32));
    }

    /// 줄 그대로 `from` 부터 MAX_ROWS 개
    pub fn rows_json(&self, from: usize) -> String {
        let total = self.total();
        let from = from.min(total);
        let end = (from + MAX_ROWS).min(total);
        let mut s = format!("{{\"start\":{},\"total\":{},\"cap\":{},\"truncated\":{},\"sig\":{},\"bad\":{},\"ret_mismatch\":{},\"rows\":[", from, total, MAX_ROWS, end < total, self.sig, self.bad, self.ret_mismatch);
        let mut cur: Option<(u16, fs::File)> = None;
        let mut first = true;
        for i in from..end {
            let (fi, off, len) = self.lines[i];
            if cur.as_ref().map(|c| c.0) != Some(fi) {
                cur = fs::File::open(self.file_path(fi as usize)).ok().map(|f| (fi, f));
            }
            let Some((_, f)) = cur.as_mut() else { continue };
            let mut buf = vec![0u8; len as usize];
            if f.seek(SeekFrom::Start(off)).is_err() || f.read_exact(&mut buf).is_err() {
                continue;
            }
            if !first {
                s.push(',');
            }
            first = false;
            s.push_str(&String::from_utf8_lossy(&buf));
        }
        s.push_str("]}");
        s
    }

    fn dim(&self, name: &str, e: &Ep) -> Option<u16> {
        Some(match name {
            "skill" => e.skill,
            "home" => e.home,
            "stage" | "curriculum" => e.stage,
            "map_mode" => e.mode,
            "driver" => e.driver,
            "bin" => {
                if e.bin == 255 {
                    return None;
                }
                e.bin as u16
            }
            _ => 0,
        })
    }
    fn labels(&self, name: &str) -> Vec<String> {
        match name {
            "skill" => self.skills.names.clone(),
            "home" => self.homes.names.clone(),
            "stage" | "curriculum" => self.stages.names.clone(),
            "map_mode" => self.modes.names.clone(),
            "driver" => self.drivers.names.clone(),
            "bin" => self.bin_names.clone(),
            _ => vec!["all".into()],
        }
    }

    /// 칸마다 최근 `last` 판(0 = 전체). 거르기: home / stage / map_mode 이름. `upto` = 이 env_steps 까지(시점 커서)
    pub fn table_json(&self, rows: &str, cols: &str, last: usize, filt: &[(&str, &str)], upto: f64) -> String {
        let rl = self.labels(rows);
        let cl = self.labels(cols);
        let nr = rl.len().max(1);
        let nc = cl.len().max(1);
        // 거르기를 번호로
        let mut fl: Vec<(&str, u16)> = vec![];
        for (k, v) in filt {
            if v.is_empty() {
                continue;
            }
            let id = self.labels(k).iter().position(|n| n == v);
            match id {
                Some(i) => fl.push((k, i as u16)),
                None => return format!("{{\"rows\":[],\"cols\":[],\"cells\":[],\"why\":\"no episodes with {}={}\"}}", k, v),
            }
        }
        #[derive(Default, Clone, Copy)]
        struct C {
            n: usize,
            ns: usize,
            s: f64,
            nt: usize,
            t: f64,
            ncon: usize,
            con: f64,
            nret: usize,
            ret: f64,
            nspl: usize,
            spl: f64,
            ncol: usize,
            col: f64,
            nto: usize,
            to: f64,
        }
        let mut cells = vec![C::default(); nr * nc];
        let cap = if last == 0 { usize::MAX } else { last };
        let mut full = 0usize;
        for e in self.eps.iter().rev() {
            if full == nr * nc {
                break;
            }
            if upto.is_finite() && e.env_steps.is_finite() && e.env_steps > upto {
                continue;
            }
            if fl.iter().any(|(k, id)| self.dim(k, e) != Some(*id)) {
                continue;
            }
            let (Some(r), Some(c)) = (self.dim(rows, e), self.dim(cols, e)) else { continue };
            let cell = &mut cells[r as usize * nc + c as usize];
            if cell.n >= cap {
                continue;
            }
            cell.n += 1;
            if cell.n == cap {
                full += 1;
            }
            if e.success >= 0 {
                cell.ns += 1;
                cell.s += e.success as f64;
            }
            if e.t.is_finite() {
                cell.nt += 1;
                cell.t += e.t as f64;
            }
            if e.contacts.is_finite() {
                cell.ncon += 1;
                cell.con += e.contacts as f64;
            }
            if e.ret.is_finite() {
                cell.nret += 1;
                cell.ret += e.ret as f64;
            }
            if e.spl.is_finite() {
                cell.nspl += 1;
                cell.spl += e.spl as f64;
            }
            if e.collided >= 0 {
                cell.ncol += 1;
                cell.col += e.collided as f64;
            }
            if e.timeout >= 0 {
                cell.nto += 1;
                cell.to += e.timeout as f64;
            }
        }
        let mut s = String::from("{\"rows\":[");
        for (i, n) in rl.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            jstr(&mut s, n);
        }
        s.push_str("],\"cols\":[");
        for (i, n) in cl.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            jstr(&mut s, n);
        }
        s.push_str(&format!("],\"total\":{},\"last\":{},\"cells\":[", self.total(), last));
        let avg = |n: usize, v: f64| if n > 0 { v / n as f64 } else { f64::NAN };
        for r in 0..nr {
            if r > 0 {
                s.push(',');
            }
            s.push('[');
            for c in 0..nc {
                if c > 0 {
                    s.push(',');
                }
                let x = cells[r * nc + c];
                s.push_str(&format!("{{\"n\":{},\"success\":", x.n));
                num64(&mut s, avg(x.ns, x.s));
                s.push_str(",\"t\":");
                num64(&mut s, avg(x.nt, x.t));
                s.push_str(",\"contacts\":");
                num64(&mut s, avg(x.ncon, x.con));
                s.push_str(",\"ret\":");
                num64(&mut s, avg(x.nret, x.ret));
                s.push_str(",\"spl\":");
                num64(&mut s, avg(x.nspl, x.spl));
                s.push_str(",\"collision\":");
                num64(&mut s, avg(x.ncol, x.col));
                s.push_str(",\"timeout\":");
                num64(&mut s, avg(x.nto, x.to));
                s.push('}');
            }
            s.push(']');
        }
        s.push_str("],\"dims\":{");
        for (i, d) in ["skill", "home", "stage", "map_mode", "driver"].iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("\"{}\":", d));
            s.push_str(&serde_json::to_string(&self.labels(d)).unwrap());
        }
        s.push_str("}}");
        s
    }
}

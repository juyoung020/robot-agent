//! 잡다한 도구: 시간, 결정적 난수, 이름 다듬기, 명령줄 인자.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn unix_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// 결정적 난수(splitmix64). 가짜 세계·웹소켓 키 등에 쓴다. 외부 크레이트 없이.
#[derive(Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn from_time() -> Self {
        let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
        Rng::new(t ^ (std::process::id() as u64).rotate_left(32))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// [0, 1)
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        if hi <= lo {
            return lo;
        }
        lo + self.next_u64() % (hi - lo)
    }
    pub fn bytes4(&mut self) -> [u8; 4] {
        let v = self.next_u64();
        [(v >> 8) as u8, (v >> 16) as u8, (v >> 24) as u8, (v >> 32) as u8]
    }
}

/// 물체 이름 → 사람이 읽는 이름.
/// - BDDL synset: `radio_receiver.n.01_1` → "radio receiver", `can__of__soda.n.01_2` → "can of soda"
/// - 시연 주석 id: `coffee_table_koagbh_0` → "coffee table", `radio_89` → "radio",
///   `half_head_cabbage_212_1` → "half head cabbage"
/// - 숫자만(scenemap 물체 id): "17" → "object 17"
pub fn display_name(id: &str) -> String {
    let id = id.trim().trim_start_matches('?');
    if id.is_empty() {
        return String::new();
    }
    if id.chars().all(|c| c.is_ascii_digit()) {
        return format!("object {id}");
    }
    if let Some(p) = id.find(".n.") {
        return id[..p].replace("__", "_").split('_').filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
    }
    let mut toks: Vec<&str> = id.split('_').filter(|s| !s.is_empty()).collect();
    // 모델 id 패턴: <종류>_<영소문자 6글자>_<한 자리 번호>
    if toks.len() >= 3 {
        let last = toks[toks.len() - 1];
        let prev = toks[toks.len() - 2];
        if last.len() == 1
            && last.chars().all(|c| c.is_ascii_digit())
            && prev.len() == 6
            && prev.chars().all(|c| c.is_ascii_lowercase())
        {
            toks.truncate(toks.len() - 2);
        }
    }
    while toks.len() > 1 && toks.last().map(|t| t.chars().all(|c| c.is_ascii_digit())).unwrap_or(false) {
        toks.pop();
    }
    toks.join(" ")
}

/// 종류 비교용 정규화: 소문자, 밑줄·공백 통일.
pub fn norm_category(s: &str) -> String {
    display_name(s).to_lowercase()
}

/// 영어 문장의 PaliGemma 토큰 수 추정(보수적). 실측과 대조는 instruction.rs 시험.
/// 단어 1 + 숫자 한 글자당 1(Gemma 는 숫자를 한 자리씩 끊는다) + 문장부호 1 + 긴 단어(>8자) 1 추가.
pub fn estimate_tokens(s: &str) -> usize {
    let mut n = 0usize;
    let mut word_len = 0usize;
    let mut prev = ' ';
    let flush = |wl: &mut usize, n: &mut usize| {
        if *wl > 0 {
            *n += 1 + if *wl > 8 { 1 } else { 0 };
        }
        *wl = 0;
    };
    for c in s.chars() {
        let p = prev;
        prev = c;
        if c.is_ascii_alphabetic() {
            word_len += 1;
        } else {
            flush(&mut word_len, &mut n);
            if c.is_ascii_digit() {
                // 공백 뒤 숫자는 "▁" 가 따로 한 토큰
                n += 1 + usize::from(p.is_whitespace());
            } else if c.is_whitespace() {
            } else if c.is_ascii() {
                n += 1;
            } else {
                n += 2;
            }
        }
    }
    flush(&mut word_len, &mut n);
    n
}

/// 예산 비교용: 추정에 10% + 1 여유. 과제 8개·형식 4가지 556문장 실측에서 추정이 실제보다 모자란 경우가 39개(최대 2토큰) 있어서.
pub fn tokens_with_margin(s: &str) -> usize {
    let e = estimate_tokens(s);
    e + e / 10 + 1
}

/// 아주 작은 명령줄 인자 해석기: `--key value`, `--flag`, 나머지는 위치 인자.
#[derive(Debug, Default, Clone)]
pub struct Args {
    pub kv: HashMap<String, String>,
    pub flags: Vec<String>,
    pub pos: Vec<String>,
}

impl Args {
    pub fn parse(it: impl IntoIterator<Item = String>) -> Args {
        let v: Vec<String> = it.into_iter().collect();
        let mut a = Args::default();
        let mut i = 0;
        while i < v.len() {
            let s = &v[i];
            if let Some(k) = s.strip_prefix("--") {
                if let Some((k, val)) = k.split_once('=') {
                    a.kv.insert(k.to_string(), val.to_string());
                } else if i + 1 < v.len() && !v[i + 1].starts_with("--") {
                    a.kv.insert(k.to_string(), v[i + 1].clone());
                    i += 1;
                } else {
                    a.flags.push(k.to_string());
                }
            } else {
                a.pos.push(s.clone());
            }
            i += 1;
        }
        a
    }
    pub fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
    pub fn str_or(&self, k: &str, d: &str) -> String {
        self.get(k).unwrap_or(d).to_string()
    }
    pub fn num<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        self.get(k).and_then(|s| s.parse().ok()).unwrap_or(d)
    }
    pub fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k) || matches!(self.get(k), Some("1") | Some("true") | Some("on"))
    }
}

pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(display_name("coffee_table_koagbh_0"), "coffee table");
        assert_eq!(display_name("radio_89"), "radio");
        assert_eq!(display_name("pillar_candle_222"), "pillar candle");
        assert_eq!(display_name("half_head_cabbage_212_1"), "half head cabbage");
        assert_eq!(display_name("radio_receiver.n.01_1"), "radio receiver");
        assert_eq!(display_name("?can__of__soda.n.01_2"), "can of soda");
        assert_eq!(display_name("top_cabinet_tynnnw_1"), "top cabinet");
        assert_eq!(display_name("17"), "object 17");
    }

    #[test]
    fn args() {
        let a = Args::parse(["x", "--a", "1", "--b", "--c=q"].iter().map(|s| s.to_string()));
        assert_eq!(a.get("a"), Some("1"));
        assert!(a.flag("b"));
        assert_eq!(a.get("c"), Some("q"));
        assert_eq!(a.pos, vec!["x"]);
    }
}

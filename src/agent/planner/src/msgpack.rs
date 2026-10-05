//! msgpack 최소 구현: 복사 없이 훑기(zero-copy scan) + 작은 값 쓰기.
//!
//! 평가기↔VLA 프로토콜은 msgpack + 넘파이 확장(`{b"__ndarray__": True, b"data": bin, b"dtype": str, b"shape": [..]}`)이다
//! (OmniGibson `eval/utils/network_utils.py` pack_data/unpack_data).
//!
//! 읽기는 [`Src`] 위에서 한다. 웹소켓 클라이언트 프레임은 XOR 마스크가 걸려 있는데, [`Masked`] 로 보면
//! 마스크를 풀지 않은 버퍼에서 필요한 바이트만 그 자리에서 XOR 해 읽는다 → 큰 영상 배열은 한 번도 건드리지 않는다.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct MpErr(pub String);
impl fmt::Display for MpErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "msgpack: {}", self.0)
    }
}
impl std::error::Error for MpErr {}
pub type R<T> = Result<T, MpErr>;

fn err<T>(s: impl Into<String>) -> R<T> {
    Err(MpErr(s.into()))
}

/// 바이트 원천. 평문 또는 마스크된 버퍼.
pub trait Src {
    fn len(&self) -> usize;
    fn at(&self, i: usize) -> u8;
    fn copy(&self, off: usize, out: &mut [u8]);
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Plain<'a>(pub &'a [u8]);
impl Src for Plain<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
    #[inline]
    fn at(&self, i: usize) -> u8 {
        self.0[i]
    }
    fn copy(&self, off: usize, out: &mut [u8]) {
        out.copy_from_slice(&self.0[off..off + out.len()]);
    }
}

/// 웹소켓 마스크가 걸린 버퍼: 논리 바이트 i = buf[i] ^ key[i % 4]
pub struct Masked<'a> {
    pub buf: &'a [u8],
    pub key: [u8; 4],
}
impl Src for Masked<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.buf.len()
    }
    #[inline]
    fn at(&self, i: usize) -> u8 {
        self.buf[i] ^ self.key[i & 3]
    }
    fn copy(&self, off: usize, out: &mut [u8]) {
        out.copy_from_slice(&self.buf[off..off + out.len()]);
        crate::ws::mask_at(out, self.key, off);
    }
}

/// 평문·마스크 버퍼 어느 쪽이든(중계기가 실행 중에 고른다).
pub enum AnySrc<'a> {
    P(Plain<'a>),
    M(Masked<'a>),
}
impl Src for AnySrc<'_> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            AnySrc::P(p) => p.len(),
            AnySrc::M(m) => m.len(),
        }
    }
    #[inline]
    fn at(&self, i: usize) -> u8 {
        match self {
            AnySrc::P(p) => p.at(i),
            AnySrc::M(m) => m.at(i),
        }
    }
    fn copy(&self, off: usize, out: &mut [u8]) {
        match self {
            AnySrc::P(p) => p.copy(off, out),
            AnySrc::M(m) => m.copy(off, out),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hdr {
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    F32(f32),
    F64(f64),
    Str(usize),
    Bin(usize),
    Arr(usize),
    Map(usize),
    Ext(i8, usize),
}

pub struct Rd<'s, S: Src> {
    pub s: &'s S,
    pub pos: usize,
}

impl<'s, S: Src> Rd<'s, S> {
    pub fn new(s: &'s S, pos: usize) -> Self {
        Rd { s, pos }
    }
    #[inline]
    fn need(&self, n: usize) -> R<()> {
        if self.pos.checked_add(n).map(|e| e <= self.s.len()).unwrap_or(false) {
            Ok(())
        } else {
            err(format!("끝을 넘음 (pos {} + {} > {})", self.pos, n, self.s.len()))
        }
    }
    #[inline]
    pub fn u8(&mut self) -> R<u8> {
        self.need(1)?;
        let b = self.s.at(self.pos);
        self.pos += 1;
        Ok(b)
    }
    fn be(&mut self, n: usize) -> R<u64> {
        self.need(n)?;
        let mut v = 0u64;
        for i in 0..n {
            v = (v << 8) | self.s.at(self.pos + i) as u64;
        }
        self.pos += n;
        Ok(v)
    }

    /// 값의 머리를 읽는다. Str/Bin/Ext 이면 pos 는 내용 시작을 가리킨다.
    pub fn hdr(&mut self) -> R<Hdr> {
        let b = self.u8()?;
        Ok(match b {
            0x00..=0x7f => Hdr::UInt(b as u64),
            0x80..=0x8f => Hdr::Map((b & 0x0f) as usize),
            0x90..=0x9f => Hdr::Arr((b & 0x0f) as usize),
            0xa0..=0xbf => Hdr::Str((b & 0x1f) as usize),
            0xc0 => Hdr::Nil,
            0xc2 => Hdr::Bool(false),
            0xc3 => Hdr::Bool(true),
            0xc4 => Hdr::Bin(self.be(1)? as usize),
            0xc5 => Hdr::Bin(self.be(2)? as usize),
            0xc6 => Hdr::Bin(self.be(4)? as usize),
            0xc7 | 0xc8 | 0xc9 => {
                let n = self.be(1 << (b - 0xc7))? as usize;
                let t = self.u8()? as i8;
                Hdr::Ext(t, n)
            }
            0xca => Hdr::F32(f32::from_bits(self.be(4)? as u32)),
            0xcb => Hdr::F64(f64::from_bits(self.be(8)?)),
            0xcc => Hdr::UInt(self.be(1)?),
            0xcd => Hdr::UInt(self.be(2)?),
            0xce => Hdr::UInt(self.be(4)?),
            0xcf => Hdr::UInt(self.be(8)?),
            0xd0 => Hdr::Int(self.be(1)? as u8 as i8 as i64),
            0xd1 => Hdr::Int(self.be(2)? as u16 as i16 as i64),
            0xd2 => Hdr::Int(self.be(4)? as u32 as i32 as i64),
            0xd3 => Hdr::Int(self.be(8)? as i64),
            0xd4..=0xd8 => {
                let t = self.u8()? as i8;
                Hdr::Ext(t, 1 << (b - 0xd4))
            }
            0xd9 => Hdr::Str(self.be(1)? as usize),
            0xda => Hdr::Str(self.be(2)? as usize),
            0xdb => Hdr::Str(self.be(4)? as usize),
            0xdc => Hdr::Arr(self.be(2)? as usize),
            0xdd => Hdr::Arr(self.be(4)? as usize),
            0xde => Hdr::Map(self.be(2)? as usize),
            0xdf => Hdr::Map(self.be(4)? as usize),
            0xe0..=0xff => Hdr::Int(b as i8 as i64),
            0xc1 => return err("0xc1 은 쓰지 않는 바이트"),
        })
    }

    pub fn advance(&mut self, n: usize) -> R<()> {
        self.need(n)?;
        self.pos += n;
        Ok(())
    }

    /// 값 하나를 통째로 건너뛴다(재귀 없이). 큰 bin 은 오프셋만 더한다 → O(구조 수).
    pub fn skip(&mut self) -> R<()> {
        let mut left: u64 = 1;
        while left > 0 {
            left -= 1;
            match self.hdr()? {
                Hdr::Str(n) | Hdr::Bin(n) | Hdr::Ext(_, n) => self.advance(n)?,
                Hdr::Arr(n) => left += n as u64,
                Hdr::Map(n) => left += 2 * n as u64,
                _ => {}
            }
            if left > (self.s.len() - self.pos) as u64 + 1 {
                return err("원소 수가 남은 바이트보다 많음");
            }
        }
        Ok(())
    }

    pub fn bytes(&mut self, n: usize) -> R<Vec<u8>> {
        self.need(n)?;
        let mut v = vec![0u8; n];
        self.s.copy(self.pos, &mut v);
        self.pos += n;
        Ok(v)
    }

    /// str 또는 bin 을 문자열로(키 비교용).
    pub fn text(&mut self) -> R<String> {
        match self.hdr()? {
            Hdr::Str(n) | Hdr::Bin(n) => Ok(String::from_utf8_lossy(&self.bytes(n)?).into_owned()),
            h => err(format!("문자열이 아님: {h:?}")),
        }
    }

    pub fn int(&mut self) -> R<i64> {
        match self.hdr()? {
            Hdr::UInt(v) => Ok(v as i64),
            Hdr::Int(v) => Ok(v),
            h => err(format!("정수가 아님: {h:?}")),
        }
    }
}

/// 최상위 맵의 항목 하나: 키와 값의 바이트 범위.
#[derive(Debug, Clone)]
pub struct Entry {
    pub key: String,
    pub key_off: usize,
    pub val_off: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct TopMap {
    pub hdr_len: usize,
    pub count: usize,
    pub entries: Vec<Entry>,
}

impl TopMap {
    pub fn find(&self, pred: impl Fn(&str) -> bool) -> Option<&Entry> {
        self.entries.iter().find(|e| pred(&e.key))
    }
}

/// 최상위가 맵인 메시지를 훑어 항목 위치를 적는다. 값 내용은 읽지 않는다.
pub fn scan_top<S: Src>(s: &S) -> R<TopMap> {
    let mut r = Rd::new(s, 0);
    let n = match r.hdr()? {
        Hdr::Map(n) => n,
        h => return err(format!("최상위가 맵이 아님: {h:?}")),
    };
    let hdr_len = r.pos;
    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        let key_off = r.pos;
        let key = r.text()?;
        let val_off = r.pos;
        r.skip()?;
        entries.push(Entry { key, key_off, val_off, end: r.pos });
    }
    Ok(TopMap { hdr_len, count: n, entries })
}

/// 넘파이 배열 참조(데이터는 원래 버퍼 안의 범위).
#[derive(Debug, Clone, PartialEq)]
pub struct NdRef {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub off: usize,
    pub len: usize,
}

impl NdRef {
    pub fn elem_size(&self) -> usize {
        self.dtype.trim_start_matches(['<', '>', '|', '=']).get(1..).and_then(|s| s.parse().ok()).unwrap_or(1)
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    /// 리틀엔디언 f32/f64 원소 i 를 f64 로.
    pub fn get_f64<S: Src>(&self, s: &S, i: usize) -> Option<f64> {
        let es = self.elem_size();
        if (i + 1) * es > self.len {
            return None;
        }
        let o = self.off + i * es;
        let mut b = [0u8; 8];
        s.copy(o, &mut b[..es.min(8)]);
        let kind = self.dtype.trim_start_matches(['<', '>', '|', '=']).chars().next().unwrap_or('?');
        Some(match (kind, es) {
            ('f', 4) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            ('f', 8) => f64::from_le_bytes(b),
            ('i', 8) => i64::from_le_bytes(b) as f64,
            ('i', 4) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            ('u', 1) | ('b', 1) => b[0] as f64,
            _ => return None,
        })
    }
}

/// `val_off` 에 있는 값이 넘파이 확장 맵이면 해석한다.
pub fn parse_nd<S: Src>(s: &S, val_off: usize) -> R<Option<NdRef>> {
    let mut r = Rd::new(s, val_off);
    let n = match r.hdr()? {
        Hdr::Map(n) => n,
        _ => return Ok(None),
    };
    let (mut is_nd, mut dtype, mut shape, mut data) = (false, String::new(), Vec::new(), None);
    for _ in 0..n {
        let k = r.text()?;
        match k.as_str() {
            "__ndarray__" => {
                is_nd = matches!(r.hdr()?, Hdr::Bool(true));
            }
            "data" => match r.hdr()? {
                Hdr::Bin(len) | Hdr::Str(len) => {
                    data = Some((r.pos, len));
                    r.advance(len)?;
                }
                h => return err(format!("data 가 bin 이 아님: {h:?}")),
            },
            "dtype" => dtype = r.text()?,
            "shape" => match r.hdr()? {
                Hdr::Arr(m) => {
                    for _ in 0..m {
                        shape.push(r.int()? as usize);
                    }
                }
                h => return err(format!("shape 가 배열이 아님: {h:?}")),
            },
            _ => r.skip()?,
        }
    }
    if !is_nd {
        return Ok(None);
    }
    let (off, len) = data.ok_or_else(|| MpErr("data 없음".into()))?;
    Ok(Some(NdRef { dtype, shape, off, len }))
}

// ---------- 쓰기 ----------

/// 맵 머리를 msgpack-python 과 같은 최소 형식으로.
pub fn map_hdr(n: usize) -> ([u8; 5], usize) {
    let mut b = [0u8; 5];
    if n < 16 {
        b[0] = 0x80 | n as u8;
        (b, 1)
    } else if n < 65536 {
        b[0] = 0xde;
        b[1..3].copy_from_slice(&(n as u16).to_be_bytes());
        (b, 3)
    } else {
        b[0] = 0xdf;
        b[1..5].copy_from_slice(&(n as u32).to_be_bytes());
        (b, 5)
    }
}

pub fn w_map(out: &mut Vec<u8>, n: usize) {
    let (b, l) = map_hdr(n);
    out.extend_from_slice(&b[..l]);
}

pub fn w_arr(out: &mut Vec<u8>, n: usize) {
    if n < 16 {
        out.push(0x90 | n as u8);
    } else if n < 65536 {
        out.push(0xdc);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else {
        out.push(0xdd);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    }
}

pub fn w_str(out: &mut Vec<u8>, s: &str) {
    let n = s.len();
    if n < 32 {
        out.push(0xa0 | n as u8);
    } else if n < 256 {
        out.push(0xd9);
        out.push(n as u8);
    } else if n < 65536 {
        out.push(0xda);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else {
        out.push(0xdb);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    }
    out.extend_from_slice(s.as_bytes());
}

pub fn w_bin(out: &mut Vec<u8>, b: &[u8]) {
    let n = b.len();
    if n < 256 {
        out.push(0xc4);
        out.push(n as u8);
    } else if n < 65536 {
        out.push(0xc5);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else {
        out.push(0xc6);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    }
    out.extend_from_slice(b);
}

pub fn w_bool(out: &mut Vec<u8>, v: bool) {
    out.push(if v { 0xc3 } else { 0xc2 });
}

pub fn w_uint(out: &mut Vec<u8>, v: u64) {
    if v < 128 {
        out.push(v as u8);
    } else if v < 256 {
        out.extend_from_slice(&[0xcc, v as u8]);
    } else if v < 65536 {
        out.push(0xcd);
        out.extend_from_slice(&(v as u16).to_be_bytes());
    } else if v < (1 << 32) {
        out.push(0xce);
        out.extend_from_slice(&(v as u32).to_be_bytes());
    } else {
        out.push(0xcf);
        out.extend_from_slice(&v.to_be_bytes());
    }
}

pub fn w_f64(out: &mut Vec<u8>, v: f64) {
    out.push(0xcb);
    out.extend_from_slice(&v.to_bits().to_be_bytes());
}

/// 넘파이 배열을 파이썬 pack_data 와 같은 순서·형식으로(키는 bin).
pub fn w_ndarray(out: &mut Vec<u8>, dtype: &str, shape: &[usize], data: &[u8]) {
    w_map(out, 4);
    w_bin(out, b"__ndarray__");
    w_bool(out, true);
    w_bin(out, b"data");
    w_bin(out, data);
    w_bin(out, b"dtype");
    w_str(out, dtype);
    w_bin(out, b"shape");
    w_arr(out, shape.len());
    for &d in shape {
        w_uint(out, d as u64);
    }
}

/// JSON 값 → msgpack (메타데이터·시험용).
pub fn w_json(out: &mut Vec<u8>, v: &serde_json::Value) {
    use serde_json::Value as V;
    match v {
        V::Null => out.push(0xc0),
        V::Bool(b) => w_bool(out, *b),
        V::Number(n) => {
            if let Some(u) = n.as_u64() {
                w_uint(out, u)
            } else if let Some(i) = n.as_i64() {
                out.push(0xd3);
                out.extend_from_slice(&i.to_be_bytes());
            } else {
                w_f64(out, n.as_f64().unwrap_or(0.0))
            }
        }
        V::String(s) => w_str(out, s),
        V::Array(a) => {
            w_arr(out, a.len());
            for x in a {
                w_json(out, x);
            }
        }
        V::Object(m) => {
            w_map(out, m.len());
            for (k, x) in m {
                w_str(out, k);
                w_json(out, x);
            }
        }
    }
}

/// msgpack → JSON (작은 메시지 기록용. bin 은 길이만).
pub fn to_json<S: Src>(r: &mut Rd<'_, S>, depth: usize) -> R<serde_json::Value> {
    use serde_json::{json, Value as V};
    if depth > 32 {
        return err("너무 깊음");
    }
    Ok(match r.hdr()? {
        Hdr::Nil => V::Null,
        Hdr::Bool(b) => V::Bool(b),
        Hdr::Int(i) => json!(i),
        Hdr::UInt(u) => json!(u),
        Hdr::F32(f) => json!(f),
        Hdr::F64(f) => json!(f),
        Hdr::Str(n) => V::String(String::from_utf8_lossy(&r.bytes(n)?).into_owned()),
        Hdr::Bin(n) => {
            r.advance(n)?;
            json!({ "bin_len": n })
        }
        Hdr::Ext(t, n) => {
            r.advance(n)?;
            json!({ "ext": t, "len": n })
        }
        Hdr::Arr(n) => {
            let mut a = Vec::with_capacity(n.min(1024));
            for _ in 0..n {
                a.push(to_json(r, depth + 1)?);
            }
            V::Array(a)
        }
        Hdr::Map(n) => {
            let mut m = serde_json::Map::new();
            for _ in 0..n {
                let k = match to_json(r, depth + 1)? {
                    V::String(s) => s,
                    other => other.to_string(),
                };
                let v = to_json(r, depth + 1)?;
                m.insert(k, v);
            }
            V::Object(m)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<u8> {
        let mut o = Vec::new();
        w_map(&mut o, 3);
        w_str(&mut o, "robot::proprio");
        let data: Vec<u8> = [0.5f32, -0.25, 0.125].iter().flat_map(|f| f.to_le_bytes()).collect();
        w_ndarray(&mut o, "<f4", &[1, 3], &data);
        w_str(&mut o, "robot::robot:zed_link:Camera:0::rgb");
        w_ndarray(&mut o, "|u1", &[1, 2, 2, 4], &[7u8; 16]);
        w_str(&mut o, "task_id");
        w_ndarray(&mut o, "<i8", &[1], &5i64.to_le_bytes());
        o
    }

    #[test]
    fn scan_plain_and_masked_agree() {
        let o = sample();
        let key = [0x12, 0x34, 0x56, 0x78];
        let m: Vec<u8> = o.iter().enumerate().map(|(i, b)| b ^ key[i & 3]).collect();
        let p = scan_top(&Plain(&o)).unwrap();
        let q = scan_top(&Masked { buf: &m, key }).unwrap();
        assert_eq!(p.count, 3);
        assert_eq!(p.entries.len(), q.entries.len());
        for (a, b) in p.entries.iter().zip(&q.entries) {
            assert_eq!((a.key.as_str(), a.val_off, a.end), (b.key.as_str(), b.val_off, b.end));
        }
        let src = Masked { buf: &m, key };
        let nd = parse_nd(&src, q.entries[0].val_off).unwrap().unwrap();
        assert_eq!(nd.shape, vec![1, 3]);
        assert_eq!(nd.get_f64(&src, 1), Some(-0.25));
        let t = parse_nd(&src, q.entries[2].val_off).unwrap().unwrap();
        assert_eq!(t.get_f64(&src, 0), Some(5.0));
    }

    #[test]
    fn skip_rejects_garbage() {
        let bad = [0xdd, 0xff, 0xff, 0xff, 0xff];
        assert!(Rd::new(&Plain(&bad), 0).skip().is_err());
    }
}

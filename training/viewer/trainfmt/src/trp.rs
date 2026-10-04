//! `.trp` — 판 하나의 궤적, 바이너리 (TRAIN_VIEWER.md 4.4). 모두 little-endian, 섹션 시작은 8 바이트 정렬.
//!
//! ```text
//! "TRP1"  u32 head_len  head(JSON, UTF-8)  [0 채움 → 8 정렬]
//! frames : n_frames × n_cols × f32                (행 우선)
//! slots  : n_frames × n_slots × n_slot_cols × f16
//! map    : [u32 frame][MAP_RECT 페이로드] 이어짐 — 페이로드 = i32 w,h; f64 res,ox,oy; i32 x0,y0,x1,y1 (48 B) + i8 cells
//!          (scenemap stream.hpp MAP_RECT 와 같은 바이트: −1 모름, 0–100 %)
//! img    : (선택) [u32 frame][u8 cam][u32 len][JPEG]
//! ```
//! 머리 JSON: `meta`(그 판의 episodes.jsonl 줄), `cols`, `slot_cols`, `n_slots`, `dt`, `stride`, `n_frames`,
//! `sections`([{name, off, len}], 파일 처음부터 바이트), `objects`, `joint_map`, `grid`.

use serde_json::{json, Value};
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"TRP1";

/// 프레임 열 1 판(4.4 표). 쓰는 쪽은 이 중 가진 것만 싣는다 — 열 이름이 머리에 있으므로 읽는 쪽은 이름으로 찾는다.
pub const FRAME_COLS_V1: &[&str] = &[
    "t", "x", "y", "yaw", "sx", "sy", "syaw", "vx", "wz", "q1", "q2", "q3", "q4", "q5", "qg", "ee_x", "ee_y", "ee_z", "a_vx", "a_wz", "a_q1",
    "a_q2", "a_q3", "a_q4", "a_q5", "a_g",
];
/// 슬롯 열(f16, 칸마다)
pub const SLOT_COLS_V1: &[&str] = &["id", "bx", "by", "bz", "px", "py", "pz", "ex", "ey", "ez", "src", "unc", "state", "is_tgt", "age"];
/// `ev` 비트
pub const EV_CONTACT: u32 = 1;
pub const EV_GRASP: u32 = 2;
pub const EV_DROP: u32 = 4;
pub const EV_SUCCESS: u32 = 8;
pub const EV_FILTER: u32 = 16;
pub const EV_RESET: u32 = 32;

/// f32 → IEEE half 비트(가장 가까운 짝수)
pub fn f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mut man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        man |= 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = man >> shift;
        let rem = man & ((1 << shift) - 1);
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1; // 넘치면 지수로 올라감(무한대까지 맞음)
    }
    sign | r as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    if e == 0 {
        s * m * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0.0 { s * f32::INFINITY } else { f32::NAN }
    } else {
        s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

/// 판 하나를 모아 두었다가 `finish` 에서 한 번에 쓴다(로그 스레드에서).
pub struct TrpWriter {
    pub cols: Vec<String>,
    pub slot_cols: Vec<String>,
    pub n_slots: usize,
    pub frames: Vec<f32>,
    pub slots: Vec<u16>,
    pub map: Vec<u8>,
    pub img: Vec<u8>,
    pub head: Value,
    n_frames: usize,
}

impl TrpWriter {
    pub fn new(cols: &[String], slot_cols: &[String], n_slots: usize, head: Value) -> TrpWriter {
        TrpWriter { cols: cols.to_vec(), slot_cols: slot_cols.to_vec(), n_slots, frames: vec![], slots: vec![], map: vec![], img: vec![], head, n_frames: 0 }
    }
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }
    /// 프레임 하나: `row` 는 cols 차례, `slots` 는 n_slots × slot_cols (없으면 빈 조각 — 슬롯 섹션 없이)
    pub fn frame(&mut self, row: &[f32], slots: &[f32]) {
        assert_eq!(row.len(), self.cols.len());
        self.frames.extend_from_slice(row);
        if !self.slot_cols.is_empty() {
            assert_eq!(slots.len(), self.n_slots * self.slot_cols.len());
            self.slots.extend(slots.iter().map(|&v| f16_bits(v)));
        }
        self.n_frames += 1;
    }
    /// 지도 변화분(MAP_RECT). `cells` = (x1−x0+1)(y1−y0+1) 개, 행 0 = 최소 y.
    #[allow(clippy::too_many_arguments)]
    pub fn map_rect(&mut self, frame: u32, w: i32, h: i32, res: f64, ox: f64, oy: f64, x0: i32, y0: i32, x1: i32, y1: i32, cells: &[i8]) {
        assert_eq!(cells.len() as i64, ((x1 - x0 + 1) as i64) * ((y1 - y0 + 1) as i64));
        self.map.extend_from_slice(&frame.to_le_bytes());
        for v in [w, h] {
            self.map.extend_from_slice(&v.to_le_bytes());
        }
        for v in [res, ox, oy] {
            self.map.extend_from_slice(&v.to_le_bytes());
        }
        for v in [x0, y0, x1, y1] {
            self.map.extend_from_slice(&v.to_le_bytes());
        }
        self.map.extend(cells.iter().map(|&c| c as u8));
    }
    pub fn image(&mut self, frame: u32, cam: u8, jpeg: &[u8]) {
        self.img.extend_from_slice(&frame.to_le_bytes());
        self.img.push(cam);
        self.img.extend_from_slice(&(jpeg.len() as u32).to_le_bytes());
        self.img.extend_from_slice(jpeg);
    }

    /// 바이트로
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut secs: Vec<(&str, Vec<u8>)> = vec![("frames", self.frames.iter().flat_map(|v| v.to_le_bytes()).collect())];
        if !self.slot_cols.is_empty() {
            secs.push(("slots", self.slots.iter().flat_map(|v| v.to_le_bytes()).collect()));
        }
        if !self.map.is_empty() {
            secs.push(("map", self.map.clone()));
        }
        if !self.img.is_empty() {
            secs.push(("img", self.img.clone()));
        }
        let pad8 = |n: usize| (n + 7) & !7;
        // 머리 길이가 오프셋에 따라 바뀌므로 자리가 멈출 때까지 다시 잰다(두세 번)
        let mut head_len = 0usize;
        let mut head = String::new();
        for _ in 0..8 {
            let mut off = pad8(8 + head_len);
            let mut sj = vec![];
            for (n, b) in &secs {
                sj.push(json!({"name": n, "off": off, "len": b.len()}));
                off = pad8(off + b.len());
            }
            let mut h = self.head.clone();
            h["cols"] = json!(self.cols);
            h["slot_cols"] = json!(self.slot_cols);
            h["n_slots"] = json!(self.n_slots);
            h["n_frames"] = json!(self.n_frames);
            h["sections"] = Value::Array(sj);
            h["format"] = json!("TRP1");
            head = h.to_string();
            if head.len() == head_len {
                break;
            }
            head_len = head.len();
        }
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(head.len() as u32).to_le_bytes());
        out.extend_from_slice(head.as_bytes());
        for (_, b) in &secs {
            out.resize(pad8(out.len()), 0);
            out.extend_from_slice(b);
        }
        out
    }

    /// `.tmp` 에 쓰고 이름 바꾸기
    pub fn finish(&self, path: &Path) -> std::io::Result<usize> {
        let b = self.to_bytes();
        crate::write_atomic(path, &b)?;
        Ok(b.len())
    }
}

/// 머리만 읽기(뷰어 목록). 파일 앞부분 바이트를 받는다.
pub fn read_head(b: &[u8]) -> Option<Value> {
    if b.len() < 8 || &b[0..4] != MAGIC {
        return None;
    }
    let n = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize;
    if b.len() < 8 + n {
        return None;
    }
    serde_json::from_slice(&b[8..8 + n]).ok()
}

/// 머리 길이(파일 앞 8 바이트에서)
pub fn head_len(b: &[u8]) -> Option<usize> {
    if b.len() < 8 || &b[0..4] != MAGIC {
        return None;
    }
    Some(8 + u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half() {
        for v in [0.0f32, 1.0, -2.5, 0.1, 31.99, 65504.0, 1e-5, 3.14159] {
            let r = f16_to_f32(f16_bits(v));
            assert!((r - v).abs() <= v.abs() * 1e-3 + 1e-7, "{} {}", v, r);
        }
    }
    #[test]
    fn roundtrip() {
        let cols: Vec<String> = ["t", "x"].iter().map(|s| s.to_string()).collect();
        let sc: Vec<String> = ["id", "bx"].iter().map(|s| s.to_string()).collect();
        let mut w = TrpWriter::new(&cols, &sc, 2, json!({"meta": {"ep": 1}}));
        w.frame(&[0.0, 1.0], &[1.0, 2.0, 3.0, 4.0]);
        w.frame(&[0.1, 1.5], &[1.0, 2.0, 3.0, 4.0]);
        w.map_rect(0, 4, 4, 0.05, 0.0, 0.0, 0, 0, 1, 1, &[-1, 0, 50, 100]);
        let b = w.to_bytes();
        let h = read_head(&b).unwrap();
        let secs = h["sections"].as_array().unwrap();
        let f = &secs[0];
        let off = f["off"].as_u64().unwrap() as usize;
        assert_eq!(off % 8, 0);
        let x = f32::from_le_bytes(b[off + 12..off + 16].try_into().unwrap());
        assert_eq!(x, 1.5);
        let m = secs.iter().find(|s| s["name"] == "map").unwrap();
        let mo = m["off"].as_u64().unwrap() as usize;
        assert_eq!(b[mo + 4 + 48 + 2] as i8, 50);
        assert_eq!(h["n_frames"], 2);
    }
}

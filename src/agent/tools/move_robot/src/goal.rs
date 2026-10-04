//! 목표 칸(goal entries, 결정 2026-10-05 — 앱 지도 두드리기): 집을 것(PICK)·놓을 곳(PLACE) 두 칸, 칸마다 값 16.
//! GPU 학습 쪽(`training/RL/map/include/map_tok.h` 목표 칸·`observation/include/obs.h` X0 432..463)과 같은 정의다
//! (docs/map_vla/VLA_INPUT.md "목표 칸").
//!
//! 칸 값(로봇 base_link 기준, 믿는 자세, m):
//! `0 있음 | 1 물체 | 2 지점 | 3 위치 앎 | 4 잃음(물체가 사라짐/지도에서 빠짐 → 마지막으로 안 자리) | 5..7 x 앞·y 왼·z(세계 z − 0.15) |
//!  8 수평 거리 | 9 sin 방위 = y/거리 | 10 cos 방위 = x/거리 (거리 < 1e-4 → 0, 1) | 11..13 위치 − 손끝 | 14, 15 = 0`.
//! 없는 칸 = 모두 0. 아직 못 본 물체 = 있음·물체만 1. 지점 = 있음·지점·앎 1.
//! 지도 좌표·id·클래스 번호는 넣지 않는다(VLA_INPUT 0절). 물체 간 관계(on/in/next to/near)도 쓰지 않는다 — 숫자와 종류뿐.
//!
//! 정규화(정책 입력 32): 깃발(0..4, 14, 15)·sin·cos 는 그대로, 위치 x·y·z 는 tok_norm.h 특징 0·1·2, 거리 6, 손끝 기준 3·4·5 로
//! clamp10((clip_log(raw) − μ) / σ), clip_log(x) = sign(x)·ln(1 + min(|x|, 5) / 0.5). μ·σ 는 tok_norm.h 를 그대로 읽고(include_str!),
//! ln 은 시뮬 `dm::lnf_d` 를 같은 연산 차례로 옮겼다. 원값은 시뮬 지도 토큰처럼 FP16 으로 반올림한 뒤 정규화한다.

use crate::verify::{to_robot, MemObject, ObjState};
use serde_json::{json, Value};
use std::sync::OnceLock;

pub const N_GV: usize = 16;
pub const PICK: usize = 0;
pub const PLACE: usize = 1;
pub const GE_PRESENT: usize = 0;
pub const GE_KOBJ: usize = 1;
pub const GE_KPT: usize = 2;
pub const GE_KNOWN: usize = 3;
pub const GE_LOST: usize = 4;
pub const GE_POS: usize = 5;
pub const GE_DIST: usize = 8;
pub const GE_SIN: usize = 9;
pub const GE_COS: usize = 10;
pub const GE_EEF: usize = 11;
/// base_footprint → base_link (URDF base_joint z)
pub const BASE_Z: f64 = 0.15;

/// 지점 놓기 성공(공유 정의, GPU 쪽 `bscene.h pred_at_point` 와 같은 수): 놓은 물체 가운데가 지점에서 수평 0.05 m 안,
/// 물체 바닥이 지점 높이 ± 0.02 m, 손에서 놓임
pub const POINT_R: f64 = 0.05;
pub const POINT_DZ: f64 = 0.02;

// ---------------------------------------------------------------- 정규화 (tok_norm.h 그대로)

const TOK_NORM_H: &str = include_str!("../../../../../training/RL/observation/include/tok_norm.h");

/// C99 16진 실수("-0x1.23c478p-2", "0x0p+0") → f32 (tok_norm.h 의 값은 f32 로 정확)
pub fn parse_hexf(s: &str) -> Option<f32> {
    let s = s.trim();
    let (neg, t) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X"))?;
    let (mant, exp) = t.split_once(['p', 'P'])?;
    let exp: i32 = exp.parse().ok()?;
    let (ip, fp) = mant.split_once('.').unwrap_or((mant, ""));
    let mut m: u64 = 0;
    for ch in ip.chars().chain(fp.chars()) {
        m = m.checked_mul(16)? + ch.to_digit(16)? as u64;
    }
    let v = (m as f64) * 2f64.powi(exp - 4 * fp.len() as i32);
    Some(if neg { -v as f32 } else { v as f32 })
}

fn parse_macro(name: &str) -> Vec<f32> {
    let key = format!("#define {name} {{");
    let i = TOK_NORM_H.find(&key).unwrap_or_else(|| panic!("tok_norm.h has no {name}")) + key.len();
    let j = i + TOK_NORM_H[i..].find('}').expect("tok_norm.h macro end");
    TOK_NORM_H[i..j].replace('\\', " ").split(',').map(|x| parse_hexf(x).unwrap_or_else(|| panic!("tok_norm.h {name}: bad number '{x}'"))).collect()
}

/// (μ, σ) — 특징 111 개(칸 0..32 | 벽 | 안 본 곳 | 방 | 경유 지점)
pub fn tok_norm() -> &'static (Vec<f32>, Vec<f32>) {
    static T: OnceLock<(Vec<f32>, Vec<f32>)> = OnceLock::new();
    T.get_or_init(|| {
        let (mu, sd) = (parse_macro("TOKN_MU"), parse_macro("TOKN_SD"));
        let n: usize = TOK_NORM_H.split("constexpr int N_FEAT = ").nth(1).and_then(|r| r.split(';').next()).and_then(|x| x.trim().parse().ok()).expect("N_FEAT");
        assert!(mu.len() == n && sd.len() == n, "tok_norm.h: {} / {} values, N_FEAT {n}", mu.len(), sd.len());
        (mu, sd)
    })
}

/// 시뮬 dm::lnf_d(detmath.h) — 같은 연산 차례
pub fn lnf_d(y: f32) -> f32 {
    let x = y.to_bits();
    let mut e = ((x >> 23) & 0xff) as i32 - 127;
    let mut m = f32::from_bits((x & 0x7f_ffff) | 0x3f80_0000);
    if m > 1.414_213_56_f32 {
        m *= 0.5;
        e += 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let mut p = 0.111_111_11_f32;
    p *= z2;
    p += 0.142_857_14;
    p *= z2;
    p += 0.2;
    p *= z2;
    p += 0.333_333_33;
    p *= z2;
    p += 1.0;
    let lm = 2.0 * z * p;
    e as f32 * 0.693_147_18 + lm
}

pub fn clamp10(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(-10.0, 10.0)
    }
}

/// sign(x)·ln(1 + min(|x|, 5) / 0.5)
pub fn clip_log(x: f32) -> f32 {
    let a = x.abs().min(5.0);
    let y = lnf_d(1.0 + a / 0.5);
    if x < 0.0 {
        -y
    } else {
        y
    }
}

/// 길이 특징 f(tok_norm 번호)의 정규화 값
pub fn norm_len(f: usize, raw: f32) -> f32 {
    if raw.is_nan() {
        return 0.0;
    }
    let (mu, sd) = tok_norm();
    clamp10((clip_log(raw) - mu[f]) / sd[f])
}

/// f32 → FP16 → f32 (가장 가까운 짝수, 시뮬 지도 토큰 f2h 와 같은 반올림)
pub fn round_f16(f: f32) -> f32 {
    let x = f.to_bits();
    let sign = (x >> 16) & 0x8000;
    let ax = x & 0x7fff_ffff;
    let h: u32 = if ax > 0x7f80_0000 {
        0x7fff
    } else if ax >= 0x477f_f000 {
        sign | 0x7c00
    } else if ax < 0x3880_0000 {
        if ax < 0x3300_0000 {
            sign
        } else {
            let mm = (ax & 0x7f_ffff) | 0x80_0000;
            let sh = 126 - (ax >> 23) as i32;
            let mut h = mm >> sh;
            let rem = mm & ((1u32 << sh) - 1);
            let half = 1u32 << (sh - 1);
            if rem > half || (rem == half && (h & 1) == 1) {
                h += 1;
            }
            sign | h
        }
    } else {
        let mut h = (ax - 0x3800_0000) >> 13;
        let rem = ax & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
            h += 1;
        }
        sign | h
    };
    // FP16 → f32
    let (s, e, m) = ((h & 0x8000) << 16, (h >> 10) & 0x1f, h & 0x3ff);
    if e == 0 {
        let v = m as f32 * (1.0 / 16_777_216.0);
        return f32::from_bits(v.to_bits() | s);
    }
    if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13));
    }
    f32::from_bits(s | ((e + 112) << 23) | (m << 13))
}

/// 원값 32 → 정규화 32 (X0 432..463 과 같은 배치)
pub fn normalize(raw: &[[f32; N_GV]; 2]) -> [f32; 2 * N_GV] {
    let mut out = [0f32; 2 * N_GV];
    for e in 0..2 {
        let r = &raw[e];
        let o = &mut out[e * N_GV..(e + 1) * N_GV];
        if r[GE_PRESENT] < 0.5 {
            continue;
        }
        for k in [GE_PRESENT, GE_KOBJ, GE_KPT, GE_KNOWN, GE_LOST] {
            o[k] = r[k];
        }
        if r[GE_KNOWN] < 0.5 {
            continue;
        }
        for a in 0..3 {
            o[GE_POS + a] = norm_len(a, r[GE_POS + a]);
            o[GE_EEF + a] = norm_len(3 + a, r[GE_EEF + a]);
        }
        o[GE_DIST] = norm_len(6, r[GE_DIST]);
        o[GE_SIN] = r[GE_SIN];
        o[GE_COS] = r[GE_COS];
    }
    out
}

// ---------------------------------------------------------------- 목표 지정·칸 만들기

#[derive(Clone, Debug, PartialEq)]
pub enum GoalRef {
    /// 기억 물체 id("O12")
    Obj(String),
    /// map 좌표 점(놓을 자리 / 갈 자리) — 시작 때 검사·옮김을 지난 값
    Point([f64; 3]),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GoalSpec {
    pub pick: Option<GoalRef>,
    pub place: Option<GoalRef>,
}

impl GoalSpec {
    pub fn get(&self, e: usize) -> Option<&GoalRef> {
        if e == PICK {
            self.pick.as_ref()
        } else {
            self.place.as_ref()
        }
    }
    pub fn pick_id(&self) -> Option<&str> {
        match &self.pick {
            Some(GoalRef::Obj(s)) => Some(s),
            _ => None,
        }
    }
    pub fn place_id(&self) -> Option<&str> {
        match &self.place {
            Some(GoalRef::Obj(s)) => Some(s),
            _ => None,
        }
    }
    pub fn place_point(&self) -> Option<[f64; 3]> {
        match &self.place {
            Some(GoalRef::Point(p)) => Some(*p),
            _ => None,
        }
    }
    pub fn to_json(&self) -> Value {
        let one = |g: &Option<GoalRef>| match g {
            None => Value::Null,
            Some(GoalRef::Obj(s)) => json!({"id": s}),
            Some(GoalRef::Point(p)) => json!({"point": p.map(|x| (x * 1000.0).round() / 1000.0)}),
        };
        json!({"pick": one(&self.pick), "place": one(&self.place)})
    }
}

/// 이번 스텝의 목표 칸(원값·정규화) + 각본 정책·확인용 값
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GoalEntries {
    pub raw: [[f32; N_GV]; 2],
    pub norm: Vec<f32>,
    /// 칸마다 map 위치(앎일 때) — 기록·확인용(정책에는 주지 않음)
    pub map_pos: [Option<[f64; 3]>; 2],
}

/// 칸 하나(원값). `p` = map 위치(None = 모름), `pose` = 믿는 자세, `eef` = 손끝(base_footprint 기준)
pub fn entry_raw(kind_point: bool, p: Option<[f64; 3]>, lost: bool, pose: [f64; 3], eef: [f64; 3]) -> [f32; N_GV] {
    let mut v = [0f32; N_GV];
    v[GE_PRESENT] = 1.0;
    v[if kind_point { GE_KPT } else { GE_KOBJ }] = 1.0;
    let Some(p) = p else { return v };
    v[GE_KNOWN] = 1.0;
    v[GE_LOST] = if lost && !kind_point { 1.0 } else { 0.0 };
    let r = to_robot(pose, p); // z = 세계 z
    let (x, y, z) = (r[0], r[1], r[2] - BASE_Z);
    let d = x.hypot(y);
    let (sn, cs) = if d < 1e-4 { (0.0, 1.0) } else { (y / d, x / d) };
    let vals = [x, y, z, d, sn, cs, x - eef[0], y - eef[1], z - (eef[2] - BASE_Z)];
    for (k, val) in vals.iter().enumerate() {
        v[GE_POS + k] = round_f16(*val as f32);
    }
    v
}

/// 마지막으로 안 자리(물체가 사라지거나 지도에서 빠져도 이 값 + 잃음 1)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GoalMemory {
    pub last: [Option<[f64; 3]>; 2],
}

/// id → 지금 지도 물체. **자리 맞춤 함수**: 실시간 지도 id 풀이(다른 작업자의 search_objects 실시간 기억)가 move_robot 에
/// 들어오면 이 함수만 바꾼다. 지금은 `mr_vla_set_objects` 로 받은 기억 스냅숏에서 id 로 찾는다.
pub fn resolve<'a>(objs: &'a [MemObject], id: &str) -> Option<&'a MemObject> {
    objs.iter().find(|o| o.id.eq_ignore_ascii_case(id))
}

/// 매 스텝: 지정 → 칸 2 개
pub fn build(spec: &GoalSpec, objs: &[MemObject], pose: [f64; 3], eef: [f64; 3], mem: &mut GoalMemory) -> GoalEntries {
    let mut g = GoalEntries::default();
    for e in 0..2 {
        match spec.get(e) {
            None => {}
            Some(GoalRef::Point(p)) => {
                g.raw[e] = entry_raw(true, Some(*p), false, pose, eef);
                g.map_pos[e] = Some(*p);
            }
            Some(GoalRef::Obj(id)) => {
                let (p, lost) = match resolve(objs, id) {
                    Some(o) if o.state != ObjState::Gone => {
                        mem.last[e] = Some(o.pos);
                        (Some(o.pos), false)
                    }
                    Some(o) => {
                        // 사라짐: 기억의 그 자리가 마지막으로 안 자리
                        let p = mem.last[e].unwrap_or(o.pos);
                        (Some(p), true)
                    }
                    None => (mem.last[e], true),
                };
                g.raw[e] = entry_raw(false, p, lost, pose, eef);
                g.map_pos[e] = p;
            }
        }
    }
    g.norm = normalize(&g.raw).to_vec();
    g
}

// ---------------------------------------------------------------- 지점 검사·옮기기

/// 놓을 면 윗면 높이(세계 z, CURRICULUM_BEHAVIOR2026 3.1 place_top 느슨 0.05–0.52 m)
pub const TOP_MIN: f64 = 0.05;
pub const TOP_MAX: f64 = 0.52;
/// 면 위 지점은 윗면 사각형을 이만큼 줄인 안쪽(물체가 가장자리에 걸리지 않게, 추정)
pub const TOP_MARGIN: f64 = 0.03;
/// 이보다 높은 면은 옆 잡기 문턱(3.1 topdown_z)
pub const TOPDOWN_Z: f64 = 0.25;
/// 몸통 가운데 → 놓을 점 닿는 거리: 낮은 곳(바닥·≤ 0.25 m 면) 0.38 m(= 옆 0.11 + 0.27), 높은 면 0.31 m(= 0.21 + 가장자리 안 0.10)
pub const REACH_LOW: f64 = 0.38;
pub const REACH_HIGH: f64 = 0.31;
/// 바닥 높이(세계 z, 가정: 한 층 지도)
pub const FLOOR_Z: f64 = 0.0;
/// 바닥 점: 이 높이 안이면 바닥으로 본다
pub const FLOOR_TOL: f64 = 0.05;
/// 옮기기 한도(m) — 이 안에 맞는 자리가 없으면 거절
pub const SNAP_MAX: f64 = 1.0;
/// 바닥 점 둘레 빈 반지름(물체 바닥 자국, 추정)
pub const FLOOR_CLEAR: f64 = 0.05;

#[derive(Clone, Debug, PartialEq)]
pub struct PointCheck {
    /// 검사·옮긴 뒤 점(map)
    pub point: [f64; 3],
    /// "floor" | "surface"
    pub on: &'static str,
    /// 받침 물체 id(면일 때) — 기록용(정책 입력 아님)
    pub support: Option<String>,
    pub snap_m: f64,
    /// 지도가 있어 바닥 닿음까지 검사했나
    pub validated: bool,
}

impl PointCheck {
    pub fn to_json(&self, asked: [f64; 3]) -> Value {
        let r3 = |p: [f64; 3]| p.map(|x| (x * 1000.0).round() / 1000.0);
        json!({"asked": r3(asked), "point": r3(self.point), "on": self.on, "support": self.support, "snap_m": (self.snap_m * 1000.0).round() / 1000.0, "validated": self.validated})
    }
}

/// 지도(격자 + 로봇이 닿는 칸)
pub struct PlaceMap<'a> {
    pub grid: &'a crate::map::Grid,
    pub a: &'a crate::map::Analysis,
}

fn top_of(o: &MemObject) -> f64 {
    o.pos[2] + 0.5 * o.extent[2]
}

/// 면 윗면 안쪽 사각형(줄인 반 크기)
fn top_rect(o: &MemObject) -> Option<(f64, f64)> {
    let (hx, hy) = (0.5 * o.extent[0] - TOP_MARGIN, 0.5 * o.extent[1] - TOP_MARGIN);
    (hx > 0.0 && hy > 0.0).then_some((hx, hy))
}

fn support_ok(o: &MemObject) -> bool {
    let t = top_of(o);
    o.state != ObjState::Gone && o.state != ObjState::Held && (TOP_MIN..=TOP_MAX).contains(&t) && top_rect(o).is_some()
}

/// 로봇이 설 수 있는 칸에서 이 점까지 닿는가(지도 없으면 None = 모름)
fn reach_ok(pm: Option<&PlaceMap>, x: f64, y: f64, reach: f64) -> Option<bool> {
    let pm = pm?;
    Some(pm.a.nearest_reachable(pm.grid, x, y, reach).is_some())
}

/// 바닥 점: 아는 빈칸, 둘레 FLOOR_CLEAR 안에 막힌 칸 없음, 다른 물체 바닥 자국 밖, 서는 칸에서 닿음
fn floor_ok(pm: &PlaceMap, objs: &[MemObject], x: f64, y: f64) -> bool {
    let g = pm.grid;
    let k = (FLOOR_CLEAR / g.res).ceil() as i64;
    let (cx, cy) = g.cell_of(x, y);
    for dy in -k..=k {
        for dx in -k..=k {
            match g.idx(cx + dx, cy + dy) {
                Some(i) if crate::map::is_free(g.cells[i]) => {}
                _ => return false,
            }
        }
    }
    let inside = objs.iter().any(|o| o.state != ObjState::Gone && (x - o.pos[0]).abs() < 0.5 * o.extent[0] + FLOOR_CLEAR && (y - o.pos[1]).abs() < 0.5 * o.extent[1] + FLOOR_CLEAR && o.pos[2] - 0.5 * o.extent[2] < FLOOR_Z + 0.3);
    !inside && reach_ok(Some(pm), x, y, REACH_LOW) == Some(true)
}

/// 지점 검사: 바닥(아는 빈칸 + 닿음) 또는 면 위(윗면 0.05–0.52 m, 줄인 사각형 안, 닿음). 아니면 SNAP_MAX 안 가장 가까운 맞는 자리로.
/// `z` 가 없으면 xy 가 면 위면 그 면, 아니면 바닥. 지도가 없으면 바닥 점은 검사 없이 받는다(validated = false).
pub fn check_point(xy: [f64; 2], z: Option<f64>, objs: &[MemObject], pm: Option<&PlaceMap>) -> Result<PointCheck, String> {
    let want_floor = z.map(|z| (z - FLOOR_Z).abs() <= FLOOR_TOL);
    // 1) 면 후보(점에서 가장 가까운 윗면 안쪽 점)
    let mut best: Option<(f64, PointCheck)> = None;
    let mut consider = |d: f64, c: PointCheck| {
        if d <= SNAP_MAX + 1e-9 && best.as_ref().map_or(true, |(b, _)| d < *b - 1e-12) {
            best = Some((d, c));
        }
    };
    if want_floor != Some(true) {
        for o in objs.iter().filter(|o| support_ok(o)) {
            let t = top_of(o);
            if let Some(z) = z {
                if (z - t).abs() > FLOOR_TOL {
                    continue;
                }
            }
            let (hx, hy) = top_rect(o).unwrap();
            let px = xy[0].clamp(o.pos[0] - hx, o.pos[0] + hx);
            let py = xy[1].clamp(o.pos[1] - hy, o.pos[1] + hy);
            let reach = if t > TOPDOWN_Z { REACH_HIGH } else { REACH_LOW };
            if reach_ok(pm, px, py, reach) == Some(false) {
                continue;
            }
            let d = (px - xy[0]).hypot(py - xy[1]);
            consider(d, PointCheck { point: [px, py, t], on: "surface", support: Some(o.id.clone()), snap_m: d, validated: pm.is_some() });
        }
    }
    // 2) 바닥 후보
    if want_floor != Some(false) {
        match pm {
            None => {
                // 지도 없음: 면 안이 아니면 그대로 받음(검사 못 함)
                let on_top = objs.iter().any(|o| o.state != ObjState::Gone && (xy[0] - o.pos[0]).abs() < 0.5 * o.extent[0] && (xy[1] - o.pos[1]).abs() < 0.5 * o.extent[1]);
                if !on_top || want_floor == Some(true) {
                    consider(0.0, PointCheck { point: [xy[0], xy[1], FLOOR_Z], on: "floor", support: None, snap_m: 0.0, validated: false });
                }
            }
            Some(m) => {
                let g = m.grid;
                if floor_ok(m, objs, xy[0], xy[1]) {
                    consider(0.0, PointCheck { point: [xy[0], xy[1], FLOOR_Z], on: "floor", support: None, snap_m: 0.0, validated: true });
                } else {
                    let r = (SNAP_MAX / g.res).ceil() as i64;
                    let (cx, cy) = g.cell_of(xy[0], xy[1]);
                    let mut fb: Option<(f64, [f64; 2])> = None;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let Some(i) = g.idx(cx + dx, cy + dy) else { continue };
                            let (px, py) = g.center(i);
                            let d = (px - xy[0]).hypot(py - xy[1]);
                            if d > SNAP_MAX || fb.map_or(false, |(b, _)| d >= b) {
                                continue;
                            }
                            if floor_ok(m, objs, px, py) {
                                fb = Some((d, [px, py]));
                            }
                        }
                    }
                    if let Some((d, p)) = fb {
                        consider(d, PointCheck { point: [p[0], p[1], FLOOR_Z], on: "floor", support: None, snap_m: d, validated: true });
                    }
                }
            }
        }
    }
    best.map(|b| b.1).ok_or_else(|| {
        format!("no placeable floor or surface point within {SNAP_MAX:.1} m of ({:.2}, {:.2}){}", xy[0], xy[1], match z {
            Some(z) => format!(" at z {z:.2}"),
            None => String::new(),
        })
    })
}

/// 지점 놓기 확인(공유 정의): 놓임 + 물체 가운데 수평 POINT_R 안 + 바닥 높이 ± POINT_DZ
pub fn at_point(o: &MemObject, p: [f64; 3]) -> (bool, f64, f64) {
    let d = (o.pos[0] - p[0]).hypot(o.pos[1] - p[1]);
    let dz = (o.pos[2] - 0.5 * o.extent[2]) - p[2];
    (o.state != ObjState::Held && d <= POINT_R && dz.abs() <= POINT_DZ, d, dz)
}

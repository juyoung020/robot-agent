//! `move_robot` — LLM 이 로봇 한 부분(베이스·몸통·팔·그리퍼)을 직접 움직이는 도구 하나.
//!
//! 이 크레이트가 하는 일은 이것뿐이다.
//! - [`definition`]: OpenAI `tools` 스키마(9B 모델용으로 작게: enum·필수 인자, 짧은 설명)
//! - [`parse`]: 도구 인자 → [`Command`] (틀리면 모델이 고칠 수 있는 오류 문장)
//! - [`Robot`]: 매 시뮬 스텝 proprio(61) 를 받아 BEHAVIOR R1Pro 행동 벡터(23)를 내는 닫힌 고리 실행기.
//!   관절은 최소 저크 보간 + 도착/막힘/시간 초과 판정, 베이스는 base_qvel 적분(오도메트리) 위 P 제어 + 가감속 한도.
//! - [`ffi`]: 같은 실행기를 C ABI 로(시뮬 쪽 파이썬은 ctypes 로 부른다)
//! - [`link`]: 에이전트 쪽 실행 경로(시뮬 TCP / 가짜 로봇)
//!
//! 단위: 길이 m, 각도 도(°, LLM 에게 보이는 값). 안에서는 rad. 그리퍼는 0 = 닫힘 … 1 = 열림.
//! 행동·관측 배치는 BEHAVIOR-1K `omnigibson/eval/utils/eval_utils.py` 의 `ACTION_QPOS_INDICES` / `PROPRIOCEPTION_INDICES`,
//! 제어기는 `omnigibson/eval/r1pro.yaml`(팔·몸통 = 절대 관절 위치, 베이스 = 로봇 기준 속도 [-1,1] → [±0.75 m/s, ±0.75, ±1 rad/s],
//! 그리퍼 = smooth [-1,1] → 손가락 0…0.05 m). 머리(카메라)는 NullJointController 라 움직일 수 없다.

pub mod ffi;
pub mod link;

use serde_json::{json, Value};
use std::f64::consts::PI;

pub const ACTION_DIM: usize = 23;
pub const PROPRIO_DIM: usize = 61;
pub const HZ: f64 = 30.0;
pub const TOOL_NAME: &str = "move_robot";

/// 행동 벡터 칸 (ACTION_QPOS_INDICES["R1Pro"])
pub mod act {
    use std::ops::Range;
    pub const BASE: Range<usize> = 0..3;
    pub const TORSO: Range<usize> = 3..7;
    pub const LEFT_ARM: Range<usize> = 7..14;
    pub const LEFT_GRIPPER: usize = 14;
    pub const RIGHT_ARM: Range<usize> = 15..22;
    pub const RIGHT_GRIPPER: usize = 22;
    /// 베이스 명령 [-1,1] 이 뜻하는 실제 속도 (r1pro.yaml command_output_limits)
    pub const BASE_OUT: [f64; 3] = [0.75, 0.75, 1.0];
}

/// proprio 칸 (PROPRIOCEPTION_INDICES["R1Pro"])
pub mod prop {
    use std::ops::Range;
    pub const BASE_QVEL: Range<usize> = 0..3;
    pub const ARM_L_QPOS: Range<usize> = 3..10;
    pub const ARM_L_QVEL: Range<usize> = 10..17;
    pub const GRIP_L_QPOS: Range<usize> = 24..26;
    pub const GRIP_L_QVEL: Range<usize> = 26..28;
    pub const ARM_R_QPOS: Range<usize> = 28..35;
    pub const ARM_R_QVEL: Range<usize> = 35..42;
    pub const GRIP_R_QPOS: Range<usize> = 49..51;
    pub const GRIP_R_QVEL: Range<usize> = 51..53;
    pub const TRUNK_QPOS: Range<usize> = 53..57;
    pub const TRUNK_QVEL: Range<usize> = 57..61;
}

/// 손가락 하나의 최대 벌림(m, URDF finger_joint 상한)
pub const FINGER_MAX: f64 = 0.05;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Part {
    Base,
    Torso,
    LeftArm,
    RightArm,
    LeftGripper,
    RightGripper,
}

pub const PARTS: [&str; 6] = ["base", "torso", "left_arm", "right_arm", "left_gripper", "right_gripper"];

impl Part {
    pub fn from_name(s: &str) -> Option<Part> {
        let n: String = s.trim().to_ascii_lowercase().chars().map(|c| if c == ' ' || c == '-' { '_' } else { c }).collect();
        Some(match n.as_str() {
            "base" => Part::Base,
            "torso" | "trunk" => Part::Torso,
            "left_arm" => Part::LeftArm,
            "right_arm" => Part::RightArm,
            "left_gripper" | "left_hand" => Part::LeftGripper,
            "right_gripper" | "right_hand" => Part::RightGripper,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        PARTS[self as usize]
    }
    pub fn dof(self) -> usize {
        match self {
            Part::Base => 3,
            Part::Torso => 4,
            Part::LeftArm | Part::RightArm => 7,
            Part::LeftGripper | Part::RightGripper => 1,
        }
    }
    pub fn is_gripper(self) -> bool {
        matches!(self, Part::LeftGripper | Part::RightGripper)
    }
    /// LLM 에게 보이는 단위
    pub fn units(self) -> &'static str {
        match self {
            Part::Base => "[forward m, left m, turn_left deg]",
            Part::Torso | Part::LeftArm | Part::RightArm => "deg",
            _ => "0=closed..1=open",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Delta,
    Absolute,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Command {
    pub part: Part,
    pub mode: Mode,
    /// LLM 단위(m, deg, 0..1) 그대로
    pub values: Vec<f64>,
    pub duration_s: Option<f64>,
}

// ---------------------------------------------------------------- 안전 한계

/// 관절 한계 안쪽 여유(rad) — URDF 한계에 붙지 않게
pub const JOINT_MARGIN: f64 = 2.0 * PI / 180.0;

/// URDF(`r1pro.urdf`) torso_joint1..4 (rad)
pub const TORSO_LIM: [(f64, f64); 4] = [(-1.1345, 1.8326), (-2.7925, 2.5307), (-1.8326, 1.5708), (-3.0543, 3.0543)];
/// left_arm_joint1..7 (rad)
pub const LEFT_ARM_LIM: [(f64, f64); 7] = [
    (-4.4506, 1.3090),
    (-0.1745, 3.1416),
    (-2.356196, 2.356196),
    (-2.0944, 0.3491),
    (-2.356196, 2.356196),
    (-1.047198, 1.047198),
    (-1.5708, 1.5708),
];
/// right_arm_joint1..7 (rad) — 2번만 왼팔과 거울
pub const RIGHT_ARM_LIM: [(f64, f64); 7] = [
    (-4.4506, 1.3090),
    (-3.1416, 0.1745),
    (-2.356196, 2.356196),
    (-2.0944, 0.3491),
    (-2.356196, 2.356196),
    (-1.047198, 1.047198),
    (-1.5708, 1.5708),
];

/// 안전 속도·가속도와 판정 문턱. 기본값은 사람이 옆에 있어도 놀라지 않을 정도로 느리게.
#[derive(Clone, Debug)]
pub struct Safety {
    pub arm_vmax: f64,     // rad/s (URDF 7~10 rad/s 의 약 1/10)
    pub torso_vmax: f64,   // rad/s
    pub gripper_vmax: f64, // 벌림 비율/s (1 = 1 초에 다 열기)
    pub base_vmax: f64,    // m/s
    pub base_wmax: f64,    // rad/s
    pub base_acc: f64,     // m/s²
    pub base_wacc: f64,    // rad/s²
    pub base_max_xy: f64,  // 한 번에 움직일 거리 상한(축마다, m)
    pub base_max_yaw: f64, // 한 번에 돌 각 상한(rad)
    pub min_duration: f64,
    pub max_duration: f64,
    /// 관절 도착 허용 오차(rad) / 그리퍼(비율) / 베이스(m, rad)
    pub joint_tol: f64,
    pub gripper_tol: f64,
    pub base_tol_xy: f64,
    pub base_tol_yaw: f64,
    /// 움직이는 중 지령과 실제 차이가 이만큼(rad) 넘게 몇 스텝 이어지면 막힘
    pub track_limit: f64,
    /// 보간 끝난 뒤 기다리는 최대 시간(s)
    pub settle_s: f64,
}

impl Default for Safety {
    fn default() -> Self {
        Safety {
            arm_vmax: 45f64.to_radians(),
            torso_vmax: 20f64.to_radians(),
            gripper_vmax: 1.0,
            base_vmax: 0.3,
            base_wmax: 35f64.to_radians(),
            base_acc: 0.6,
            base_wacc: 90f64.to_radians(),
            base_max_xy: 2.0,
            base_max_yaw: PI,
            min_duration: 0.1,
            max_duration: 20.0,
            joint_tol: 1.5f64.to_radians(),
            gripper_tol: 0.05,
            base_tol_xy: 0.02,
            base_tol_yaw: 2f64.to_radians(),
            track_limit: 20f64.to_radians(),
            settle_s: 1.5,
        }
    }
}

/// 부분의 관절 한계(안쪽 여유 포함, 안 단위: rad / 비율)
pub fn joint_limits(part: Part) -> Vec<(f64, f64)> {
    let shrink = |l: &[(f64, f64)]| l.iter().map(|&(a, b)| (a + JOINT_MARGIN, b - JOINT_MARGIN)).collect();
    match part {
        Part::Torso => shrink(&TORSO_LIM),
        Part::LeftArm => shrink(&LEFT_ARM_LIM),
        Part::RightArm => shrink(&RIGHT_ARM_LIM),
        Part::LeftGripper | Part::RightGripper => vec![(0.0, 1.0)],
        Part::Base => vec![],
    }
}

/// LLM 단위 → 안 단위 (deg → rad; m·비율은 그대로)
pub fn to_internal(part: Part, v: &[f64]) -> Vec<f64> {
    match part {
        Part::Torso | Part::LeftArm | Part::RightArm => v.iter().map(|x| x.to_radians()).collect(),
        Part::Base => vec![v[0], v[1], v[2].to_radians()],
        _ => v.to_vec(),
    }
}

/// 안 단위 → LLM 단위(반올림 포함: 각도 0.1°, 길이 1 mm, 비율 0.01)
pub fn to_user(part: Part, v: &[f64]) -> Vec<f64> {
    let r = |x: f64, k: f64| (x * k).round() / k;
    match part {
        Part::Torso | Part::LeftArm | Part::RightArm => v.iter().map(|x| r(x.to_degrees(), 10.0)).collect(),
        Part::Base => vec![r(v[0], 1000.0), r(v[1], 1000.0), r(v[2].to_degrees(), 10.0)],
        _ => v.iter().map(|x| r(*x, 100.0)).collect(),
    }
}

// ---------------------------------------------------------------- 스키마

pub const DESCRIPTION: &str = "Directly move ONE robot part and wait until it stops. Units: meters, degrees. \
values order: base [forward_m, left_m, turn_left_deg] (delta only, robot frame); torso [j1..j4]; left_arm/right_arm [j1..j7] shoulder to wrist; \
gripper [opening] 0=closed..1=open (use absolute). mode delta = add to current, absolute = go to value. \
delta with all zeros only reads the state. Joint limits and safe speeds are enforced.";

/// OpenAI Chat Completions `tools` 항목 하나
pub fn definition() -> Value {
    json!({"type": "function", "function": {"name": TOOL_NAME, "description": DESCRIPTION, "parameters": {
        "type": "object",
        "properties": {
            "part": {"type": "string", "enum": PARTS},
            "mode": {"type": "string", "enum": ["delta", "absolute"]},
            "values": {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 7,
                       "description": "base 3, torso 4, arm 7, gripper 1"},
            "duration_s": {"type": "number", "description": "optional; slower if too fast for safe speed"}
        },
        "required": ["part", "mode", "values"]
    }}})
}

/// 도구 인자 → 명령. 인자는 객체 또는 JSON 문자열(OpenAI `arguments`). 오류는 모델이 고칠 수 있게 짧게.
pub fn parse(args: &Value) -> Result<Command, String> {
    let owned;
    let a = match args {
        Value::String(s) => {
            owned = serde_json::from_str::<Value>(s).map_err(|e| format!("arguments are not JSON: {e}"))?;
            &owned
        }
        v => v,
    };
    if !a.is_object() {
        return Err("arguments must be an object {part, mode, values}".into());
    }
    let part_s = a.get("part").and_then(|v| v.as_str()).unwrap_or("");
    let part = Part::from_name(part_s).ok_or_else(|| format!("part must be one of {} (got '{part_s}')", PARTS.join(", ")))?;
    let mode = match a.get("mode").and_then(|v| v.as_str()).map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("delta") | Some("relative") => Mode::Delta,
        Some("absolute") => Mode::Absolute,
        Some(m) => return Err(format!("mode must be delta or absolute (got '{m}')")),
        None => return Err("mode is required: delta or absolute".into()),
    };
    if part == Part::Base && mode == Mode::Absolute {
        return Err("base supports only mode=delta: [forward_m, left_m, turn_left_deg] from where it is now".into());
    }
    let num = |v: &Value| -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.trim().parse::<f64>().ok(),
            _ => None,
        }
    };
    let values: Vec<f64> = match a.get("values") {
        Some(Value::Array(xs)) => {
            let mut out = Vec::with_capacity(xs.len());
            for (i, x) in xs.iter().enumerate() {
                out.push(num(x).ok_or_else(|| format!("values[{i}] is not a number"))?);
            }
            out
        }
        Some(x) if part.dof() == 1 && num(x).is_some() => vec![num(x).unwrap()],
        Some(_) => return Err("values must be an array of numbers".into()),
        None => return Err(format!("values is required: {} numbers, {}", part.dof(), part.units())),
    };
    if values.len() != part.dof() {
        return Err(format!("{} needs exactly {} values ({}), got {}", part.name(), part.dof(), part.units(), values.len()));
    }
    if let Some(i) = values.iter().position(|x| !x.is_finite()) {
        return Err(format!("values[{i}] is not finite"));
    }
    let duration_s = match a.get("duration_s") {
        None | Some(Value::Null) => None,
        Some(v) => Some(num(v).filter(|d| d.is_finite() && *d > 0.0).ok_or("duration_s must be a positive number")?),
    };
    Ok(Command { part, mode, values, duration_s })
}

// ---------------------------------------------------------------- 관측

/// proprio 에서 이 도구가 쓰는 값만 (안 단위)
#[derive(Clone, Debug, Default)]
pub struct State {
    pub base_v: [f64; 3],
    pub torso: [f64; 4],
    pub torso_v: [f64; 4],
    pub arm_l: [f64; 7],
    pub arm_l_v: [f64; 7],
    pub arm_r: [f64; 7],
    pub arm_r_v: [f64; 7],
    /// 벌림 비율 0..1 (두 손가락 평균 / FINGER_MAX)
    pub grip_l: f64,
    pub grip_l_v: f64,
    pub grip_r: f64,
    pub grip_r_v: f64,
}

fn arr<const N: usize>(p: &[f32], r: std::ops::Range<usize>) -> [f64; N] {
    let mut o = [0.0; N];
    for (k, i) in r.enumerate() {
        o[k] = p[i] as f64;
    }
    o
}

fn mean(p: &[f32], r: std::ops::Range<usize>) -> f64 {
    let n = r.len() as f64;
    p[r].iter().map(|x| *x as f64).sum::<f64>() / n
}

impl State {
    pub fn from_proprio(p: &[f32]) -> Result<State, String> {
        if p.len() < PROPRIO_DIM {
            return Err(format!("proprio has {} values, need {PROPRIO_DIM} (R1Pro)", p.len()));
        }
        if p[..PROPRIO_DIM].iter().any(|x| !x.is_finite()) {
            return Err("proprio has NaN/inf".into());
        }
        Ok(State {
            base_v: arr(p, prop::BASE_QVEL),
            torso: arr(p, prop::TRUNK_QPOS),
            torso_v: arr(p, prop::TRUNK_QVEL),
            arm_l: arr(p, prop::ARM_L_QPOS),
            arm_l_v: arr(p, prop::ARM_L_QVEL),
            arm_r: arr(p, prop::ARM_R_QPOS),
            arm_r_v: arr(p, prop::ARM_R_QVEL),
            grip_l: mean(p, prop::GRIP_L_QPOS) / FINGER_MAX,
            grip_l_v: mean(p, prop::GRIP_L_QVEL) / FINGER_MAX,
            grip_r: mean(p, prop::GRIP_R_QPOS) / FINGER_MAX,
            grip_r_v: mean(p, prop::GRIP_R_QVEL) / FINGER_MAX,
        })
    }
    /// 관절 부분의 위치·속도 (베이스는 빈 값)
    pub fn joints(&self, part: Part) -> (Vec<f64>, Vec<f64>) {
        match part {
            Part::Torso => (self.torso.to_vec(), self.torso_v.to_vec()),
            Part::LeftArm => (self.arm_l.to_vec(), self.arm_l_v.to_vec()),
            Part::RightArm => (self.arm_r.to_vec(), self.arm_r_v.to_vec()),
            Part::LeftGripper => (vec![self.grip_l], vec![self.grip_l_v]),
            Part::RightGripper => (vec![self.grip_r], vec![self.grip_r_v]),
            Part::Base => (vec![], vec![]),
        }
    }
}

/// 그리퍼 벌림 비율(0..1) ↔ 행동 값([-1,1], smooth 모드: -1 → 0 m, +1 → 0.05 m)
pub fn gripper_to_action(frac: f64) -> f64 {
    (2.0 * frac - 1.0).clamp(-1.0, 1.0)
}

/// 행동 벡터에서 부분이 차지하는 칸
pub fn action_slots(part: Part) -> std::ops::Range<usize> {
    match part {
        Part::Base => act::BASE,
        Part::Torso => act::TORSO,
        Part::LeftArm => act::LEFT_ARM,
        Part::RightArm => act::RIGHT_ARM,
        Part::LeftGripper => act::LEFT_GRIPPER..act::LEFT_GRIPPER + 1,
        Part::RightGripper => act::RIGHT_GRIPPER..act::RIGHT_GRIPPER + 1,
    }
}

// ---------------------------------------------------------------- 보간

/// 최소 저크 s(τ) = 10τ³ − 15τ⁴ + 6τ⁵ (τ ∈ [0,1]), 최고 속도 = 1.875 · Δ / T
pub fn min_jerk(tau: f64) -> f64 {
    let t = tau.clamp(0.0, 1.0);
    t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
}
pub const MIN_JERK_PEAK: f64 = 1.875;

/// 움직임 시간: 요청 시간과 "가장 바쁜 관절이 안전 속도를 넘지 않는 시간" 중 긴 쪽. (시간, 늦췄나)
pub fn plan_duration(delta: &[f64], vmax: f64, requested: Option<f64>, s: &Safety) -> (f64, bool) {
    let dmax = delta.iter().fold(0.0f64, |m, d| m.max(d.abs()));
    let need = (MIN_JERK_PEAK * dmax / vmax).max(s.min_duration);
    match requested {
        Some(r) => {
            let r = r.clamp(s.min_duration, s.max_duration);
            if r + 1e-9 < need {
                (need, true) // 안전 속도가 우선: 늦춘다
            } else {
                (r, false)
            }
        }
        None => (need, false),
    }
}

/// 목표를 한계 안으로. 잘린 칸 번호를 돌려준다.
pub fn clamp_target(target: &mut [f64], lim: &[(f64, f64)]) -> Vec<usize> {
    let mut clamped = vec![];
    for (i, (t, &(lo, hi))) in target.iter_mut().zip(lim).enumerate() {
        let c = t.clamp(lo, hi);
        if (c - *t).abs() > 1e-9 {
            clamped.push(i);
            *t = c;
        }
    }
    clamped
}

// ---------------------------------------------------------------- 실행기

#[derive(Clone, Debug)]
struct JointMove {
    part: Part,
    start: Vec<f64>,
    target: Vec<f64>,
    t_move: f64,
    still: u32,
    off_track: u32,
}

#[derive(Clone, Debug)]
struct BaseMove {
    target: [f64; 3],
    pose: [f64; 3],
    v_cmd: [f64; 3],
    t_max: f64,
    stall: u32,
}

#[derive(Clone, Debug)]
enum Motion {
    Joints(JointMove),
    Base(BaseMove),
}

#[derive(Clone, Debug)]
struct Active {
    cmd: Command,
    motion: Motion,
    clamped: Vec<usize>,
    slowed: bool,
    steps: u64,
}

/// 매 시뮬 스텝 하나씩 부르는 닫힌 고리 실행기. 명령이 없을 때는 마지막 목표를 유지(그 자리 버티기, 베이스 0 속도).
pub struct Robot {
    pub safety: Safety,
    pub dt: f64,
    /// 유지 중인 행동(팔·몸통 = 관절 목표 rad, 그리퍼 = [-1,1], 베이스 = 0)
    hold: [f64; ACTION_DIM],
    state: Option<State>,
    active: Option<Active>,
    result: Option<Value>,
    pub ticks: u64,
}

/// [`Robot::tick`] 결과
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tick {
    Idle,
    Moving,
    /// 이번 스텝에 명령이 끝났다 → [`Robot::take_result`]
    Done,
    /// proprio 가 이상했다(행동 = 유지값, 베이스 0)
    BadObs,
}

impl Default for Robot {
    fn default() -> Self {
        Robot::new(HZ)
    }
}

impl Robot {
    pub fn new(hz: f64) -> Robot {
        Robot { safety: Safety::default(), dt: 1.0 / hz, hold: [0.0; ACTION_DIM], state: None, active: None, result: None, ticks: 0 }
    }

    pub fn busy(&self) -> bool {
        self.active.is_some()
    }
    pub fn has_state(&self) -> bool {
        self.state.is_some()
    }
    pub fn take_result(&mut self) -> Option<Value> {
        self.result.take()
    }
    pub fn put_result(&mut self, v: Value) {
        self.result = Some(v);
    }
    pub fn hold(&self) -> &[f64; ACTION_DIM] {
        &self.hold
    }

    /// 판이 새로 시작될 때: 다음 관측에서 유지값을 다시 잡는다
    pub fn reset(&mut self) {
        if self.active.is_some() {
            self.result = Some(json!({"status": "error", "message": "episode reset while moving"}));
        }
        self.active = None;
        self.state = None;
    }

    /// 지금 관측으로 유지값을 잡는다(첫 관측)
    fn init_hold(&mut self, s: &State) {
        let h = &mut self.hold;
        h[act::BASE].fill(0.0);
        h[act::TORSO].copy_from_slice(&s.torso);
        h[act::LEFT_ARM].copy_from_slice(&s.arm_l);
        h[act::RIGHT_ARM].copy_from_slice(&s.arm_r);
        h[act::LEFT_GRIPPER] = gripper_to_action(s.grip_l);
        h[act::RIGHT_GRIPPER] = gripper_to_action(s.grip_r);
    }

    fn set_hold(&mut self, part: Part, q: &[f64]) {
        let r = action_slots(part);
        if part.is_gripper() {
            self.hold[r.start] = gripper_to_action(q[0]);
        } else if part != Part::Base {
            self.hold[r].copy_from_slice(q);
        }
    }

    /// 지금 상태(LLM 단위) — 관절 부분은 위치, 베이스는 [0,0,0](오도메트리는 명령마다 0 에서 시작)
    fn user_state(&self, part: Part) -> Vec<f64> {
        match (part, &self.state) {
            (Part::Base, _) | (_, None) => vec![0.0; part.dof()],
            (p, Some(s)) => to_user(p, &s.joints(p).0),
        }
    }

    /// 명령 시작. `Ok(true)` = 바로 끝남(읽기·오류 → 결과 준비됨), `Ok(false)` = 움직이기 시작.
    pub fn command(&mut self, args: &Value) -> bool {
        match self.start(args) {
            Ok(done) => done,
            Err(e) => {
                self.result = Some(error_obs(&e));
                true
            }
        }
    }

    fn start(&mut self, args: &Value) -> Result<bool, String> {
        if self.active.is_some() {
            return Err("still executing the previous move_robot call".into());
        }
        let cmd = parse(args)?;
        let Some(st) = self.state.clone() else { return Err("no robot observation yet; try again in a moment".into()) };
        let s = self.safety.clone();
        let part = cmd.part;
        let v = to_internal(part, &cmd.values);

        // 읽기만: delta 이고 전부 0
        if cmd.mode == Mode::Delta && cmd.values.iter().all(|x| *x == 0.0) {
            self.result = Some(json!({"status": "reached", "part": part.name(), "state": self.user_state(part), "units": part.units(), "steps": 0}));
            return Ok(true);
        }

        let (motion, clamped, slowed) = if part == Part::Base {
            let mut t = [v[0], v[1], v[2]];
            let mut clamped = vec![];
            for (i, lim) in [s.base_max_xy, s.base_max_xy, s.base_max_yaw].into_iter().enumerate() {
                if t[i].abs() > lim {
                    t[i] = t[i].clamp(-lim, lim);
                    clamped.push(i);
                }
            }
            let need = t[0].hypot(t[1]) / s.base_vmax + t[2].abs() / s.base_wmax;
            let t_max = need * 1.5 + 3.0 + cmd.duration_s.unwrap_or(0.0);
            (Motion::Base(BaseMove { target: t, pose: [0.0; 3], v_cmd: [0.0; 3], t_max, stall: 0 }), clamped, false)
        } else {
            let (q, _) = st.joints(part);
            let mut target: Vec<f64> = match cmd.mode {
                Mode::Delta => q.iter().zip(&v).map(|(a, b)| a + b).collect(),
                Mode::Absolute => v.clone(),
            };
            let clamped = clamp_target(&mut target, &joint_limits(part));
            // 보간은 지금 지령(유지값)에서 시작한다 — 측정값에서 시작하면 처진 만큼 한 스텝에 튄다
            let start: Vec<f64> = if part.is_gripper() {
                vec![(self.hold[action_slots(part).start] + 1.0) / 2.0]
            } else {
                self.hold[action_slots(part)].to_vec()
            };
            let delta: Vec<f64> = target.iter().zip(&start).map(|(t, a)| t - a).collect();
            let vmax = match part {
                Part::Torso => s.torso_vmax,
                Part::LeftArm | Part::RightArm => s.arm_vmax,
                _ => s.gripper_vmax,
            };
            let (t_move, slowed) = plan_duration(&delta, vmax, cmd.duration_s, &s);
            (Motion::Joints(JointMove { part, start, target, t_move, still: 0, off_track: 0 }), clamped, slowed)
        };
        self.active = Some(Active { cmd, motion, clamped, slowed, steps: 0 });
        Ok(false)
    }

    /// 시뮬 한 스텝: proprio(61) → 행동(23). 행동은 언제나 채워진다.
    pub fn tick(&mut self, proprio: &[f32], action: &mut [f32]) -> Tick {
        self.ticks += 1;
        let st = match State::from_proprio(proprio) {
            Ok(s) => s,
            Err(_) => {
                self.write_action(action, None);
                return Tick::BadObs;
            }
        };
        if self.state.is_none() {
            self.init_hold(&st);
        }
        self.state = Some(st.clone());
        let Some(mut a) = self.active.take() else {
            self.write_action(action, None);
            return Tick::Idle;
        };
        a.steps += 1;
        let t = a.steps as f64 * self.dt;
        let s = self.safety.clone();
        let mut base_cmd = None;
        let mut outcome: Option<&'static str> = None;
        let mut final_state: Vec<f64> = vec![];
        let mut target_user: Vec<f64> = vec![];
        let mut err_user: Option<Value> = None;
        match &mut a.motion {
            Motion::Joints(m) => {
                let (q, qd) = st.joints(m.part);
                let tau = t / m.t_move;
                let sj = min_jerk(tau);
                let cmd: Vec<f64> = m.start.iter().zip(&m.target).map(|(a0, b)| a0 + sj * (b - a0)).collect();
                let err = m.target.iter().zip(&q).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                let track = cmd.iter().zip(&q).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                let gr = m.part.is_gripper();
                let tol = if gr { s.gripper_tol } else { s.joint_tol };
                let vel = qd.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
                let still_eps = if gr { 0.02 } else { 0.5f64.to_radians() };
                m.still = if vel < still_eps { m.still + 1 } else { 0 };
                // 그리퍼는 물체를 쥐면 원래 못 닫으므로 움직이는 중 막힘 판정을 하지 않는다
                m.off_track = if !gr && tau < 1.0 && track > s.track_limit { m.off_track + 1 } else { 0 };
                let mut hold_q = cmd.clone();
                if m.off_track >= 6 {
                    outcome = Some("blocked");
                    hold_q = q.clone(); // 미는 것을 멈춘다
                } else if tau >= 1.0 {
                    hold_q = m.target.clone();
                    if err < tol {
                        outcome = Some("reached");
                    } else if m.still >= 8 && t > m.t_move + 0.2 {
                        outcome = Some("blocked");
                        if !gr {
                            hold_q = q.clone();
                        } // 그리퍼는 계속 쥔다(지령 유지)
                    } else if t > m.t_move + s.settle_s {
                        outcome = Some("timeout");
                    }
                }
                self.set_hold(m.part, &hold_q);
                if outcome.is_some() {
                    final_state = to_user(m.part, &q);
                    target_user = to_user(m.part, &m.target);
                    err_user = Some(json!(to_user(m.part, &[err])[0]));
                }
            }
            Motion::Base(m) => {
                // 오도메트리: 측정 base_qvel(로봇 기준)을 명령 시작 자세 기준으로 적분 (planner odom.rs 와 같은 식)
                let [vx, vy, wz] = st.base_v;
                let (sn, cs) = m.pose[2].sin_cos();
                m.pose[0] += (cs * vx - sn * vy) * self.dt;
                m.pose[1] += (sn * vx + cs * vy) * self.dt;
                m.pose[2] += wz * self.dt;
                let (ex_w, ey_w) = (m.target[0] - m.pose[0], m.target[1] - m.pose[1]);
                let (sn, cs) = m.pose[2].sin_cos();
                let (ex, ey) = (cs * ex_w + sn * ey_w, -sn * ex_w + cs * ey_w);
                let eyaw = m.target[2] - m.pose[2];
                let exy = ex.hypot(ey);
                let mut des = [0.0; 3];
                if exy > 1e-6 {
                    let sp = s.base_vmax.min((2.0 * s.base_acc * exy).sqrt()).min(2.0 * exy);
                    des[0] = ex / exy * sp;
                    des[1] = ey / exy * sp;
                }
                let w = s.base_wmax.min((2.0 * s.base_wacc * eyaw.abs()).sqrt()).min(2.0 * eyaw.abs());
                des[2] = eyaw.signum() * w;
                let reached = exy < s.base_tol_xy && eyaw.abs() < s.base_tol_yaw;
                if reached {
                    des = [0.0; 3];
                }
                let lim = [s.base_acc * self.dt, s.base_acc * self.dt, s.base_wacc * self.dt];
                for i in 0..3 {
                    m.v_cmd[i] += (des[i] - m.v_cmd[i]).clamp(-lim[i], lim[i]);
                }
                let cmd_sp = m.v_cmd[0].hypot(m.v_cmd[1]);
                let meas_sp = vx.hypot(vy);
                let moving_cmd = cmd_sp > 0.05 || m.v_cmd[2].abs() > 0.1;
                let barely = meas_sp < 0.2 * cmd_sp.max(0.05) && wz.abs() < 0.2 * m.v_cmd[2].abs().max(0.1);
                m.stall = if moving_cmd && barely { m.stall + 1 } else { 0 };
                if reached {
                    outcome = Some("reached");
                } else if m.stall >= 30 {
                    outcome = Some("blocked");
                } else if t > m.t_max {
                    outcome = Some("timeout");
                }
                if outcome.is_some() {
                    m.v_cmd = [0.0; 3];
                    final_state = to_user(Part::Base, &m.pose);
                    target_user = to_user(Part::Base, &m.target);
                    // 남은 거리(m)와 남은 회전(deg)
                    err_user = Some(json!([(exy * 1000.0).round() / 1000.0, (eyaw.to_degrees() * 10.0).round() / 10.0]));
                }
                base_cmd = Some(m.v_cmd);
            }
        }
        self.write_action(action, base_cmd);
        match outcome {
            None => {
                self.active = Some(a);
                Tick::Moving
            }
            Some(o) => {
                let part = a.cmd.part;
                let mut r = json!({"status": o, "part": part.name(), "state": final_state, "target": target_user, "units": part.units(),
                                   "steps": a.steps, "time_s": (t * 100.0).round() / 100.0});
                if o != "reached" {
                    r["error"] = json!(err_user);
                    r["hint"] = json!(hint(part, o, &a.cmd));
                }
                if !a.clamped.is_empty() {
                    r["clamped"] = json!(a.clamped);
                    if part != Part::Base && !part.is_gripper() {
                        // 잘린 관절의 허용 범위(LLM 단위) — 방향(부호)을 고치게
                        let lim = joint_limits(part);
                        let l: Vec<Value> = a.clamped.iter().map(|&i| json!(to_user(part, &[lim[i].0, lim[i].1]))).collect();
                        r["limits"] = json!(l);
                    }
                    if r.get("hint").is_none() {
                        r["hint"] = json!("some values were outside joint limits and were clamped");
                    }
                }
                if a.slowed {
                    r["slowed"] = json!(true);
                }
                self.result = Some(r);
                Tick::Done
            }
        }
    }

    fn write_action(&self, action: &mut [f32], base: Option<[f64; 3]>) {
        for (o, h) in action.iter_mut().zip(self.hold.iter()) {
            *o = *h as f32;
        }
        let b = base.unwrap_or([0.0; 3]);
        for i in 0..3 {
            action[act::BASE.start + i] = (b[i] / act::BASE_OUT[i]).clamp(-1.0, 1.0) as f32;
        }
    }
}

fn hint(part: Part, outcome: &str, cmd: &Command) -> &'static str {
    match (part, outcome) {
        (Part::LeftGripper | Part::RightGripper, "blocked") if cmd.values[0] < 0.5 => "gripper stopped before closing: probably holding an object",
        (Part::LeftGripper | Part::RightGripper, _) => "gripper did not reach the opening",
        (Part::Base, "blocked") => "base is not moving: probably an obstacle; try another direction",
        (Part::Base, _) => "base did not arrive in time; call again with the remaining distance",
        (_, "blocked") => "joint stopped early: probably contact or self-collision; it now holds where it stopped",
        _ => "not settled at the target yet; it keeps holding the target",
    }
}

pub fn error_obs(msg: &str) -> Value {
    json!({"status": "error", "message": msg, "hint": "fix the arguments and call move_robot again"})
}

#[cfg(test)]
mod tests;

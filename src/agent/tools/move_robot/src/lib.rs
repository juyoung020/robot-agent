//! `move_robot` — LLM 이 로봇 한 부분(베이스·팔·그리퍼)을 직접 움직이는 도구 하나. 로봇은 AgileX LIMO(4륜 차동(스키드 스티어) 베이스) + ROBOTIS OMX-F
//! (관절 5 + 그리퍼) 하나뿐이다.
//!
//! 이 크레이트가 하는 일은 이것뿐이다.
//! - [`definition`]: OpenAI `tools` 스키마(9B 모델용으로 작게: enum·필수 인자, 짧은 설명)
//! - [`parse`]: 도구 인자 → [`Command`] (틀리면 모델이 고칠 수 있는 오류 문장)
//! - [`Robot`]: 매 시뮬 스텝 LIMO proprio(24)를 받아 행동 벡터(8 = [vx, wz, j1..j5, 그리퍼])를 내는 닫힌 고리 실행기.
//!   관절은 최소 저크 보간 + 도착/막힘/시간 초과 판정, 베이스는 base_qvel 적분(오도메트리) 위 P 제어 + 가감속 한도.
//! - [`ffi`]: 같은 실행기를 C ABI 로(시뮬 쪽 파이썬은 ctypes 로 부른다)
//! - [`link`]: 에이전트 쪽 실행 경로(시뮬 TCP / 가짜 로봇)
//!
//! 단위: 길이 m, 각도 도(°, LLM 에게 보이는 값). 안에서는 rad. 그리퍼는 0 = 닫힘 … 1 = 열림.
//! 행동·관측 배치와 관절 한계는 [`limo`](관절 이름 omx_joint1..5, 한계 = `src/robot/real_limits.json`), 평가기 설정은 `src/robot/og/limo_omx_eval.yaml`.
//! 베이스는 4륜 차동(스키드 스티어)이라 앞뒤(vx)와 제자리 돌기(wz)만 낸다 — 옆으로 가지 못한다. 머리(카메라)는 몸에 붙어 앞만 본다.

pub mod ffi;
pub mod frontier;
pub mod goal;
pub mod limo;
pub mod limo_mock;
pub mod link;
pub mod llm_view;
pub mod map;
pub mod mock_eval;
pub mod nav;
pub mod robot_nav;
mod robot_vla;
pub mod verify;
pub mod vla;

pub use limo::{act, prop, LimoState as State, Part, ACTION_DIM, JOINT_MARGIN, PARTS, PROPRIO_DIM};
pub use map::{Grid, MapIn, NavParams, RoomGrid, Scan};

use serde_json::{json, Value};
use std::f64::consts::PI;

pub const HZ: f64 = 30.0;
pub const TOOL_NAME: &str = "move_robot";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Delta,
    Absolute,
    /// 베이스: 아는 빈칸만 지나는 경로(A*/Dijkstra)로 목표(프런티어·방 id 또는 로봇 기준 [앞 m, 왼쪽 m])까지
    GoTo,
    /// 베이스: [왼쪽으로 돌기 °, 앞 m] — 돌고 나서 천천히 앞으로, 살아 있는 깊이로 막히면 멈춤(모르는 곳 살피기)
    Probe,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Command {
    pub part: Part,
    pub mode: Mode,
    /// LLM 단위(m, deg, 0..1) 그대로
    pub values: Vec<f64>,
    pub duration_s: Option<f64>,
    /// go_to 목표 id ("F1", "R2")
    pub target: Option<String>,
}

// ---------------------------------------------------------------- 안전 한계

/// 안전 속도·가속도와 판정 문턱. 기본값은 사람이 옆에 있어도 놀라지 않을 정도로 느리게.
#[derive(Clone, Debug)]
pub struct Safety {
    pub arm_vmax: f64,     // rad/s (OMX-F XL430 최고 4.8 rad/s 의 약 1/6)
    pub gripper_vmax: f64, // 벌림 비율/s (1 = 1 초에 다 열기)
    pub base_vmax: f64,    // m/s
    pub base_wmax: f64,    // rad/s
    pub base_acc: f64,     // m/s²
    pub base_wacc: f64,    // rad/s²
    pub base_max_fwd: f64, // delta 한 번에 갈 거리 상한(m)
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
    /// go_to(아는 빈칸 경로) 최고 속도 m/s · 회전 rad/s
    pub nav_vmax: f64,
    pub nav_wmax: f64,
    /// probe(모르는 곳 살피기) 앞으로 속도 m/s, 한 번에 최대 거리 m
    pub probe_vmax: f64,
    pub probe_max_m: f64,
    /// 베이스 마지막 접근: 남은 거리가 있으면 이 속도 아래로는 안 줄임(덜 가서 멈추는 것 막기)
    pub base_creep: f64,
    pub base_wcreep: f64,
}

impl Default for Safety {
    fn default() -> Self {
        Safety {
            arm_vmax: 45f64.to_radians(),
            gripper_vmax: 1.0,
            base_vmax: 0.3,
            base_wmax: 35f64.to_radians(),
            base_acc: 0.6,
            base_wacc: 90f64.to_radians(),
            base_max_fwd: 2.0,
            base_max_yaw: PI,
            min_duration: 0.1,
            max_duration: 20.0,
            joint_tol: 1.5f64.to_radians(),
            gripper_tol: 0.05,
            base_tol_xy: 0.01,
            base_tol_yaw: 1f64.to_radians(),
            track_limit: 20f64.to_radians(),
            settle_s: 1.5,
            nav_vmax: 0.5,
            nav_wmax: 50f64.to_radians(),
            probe_vmax: 0.25,
            probe_max_m: 1.5,
            base_creep: 0.03,
            base_wcreep: 3f64.to_radians(),
        }
    }
}

/// 부분의 관절 한계(안쪽 여유 포함, 안 단위: rad / 비율)
pub fn joint_limits(part: Part) -> Vec<(f64, f64)> {
    match part {
        Part::Arm => limo::arm_limits().to_vec(),
        Part::Gripper => vec![(0.0, 1.0)],
        Part::Base => vec![],
    }
}

/// LLM 단위 → 안 단위 (deg → rad; m·비율은 그대로)
pub fn to_internal(part: Part, v: &[f64]) -> Vec<f64> {
    match part {
        Part::Arm => v.iter().map(|x| x.to_radians()).collect(),
        Part::Base => vec![v[0], v[1].to_radians()],
        Part::Gripper => v.to_vec(),
    }
}

/// 안 단위 → LLM 단위(반올림 포함: 각도 0.1°, 길이 1 mm, 비율 0.01)
pub fn to_user(part: Part, v: &[f64]) -> Vec<f64> {
    let r = |x: f64, k: f64| (x * k).round() / k;
    match part {
        Part::Arm => v.iter().map(|x| r(x.to_degrees(), 10.0)).collect(),
        Part::Base => vec![r(v[0], 1000.0), r(v[1].to_degrees(), 10.0)],
        Part::Gripper => v.iter().map(|x| r(*x, 100.0)).collect(),
    }
}

// ---------------------------------------------------------------- 스키마

pub const DESCRIPTION: &str = "Move ONE robot part and wait until it stops. Units: meters, degrees; left/counter-clockwise is positive. \
The base is a 4-wheel differential (skid-steer) drive: it drives forward/backward and turns in place, it cannot move sideways. \
base modes: go_to = drive along a planned path through KNOWN free space to target id (\"F1\" frontier, \"R2\" room) or values [forward_m, left_m] (a point in front/left of the robot, must be known free); \
probe = values [turn_left_deg, forward_m]: turn in place, then creep forward into UNKNOWN space, stopping before obstacles seen by the camera (forward_m 0 = just turn/look); \
delta = values [forward_m, turn_left_deg] drive straight, then turn in place (no planning). Base results include a map summary. \
arm [j1..j5] = OMX-F joints base to wrist (deg; delta or absolute); gripper [opening] 0=closed..1=open (absolute). \
delta with all zeros only reads the state. Joint limits, safe speeds and obstacle stops are enforced.";

/// OpenAI Chat Completions `tools` 항목 하나 (설명은 기본 [`DESCRIPTION`])
pub fn definition() -> Value {
    definition_with(DESCRIPTION)
}

/// 설명만 바꾼 도구 정의(에이전트가 프롬프트 폴더의 도구 설명을 쓸 때)
pub fn definition_with(desc: &str) -> Value {
    definition_modes(desc, &["go_to", "probe", "delta", "absolute"])
}

/// 설명과 LLM 에게 보일 mode enum 을 고른 도구 정의. 스킬 설정(`skill.json` 의 `tools[].modes`)이 정한다 —
/// 에이전트 쪽 모드 `explore`(프런티어 탐사, [`frontier`])는 이 목록에 넣어야 보인다.
pub fn definition_modes(desc: &str, modes: &[&str]) -> Value {
    let mut d = json!({"type": "function", "function": {"name": TOOL_NAME, "description": desc, "parameters": {
        "type": "object",
        "properties": {
            "part": {"type": "string", "enum": PARTS},
            "mode": {"type": "string", "enum": modes},
            "target": {"type": "string", "description": "go_to only: id from the last map summary, e.g. F1 or R2"},
            "values": {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 5,
                       "description": "probe 2, go_to 2 (if no target), base delta 2, arm 5, gripper 1"},
            "duration_s": {"type": "number", "description": "optional; slower if too fast for safe speed"}
        },
        "required": ["part", "mode"]
    }}});
    if modes.contains(&frontier::MODE) {
        d["function"]["parameters"]["properties"]["max_steps"] =
            json!({"type": "integer", "minimum": 1, "maximum": frontier::MAX_STEPS, "description": "explore only: frontier go_to moves in this call (default 1)"});
    }
    d
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
    let mode = match a.get("mode").and_then(|v| v.as_str()).map(|s| s.trim().to_ascii_lowercase().replace(['-', ' '], "_")).as_deref() {
        Some("delta") | Some("relative") => Mode::Delta,
        Some("absolute") => Mode::Absolute,
        Some("go_to") | Some("goto") => Mode::GoTo,
        Some("probe") => Mode::Probe,
        Some(m) => return Err(format!("mode must be go_to, probe, delta or absolute (got '{m}')")),
        None => return Err("mode is required: go_to, probe, delta or absolute".into()),
    };
    if part == Part::Base && mode == Mode::Absolute {
        return Err("base has no absolute mode: use go_to with a target id (known space) or probe [turn_left_deg, forward_m] (unknown space)".into());
    }
    if part != Part::Base && matches!(mode, Mode::GoTo | Mode::Probe) {
        return Err(format!("{} supports only delta or absolute; go_to and probe are base modes", part.name()));
    }
    let target = a.get("target").and_then(|v| v.as_str()).map(|t| t.trim().to_ascii_uppercase()).filter(|t| !t.is_empty());
    if mode == Mode::GoTo && target.is_some() {
        let t = target.clone().unwrap();
        let ok = t.len() >= 2 && (t.starts_with('F') || t.starts_with('R')) && t[1..].chars().all(|c| c.is_ascii_digit());
        if !ok {
            return Err(format!("target must be an id from the last map summary like F1 or R2 (got '{t}')"));
        }
        return Ok(Command { part, mode, values: vec![], duration_s: None, target });
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
        None if mode == Mode::GoTo => return Err("go_to needs target (an id like F1 or R2 from the last map summary) or values [forward_m, left_m]".into()),
        None if mode == Mode::Probe => return Err("probe needs values [turn_left_deg, forward_m]".into()),
        None => return Err(format!("values is required: {} numbers, {}", part.dof(), part.units())),
    };
    let need = match mode {
        Mode::GoTo | Mode::Probe => 2,
        _ => part.dof(),
    };
    if values.len() != need {
        return Err(match mode {
            Mode::GoTo => format!("go_to needs target (e.g. \"F1\") or values [forward_m, left_m], got {} values", values.len()),
            Mode::Probe => format!("probe needs exactly 2 values [turn_left_deg, forward_m], got {}", values.len()),
            _ => format!("{} needs exactly {} values ({}), got {}", part.name(), part.dof(), part.units(), values.len()),
        });
    }
    if let Some(i) = values.iter().position(|x| !x.is_finite()) {
        return Err(format!("values[{i}] is not finite"));
    }
    let duration_s = match a.get("duration_s") {
        None | Some(Value::Null) => None,
        Some(v) => Some(num(v).filter(|d| d.is_finite() && *d > 0.0).ok_or("duration_s must be a positive number")?),
    };
    if mode == Mode::Probe && (values[1] < 0.0) {
        return Err("probe forward_m must be >= 0 (turn first to face where you want to go)".into());
    }
    Ok(Command { part, mode, values, duration_s, target })
}

// ---------------------------------------------------------------- 관측

/// 행동 벡터에서 부분이 차지하는 칸
pub fn action_slots(part: Part) -> std::ops::Range<usize> {
    part.slots()
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
    /// [앞 m(뒤는 음수), 왼쪽으로 돌기 rad] — 먼저 곧게 가고(phase 0), 멈춘 뒤 제자리에서 돈다(phase 1)
    target: [f64; 2],
    /// 이 명령에서 간 거리(m, 부호 있음)와 돈 각(rad) — base_qvel 적분(오도메트리)
    dist: f64,
    yaw: f64,
    phase: u8,
    /// 지령 [vx, wz]
    v_cmd: [f64; 2],
    t_max: f64,
    stall: u32,
    /// 도착 판정 뒤 0 지령으로 멈출 때까지 기다린 스텝(관성으로 더 가거나 덜 간 것까지 재서 보고)
    settle: u32,
    retries: u32,
    /// 안전 정지(지도·깊이): (이유, 남은 여유 m)
    stop: Option<(&'static str, f64)>,
    /// 명령 시작 map 위치(발밑 모름 칸 허용 기준)
    anchor: [f64; 2],
}

#[derive(Clone, Debug)]
pub(crate) enum Motion {
    Joints(JointMove),
    Base(BaseMove),
    Nav(robot_nav::NavMove),
}

#[derive(Clone, Debug)]
pub(crate) struct Active {
    pub(crate) cmd: Command,
    pub(crate) motion: Motion,
    clamped: Vec<usize>,
    slowed: bool,
    steps: u64,
}

/// 매 시뮬 스텝 하나씩 부르는 닫힌 고리 실행기. 명령이 없을 때는 마지막 목표를 유지(그 자리 버티기, 베이스 0 속도).
pub struct Robot {
    pub safety: Safety,
    pub dt: f64,
    /// 유지 중인 행동(팔 = 관절 목표 rad, 그리퍼 = 벌림 0..1, 베이스 = 0)
    hold: [f64; ACTION_DIM],
    state: Option<State>,
    active: Option<Active>,
    result: Option<Value>,
    pub ticks: u64,
    /// 지도·주행 층(지도를 받은 적이 있을 때만 쓰임)
    pub nav: robot_nav::NavState,
    /// 이번 스텝의 GT 자세(map 틀). 다음 tick 한 번에서 base_qvel 적분 대신 쓰인다(Map_Vla)
    gt_pose: Option<[f64; 3]>,
    /// VLA 실행기(LIMO + OMX-F, POLICY 1.3): 기억 물체·거르개·지금 단계
    pub vla: vla::VlaCtx,
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
        Robot { safety: Safety::default(), dt: 1.0 / hz, hold: [0.0; ACTION_DIM], state: None, active: None, result: None, ticks: 0, nav: robot_nav::NavState::with_body(&Self::body()), gt_pose: None, vla: vla::VlaCtx::default() }
    }

    /// 몸 크기: LIMO + OMX-F 하나([`nav::Body::limo_omx`])
    fn body() -> nav::Body {
        nav::Body::limo_omx()
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

    /// 지도 받기(평가기: keyframe 마다 sgrt_map → mr_set_map, 가짜 로봇: MockWorld). 할당은 격자 크기가 바뀔 때만.
    pub fn set_map(&mut self, m: MapIn, d: &nav::MapDelta) {
        self.nav.set_map(m, d);
    }
    /// 정답 기준(측정용, LLM 에 안 보임): map 좌표 격자, 값 1 = 닿을 수 있는 바닥
    pub fn set_reference(&mut self, g: Grid) {
        self.nav.set_reference(g);
    }

    /// 판이 새로 시작될 때: 다음 관측에서 유지값을 다시 잡는다
    pub fn reset(&mut self) {
        if self.active.is_some() {
            self.result = Some(json!({"status": "error", "message": "episode reset while moving"}));
        }
        self.active = None;
        if self.vla.run.is_some() {
            self.vla_stop("reset");
        }
        self.vla.filter.init = false;
        self.state = None;
        self.gt_pose = None;
        self.nav.gt_prev = None; // 새 판의 첫 GT 자세를 지난 판 끝에서 잰 거리로 세지 않게
    }

    /// 이번 스텝의 GT 자세(map 틀: x, y, yaw rad). 다음 [`Robot::tick`] 한 번에서 base_qvel 적분 대신 쓰인다.
    /// 시뮬 GT 모드(SGRT_POSE=gt, map = world)에서 매 스텝 tick 앞에 부른다. 부르지 않은 스텝은 예전처럼 속도 적분.
    pub fn set_gt_pose(&mut self, pose: [f64; 3]) {
        self.gt_pose = Some(pose);
    }

    /// 지금 관측으로 유지값을 잡는다(첫 관측)
    fn init_hold(&mut self, s: &State) {
        let h = &mut self.hold;
        h[act::BASE].fill(0.0);
        h[act::ARM].copy_from_slice(&s.arm);
        h[act::GRIPPER] = s.grip;
    }

    fn set_hold(&mut self, part: Part, q: &[f64]) {
        let r = action_slots(part);
        if part.is_gripper() {
            self.hold[r.start] = q[0];
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
        if self.vla.run.is_some() {
            return Err("a VLA step is running; wait for its result".into());
        }
        let cmd = parse(args)?;
        let Some(st) = self.state.clone() else { return Err("no robot observation yet; try again in a moment".into()) };
        let s = self.safety.clone();
        let part = cmd.part;
        if matches!(cmd.mode, Mode::GoTo | Mode::Probe) {
            let nm = self.start_nav(&cmd)?;
            self.active = Some(Active { cmd, motion: Motion::Nav(nm), clamped: vec![], slowed: false, steps: 0 });
            return Ok(false);
        }
        let v = to_internal(part, &cmd.values);

        // 읽기만: delta 이고 전부 0
        if cmd.mode == Mode::Delta && cmd.values.iter().all(|x| *x == 0.0) {
            let mut r = json!({"status": "reached", "part": part.name(), "state": self.user_state(part), "units": part.units(), "steps": 0});
            if part == Part::Base {
                self.attach_map(&mut r);
            }
            self.result = Some(r);
            return Ok(true);
        }

        let (motion, clamped, slowed) = if part == Part::Base {
            let mut t = [v[0], v[1]];
            let mut clamped = vec![];
            for (i, lim) in [s.base_max_fwd, s.base_max_yaw].into_iter().enumerate() {
                if t[i].abs() > lim {
                    t[i] = t[i].clamp(-lim, lim);
                    clamped.push(i);
                }
            }
            let need = t[0].abs() / s.base_vmax + t[1].abs() / s.base_wmax;
            let t_max = need * 1.5 + 3.0 + cmd.duration_s.unwrap_or(0.0);
            (Motion::Base(BaseMove { target: t, dist: 0.0, yaw: 0.0, phase: 0, v_cmd: [0.0; 2], t_max, stall: 0, settle: 0, retries: 0, stop: None, anchor: [self.nav.pose[0], self.nav.pose[1]] }), clamped, false)
        } else {
            let (q, _) = st.joints(part);
            let mut target: Vec<f64> = match cmd.mode {
                Mode::Delta => q.iter().zip(&v).map(|(a, b)| a + b).collect(),
                _ => v.clone(),
            };
            let clamped = clamp_target(&mut target, &joint_limits(part));
            // 보간은 지금 지령(유지값)에서 시작한다 — 측정값에서 시작하면 처진 만큼 한 스텝에 튄다
            let start: Vec<f64> = if part.is_gripper() {
                vec![self.hold[action_slots(part).start]]
            } else {
                self.hold[action_slots(part)].to_vec()
            };
            let delta: Vec<f64> = target.iter().zip(&start).map(|(t, a)| t - a).collect();
            let vmax = if part.is_gripper() { s.gripper_vmax } else { s.arm_vmax };
            let (t_move, slowed) = plan_duration(&delta, vmax, cmd.duration_s, &s);
            (Motion::Joints(JointMove { part, start, target, t_move, still: 0, off_track: 0 }), clamped, slowed)
        };
        self.active = Some(Active { cmd, motion, clamped, slowed, steps: 0 });
        Ok(false)
    }

    /// 시뮬 한 스텝: proprio(24) → 행동(8). 행동은 언제나 채워진다.
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
        match self.gt_pose.take() {
            Some(g) => self.nav.integrate_gt(g, self.dt),
            None => self.nav.integrate([st.base_v[0], st.base_v[2]], self.dt),
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
        let mut nav_done: Option<Value> = None;
        let mut base_stop: Option<(&'static str, f64)> = None;
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
                // 오도메트리: 측정 base_qvel 의 vx·wz(차동 — vy 는 쓰지 않음)를 명령 시작부터 적분
                let (vx, wz) = (st.base_v[0], st.base_v[2]);
                m.dist += vx * self.dt;
                m.yaw += wz * self.dt;
                let ex = m.target[0] - m.dist;
                let eyaw = m.target[1] - m.yaw;
                let dist_ok = ex.abs() < s.base_tol_xy;
                let yaw_ok = eyaw.abs() < s.base_tol_yaw;
                let mut des = [0.0; 2];
                if m.phase == 0 {
                    if dist_ok {
                        // 멈추고 나서 돈다
                        if vx.abs() < 0.02 {
                            m.phase = 1;
                        }
                    } else {
                        // 감속 곡선 + 마지막 접근 최소 속도(덜 가서 멈추는 것 막기)
                        let mut sp = s.base_vmax.min((2.0 * s.base_acc * ex.abs()).sqrt()).max(s.base_creep.min(ex.abs() * 4.0));
                        // 안전 정지: 지도가 있으면 가는 방향 몸통 통로(지도 장애물·모르는 곳(카메라 밖)·깊이)
                        if self.nav.have_map {
                            let heading = self.nav.pose[2] + if ex < 0.0 { PI } else { 0.0 };
                            let (free, by) = self.free_ahead(self.nav.pose, heading, true, m.anchor);
                            let room = free - self.nav.params.stop_margin;
                            if room < ex.abs() {
                                sp = sp.min((2.0 * s.base_acc * room.max(0.0)).sqrt());
                                if room <= 0.01 {
                                    m.stop = Some((by, (free.max(0.0) * 100.0).round() / 100.0));
                                }
                            }
                        }
                        des[0] = ex.signum() * sp;
                        // 곧게: 처음 방향에서 벗어난 만큼 되돌린다(돌 각은 이 단계 뒤)
                        des[1] = (-2.0 * m.yaw).clamp(-s.base_wmax, s.base_wmax);
                    }
                }
                if m.phase == 1 && !yaw_ok {
                    let w = s.base_wmax.min((2.0 * s.base_wacc * eyaw.abs()).sqrt()).max(s.base_wcreep.min(eyaw.abs() * 4.0));
                    des[1] = eyaw.signum() * w;
                }
                let lim = [s.base_acc * self.dt, s.base_wacc * self.dt];
                for i in 0..2 {
                    m.v_cmd[i] += (des[i] - m.v_cmd[i]).clamp(-lim[i], lim[i]);
                }
                let within = dist_ok && yaw_ok && m.phase == 1;
                if m.stop.is_some() || within || m.settle > 0 {
                    m.v_cmd = [0.0; 2];
                }
                let moving_cmd = m.v_cmd[0].abs() > 0.05 || m.v_cmd[1].abs() > 0.1;
                let barely = vx.abs() < 0.2 * m.v_cmd[0].abs().max(0.05) && wz.abs() < 0.2 * m.v_cmd[1].abs().max(0.1);
                m.stall = if moving_cmd && barely { m.stall + 1 } else { 0 };
                if within || m.settle > 0 {
                    // 멈출 때까지(관성) 기다려 실제 도착 자리를 잰다. 너무 벗어나면 한 번 더 접근
                    m.settle += 1;
                    let still = vx.abs() < 0.005 && wz.abs() < 0.01;
                    if (still && m.settle >= 3) || m.settle >= 15 {
                        let good = ex.abs() < 2.0 * s.base_tol_xy && eyaw.abs() < 2.0 * s.base_tol_yaw;
                        if good || m.retries >= 2 {
                            outcome = Some(if good { "reached" } else { "timeout" });
                        } else {
                            m.retries += 1;
                            m.settle = 0;
                            m.phase = if ex.abs() >= 2.0 * s.base_tol_xy { 0 } else { 1 };
                        }
                    }
                } else if m.stop.is_some() {
                    outcome = Some("blocked");
                } else if m.stall >= 30 {
                    outcome = Some("blocked");
                    self.nav.n_stall += 1;
                } else if t > m.t_max {
                    outcome = Some("timeout");
                }
                if outcome.is_some() {
                    m.v_cmd = [0.0; 2];
                    final_state = to_user(Part::Base, &[m.dist, m.yaw]);
                    target_user = to_user(Part::Base, &m.target);
                    // 남은 거리(m)와 남은 회전(deg)
                    err_user = Some(json!([(ex * 1000.0).round() / 1000.0, (eyaw.to_degrees() * 10.0).round() / 10.0]));
                    if outcome == Some("blocked") {
                        self.nav.n_blocked += 1;
                    }
                }
                base_cmd = Some(m.v_cmd);
            }
            Motion::Nav(m) => {
                if let Some(r) = self.tick_nav(m, t, [st.base_v[0], st.base_v[2]]) {
                    nav_done = Some(r);
                    base_cmd = Some([0.0; 2]);
                } else {
                    base_cmd = Some(m.v_cmd);
                }
            }
        }
        self.write_action(action, base_cmd);
        if let Some(r) = nav_done {
            self.result = Some(r);
            return Tick::Done;
        }
        if let Motion::Base(m) = &a.motion {
            base_stop = m.stop;
        }
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
                    if part == Part::Arm {
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
                if let Some((by, d)) = base_stop {
                    r["stopped_by"] = json!(by);
                    r["clear_m"] = json!(d);
                    r["hint"] = json!("stopped before an obstacle or unseen space; turn to face where you go (probe) or go_to a frontier id");
                }
                if part == Part::Base {
                    self.attach_map(&mut r);
                }
                self.result = Some(r);
                Tick::Done
            }
        }
    }

    /// 행동 8 = [vx m/s, wz rad/s, j1..j5 rad, 그리퍼 0..1] (물리 단위 — 안전 거르개 출력과 같은 배치)
    fn write_action(&self, action: &mut [f32], base: Option<[f64; 2]>) {
        for (o, h) in action.iter_mut().zip(self.hold.iter()) {
            *o = *h as f32;
        }
        let b = base.unwrap_or([0.0; 2]);
        action[act::BASE.start] = b[0] as f32;
        action[act::BASE.start + 1] = b[1] as f32;
    }
}

fn hint(part: Part, outcome: &str, cmd: &Command) -> &'static str {
    match (part, outcome) {
        (Part::Gripper, "blocked") if cmd.values[0] < 0.5 => "gripper stopped before closing: probably holding an object",
        (Part::Gripper, _) => "gripper did not reach the opening",
        (Part::Base, "blocked") => "base is not moving: probably an obstacle; turn to face another direction (probe) and try again",
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
#[cfg(test)]
mod tests_vla;

/// 관측 모양 변형(실험): 주변 여유 빼기
pub fn robot_nav_style_compact() -> robot_nav::ObsStyle {
    robot_nav::ObsStyle::Compact
}

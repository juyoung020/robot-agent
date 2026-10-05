//! 자동 확인(증거). 단계가 끝났는지 기억(scenemap 물체) + 그리퍼 + 손끝으로 판정한다. LLM 은 이 판정을 바꾸지 못한다(plan.md 8절).
//!
//! 판정 규칙과 문턱은 기존 계획기 `src/agent/planner/src/planner.rs` `Core::evidence` 와
//! `docs/에이전트_설계.md` 1.7 을 그대로 옮겼다(plan.md 3.2 의 `task/verify.rs` 자리, POLICY 3.1 성공 표):
//! - 집기: 그리퍼가 무언가에 닫힘(빈손으로 끝까지 닫히지 않음) + 대상이 0.1 m 넘게 들림(또는 기억 상태 `held`) → done,
//!   끝까지 닫힘(빈손) → failed. "손가락 폭" 대신 OMX-F 의 벌림 비율 [`EMPTY_FRAC`] 로 본다(추정).
//! - 놓기: 그리퍼 열림 + 물체가 받침에서 수평 0.6 m 안 + 손끝이 물체에서 0.05 m 넘게 물러남 → done,
//!   열렸는데 받침에서 멀면 failed.
//! - 다가가기: 대상이 팔 닿는 거리 안 + 베이스 멈춤 → done.
//! - 서랍·문·스위치: 기억·고유감각으로 못 잼 → unknown(영상 확인 몫).
//! "유지 시간"(집기 1 s 등)은 부르는 쪽([`crate::vla`])이 센다.

use serde_json::{json, Value};

/// 그리퍼 벌림 비율이 이 아래면 빈손으로 끝까지 닫힘(추정)
pub const EMPTY_FRAC: f64 = 0.03;
/// 이 위면 "열림"(OMX-F 는 100° 다 안 열어도 놓임, 추정)
pub const OPEN_FRAC: f64 = 0.6;
/// 집기 성공 들림(m) — 에이전트_설계 1.7
pub const LIFT_M: f64 = 0.1;
/// 놓기: 물체 ↔ 받침 수평 거리(m) — 계획기 0.6
pub const PLACE_NEAR_M: f64 = 0.6;
/// 놓기: 손끝이 물체에서 물러난 거리(m) — POLICY 3.1
pub const RETREAT_M: f64 = 0.05;
/// 다가가기: 팔 받침(omx_link0, base_footprint x −0.04)에서 물체까지 수평 거리 — OMX-F 손끝 영점 거리 0.313 m + 여유(추정)
pub const REACH_M: f64 = 0.38;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObjState {
    Seen,
    Gone,
    Moved,
    Held,
}

impl ObjState {
    pub fn from_i32(s: i32) -> ObjState {
        match s {
            1 => ObjState::Gone,
            2 => ObjState::Moved,
            3 => ObjState::Held,
            _ => ObjState::Seen,
        }
    }
    pub fn from_name(s: &str) -> ObjState {
        match s {
            "gone" => ObjState::Gone,
            "moved" => ObjState::Moved,
            "held" => ObjState::Held,
            _ => ObjState::Seen,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            ObjState::Seen => "seen",
            ObjState::Gone => "gone",
            ObjState::Moved => "moved",
            ObjState::Held => "held",
        }
    }
}

/// 기억 속 물체 하나(scenemap `sm_object` 와 같은 값, map 좌표)
#[derive(Clone, Debug, PartialEq)]
pub struct MemObject {
    /// "O<scenemap id>" 또는 시험·가짜 세계의 이름 id
    pub id: String,
    pub name: String,
    pub score: f64,
    pub pos: [f64; 3],
    pub extent: [f64; 3],
    pub first_pos: [f64; 3],
    pub n_obs: u32,
    /// 시뮬 시각(s)
    pub last_seen: f64,
    pub state: ObjState,
}

impl MemObject {
    pub fn from_json(v: &Value) -> Option<MemObject> {
        let id = match v.get("id")? {
            Value::String(s) => norm_id(s),
            Value::Number(n) => format!("O{}", n.as_u64()?),
            _ => return None,
        };
        let a3 = |k: &str, d: [f64; 3]| -> [f64; 3] {
            v.get(k).and_then(|x| x.as_array()).filter(|x| x.len() == 3).map(|x| [0, 1, 2].map(|i| x[i].as_f64().unwrap_or(0.0))).unwrap_or(d)
        };
        let pos = a3("pos", [f64::NAN; 3]);
        if pos.iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some(MemObject {
            id,
            name: v.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            score: v.get("score").and_then(|x| x.as_f64()).unwrap_or(1.0),
            pos,
            extent: a3("extent", [0.1; 3]),
            first_pos: a3("first_pos", pos),
            n_obs: v.get("n_obs").and_then(|x| x.as_u64()).unwrap_or(1) as u32,
            last_seen: v.get("last_seen").and_then(|x| x.as_f64()).unwrap_or(0.0),
            state: ObjState::from_name(v.get("state").and_then(|x| x.as_str()).unwrap_or("seen")),
        })
    }
    pub fn lifted(&self) -> f64 {
        self.pos[2] - self.first_pos[2]
    }
    pub fn moved(&self) -> f64 {
        let d = [0, 1, 2].map(|i| self.pos[i] - self.first_pos[i]);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }
}

/// 물체 id 정규화: "12" → "O12", "o12" → "O12", 그 밖은 그대로(앞뒤 공백 제거)
pub fn norm_id(s: &str) -> String {
    let t = s.trim();
    if !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()) {
        return format!("O{t}");
    }
    if t.len() > 1 && (t.starts_with('o') || t.starts_with('O')) && t[1..].chars().all(|c| c.is_ascii_digit()) {
        return format!("O{}", &t[1..]);
    }
    t.to_string()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    Done,
    Failed,
    Running,
    Unknown,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Done => "done",
            Verdict::Failed => "failed",
            Verdict::Running => "running",
            Verdict::Unknown => "unknown",
        }
    }
}

/// 확인에 쓰는 순간 값
pub struct Evidence<'a> {
    pub kind: crate::vla::SkillKind,
    /// 대상(집기·다가가기: 집을 물체, 놓기: 든 물체, 서랍·문: 가구)
    pub target: Option<&'a MemObject>,
    /// 놓기의 받침·그릇
    pub support: Option<&'a MemObject>,
    /// 놓기·다가가기의 지점(map, 검사·옮긴 뒤) — 있으면 받침 대신 이 점으로 판정(goal.rs `at_point`)
    pub point: Option<[f64; 3]>,
    /// 로봇 map 자세 (x, y, yaw)
    pub pose: [f64; 3],
    /// 그리퍼 벌림 비율(측정)과 지령
    pub grip: f64,
    pub grip_cmd: f64,
    /// 손끝 위치(로봇 기준)
    pub eef: [f64; 3],
    /// 베이스 속력(m/s), 회전(rad/s)
    pub base_speed: f64,
    pub base_turn: f64,
}

/// map 점 → 로봇 기준
pub fn to_robot(pose: [f64; 3], p: [f64; 3]) -> [f64; 3] {
    let (s, c) = pose[2].sin_cos();
    let (dx, dy) = (p[0] - pose[0], p[1] - pose[1]);
    [c * dx + s * dy, -s * dx + c * dy, p[2]]
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// 판정과 근거(짧은 영어 글 + 잰 값)
pub fn evidence(e: &Evidence) -> (Verdict, Value) {
    use crate::vla::SkillKind as K;
    // 닫으려는데(지령이 측정보다 작음) 빈손 문턱 위에서 멈춤 = 무언가를 쥠
    let closing = e.grip_cmd < e.grip - 0.05 || e.grip_cmd < EMPTY_FRAC;
    let closed_on = e.grip > EMPTY_FRAC && e.grip < OPEN_FRAC && closing;
    let closed_empty = e.grip <= EMPTY_FRAC;
    let opened = e.grip >= OPEN_FRAC;
    let g = r2(e.grip);
    match e.kind {
        K::Pick => {
            let Some(t) = e.target else { return (Verdict::Unknown, json!({"why": "target not in memory", "gripper": g})) };
            let lifted = t.lifted();
            let held = t.state == ObjState::Held;
            let m = json!({"gripper": g, "lifted_m": r2(lifted), "moved_m": r2(t.moved()), "state": t.state.name()});
            if closed_on && (lifted > LIFT_M || held) {
                return (Verdict::Done, json!({"why": format!("gripper closed on {}; lifted {:.2} m", t.id, lifted), "m": m}));
            }
            if closed_empty {
                return (Verdict::Failed, json!({"why": "gripper closed on nothing (grasp missed)", "m": m}));
            }
            if closed_on {
                return (Verdict::Running, json!({"why": "gripper closed on something; target not lifted yet", "m": m}));
            }
            (Verdict::Running, json!({"why": "gripper still open", "m": m}))
        }
        K::Place if e.point.is_some() => {
            let p = e.point.unwrap();
            let Some(t) = e.target else { return (Verdict::Unknown, json!({"why": "held object not in memory", "gripper": g})) };
            let (ok, d, dz) = crate::goal::at_point(t, p);
            let rel = to_robot(e.pose, t.pos);
            let retreat = ((rel[0] - e.eef[0]).powi(2) + (rel[1] - e.eef[1]).powi(2) + (rel[2] - e.eef[2]).powi(2)).sqrt();
            let m = json!({"gripper": g, "to_point_m": (d * 1000.0).round() / 1000.0, "bottom_dz_m": (dz * 1000.0).round() / 1000.0, "eef_to_object_m": r2(retreat), "state": t.state.name()});
            if opened && ok && retreat > RETREAT_M {
                return (Verdict::Done, json!({"why": format!("gripper opened; {} rests within {:.2} m of the point", t.id, crate::goal::POINT_R), "m": m}));
            }
            if opened && t.state != ObjState::Held && !ok && retreat > RETREAT_M {
                return (Verdict::Failed, json!({"why": "gripper opened but the object is not at the destination point", "m": m}));
            }
            (Verdict::Running, json!({"why": if opened { "gripper open; hand not withdrawn yet" } else { "still holding" }, "m": m}))
        }
        K::Approach if e.point.is_some() => {
            let rel = to_robot(e.pose, e.point.unwrap());
            let d = (rel[0] + 0.04).hypot(rel[1]);
            let m = json!({"reach_m": r2(d), "base_speed": r2(e.base_speed)});
            if d <= REACH_M && e.base_speed < 0.02 && e.base_turn.abs() < 0.05 {
                return (Verdict::Done, json!({"why": format!("point within arm reach ({d:.2} m), base stopped"), "m": m}));
            }
            (Verdict::Running, json!({"why": format!("{d:.2} m from the arm base"), "m": m}))
        }
        K::Place => {
            let Some(t) = e.target else { return (Verdict::Unknown, json!({"why": "held object not in memory", "gripper": g})) };
            let near = e.support.map(|s| (t.pos[0] - s.pos[0]).hypot(t.pos[1] - s.pos[1]));
            let rel = to_robot(e.pose, t.pos);
            let retreat = ((rel[0] - e.eef[0]).powi(2) + (rel[1] - e.eef[1]).powi(2) + (rel[2] - e.eef[2]).powi(2)).sqrt();
            let m = json!({"gripper": g, "to_support_m": near.map(r2), "eef_to_object_m": r2(retreat), "state": t.state.name()});
            let on_support = near.map_or(true, |d| d < PLACE_NEAR_M);
            if opened && on_support && retreat > RETREAT_M && t.state != ObjState::Held {
                return (Verdict::Done, json!({"why": format!("gripper opened; {} is at the destination", t.id), "m": m}));
            }
            if opened && !on_support {
                return (Verdict::Failed, json!({"why": "gripper opened but the object is not at the destination", "m": m}));
            }
            (Verdict::Running, json!({"why": if opened { "gripper open; hand not withdrawn yet" } else { "still holding" }, "m": m}))
        }
        K::Approach => {
            let Some(t) = e.target else { return (Verdict::Unknown, json!({"why": "target not in memory"})) };
            let rel = to_robot(e.pose, t.pos);
            let d = (rel[0] + 0.04).hypot(rel[1]);
            let m = json!({"reach_m": r2(d), "base_speed": r2(e.base_speed)});
            if d <= REACH_M && e.base_speed < 0.02 && e.base_turn.abs() < 0.05 {
                return (Verdict::Done, json!({"why": format!("{} within arm reach ({d:.2} m), base stopped", t.id), "m": m}));
            }
            (Verdict::Running, json!({"why": format!("{d:.2} m from the arm base"), "m": m}))
        }
        K::OpenClose | K::Press => (Verdict::Unknown, json!({"why": "not measurable from memory or proprioception; check the image", "gripper": g})),
        K::MoveTo | K::Explore | K::Other => (Verdict::Unknown, json!({"why": "not a VLA skill"})),
    }
}

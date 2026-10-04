//! VLA 실행기(LIMO + OMX-F): 조작 단계 하나를 VLA 정책으로 끝까지 하고, 끝·확인·예산·안전을 여기서 정한다.
//! 설계는 `docs/map_vla/POLICY.md` 1.2(실행기 나눔)·1.3(접점)·7.1(안전 거르개)·7.2(끝내거나 돌려주는 조건).
//!
//! 호출(도구 인자 한 줄): `{"executor":"vla","skill":"pick up cup from dining table","objects":["O12","O3"],"max_s":30}`
//! - `skill`: skillspec 문장. 종류([`SkillKind`])로 실행기를 고른다([`route`]): `move to`·탐사는 `move_robot` 몫이라 거절.
//! - `objects`: set_plan 의 물체 id(scenemap "O<id>"). 이 id 의 물체 칸에 **목표인지** 값을 켠다([`build_slots`]).
//! - `max_s`: 예산(시뮬 초). 없으면 스킬별 기본([`default_budget_s`], 추정 — 교사 p90 × 1.5 가 생기면 바꾼다).
//! - `policy`(선택): `scripted`(기본, 각본 흉내) | `replay:<jsonl>` | `external`(행동을 C ABI 로 밖에서 넣음).
//!   **진짜 학습된 VLA 는 아직 없다.** 각본·재생 정책은 배관·끝 신호·결과·안전 거르기를 끝까지 시험하는 대역이다.
//!
//! 시작 조건(1.2): 대상이 기억에 있고, 1.5 m 안이고, 지금 보이거나 기억 불확실도가 작을 때. 아니면 `handback`
//! (`not_in_map` / `too_far` / `not_visible`) — 상태 기계가 `move_robot go_to` 를 먼저 넣는다.
//!
//! 매 스텝: 몸 상태 → (10 Hz 마다) 정책 → 행동 8 → **안전 거르개**([`SafetyFilter`]) → 평가기.
//! 끝(7.2): 끝 신호 > 0.8 이 0.5 s + 자동 확인([`crate::verify`]) → `done` / 확인이 실패 → `failed`, 예산 → `timeout`,
//! 진척 없음 예산 1/3 → `handback("stalled")`, 대상 사라짐 2 s → `handback("not_visible")`, 확신 낮음 1 s → `handback("unsure")`,
//! 몸통 접촉·거르개 정지 되풀이 → `failed("unsafe")`.
//! 결과: `{"status":"done|failed|timeout|handback","reason","evidence","steps","min_clear_m","contacts", …}`.

use crate::limo::{self, LimoState, ACTION_DIM as A8};
use crate::goal::{GoalEntries, GoalMemory, GoalRef, GoalSpec, PointCheck};
use crate::verify::{self, MemObject, ObjState, Verdict};
use serde_json::{json, Value};
use std::f64::consts::PI;

// ---------------------------------------------------------------- 단계 종류·실행기 나눔 (1.2)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SkillKind {
    MoveTo,
    Explore,
    /// 마지막 다가가기(물체 1.5 m 안)
    Approach,
    Pick,
    Place,
    /// 서랍·가구 문 열기/닫기
    OpenClose,
    /// press, turn on/off
    Press,
    Other,
}

impl SkillKind {
    /// skillspec 문장(plan.md 2.2 틀)에서 종류
    pub fn from_sentence(s: &str) -> SkillKind {
        let t = s.trim().to_ascii_lowercase();
        let w = |p: &str| t == p || t.starts_with(&format!("{p} "));
        if w("move to") || w("go to") || w("navigate to") {
            SkillKind::MoveTo
        } else if w("explore") || w("look around") {
            SkillKind::Explore
        } else if w("approach") {
            SkillKind::Approach
        } else if w("pick up") || w("pick") || w("grasp") || w("lift") || w("hold") {
            SkillKind::Pick
        } else if w("place") || w("put") || w("release") || w("insert") {
            SkillKind::Place
        } else if (w("open") || w("close")) && (t.contains("drawer") || t.contains(" door of ")) {
            SkillKind::OpenClose
        } else if w("press") || w("turn on") || w("turn off") {
            SkillKind::Press
        } else {
            SkillKind::Other
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            SkillKind::MoveTo => "move_to",
            SkillKind::Explore => "explore",
            SkillKind::Approach => "approach",
            SkillKind::Pick => "pick",
            SkillKind::Place => "place",
            SkillKind::OpenClose => "open_close",
            SkillKind::Press => "press",
            SkillKind::Other => "other",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Route {
    /// 지도 기반 고전 제어(`move_robot` go_to·probe)
    MoveRobot,
    Vla,
    /// 아직 안 정함(사람이 지나는 방문 등, POLICY 1.2 마지막 줄)
    Undecided,
}

/// "어디로 가나" 는 move_robot, "어떻게 손대나" 는 VLA (POLICY 1.2 표)
pub fn route(sentence: &str) -> (Route, &'static str) {
    let k = SkillKind::from_sentence(sentence);
    let t = sentence.trim().to_ascii_lowercase();
    match k {
        SkillKind::MoveTo | SkillKind::Explore => (Route::MoveRobot, "navigation and exploration run on move_robot go_to / probe (known-map planning, contact 0)"),
        SkillKind::Approach | SkillKind::Pick | SkillKind::Place | SkillKind::OpenClose | SkillKind::Press => (Route::Vla, "contact-rich manipulation near the object"),
        SkillKind::Other if (t.starts_with("open") || t.starts_with("close")) && t.contains("door") => {
            (Route::Undecided, "room doors: handle height is probably outside the OMX-F workspace (POLICY 9)")
        }
        SkillKind::Other => (Route::Undecided, "unknown skill sentence (use a skillspec template)"),
    }
}

/// 스킬별 기본 예산(시뮬 s, 추정). 교사 성공 판 길이 p90 × 1.5(POLICY 3.1)가 생기면 그 값으로 바꾼다.
pub fn default_budget_s(k: SkillKind) -> f64 {
    match k {
        SkillKind::Approach => 20.0,
        SkillKind::Pick | SkillKind::Place => 40.0,
        SkillKind::OpenClose => 45.0,
        SkillKind::Press => 25.0,
        _ => 20.0,
    }
}

// ---------------------------------------------------------------- 호출

#[derive(Clone, Debug, PartialEq)]
pub struct VlaCall {
    pub skill: String,
    pub kind: SkillKind,
    /// 정규화한 물체 id ("O12"): [집을 것, 놓을 곳 id] 차례(옛 꼴과 같음). 지점 놓기·지점 가기면 [집을 것] 또는 빈 목록
    pub objects: Vec<String>,
    pub budget_s: f64,
    pub policy: String,
    /// 목표 칸 지정(goal.rs). 지점은 시작 때 검사·옮긴 뒤 값으로 바뀜
    pub goal: GoalSpec,
    /// 부른 그대로의 놓을 지점(map x, y, z 없으면 None) — 시작 때 검사
    pub point_req: Option<([f64; 2], Option<f64>)>,
}

impl VlaCall {
    /// 시작 조건을 재는 물체: 놓기는 받침(objects[1]), 그 밖은 objects[0]. 지점이 기준이면 None([`VlaCall::anchor_point`])
    pub fn anchor_id(&self) -> Option<&str> {
        if self.anchor_point().is_some() {
            return None;
        }
        if self.kind == SkillKind::Place && self.objects.len() > 1 {
            Some(&self.objects[1])
        } else {
            self.objects.first().map(|s| s.as_str())
        }
    }
    /// 지점이 기준인 단계(지점에 놓기, 지점으로 다가가기)
    pub fn anchor_point(&self) -> Option<[f64; 3]> {
        match self.kind {
            SkillKind::Place | SkillKind::Approach => self.goal.place_point(),
            _ => None,
        }
    }
    pub fn pick_id(&self) -> Option<&str> {
        self.goal.pick_id()
    }
}

pub fn is_vla_call(args: &Value) -> bool {
    let v = match args {
        Value::String(s) => serde_json::from_str::<Value>(s).ok(),
        v => Some(v.clone()),
    };
    v.as_ref().and_then(|v| v.get("executor")).and_then(|e| e.as_str()) == Some("vla")
}

/// 호출 검사. 실행기 나눔에 어긋나면 고치는 방법을 담은 오류.
pub fn parse_call(args: &Value) -> Result<VlaCall, Value> {
    let owned;
    let a = match args {
        Value::String(s) => {
            owned = serde_json::from_str::<Value>(s).map_err(|e| err(&format!("arguments are not JSON: {e}"), None))?;
            &owned
        }
        v => v,
    };
    let skill = a.get("skill").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    if skill.is_empty() {
        return Err(err("skill is required: the step sentence from set_plan (skillspec)", None));
    }
    let nid = |x: &Value| -> Option<String> {
        match x {
            Value::String(s) => Some(verify::norm_id(s)).filter(|s| !s.is_empty()),
            Value::Number(n) => n.as_u64().map(|k| format!("O{k}")),
            _ => None,
        }
    };
    let mut objects: Vec<String> = match a.get("objects") {
        Some(Value::Array(xs)) => xs.iter().filter_map(nid).collect(),
        _ => vec![],
    };
    // 통합 목표 지정: {"goal": {"pick": {"id"}, "place": {"id"} | {"point": [x, y(, z)]}}}
    let mut goal = GoalSpec::default();
    let mut point_req = None;
    if let Some(g) = a.get("goal").filter(|g| !g.is_null()) {
        let bad = |m: &str| err(&format!("goal: {m} — use {{\"pick\": {{\"id\": \"O12\"}}, \"place\": {{\"id\": \"O3\"}} | {{\"point\": [x, y, z]}}}}"), None);
        if !g.is_object() {
            return Err(bad("must be an object"));
        }
        if let Some(p) = g.get("pick").filter(|p| !p.is_null()) {
            match p.get("id").and_then(nid) {
                Some(id) => goal.pick = Some(GoalRef::Obj(id)),
                None => return Err(bad("pick needs an object id (points are place-only)")),
            }
        }
        if let Some(p) = g.get("place").filter(|p| !p.is_null()) {
            if let Some(id) = p.get("id").and_then(nid) {
                goal.place = Some(GoalRef::Obj(id));
            } else if let Some(xs) = p.get("point").and_then(|x| x.as_array()) {
                let v: Vec<f64> = xs.iter().filter_map(|x| x.as_f64()).collect();
                if !(v.len() == 2 || v.len() == 3) || v.len() != xs.len() || v.iter().any(|x| !x.is_finite()) {
                    return Err(bad("place.point must be [x, y] or [x, y, z] (map frame, m)"));
                }
                point_req = Some(([v[0], v[1]], v.get(2).copied()));
                goal.place = Some(GoalRef::Point([v[0], v[1], v.get(2).copied().unwrap_or(0.0)]));
            } else {
                return Err(bad("place needs an id or a point"));
            }
        }
        if goal.pick.is_none() && goal.place.is_none() {
            return Err(bad("give pick and/or place"));
        }
        // objects(옛 꼴)를 goal 에서: [집을 것, 놓을 곳 id]
        objects = vec![];
        if let Some(id) = goal.pick_id() {
            objects.push(id.to_string());
        }
        if let Some(id) = goal.place_id() {
            if objects.is_empty() {
                return Err(bad("a place object needs the picked object id too (pick.id)"));
            }
            objects.push(id.to_string());
        }
    } else {
        goal.pick = objects.first().map(|s| GoalRef::Obj(s.clone()));
        goal.place = objects.get(1).map(|s| GoalRef::Obj(s.clone()));
    }
    // 실행기 나눔: 지점이 있으면 "go here"·"move to"·"approach" 문장은 지점까지의 마지막 다가가기(VLA, 1.5 m 안 — 멀면 handback too_far → go_to)
    let mut kind = SkillKind::from_sentence(&skill);
    if point_req.is_some() && matches!(kind, SkillKind::MoveTo | SkillKind::Approach | SkillKind::Other) {
        kind = SkillKind::Approach;
    } else {
        let (r, why) = route(&skill);
        match r {
            Route::MoveRobot => return Err(err(&format!("'{skill}' is not a VLA step: {why}"), Some("move_robot"))),
            Route::Undecided => return Err(err(&format!("'{skill}': no executor ({why})"), None)),
            Route::Vla => {}
        }
    }
    if objects.is_empty() && point_req.is_none() {
        return Err(err("objects is required: the set_plan object ids of this step, e.g. [\"O12\"] (target first; place: [object, support]) — or goal {pick, place}", None));
    }
    if kind == SkillKind::Place && (objects.is_empty() || (objects.len() < 2 && point_req.is_none())) {
        return Err(err("place needs objects [held object id, support/container id] or goal {pick: {id}, place: {id} | {point}}", None));
    }
    if !matches!(kind, SkillKind::Place | SkillKind::Approach | SkillKind::Pick) && point_req.is_some() {
        return Err(err("a place point is only for place / approach (go here) / pick steps", None));
    }
    if kind != SkillKind::Approach && objects.is_empty() {
        return Err(err("this step needs the object id (goal.pick.id)", None));
    }
    let num = |k: &str| a.get(k).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())));
    let budget_s = num("budget_s").or(num("max_s")).unwrap_or_else(|| default_budget_s(kind));
    if !(budget_s.is_finite() && budget_s > 0.0) {
        return Err(err("max_s must be a positive number of seconds", None));
    }
    let policy = a
        .get("policy")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| std::env::var("MR_VLA_POLICY").ok())
        .unwrap_or_else(|| "scripted".into());
    Ok(VlaCall { skill, kind, objects, budget_s: budget_s.min(600.0), policy, goal, point_req })
}

fn err(msg: &str, route: Option<&str>) -> Value {
    let mut v = json!({"status": "error", "executor": "vla", "message": msg});
    if let Some(r) = route {
        v["route"] = json!(r);
        v["hint"] = json!("use move_robot (part base, mode go_to) for this step");
    } else {
        v["hint"] = json!("fix the call: {\"executor\":\"vla\",\"skill\":<step sentence>,\"objects\":[ids],\"max_s\":<s>}");
    }
    v
}

// ---------------------------------------------------------------- 물체 칸 (VLA_INPUT 3절 일부)

pub const MAX_SLOTS: usize = 16;

/// 정책에 주는 물체 칸 하나. 지도마다 다른 값(map 좌표·id 순서)은 정책 쪽 값에 쓰지 않는다 — `id` 는 기록·시험용.
#[derive(Clone, Debug, PartialEq)]
pub struct Slot {
    pub id: String,
    /// 로봇 기준 위치(m)
    pub rel: [f64; 3],
    pub dist: f64,
    pub bearing: f64,
    pub extent: [f64; 3],
    pub state: ObjState,
    /// 목표인지(set_plan 물체 id 로 켬, 순서가 아니라 이 값으로)
    pub is_goal: bool,
    /// 출처: 지금 보는 중(true) / 기억(false)
    pub visible: bool,
    /// 위치 불확실도(m)
    pub unc_m: f64,
    pub age_s: f64,
}

/// 칸 고르기: 목표 물체는 항상(호출 순서대로), 나머지는 로봇에서 가까운 순으로 최대 16.
pub fn build_slots(objs: &[MemObject], pose: [f64; 3], goals: &[String], now: f64, unc: &dyn Fn(&MemObject) -> f64, visible_s: f64) -> Vec<Slot> {
    let mk = |o: &MemObject, goal: bool| {
        let rel = verify::to_robot(pose, o.pos);
        let age = (now - o.last_seen).max(0.0);
        Slot {
            id: o.id.clone(),
            rel,
            dist: rel[0].hypot(rel[1]),
            bearing: rel[1].atan2(rel[0]),
            extent: o.extent,
            state: o.state,
            is_goal: goal,
            visible: age <= visible_s && o.state != ObjState::Gone,
            unc_m: unc(o),
            age_s: age,
        }
    };
    let mut out: Vec<Slot> = vec![];
    for g in goals {
        if let Some(o) = objs.iter().find(|o| o.id.eq_ignore_ascii_case(g)) {
            if !out.iter().any(|s| s.id == o.id) {
                out.push(mk(o, true));
            }
        }
    }
    let mut rest: Vec<Slot> = objs.iter().filter(|o| !out.iter().any(|s| s.id == o.id)).map(|o| mk(o, false)).collect();
    rest.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    for s in rest {
        if out.len() >= MAX_SLOTS {
            break;
        }
        out.push(s);
    }
    out
}

// ---------------------------------------------------------------- 정책 접점

/// 정책이 한 번에 보는 것(10 Hz)
pub struct VlaObs<'a> {
    pub skill: &'a str,
    pub kind: SkillKind,
    pub body: &'a LimoState,
    pub slots: &'a [Slot],
    /// 단계 시작 뒤 시간(s)
    pub t: f64,
    /// 직전 명령(거른 뒤)
    pub last_cmd: &'a [f64; A8],
    /// 목표 칸 2 × 16(집을 것·놓을 곳, goal.rs): 원값 `raw` 와 정책 입력 `norm`(X0 432..463 과 같은 배치)
    pub goal: &'a GoalEntries,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VlaOut {
    /// [vx m/s, wz rad/s, j1..j5 rad(목표 위치), 그리퍼 벌림 0..1]
    pub action: [f64; A8],
    /// "이 단계 끝" 확률(POLICY 5.3 끝 신호 머리)
    pub end_prob: f64,
    /// 확신 낮음 0..1(묶음 퍼짐 등, 없으면 0)
    pub unsure: f64,
}

pub trait VlaPolicy {
    fn name(&self) -> String;
    fn reset(&mut self, _skill: &str) {}
    fn act(&mut self, o: &VlaObs) -> VlaOut;
    /// 밖에서 넣는 정책(external)은 act 대신 이 값을 쓴다
    fn is_external(&self) -> bool {
        false
    }
}

/// 각본 정책(대역): 목표 칸만 보고 다가가기 → 팔 뻗기(수치 역기구학) → 잡기/놓기 → 들기/물러나기 → 끝 신호.
/// 학습된 VLA 가 아니다. 목표인지 값을 쓰므로 칸 순서·문장 대신 id 표시가 실제로 행동을 고르는지 시험할 수 있다.
pub struct ScriptedPolicy {
    phase: u8,
    t_phase: f64,
    arm_goal: Option<[f64; 5]>,
    /// 끝 신호 대신 계속 0(시간 초과·진척 없음 시험)
    pub never_end: bool,
    /// 아무것도 안 함(진척 없음 시험)
    pub idle: bool,
    /// 앞으로만 달림(안전 시험)
    pub ram: bool,
}

impl ScriptedPolicy {
    pub fn new() -> ScriptedPolicy {
        ScriptedPolicy { phase: 0, t_phase: 0.0, arm_goal: None, never_end: false, idle: false, ram: false }
    }
    fn next(&mut self, t: f64) {
        self.phase += 1;
        self.t_phase = t;
        self.arm_goal = None;
    }
}

impl Default for ScriptedPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// 수치 역기구학(감쇠 최소제곱, j1..j4): 손끝을 base_footprint 기준 `p` 로. j5(손목 돌림)는 지금 값.
pub fn ik(p: [f64; 3], q0: &[f64; 5]) -> [f64; 5] {
    let lim = limo::arm_limits();
    let mut q = *q0;
    for _ in 0..80 {
        let f = limo::fk_eef(&q);
        let e = [p[0] - f[0], p[1] - f[1], p[2] - f[2]];
        if e.iter().map(|x| x * x).sum::<f64>().sqrt() < 1e-3 {
            break;
        }
        let mut jac = [[0.0; 4]; 3];
        for j in 0..4 {
            let mut qq = q;
            qq[j] += 1e-4;
            let g = limo::fk_eef(&qq);
            for i in 0..3 {
                jac[i][j] = (g[i] - f[i]) / 1e-4;
            }
        }
        // dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e
        let lam2 = 1e-3;
        let mut m = [[0.0; 3]; 3];
        for i in 0..3 {
            for k in 0..3 {
                m[i][k] = (0..4).map(|j| jac[i][j] * jac[k][j]).sum::<f64>() + if i == k { lam2 } else { 0.0 };
            }
        }
        let Some(mi) = inv3(&m) else { break };
        let y = [0, 1, 2].map(|i| (0..3).map(|k| mi[i][k] * e[k]).sum::<f64>());
        for j in 0..4 {
            let dq = (0..3).map(|i| jac[i][j] * y[i]).sum::<f64>().clamp(-0.2, 0.2);
            q[j] = (q[j] + dq).clamp(lim[j].0, lim[j].1);
        }
    }
    q
}

fn inv3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if d.abs() < 1e-12 {
        return None;
    }
    let c = |a: usize, b: usize, x: usize, y: usize| m[a][b] * m[x][y] - m[a][y] * m[x][b];
    Some([
        [c(1, 1, 2, 2) / d, -c(0, 1, 2, 2) / d, c(0, 1, 1, 2) / d],
        [-c(1, 0, 2, 2) / d, c(0, 0, 2, 2) / d, -c(0, 0, 1, 2) / d],
        [c(1, 0, 2, 1) / d, -c(0, 0, 2, 1) / d, c(0, 0, 1, 1) / d],
    ])
}

impl VlaPolicy for ScriptedPolicy {
    fn name(&self) -> String {
        "scripted".into()
    }
    fn reset(&mut self, _skill: &str) {
        let (n, i, r) = (self.never_end, self.idle, self.ram);
        *self = ScriptedPolicy::new();
        self.never_end = n;
        self.idle = i;
        self.ram = r;
    }
    fn act(&mut self, o: &VlaObs) -> VlaOut {
        let b = o.body;
        let mut a = [0.0; A8];
        a[2..7].copy_from_slice(&b.arm);
        a[7] = o.last_cmd[7];
        if o.t < 1e-9 {
            a[7] = b.grip;
        }
        let out = |a: [f64; A8], e: f64| VlaOut { action: a, end_prob: e, unsure: 0.0 };
        if self.idle {
            a[2..7].copy_from_slice(&o.last_cmd[2..7]);
            return out(a, 0.0);
        }
        if self.ram {
            a[0] = 1.0;
            return out(a, 0.0);
        }
        // 다가갈 칸: 놓기는 받침(두 번째 목표), 그 밖은 첫 목표
        let goals: Vec<&Slot> = o.slots.iter().filter(|s| s.is_goal).collect();
        // 놓을 곳·다가갈 곳이 지점이면 목표 칸(놓을 곳)의 로봇 기준 자리로(크기 0)
        let pe = &o.goal.raw[crate::goal::PLACE];
        let pt_slot = (matches!(o.kind, SkillKind::Place | SkillKind::Approach) && pe[crate::goal::GE_KPT] > 0.5).then(|| Slot {
            id: "point".into(),
            rel: [pe[crate::goal::GE_POS] as f64, pe[crate::goal::GE_POS + 1] as f64, pe[crate::goal::GE_POS + 2] as f64 + crate::goal::BASE_Z],
            dist: pe[crate::goal::GE_DIST] as f64,
            bearing: (pe[crate::goal::GE_SIN] as f64).atan2(pe[crate::goal::GE_COS] as f64),
            extent: [0.0; 3],
            state: ObjState::Seen,
            is_goal: true,
            visible: true,
            unc_m: 0.0,
            age_s: 0.0,
        });
        let tgt = if pt_slot.is_some() { pt_slot.as_ref() } else if o.kind == SkillKind::Place { goals.get(1).or(goals.first()).copied() } else { goals.first().copied() };
        let Some(tgt) = tgt else {
            return out(a, 0.0);
        };
        let tp = o.t - self.t_phase;
        let arm_err = |g: &[f64; 5]| g.iter().zip(&b.arm).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        // 팔 받침(omx_link0 = base_footprint x −0.04) 기준 수평 거리·방향
        let (ax, ay) = (tgt.rel[0] + 0.04, tgt.rel[1]);
        let d = ax.hypot(ay);
        let head = ay.atan2(ax);
        let want = 0.27;
        let end = if self.never_end { 0.0 } else { 1.0 };
        match (o.kind, self.phase) {
            // 0: 다가가기 — 물체 쪽으로 돌고 팔 닿는 거리까지
            (_, 0) => {
                a[1] = (1.5 * head).clamp(-0.6, 0.6);
                if head.abs() < 0.35 {
                    a[0] = (0.8 * (d - want)).clamp(-0.15, 0.25);
                }
                if o.kind != SkillKind::Place {
                    a[7] = 1.0; // 손 펴고 다가감
                }
                let stopped = b.base_v[0].abs() < 0.01 && tp > 1.0;
                if (d < want + 0.02 || (d < verify::REACH_M - 0.02 && stopped)) && head.abs() < 0.1 {
                    a[0] = 0.0;
                    a[1] = 0.0;
                    if o.kind == SkillKind::Approach {
                        return out(a, end);
                    }
                    self.next(o.t);
                }
                out(a, 0.0)
            }
            // 1: 팔 뻗기 — 집기는 물체 중심, 놓기는 받침 윗면 위 5 cm, 그 밖은 물체 중심
            (_, 1) => {
                let z = match o.kind {
                    SkillKind::Place => tgt.rel[2] + tgt.extent[2] * 0.5 + 0.06,
                    _ => tgt.rel[2],
                };
                let g = *self.arm_goal.get_or_insert_with(|| ik([tgt.rel[0], tgt.rel[1], z], &b.arm));
                a[2..7].copy_from_slice(&g);
                if (arm_err(&g) < 0.04 && tp > 0.5) || tp > 4.0 {
                    self.next(o.t);
                }
                out(a, 0.0)
            }
            // 2: 집기 = 닫기 / 놓기 = 열기 / 그 밖 = 그대로
            (k, 2) => {
                a[2..7].copy_from_slice(&o.last_cmd[2..7]);
                a[7] = match k {
                    SkillKind::Pick => 0.0,
                    SkillKind::Place => 1.0,
                    _ => a[7],
                };
                if tp > 1.2 {
                    self.next(o.t);
                }
                out(a, 0.0)
            }
            // 3: 집기 = 15 cm 들기 / 놓기·그 밖 = 홈 자세로 물러나기
            (k, 3) => {
                let g = *self.arm_goal.get_or_insert_with(|| {
                    if k == SkillKind::Pick {
                        let f = limo::fk_eef(&b.arm);
                        ik([f[0] - 0.03, f[1], f[2] + 0.15], &b.arm)
                    } else {
                        limo::ARM_HOME
                    }
                });
                a[2..7].copy_from_slice(&g);
                a[7] = o.last_cmd[7];
                if (arm_err(&g) < 0.04 && tp > 0.5) || tp > 4.0 {
                    self.next(o.t);
                }
                out(a, 0.0)
            }
            // 4: 그대로 들고 끝 신호
            _ => {
                a[2..7].copy_from_slice(&o.last_cmd[2..7]);
                a[7] = o.last_cmd[7];
                out(a, end)
            }
        }
    }
}

/// 재생 정책(대역): JSONL 한 줄에 `{"a":[8], "end":p, "unsure":u}`. 10 Hz 로 한 줄씩, 끝나면 마지막 줄을 유지.
pub struct ReplayPolicy {
    pub rows: Vec<VlaOut>,
    i: usize,
    pub path: String,
}

impl ReplayPolicy {
    pub fn from_file(path: &str) -> Result<ReplayPolicy, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("replay {path}: {e}"))?;
        let mut rows = vec![];
        for (n, l) in s.lines().enumerate() {
            let l = l.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            let v: Value = serde_json::from_str(l).map_err(|e| format!("replay {path}:{}: {e}", n + 1))?;
            rows.push(Self::row(&v).ok_or_else(|| format!("replay {path}:{}: need {{\"a\":[8 numbers]}}", n + 1))?);
        }
        if rows.is_empty() {
            return Err(format!("replay {path}: no rows"));
        }
        Ok(ReplayPolicy { rows, i: 0, path: path.into() })
    }
    pub fn row(v: &Value) -> Option<VlaOut> {
        let a = v.get("a")?.as_array()?;
        if a.len() != A8 {
            return None;
        }
        let mut action = [0.0; A8];
        for (k, x) in a.iter().enumerate() {
            action[k] = x.as_f64()?;
        }
        Some(VlaOut { action, end_prob: v.get("end").and_then(|x| x.as_f64()).unwrap_or(0.0), unsure: v.get("unsure").and_then(|x| x.as_f64()).unwrap_or(0.0) })
    }
}

impl VlaPolicy for ReplayPolicy {
    fn name(&self) -> String {
        format!("replay:{}", self.path)
    }
    fn reset(&mut self, _s: &str) {
        self.i = 0;
    }
    fn act(&mut self, _o: &VlaObs) -> VlaOut {
        let r = self.rows[self.i.min(self.rows.len() - 1)];
        self.i += 1;
        r
    }
}

/// 밖에서 넣는 정책(학습된 엔진 자리): 행동·끝 신호는 [`crate::Robot::vla_tick_ext`] 로 들어온다.
pub struct ExternalPolicy;

impl VlaPolicy for ExternalPolicy {
    fn name(&self) -> String {
        "external".into()
    }
    fn act(&mut self, o: &VlaObs) -> VlaOut {
        VlaOut { action: *o.last_cmd, end_prob: 0.0, unsure: 0.0 }
    }
    fn is_external(&self) -> bool {
        true
    }
}

pub fn make_policy(spec: &str) -> Result<Box<dyn VlaPolicy>, String> {
    let s = spec.trim();
    match s {
        "scripted" | "" => Ok(Box::new(ScriptedPolicy::new())),
        "scripted:never_end" => Ok(Box::new(ScriptedPolicy { never_end: true, ..ScriptedPolicy::new() })),
        "scripted:idle" => Ok(Box::new(ScriptedPolicy { idle: true, ..ScriptedPolicy::new() })),
        "scripted:ram" => Ok(Box::new(ScriptedPolicy { ram: true, ..ScriptedPolicy::new() })),
        "external" => Ok(Box::new(ExternalPolicy)),
        _ if s.starts_with("replay:") => Ok(Box::new(ReplayPolicy::from_file(&s[7..])?)),
        _ => Err(format!("policy must be scripted | scripted:never_end | scripted:idle | scripted:ram | replay:<jsonl> | external (got '{s}')")),
    }
}

// ---------------------------------------------------------------- 한계·문턱

#[derive(Clone, Debug)]
pub struct VlaParams {
    /// 정책 주기(Hz) — 실행기 스텝(30 Hz)마다 거르개는 돌고, 정책은 이 주기로
    pub policy_hz: f64,
    pub end_prob: f64,
    pub end_hold_s: f64,
    /// 집기: 확인이 이만큼 이어져야 done(들림 1 s 유지, POLICY 3.1)
    pub verify_hold_s: f64,
    pub start_max_m: f64,
    pub unc_max_m: f64,
    pub visible_s: f64,
    pub lost_s: f64,
    pub unsure: f64,
    pub unsure_s: f64,
    /// 거르개가 크게 바꾸는(정지·막힘) 상태가 이만큼 이어지면 failed(unsafe)
    pub unsafe_filter_s: f64,
    // 거르개 한계 (POLICY 7.1)
    pub arm_vmax: f64,
    pub grip_vmax: f64,
    pub base_vmax: f64,
    pub base_wmax: f64,
    pub base_acc: f64,
    pub base_wacc: f64,
    pub track_limit: f64,
    /// 진척 판정 문턱: 관절(rad), 그리퍼(비율), 베이스(m), 물체(m)
    pub prog_joint: f64,
    pub prog_grip: f64,
    pub prog_base: f64,
    pub prog_obj: f64,
}

impl Default for VlaParams {
    fn default() -> Self {
        VlaParams {
            policy_hz: 10.0,
            end_prob: 0.8,
            end_hold_s: 0.5,
            verify_hold_s: 1.0,
            start_max_m: 1.5,
            unc_max_m: 0.3,
            visible_s: 1.0,
            lost_s: 2.0,
            unsure: 0.5,
            unsure_s: 1.0,
            unsafe_filter_s: 2.0,
            // OMX-F XL430 최고 4.8 rad/s(URDF velocity)의 약 1/5 — R1 기본 45°/s 를 OMX 에 맞게(추정)
            arm_vmax: 60f64.to_radians(),
            grip_vmax: 1.0,
            // VLA 베이스는 `delta` 수준 0.3 m/s 이하(POLICY 7.1)
            base_vmax: 0.3,
            base_wmax: 35f64.to_radians(),
            base_acc: 0.6,
            base_wacc: 90f64.to_radians(),
            track_limit: 20f64.to_radians(),
            prog_joint: 3f64.to_radians(),
            prog_grip: 0.05,
            prog_base: 0.03,
            prog_obj: 0.02,
        }
    }
}

// ---------------------------------------------------------------- 안전 거르개 (7.1)

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FilterInfo {
    /// 관절 한계·속도 상한으로 잘림(정책 행동이 범위 밖)
    pub clipped: bool,
    /// 베이스가 장애물 앞에서 멈춤/줄임: (무엇 때문, 남은 여유 m)
    pub base_stop: Option<(&'static str, f64)>,
    /// 팔이 지령을 못 따라옴(접촉·자기 충돌) → 측정 자리에 멈춤
    pub arm_blocked: bool,
    /// 입력에 NaN
    pub bad_input: bool,
}

impl FilterInfo {
    /// 거르개가 행동을 "크게" 바꿨나(되풀이되면 unsafe)
    pub fn heavy(&self) -> bool {
        self.base_stop.map_or(false, |(_, room)| room <= 0.01) || self.arm_blocked || self.bad_input
    }
}

/// 매 스텝 VLA 행동을 거르는 상태(직전 거른 명령·막힘 셈)
#[derive(Clone, Debug)]
pub struct SafetyFilter {
    pub prev: [f64; A8],
    pub init: bool,
    off_track: u32,
}

impl Default for SafetyFilter {
    fn default() -> Self {
        SafetyFilter { prev: [0.0; A8], init: false, off_track: 0 }
    }
}

/// 베이스 앞·뒤 여유(m)와 무엇이 막나 — 지도가 없으면 None
pub type FreeFn<'a> = &'a dyn Fn(bool) -> Option<(f64, &'static str)>;

impl SafetyFilter {
    /// 지금 측정으로 유지값을 잡는다(베이스 0, 팔·그리퍼 = 지금 자리)
    pub fn hold_from(&mut self, st: &LimoState) {
        self.prev = [0.0; A8];
        self.prev[2..7].copy_from_slice(&st.arm);
        self.prev[7] = st.grip;
        self.init = true;
        self.off_track = 0;
    }

    /// 행동 8 하나를 거른다. `free(forward)` = 앞(true)/뒤(false) 여유, `stop_margin` = 그 앞에서 멈출 거리.
    pub fn filter(&mut self, input: &[f64; A8], st: &LimoState, dt: f64, p: &VlaParams, free: FreeFn, stop_margin: f64) -> ([f64; A8], FilterInfo) {
        if !self.init {
            self.hold_from(st);
        }
        let mut info = FilterInfo::default();
        let mut a = *input;
        for (k, x) in a.iter_mut().enumerate() {
            if !x.is_finite() {
                *x = if k < 2 { 0.0 } else { self.prev[k] };
                info.bad_input = true;
            }
        }
        // 베이스: 속도 상한 → 가감속 → 장애물 앞 멈춤(앞으로는 지도·깊이, 뒤로는 지도)
        let (vx_in, wz_in) = (a[0], a[1]);
        a[0] = a[0].clamp(-p.base_vmax, p.base_vmax);
        a[1] = a[1].clamp(-p.base_wmax, p.base_wmax);
        if (a[0] - vx_in).abs() > 1e-9 || (a[1] - wz_in).abs() > 1e-9 {
            info.clipped = true;
        }
        if a[0].abs() > 1e-4 {
            if let Some((d, by)) = free(a[0] > 0.0) {
                let room = d - stop_margin;
                let vmax = (2.0 * p.base_acc * room.max(0.0)).sqrt();
                if a[0].abs() > vmax {
                    a[0] = a[0].signum() * vmax;
                    info.base_stop = Some((by, (room.max(0.0) * 100.0).round() / 100.0));
                }
                if room <= 0.01 {
                    a[0] = 0.0;
                }
            }
        }
        let lim = [p.base_acc * dt, p.base_wacc * dt];
        for i in 0..2 {
            // 멈추는 쪽은 바로(정지는 가속 한도에 안 묶음)
            let toward_zero = a[i].abs() < self.prev[i].abs() && a[i] * self.prev[i] >= 0.0;
            if !toward_zero {
                a[i] = self.prev[i] + (a[i] - self.prev[i]).clamp(-lim[i], lim[i]);
            }
        }
        // 팔: 실제 한계 안쪽 → 속도 상한(직전 지령에서 한 스텝 몫) → 막힘이면 측정 자리
        let al = limo::arm_limits();
        for j in 0..5 {
            let k = 2 + j;
            let c = a[k].clamp(al[j].0, al[j].1);
            if (c - a[k]).abs() > 1e-9 {
                info.clipped = true;
            }
            let step = p.arm_vmax * dt;
            a[k] = self.prev[k] + (c - self.prev[k]).clamp(-step, step);
        }
        let track = (0..5).map(|j| (self.prev[2 + j] - st.arm[j]).abs()).fold(0.0, f64::max);
        self.off_track = if track > p.track_limit { self.off_track + 1 } else { 0 };
        if self.off_track >= 6 {
            // 더 밀지 않는다: 지금 자리에 멈춤(POLICY 7.1 막힘 판정)
            a[2..7].copy_from_slice(&st.arm);
            info.arm_blocked = true;
            self.off_track = 0;
        }
        // 그리퍼: 0..1, 속도 상한. 닫다가 막히면 지령을 그대로 둔다(쥔 채 유지, 지금 규칙)
        let g = a[7].clamp(0.0, 1.0);
        if (g - a[7]).abs() > 1e-9 {
            info.clipped = true;
        }
        a[7] = self.prev[7] + (g - self.prev[7]).clamp(-p.grip_vmax * dt, p.grip_vmax * dt);
        self.prev = a;
        (a, info)
    }
}

// ---------------------------------------------------------------- 실행 상태

pub struct VlaRun {
    pub call: VlaCall,
    pub policy: Box<dyn VlaPolicy>,
    pub steps: u64,
    pub policy_steps: u64,
    pub t: f64,
    pub out: VlaOut,
    pub ext: Option<VlaOut>,
    end_since: Option<f64>,
    verify_since: Option<f64>,
    lost_since: Option<f64>,
    unsure_since: Option<f64>,
    heavy_since: Option<f64>,
    /// 진척 기준점(시각, 팔, 그리퍼, 자세, 대상 위치)
    pub(crate) prog: (f64, [f64; 5], f64, [f64; 3], Option<[f64; 3]>),
    pub n_clipped: u64,
    pub n_base_stops: u64,
    pub n_arm_blocked: u64,
    pub min_clear: f64,
    pub contacts0: (u64, u64),
    pub last_verdict: (Verdict, Value),
    /// 목표 칸: 마지막으로 안 자리(잃음 표시용), 이번 스텝 칸, 시작 때 지점 검사 결과
    pub goal_mem: GoalMemory,
    pub goal_now: GoalEntries,
    pub point_check: Option<(PointCheck, [f64; 3])>,
}

/// 실행기 문맥(Robot 안에 늘 있음): 기억 물체·시각 맞추기·접촉·거르개·지금 실행
pub struct VlaCtx {
    pub params: VlaParams,
    pub objects: Vec<MemObject>,
    /// 시뮬 시각 = nav.now + offset (set_objects 때 맞춤)
    pub time_offset: f64,
    /// (시뮬 시각, 누적 이동 m, 누적 회전 rad) — 기억 불확실도용
    pub hist: Vec<(f64, f64, f64)>,
    turn_cum: f64,
    last_yaw: Option<f64>,
    /// 접촉 누적: 몸통(바퀴·차체), 팔·그리퍼
    pub contacts_body: u64,
    pub contacts_arm: u64,
    pub filter: SafetyFilter,
    pub run: Option<VlaRun>,
    pub last_cmd: Option<[f64; A8]>,
}

impl Default for VlaCtx {
    fn default() -> Self {
        VlaCtx {
            params: VlaParams::default(),
            objects: vec![],
            time_offset: 0.0,
            hist: vec![],
            turn_cum: 0.0,
            last_yaw: None,
            contacts_body: 0,
            contacts_arm: 0,
            filter: SafetyFilter::default(),
            run: None,
            last_cmd: None,
        }
    }
}

fn wrap(a: f64) -> f64 {
    let mut x = a % (2.0 * PI);
    if x > PI {
        x -= 2.0 * PI
    }
    if x < -PI {
        x += 2.0 * PI
    }
    x
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

impl VlaCtx {
    /// 기억 물체를 바꾼다(keyframe 마다). `now` = 시뮬 시각(s), `nav_now`·`pose`·`odo_m` = 지금 주행 상태.
    pub fn set_objects(&mut self, objs: Vec<MemObject>, now: f64, nav_now: f64, pose: [f64; 3], odo_m: f64) {
        self.objects = objs;
        self.time_offset = now - nav_now;
        if let Some(y) = self.last_yaw {
            self.turn_cum += wrap(pose[2] - y).abs();
        }
        self.last_yaw = Some(pose[2]);
        self.hist.push((now, odo_m, self.turn_cum));
        if self.hist.len() > 4096 {
            self.hist.drain(..2048);
        }
    }

    /// 기억 불확실도(m): 마지막으로 본 뒤 움직인 거리 × 10 % + 회전 × 거리 × 5 % + 2 cm (바퀴 오도메트리 오차, 추정)
    pub fn uncertainty(&self, o: &MemObject, now: f64, pose: [f64; 3], odo_m: f64) -> f64 {
        if now - o.last_seen <= self.params.visible_s {
            return 0.02;
        }
        let (mut d0, mut r0) = (odo_m, self.turn_cum);
        for &(t, d, r) in self.hist.iter().rev() {
            if t <= o.last_seen {
                d0 = d;
                r0 = r;
                break;
            }
            d0 = d;
            r0 = r;
        }
        let dist = (o.pos[0] - pose[0]).hypot(o.pos[1] - pose[1]);
        0.02 + 0.1 * (odo_m - d0).max(0.0) + 0.05 * (self.turn_cum - r0).max(0.0) * dist
    }

    pub fn find(&self, id: &str) -> Option<&MemObject> {
        self.objects.iter().find(|o| o.id.eq_ignore_ascii_case(id))
    }
}

/// 결과 JSON 의 공통 모양
pub fn result(status: &str, reason: &str, evidence: Value, run: Option<&VlaRun>, call: &VlaCall, contacts: u64, min_clear: Option<f64>) -> Value {
    let mut r = json!({
        "status": status, "reason": reason, "evidence": evidence,
        "steps": run.map_or(0, |r| r.policy_steps), "min_clear_m": min_clear.filter(|x| x.is_finite()).map(r2), "contacts": contacts,
        "executor": "vla", "skill": call.skill, "kind": call.kind.name(), "objects": call.objects, "goal": call.goal.to_json(),
    });
    if let Some(run) = run {
        r["policy"] = json!(run.policy.name());
        r["sim_steps"] = json!(run.steps);
        r["time_s"] = json!(r2(run.t));
        r["budget_s"] = json!(call.budget_s);
        r["end_prob"] = json!(r2(run.out.end_prob));
        r["filter"] = json!({"clipped_steps": run.n_clipped, "base_stops": run.n_base_stops, "arm_blocked": run.n_arm_blocked});
        if let Some((c, asked)) = &run.point_check {
            r["point"] = c.to_json(*asked);
        }
    }
    match status {
        "handback" => {
            r["hint"] = json!(match reason {
                "too_far" | "not_in_map" => "move closer with move_robot go_to first, then call the VLA step again",
                "not_visible" => "look around (move_robot probe) or go_to where it was last seen, then retry",
                "stalled" => "resend the same step once; if it stalls again, replan",
                "unsure" => "look around once and retry, or ask the planner",
                _ => "the state machine decides the next step",
            })
        }
        "failed" | "timeout" => r["hint"] = json!("recover per plan.md 3.4 (retry the same sentence up to 2 times, then approach again, then replan)"),
        _ => {}
    }
    r
}

/// 진척 판정(예산 1/3 동안 변화 없음 = 멈춤)
pub(crate) fn progress_check(run: &mut VlaRun, p: &VlaParams, st: &LimoState, pose: [f64; 3], tgt: Option<[f64; 3]>) -> bool {
    let (t0, arm0, g0, pose0, obj0) = run.prog;
    let joint = arm0.iter().zip(&st.arm).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let base = (pose[0] - pose0[0]).hypot(pose[1] - pose0[1]) + 0.2 * wrap(pose[2] - pose0[2]).abs();
    let obj = match (obj0, tgt) {
        (Some(a), Some(b)) => ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt(),
        _ => 0.0,
    };
    if joint > p.prog_joint || (st.grip - g0).abs() > p.prog_grip || base > p.prog_base || obj > p.prog_obj {
        run.prog = (run.t, st.arm, st.grip, pose, tgt);
        return false;
    }
    run.t - t0 > run.call.budget_s / 3.0
}

impl VlaRun {
    pub fn new(call: VlaCall, mut policy: Box<dyn VlaPolicy>, st: &LimoState, pose: [f64; 3], tgt: Option<[f64; 3]>, contacts0: (u64, u64)) -> VlaRun {
        policy.reset(&call.skill);
        VlaRun {
            call,
            policy,
            steps: 0,
            policy_steps: 0,
            t: 0.0,
            out: VlaOut { action: [0.0; A8], end_prob: 0.0, unsure: 0.0 },
            ext: None,
            end_since: None,
            verify_since: None,
            lost_since: None,
            unsure_since: None,
            heavy_since: None,
            prog: (0.0, st.arm, st.grip, pose, tgt),
            n_clipped: 0,
            n_base_stops: 0,
            n_arm_blocked: 0,
            min_clear: f64::INFINITY,
            contacts0,
            last_verdict: (Verdict::Unknown, Value::Null),
            goal_mem: GoalMemory::default(),
            goal_now: GoalEntries::default(),
            point_check: None,
        }
    }

    /// 끝 조건 판정(7.2 표 순서: 안전 → 대상 사라짐 → 끝 신호 + 확인 → 확신 낮음 → 진척 없음 → 시간)
    /// 반환: Some((status, reason, evidence))
    /// `point_anchor`: 기준이 지점(지점에 놓기·지점으로 다가가기) — 사라질 대상이 없다
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decide(&mut self, p: &VlaParams, info: &FilterInfo, body_contacts: u64, goal: Option<&MemObject>, point_anchor: bool, goal_visible: bool, goal_unc: f64, verdict: (Verdict, Value), stalled: bool) -> Option<(&'static str, String, Value)> {
        let t = self.t;
        self.last_verdict = verdict.clone();
        if body_contacts > 0 {
            return Some(("failed", "unsafe".into(), json!({"why": format!("{body_contacts} new body contact(s) with something other than the floor"), "verify": verdict.1})));
        }
        if info.heavy() {
            let s = *self.heavy_since.get_or_insert(t);
            if t - s >= p.unsafe_filter_s {
                return Some(("failed", "unsafe".into(), json!({"why": format!("safety filter kept stopping the policy for {:.1} s ({})", t - s,
                    if info.arm_blocked { "arm blocked" } else if info.base_stop.is_some() { "base at an obstacle" } else { "bad actions" }), "verify": verdict.1})));
            }
        } else {
            self.heavy_since = None;
        }
        let held = goal.map_or(false, |g| g.state == ObjState::Held);
        let lost = !point_anchor && match goal {
            None => true,
            Some(g) => g.state == ObjState::Gone || (!goal_visible && goal_unc > p.unc_max_m && !held),
        };
        if lost {
            let s = *self.lost_since.get_or_insert(t);
            if t - s >= p.lost_s {
                return Some(("handback", "not_visible".into(), json!({"why": "target slot gone or only remembered with large uncertainty", "unc_m": r2(goal_unc), "verify": verdict.1})));
            }
        } else {
            self.lost_since = None;
        }
        if self.out.end_prob > p.end_prob {
            let s = *self.end_since.get_or_insert(t);
            if t - s >= p.end_hold_s - 1e-9 {
                match verdict.0 {
                    Verdict::Done => {
                        let vs = *self.verify_since.get_or_insert(t);
                        let need = if self.call.kind == SkillKind::Pick { p.verify_hold_s } else { 0.0 };
                        if t - vs >= need - 1e-9 {
                            return Some(("done", "verified".into(), verdict.1));
                        }
                    }
                    Verdict::Failed => {
                        let why = verdict.1.get("why").and_then(|w| w.as_str()).unwrap_or("").to_string();
                        let reason = if why.contains("grasp missed") { "grasp_missed" } else if why.contains("not at the destination") { "place_missed" } else { "verify_failed" };
                        return Some(("failed", reason.into(), verdict.1));
                    }
                    Verdict::Unknown => return Some(("handback", "unverified".into(), verdict.1)),
                    Verdict::Running => {
                        self.verify_since = None;
                    }
                }
            }
        } else {
            self.end_since = None;
            self.verify_since = None;
        }
        if self.out.unsure > p.unsure {
            let s = *self.unsure_since.get_or_insert(t);
            if t - s >= p.unsure_s {
                return Some(("handback", "unsure".into(), json!({"why": "policy confidence stayed low", "unsure": r2(self.out.unsure), "verify": verdict.1})));
            }
        } else {
            self.unsure_since = None;
        }
        if stalled {
            return Some(("handback", "stalled".into(), json!({"why": format!("no arm, gripper, base or object progress for {:.1} s (budget/3)", self.call.budget_s / 3.0), "verify": verdict.1})));
        }
        if t >= self.call.budget_s {
            return Some(("timeout", "budget".into(), json!({"why": format!("skill budget {:.0} s used", self.call.budget_s), "verify": verdict.1})));
        }
        None
    }
}

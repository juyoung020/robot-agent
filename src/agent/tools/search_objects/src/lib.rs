//! `search_objects` · `confirm_object` — LLM 이 물체 기억에서 물건을 찾고, 확인되면 이름을 고치는 도구 두 개.
//!
//! LLM(Qwen3.5-9B, KAU API)은 글만 주고받으므로 벡터는 도구 **안에서만** 쓴다. 찾기·이름 확률·속성 낱말은 공용 물체 색인
//! (`src/scene_graph/clip` 의 `sgsearch.h`, C++ — RecallVLA 실행기도 자기 질의 벡터로 같은 색인을 씀)이 하고,
//! 이 크레이트는 그 결과에 기억의 자리 정보(방·기준물·상태·마지막으로 본 때·로봇까지 거리)를 붙여 짧은 JSON 글로 만든다.
//!
//! 3 단계(plan.md 3.3, docs/map_vla/MAPVLA_SPEC.md "공용 물체 찾기"):
//! 1. 이름 검색 — 등록·확인된 이름 / 동의어 / 상위어(질의 "chair" → 등록 "straight chair").
//! 2. 이름이 없거나 약하면 **자동으로** 이름을 무시한 생김새 재검색 — 물체마다 저장된 시점 벡터로 P(질의어 | 모습)
//!    (라벨 표 안에서 등록 이름과 견준 물체 안 상대 확률). 결과에 `match_type: "appearance"`, `p_query`, `p_registered`.
//! 3. LLM 이 묻거나(`ask_user` 칸이 언제 물을지 알려 줌) id 를 VLA 에 힌트로 넘기고, 확인되면 `confirm_object`
//!    → 이름 사후에 강한 관측(베이즈 갱신) → 다음엔 1 에서 바로 찾음. 확인은 모두 기억 폴더 `confirmations.jsonl` 에 남는다.
//!
//! - [`definitions`]: OpenAI `tools` 스키마 두 개(9B 모델용으로 작게)
//! - [`parse_search`] · [`parse_confirm`]: 인자 → 구조체(틀리면 모델이 고칠 수 있는 문장)
//! - [`ObjectSearch`]: 색인 + 기억 보기 하나, [`ObjectSearch::run_tool`] 이 도구 호출 하나를 결과 JSON 으로
//! - [`format_search`]: 색인 결과 + 기억 → LLM 이 받는 글(순수 함수, 시험 가능)
//! - [`ffi`]: 같은 것을 C ABI 로

pub mod ffi;
pub mod landmark;
pub mod live;
pub mod livecheck;
pub mod memview;
pub mod sys;

use memview::{MemSource, Memory, ObjInfo, ViewJson};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const SEARCH: &str = "search_objects";
pub const CONFIRM: &str = "confirm_object";
pub const LIST: &str = "list_place";
/// list_place 가 돌려주는 최대 물체 수
pub const PLACE_MAX: usize = 15;
/// list_place(물체) 가 둘레를 훑는 반경(m, 기준 물체 상자까지 수평 거리) — search_objects 에는 쓰지 않는다
pub const NEAR_M: f64 = 1.5;
pub const STATES: [&str; 4] = ["seen", "moved", "held", "gone"];
pub const SOURCES: [&str; 2] = ["user", "close_look"];
/// 두 후보 점수가 이만큼 안이고 1 m 넘게 떨어져 있으면 어느 것인지 물어야 함
const TIE: f64 = 0.1;
/// 이름 후보 점수가 이보다 낮으면 약함
const WEAK: f64 = 0.5;

const SEARCH_DESC: &str = "Find objects in the robot's object memory. Searches registered names (synonyms, broader terms) and, if none or weak, \
automatically re-searches by appearance ignoring names (an object may be registered under a wrong name). \
query: a short noun, English preferred, Korean works (e.g. \"radio\", \"라디오\", \"red mug\"). \
Each match: id, name with probability, alternative names, match_type name|appearance, registered (the stored name, if different), \
attrs, room, landmark (nearest fixed furniture: id, name, dist_m, dz_m = height above its top), state, last_seen_ago_s, \
pos/size [x,y,z] m (map frame), pos_sd (position uncertainty, m), rel {x forward, y left, z, dist_m, bearing_deg} from the robot now, match. \
Refer to objects by id in plans; coordinates are for your reasoning only. \
If ask_user is present, ask the user (or verify) before acting; use confirm_object once the identity is confirmed.";

const LIST_DESC: &str = "List what the memory has at a place: a room (name or id like R2) -> objects in that room, \
or a furniture/object id (e.g. O12) -> objects within 1.5 m of its box (dist_m to its box, dz_m = height above its top). \
Max 15, nearest first, same row fields as search_objects (id, name, state, pos, size, rel …).";

const CONFIRM_DESC: &str = "Record that an object's identity is confirmed, so later searches find it by this name. \
Call it after the user confirms (source user) or after the robot looked closely and verified it (source close_look). \
id from search_objects (e.g. \"O27\"), name = what it really is (e.g. \"radio\").";

pub fn search_definition() -> Value {
    json!({"type": "function", "function": {"name": SEARCH, "description": SEARCH_DESC, "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "object to find, a short noun (English or Korean)"},
            "k": {"type": "integer", "minimum": 1, "maximum": 10, "description": "max matches (default 5)"},
            "room": {"type": "string", "description": "optional room name or id (e.g. kitchen, R2)"},
            "state": {"type": "string", "enum": STATES},
            "max_age_s": {"type": "number", "description": "only objects seen within this many seconds"},
            "seen_after_s": {"type": "number", "description": "only objects seen at or after this memory time (now_s of an earlier result)"}
        },
        "required": ["query"]
    }}})
}

pub fn confirm_definition() -> Value {
    json!({"type": "function", "function": {"name": CONFIRM, "description": CONFIRM_DESC, "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "object id, e.g. O27"},
            "name": {"type": "string", "description": "what the object really is"},
            "source": {"type": "string", "enum": SOURCES}
        },
        "required": ["id", "name", "source"]
    }}})
}

pub fn list_definition() -> Value {
    json!({"type": "function", "function": {"name": LIST, "description": LIST_DESC, "parameters": {
        "type": "object",
        "properties": {"place": {"type": "string", "description": "room name or id (kitchen, R2) or furniture id (O12)"}},
        "required": ["place"]
    }}})
}

pub fn definitions() -> Vec<Value> {
    vec![search_definition(), confirm_definition(), list_definition()]
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchArgs {
    pub query: String,
    pub k: usize,
    pub room: Option<String>,
    pub state: Option<String>,
    /// 마지막으로 본 뒤 이 초 안
    pub max_age_s: Option<f64>,
    /// 기억 시각(now_s) 이 값 이후에 본 것
    pub seen_after_s: Option<f64>,
}

/// list_place 자리: 방 / 물체(가구) id
#[derive(Clone, Debug, PartialEq)]
pub enum Place {
    Room(String),
    Object(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConfirmArgs {
    pub id: u32,
    pub name: String,
    pub source: String,
}

fn obj_args(args: &Value) -> Result<Value, String> {
    let a = match args {
        Value::String(s) => serde_json::from_str::<Value>(s).map_err(|e| format!("arguments are not JSON: {e}"))?,
        v => v.clone(),
    };
    if a.is_object() { Ok(a) } else { Err("arguments must be a JSON object".into()) }
}

fn opt_str(a: &Value, k: &str) -> Option<String> {
    match &a[k] {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// "O27" · "o27" · "27" · 27 → 27
pub fn parse_id(v: &Value) -> Option<u32> {
    match v {
        Value::Number(n) => n.as_u64().map(|x| x as u32),
        Value::String(s) => s.trim().trim_start_matches(['O', 'o']).parse().ok(),
        _ => None,
    }
}

pub fn parse_search(args: &Value) -> Result<SearchArgs, String> {
    let a = obj_args(args)?;
    let query = opt_str(&a, "query").ok_or("query is required: a short noun such as \"radio\" or \"라디오\"")?;
    let k = match &a["k"] {
        Value::Null => 5,
        v => v.as_u64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())).ok_or("k must be an integer 1..10")? as usize,
    }
    .clamp(1, 10);
    let state = opt_str(&a, "state").map(|s| s.to_lowercase());
    if let Some(s) = &state {
        if !STATES.contains(&s.as_str()) {
            return Err(format!("state must be one of {} (got '{s}')", STATES.join(", ")));
        }
    }
    let num = |k: &str| -> Result<Option<f64>, String> {
        match &a[k] {
            Value::Null => Ok(None),
            v => v
                .as_f64()
                .or_else(|| v.as_str().and_then(|s| s.trim().trim_end_matches('s').trim().parse().ok()))
                .filter(|x| x.is_finite() && *x >= 0.0)
                .map(Some)
                .ok_or_else(|| format!("{k} must be a number of seconds >= 0 (got {v})")),
        }
    };
    let (max_age_s, seen_after_s) = (num("max_age_s")?, num("seen_after_s")?);
    Ok(SearchArgs { query, k, room: opt_str(&a, "room"), state, max_age_s, seen_after_s })
}

pub fn parse_list(args: &Value) -> Result<Place, String> {
    let a = obj_args(args)?;
    let p = opt_str(&a, "place").ok_or("place is required: a room name or id (kitchen, R2) or a furniture id (O12)")?;
    let t = p.trim();
    if t.len() > 1 && t.starts_with(['O', 'o']) && t[1..].chars().all(|c| c.is_ascii_digit()) {
        return Ok(Place::Object(t[1..].parse().map_err(|_| format!("bad object id {t}"))?));
    }
    Ok(Place::Room(t.to_string()))
}

pub fn parse_confirm(args: &Value) -> Result<ConfirmArgs, String> {
    let a = obj_args(args)?;
    let id = parse_id(&a["id"]).ok_or("id is required: an object id from search_objects, e.g. O27")?;
    let name = opt_str(&a, "name").ok_or("name is required: what the object really is, e.g. radio")?;
    let source = opt_str(&a, "source").map(|s| s.to_lowercase().replace(['-', ' '], "_")).unwrap_or_default();
    if !SOURCES.contains(&source.as_str()) {
        return Err(format!("source must be user or close_look (got '{source}')"));
    }
    Ok(ConfirmArgs { id, name, source })
}

pub fn error_obs(tool: &str, msg: &str) -> Value {
    json!({"status": "error", "message": msg, "hint": format!("fix the arguments and call {tool} again")})
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// 가장 긴 변으로 크기 낱말(색인의 색·재질 낱말 뒤에 붙임)
pub fn size_word(extent: &[f64; 3]) -> &'static str {
    let m = extent.iter().cloned().fold(0.0, f64::max);
    match m {
        x if x < 0.1 => "tiny",
        x if x < 0.3 => "small",
        x if x < 0.8 => "medium",
        x if x < 1.5 => "large",
        _ => "very large",
    }
}

fn r3v(v: &[f64; 3]) -> Value {
    json!([r2(v[0]), r2(v[1]), r2(v[2])])
}

/// 자리 칸(cm 로 반올림): `pos`·`size`(map, m), `pos_sd`(objprob 있으면), `rel`(지금 로봇 기준: x 앞, y 왼쪽, z 높이, 수평 거리, 방위 °).
/// LLM 은 숫자로 추론만 하고 계획에는 id 로 말한다 — map 좌표는 VLA 에 넘기지 않는다(plan.md 3.3).
pub fn geo(m: &mut serde_json::Map<String, Value>, o: &ObjInfo, pose: Option<[f64; 3]>) {
    m.insert("pos".into(), r3v(&o.pos));
    m.insert("size".into(), r3v(&o.extent));
    if let Some(sd) = &o.pos_sd {
        m.insert("pos_sd".into(), r3v(sd));
    }
    if let Some(p) = pose {
        let (s, c) = p[2].sin_cos();
        let (dx, dy) = (o.pos[0] - p[0], o.pos[1] - p[1]);
        let (x, y) = (c * dx + s * dy, -s * dx + c * dy);
        m.insert("rel".into(), json!({"x": r2(x), "y": r2(y), "z": r2(o.pos[2]), "dist_m": r2(x.hypot(y)), "bearing_deg": y.atan2(x).to_degrees().round()}));
    }
}

/// 시간 거르기(max_age_s · seen_after_s)
pub fn time_ok(o: &ObjInfo, now: f64, max_age_s: Option<f64>, seen_after_s: Option<f64>) -> bool {
    max_age_s.is_none_or(|a| now - o.last_seen <= a + 1e-9) && seen_after_s.is_none_or(|t| o.last_seen >= t - 1e-9)
}

/// 색인 찾기 결과(`sgs_search_json`) + 기억 → LLM 이 받는 결과. `name_of` = 색인이 보는 물체 이름(기준물 이름용).
pub fn format_search(core: &Value, mem: &dyn Memory, a: &SearchArgs, name_of: &dyn Fn(u32) -> Option<String>) -> Value {
    let rooms: Option<Vec<i64>> = a.room.as_ref().map(|r| mem.find_rooms(r));
    let pose = mem.pose();
    let mut out = vec![];
    let mut kept: Vec<(&Value, &ObjInfo)> = vec![];
    for h in core["hits"].as_array().into_iter().flatten() {
        let Some(o) = h["id"].as_u64().and_then(|id| mem.get(id as u32)) else { continue };
        if let Some(rs) = &rooms {
            if !o.room.is_some_and(|r| rs.contains(&r)) {
                continue;
            }
        }
        if a.state.as_ref().is_some_and(|s| *s != o.state) {
            continue;
        }
        if !time_ok(o, mem.now(), a.max_age_s, a.seen_after_s) {
            continue;
        }
        kept.push((h, o));
        if kept.len() >= a.k {
            break;
        }
    }
    for (h, o) in &kept {
        let name = h["name"].as_str().unwrap_or(&o.name).to_string();
        let mut m = serde_json::Map::new();
        m.insert("id".into(), json!(o.key()));
        m.insert("name".into(), json!(name));
        m.insert("name_p".into(), json!(r2(h["name_p"].as_f64().unwrap_or(0.0))));
        let alt: Vec<Value> = h["alt"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|x| x[1].as_f64().unwrap_or(0.0) >= 0.03)
            .take(3)
            .map(|x| json!({"name": x[0], "p": r2(x[1].as_f64().unwrap_or(0.0))}))
            .collect();
        if !alt.is_empty() {
            m.insert("alt".into(), json!(alt));
        }
        let mt = h["match_type"].as_str().unwrap_or("name");
        m.insert("match_type".into(), json!(mt));
        if h["unindexed"].as_bool() == Some(true) {
            // 실시간 지도에 막 생겨 색인(저장 주기)에 아직 없음: 이름은 검출 이름 그대로, 확률 모름
            m.insert("unindexed".into(), json!(true));
        }
        if !o.name.is_empty() && !o.name.eq_ignore_ascii_case(&name) {
            m.insert("registered".into(), json!(o.name));
        }
        if mt == "appearance" {
            m.insert("p_query".into(), json!(r2(h["p_query"].as_f64().unwrap_or(0.0))));
            m.insert("p_registered".into(), json!(r2(h["p_registered"].as_f64().unwrap_or(0.0))));
            if let Some(l) = h["like"].as_u64() {
                m.insert("looks_like".into(), json!(format!("O{l}")));
            }
        }
        let mut attrs: Vec<Value> = h["attrs"].as_array().cloned().unwrap_or_default();
        attrs.push(json!(size_word(&o.extent)));
        m.insert("attrs".into(), json!(attrs));
        if let Some(r) = o.room.and_then(|r| mem.room_name(r)) {
            m.insert("room".into(), json!(r));
        }
        if let Some(l) = landmark::nearest_fixed(mem, o) {
            let ln = name_of(l.id).or_else(|| mem.get(l.id).map(|x| x.name.clone())).unwrap_or_default();
            m.insert("landmark".into(), json!({"id": format!("O{}", l.id), "name": ln, "dist_m": r2(l.dist_m), "dz_m": r2(l.dz_m)}));
        }
        m.insert("state".into(), json!(o.state));
        m.insert("last_seen_ago_s".into(), json!((mem.now() - o.last_seen).max(0.0).round()));
        geo(&mut m, o, pose);
        m.insert("match".into(), json!(r2(h["match"].as_f64().unwrap_or(0.0))));
        out.push(Value::Object(m));
    }
    let mut res = serde_json::Map::new();
    res.insert("query".into(), json!(a.query));
    res.insert("searched".into(), json!(if core["step2"].as_bool().unwrap_or(false) { "name+appearance" } else { "name" }));
    res.insert("now_s".into(), json!(mem.now().round()));
    if let Some((ask, hint)) = advice(&a.query, &kept, mem) {
        if let Some(q) = ask {
            res.insert("ask_user".into(), json!(q));
        }
        res.insert("hint".into(), json!(hint));
    }
    res.insert("matches".into(), json!(out));
    Value::Object(res)
}

/// 언제 물을지: (ask_user, hint)
fn advice(q: &str, kept: &[(&Value, &ObjInfo)], mem: &dyn Memory) -> Option<(Option<String>, String)> {
    let Some((h, o)) = kept.first() else {
        return Some((
            Some(format!("nothing in memory matches '{q}' by name or appearance")),
            "ask the user where it is, or explore to find it".into(),
        ));
    };
    let m0 = h["match"].as_f64().unwrap_or(0.0);
    if h["match_type"] == "appearance" {
        let reg = if o.name.is_empty() { "no name".to_string() } else { format!("'{}'", o.name) };
        return Some((
            Some(format!(
                "nothing is registered as '{q}'; {} is registered as {reg} but looks like '{q}' (p {:.2} vs {:.2})",
                o.key(),
                h["p_query"].as_f64().unwrap_or(0.0),
                h["p_registered"].as_f64().unwrap_or(0.0)
            )),
            format!("ask the user, or pass {} as a hint and verify by a close look; then call confirm_object", o.key()),
        ));
    }
    if let Some((h2, o2)) = kept.get(1) {
        let m1 = h2["match"].as_f64().unwrap_or(0.0);
        let apart = (o.pos[0] - o2.pos[0]).hypot(o.pos[1] - o2.pos[1]);
        if m0 - m1 < TIE && apart > 1.0 {
            let room = |x: &ObjInfo| x.room.and_then(|r| mem.room_name(r)).map(|r| format!(" in {r}")).unwrap_or_default();
            return Some((
                Some(format!("{} candidates score alike: {}{} and {}{} ({apart:.1} m apart)", kept.len(), o.key(), room(o), o2.key(), room(o2))),
                "ask which one, or use the nearer one".into(),
            ));
        }
    }
    if m0 < WEAK {
        return Some((Some(format!("weak match for '{q}' ({} p {m0:.2})", o.key())), "confirm with the user or look closely first".into()));
    }
    if o.state == "gone" {
        return Some((None, format!("{} was seen there but is gone now", o.key())));
    }
    None
}

/// 방 이름 목록(오류 글용)
fn room_list(mem: &dyn Memory) -> String {
    let rs: std::collections::BTreeSet<i64> = mem.objects().iter().filter_map(|o| o.room).collect();
    rs.into_iter().filter_map(|r| mem.room_name(r).map(|n| format!("{n} (R{r})"))).collect::<Vec<_>>().join(", ")
}

/// list_place 결과(순수 함수). `info` = 색인이 보는 물체 이름·확률·속성(`sgs_object_json`), 없으면 기억의 등록 이름.
/// 방: 그 방 물체(구조물 빼고), 로봇에서 가까운 순. 물체 id: 그 상자에서 수평 1.5 m 안, 상자에서 가까운 순(dist_m·dz_m).
/// 관계말(on/in/next to)은 만들지 않는다 — 숫자만.
pub fn format_list(place: &Place, mem: &dyn Memory, info: &dyn Fn(u32) -> Option<Value>) -> Value {
    let pose = mem.pose();
    let robot_d = |o: &ObjInfo| pose.map_or(o.id as f64, |p| (o.pos[0] - p[0]).hypot(o.pos[1] - p[1]));
    let mut head = serde_json::Map::new();
    let mut cand: Vec<(&ObjInfo, f64, Option<f64>)> = vec![];
    match place {
        Place::Room(r) => {
            let rs = mem.find_rooms(r);
            if rs.is_empty() {
                return error_obs(LIST, &format!("unknown room '{r}'; rooms: {}", room_list(mem)));
            }
            head.insert("room".into(), json!(mem.room_name(rs[0]).unwrap_or_default()));
            head.insert("id".into(), json!(format!("R{}", rs[0])));
            for o in mem.objects().iter().filter(|o| !o.structural || o.movable).filter(|o| o.room.is_some_and(|x| rs.contains(&x))) {
                cand.push((o, robot_d(o), None));
            }
        }
        Place::Object(id) => {
            let Some(f) = mem.get(*id) else { return error_obs(LIST, &format!("unknown object id O{id}")) };
            head.insert("id".into(), json!(f.key()));
            head.insert("name".into(), json!(info(*id).and_then(|v| v["name"].as_str().map(String::from)).unwrap_or_else(|| f.name.clone())));
            let mut hm = serde_json::Map::new();
            geo(&mut hm, f, pose);
            head.extend(hm);
            for o in mem.objects().iter().filter(|o| o.id != *id && (!o.structural || o.movable)) {
                let d = f.xy_dist_to_box(o.pos[0], o.pos[1]);
                if d <= NEAR_M {
                    cand.push((o, d, Some(o.pos[2] - f.top_z())));
                }
            }
        }
    }
    // 옮길 수 있는 것 먼저, 그다음 거리
    cand.sort_by(|a, b| (!a.0.movable).cmp(&!b.0.movable).then(a.1.total_cmp(&b.1)));
    let n = cand.len();
    let rows: Vec<Value> = cand
        .iter()
        .take(PLACE_MAX)
        .map(|(o, d, dz)| {
            let mut m = serde_json::Map::new();
            m.insert("id".into(), json!(o.key()));
            let iv = info(o.id);
            let name = iv.as_ref().and_then(|v| v["name"].as_str().map(String::from)).unwrap_or_else(|| o.name.clone());
            if let Some(p) = iv.as_ref().and_then(|v| v["name_p"].as_f64()) {
                m.insert("name_p".into(), json!(r2(p)));
            }
            if !o.name.is_empty() && !o.name.eq_ignore_ascii_case(&name) {
                m.insert("registered".into(), json!(o.name));
            }
            m.insert("name".into(), json!(name));
            let mut attrs: Vec<Value> = iv.as_ref().and_then(|v| v["attrs"].as_array().cloned()).unwrap_or_default();
            attrs.push(json!(size_word(&o.extent)));
            m.insert("attrs".into(), json!(attrs));
            if !o.movable {
                m.insert("fixed".into(), json!(true));
            }
            if let Some(z) = dz {
                m.insert("dist_m".into(), json!(r2(*d)));
                m.insert("dz_m".into(), json!(r2(*z)));
            }
            m.insert("state".into(), json!(o.state));
            m.insert("last_seen_ago_s".into(), json!((mem.now() - o.last_seen).max(0.0).round()));
            geo(&mut m, o, pose);
            Value::Object(m)
        })
        .collect();
    let mut res = serde_json::Map::new();
    res.insert("place".into(), Value::Object(head));
    res.insert("now_s".into(), json!(mem.now().round()));
    res.insert("n".into(), json!(n));
    if n > PLACE_MAX {
        res.insert("hint".into(), json!(format!("{} more not shown; narrow with search_objects (room, max_age_s)", n - PLACE_MAX)));
    } else if n == 0 {
        res.insert("hint".into(), json!("nothing remembered there; explore or ask the user"));
    }
    res.insert("objects".into(), json!(rows));
    Value::Object(res)
}

/// 도구 세 개의 실행기: 공용 색인 + 기억(오프라인 view.json 또는 실시간 scenemap 스냅숏).
pub struct ObjectSearch {
    pub index: sys::Index,
    pub mem: Box<dyn MemSource>,
    pub mem_dir: PathBuf,
    last_query: String,
    /// 색인이 마지막으로 읽은 view.json 시각(바뀌면 sgs_reload)
    index_mtime: Option<std::time::SystemTime>,
}

impl ObjectSearch {
    /// 오프라인: 기억 폴더의 view.json
    pub fn open(mem_dir: &Path, paths: &sys::Paths) -> Result<ObjectSearch, String> {
        let mem = ViewJson::load(mem_dir)?;
        Self::with_mem(Box::new(mem), mem_dir, paths)
    }

    /// 실시간: scenemap 스냅숏 원천 + 지도 저장 폴더(색인 벡터·확인 기록·pos_sd). 폴더에 view.json 이 있어야 한다(sm_save_dsg 한 번 뒤)
    pub fn open_live(api: Box<dyn live::SnapApi>, mem_dir: &Path, paths: &sys::Paths) -> Result<ObjectSearch, String> {
        let mem = live::LiveMem::new(api, Some(mem_dir.join("view.json")))?;
        Self::with_mem(Box::new(mem), mem_dir, paths)
    }

    pub fn with_mem(mem: Box<dyn MemSource>, mem_dir: &Path, paths: &sys::Paths) -> Result<ObjectSearch, String> {
        let index_mtime = view_mtime(mem_dir);
        let index = sys::Index::open(&mem_dir.to_string_lossy(), paths)?;
        Ok(ObjectSearch { index, mem, mem_dir: mem_dir.to_path_buf(), last_query: String::new(), index_mtime })
    }

    /// 기억을 새로 읽고(실시간은 새 스냅숏), view.json 이 바뀌었으면 색인도 다시 읽음
    fn refresh(&mut self) -> Result<(), String> {
        self.mem.refresh()?;
        let m = view_mtime(&self.mem_dir);
        if m.is_some() && m != self.index_mtime {
            self.index.reload()?;
            self.index_mtime = m;
        }
        Ok(())
    }

    fn info(&self, id: u32) -> Option<Value> {
        self.index.object(id).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()).filter(|v| v.get("status").is_none())
    }

    pub fn list_place(&mut self, p: &Place) -> Value {
        if let Err(e) = self.refresh() {
            return json!({"status": "error", "message": e});
        }
        format_list(p, self.mem.mem(), &|id| self.info(id))
    }

    /// 도구 호출 하나 → 결과 JSON(실패도 관찰값)
    pub fn run_tool(&mut self, name: &str, args: &Value) -> Value {
        match name {
            SEARCH => match parse_search(args) {
                Ok(a) => self.search(&a),
                Err(e) => error_obs(SEARCH, &e),
            },
            CONFIRM => match parse_confirm(args) {
                Ok(a) => self.confirm(&a),
                Err(e) => error_obs(CONFIRM, &e),
            },
            LIST => match parse_list(args) {
                Ok(p) => self.list_place(&p),
                Err(e) => error_obs(LIST, &e),
            },
            _ => error_obs(name, &format!("unknown tool '{name}'; tools: {SEARCH}, {CONFIRM}, {LIST}")),
        }
    }

    pub fn search(&mut self, a: &SearchArgs) -> Value {
        if let Err(e) = self.refresh() {
            return json!({"status": "error", "message": e});
        }
        let mem = self.mem.mem();
        if let Some(r) = &a.room {
            if mem.find_rooms(r).is_empty() {
                return error_obs(SEARCH, &format!("unknown room '{r}'; rooms: {}", room_list(mem)));
            }
        }
        self.last_query = a.query.clone();
        // 필터는 여기서 하므로 색인에서는 기준을 넘는 것 전부(k = 0)
        let mut core = match self.index.search(&a.query, 0, false).and_then(|s| serde_json::from_str::<Value>(&s).map_err(|e| e.to_string())) {
            Ok(v) => v,
            Err(e) => return json!({"status": "error", "message": e}),
        };
        // 실시간: 지도에 막 생겨 색인(지도 저장 주기)에 아직 없는 물체 — 검출 이름의 라벨이 질의 라벨과 같으면 뒤에 붙임
        if self.mem.kind() == "live" {
            let ql = self.index.label_of(&a.query);
            let extra: Vec<Value> = self
                .mem
                .mem()
                .objects()
                .iter()
                .filter(|o| !o.name.is_empty() && self.index.object(o.id).ok().is_none_or(|s| s.contains("\"error\"")))
                .filter(|o| self.index.label_of(&o.name) == ql)
                .map(|o| json!({"id": o.id, "name": o.name, "name_p": 0.5, "match_type": "name", "match": 0.5, "unindexed": true}))
                .collect();
            if let Some(h) = core["hits"].as_array_mut() {
                h.extend(extra);
            }
        }
        let idx = &self.index;
        let name_of = |id: u32| idx.object(id).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()).and_then(|v| v["name"].as_str().map(String::from));
        format_search(&core, self.mem.mem(), a, &name_of)
    }

    pub fn confirm(&mut self, a: &ConfirmArgs) -> Value {
        if let Err(e) = self.refresh() {
            return json!({"status": "error", "message": e});
        }
        if self.mem.mem().get(a.id).is_none() {
            return error_obs(CONFIRM, &format!("unknown object id O{}", a.id));
        }
        // 실시간이면 지도(scenemap 확률 모드(objprob))에도 이름 관측을 넣는다: 라벨 = 색인 라벨 표 영어 이름, log 우도비 = 색인과 같은 값
        let (ul, ll) = sys::confirm_lrs();
        let lr = if a.source == "user" { ul } else { ll };
        let label = self.index.label_of(&a.name);
        let map = self.mem.observe_name(a.id, &label, lr.max(1.0).ln()).map(|rc| match rc {
            0 => "applied",
            -2 => "unknown_label",
            -3 => "not_objprob",
            _ => "error",
        });
        let extra = map.map_or(String::new(), |m| json!({"map": m, "map_label": label}).to_string());
        match self.index.confirm_ex(a.id, &a.name, &a.source, &self.last_query, &extra).and_then(|s| serde_json::from_str::<Value>(&s).map_err(|e| e.to_string())) {
            Ok(mut v) => {
                if v["status"] == "ok" {
                    v["id"] = json!(format!("O{}", a.id));
                    if let Some(l) = v.as_object_mut() {
                        l.remove("logged");
                    }
                    if let Some(m) = map {
                        v["map"] = json!(m);
                    }
                }
                v
            }
            Err(e) => json!({"status": "error", "message": e}),
        }
    }
}

fn view_mtime(dir: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(dir.join("view.json")).and_then(|m| m.modified()).ok()
}

#[cfg(test)]
mod tests;

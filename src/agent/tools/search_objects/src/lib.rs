//! `search_objects` · `confirm_object` — LLM 이 물체 기억에서 물건을 찾고, 확인되면 이름을 고치는 도구 두 개.
//!
//! LLM(Qwen3.5-9B, KAU API)은 글만 주고받으므로 벡터는 도구 **안에서만** 쓴다. 찾기·이름 확률·속성 낱말은 공용 물체 색인
//! (behavior-2026 `src/scene_graph/clip` 의 `sgsearch.h`, C++ — RecallVLA 실행기도 자기 질의 벡터로 같은 색인을 씀)이 하고,
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
pub mod memview;
pub mod sys;

use memview::{Memory, ObjInfo, ViewJson};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const SEARCH: &str = "search_objects";
pub const CONFIRM: &str = "confirm_object";
pub const STATES: [&str; 4] = ["seen", "moved", "held", "gone"];
pub const SOURCES: [&str; 2] = ["user", "close_look"];
/// `near` 필터 반경(m, 기준 물체 상자까지 수평 거리)
pub const NEAR_M: f64 = 1.5;
/// 두 후보 점수가 이만큼 안이고 1 m 넘게 떨어져 있으면 어느 것인지 물어야 함
const TIE: f64 = 0.1;
/// 이름 후보 점수가 이보다 낮으면 약함
const WEAK: f64 = 0.5;

const SEARCH_DESC: &str = "Find objects in the robot's object memory. Searches registered names (synonyms, broader terms) and, if none or weak, \
automatically re-searches by appearance ignoring names (an object may be registered under a wrong name). \
query: a short noun, English preferred, Korean works (e.g. \"radio\", \"라디오\", \"red mug\"). \
Each match: id, name with probability, alternative names, match_type name|appearance, registered (the stored name, if different), \
attrs, room, landmark (nearest fixed furniture: id, name, dist_m, dz_m = height above its top), state, last_seen_ago_s, dist_m, match. \
If ask_user is present, ask the user (or verify) before acting; use confirm_object once the identity is confirmed.";

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
            "near": {"type": "string", "description": "optional object id: only objects within 1.5 m of it"}
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

pub fn definitions() -> Vec<Value> {
    vec![search_definition(), confirm_definition()]
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchArgs {
    pub query: String,
    pub k: usize,
    pub room: Option<String>,
    pub state: Option<String>,
    pub near: Option<u32>,
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
    let near = match &a["near"] {
        Value::Null => None,
        v => Some(parse_id(v).ok_or_else(|| format!("near must be an object id like O12 (got {v})"))?),
    };
    Ok(SearchArgs { query, k, room: opt_str(&a, "room"), state, near })
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

/// 색인 찾기 결과(`sgs_search_json`) + 기억 → LLM 이 받는 결과. `name_of` = 색인이 보는 물체 이름(기준물 이름용).
pub fn format_search(core: &Value, mem: &dyn Memory, a: &SearchArgs, name_of: &dyn Fn(u32) -> Option<String>) -> Value {
    let rooms: Option<Vec<i64>> = a.room.as_ref().map(|r| mem.find_rooms(r));
    let pose = mem.pose();
    let near = a.near.and_then(|id| mem.get(id).cloned());
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
        if let Some(n) = &near {
            if n.id == o.id || n.xy_dist_to_box(o.pos[0], o.pos[1]) > NEAR_M {
                continue;
            }
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
        if let Some(p) = pose {
            m.insert("dist_m".into(), json!(r2((o.pos[0] - p[0]).hypot(o.pos[1] - p[1]))));
        }
        m.insert("match".into(), json!(r2(h["match"].as_f64().unwrap_or(0.0))));
        out.push(Value::Object(m));
    }
    let mut res = serde_json::Map::new();
    res.insert("query".into(), json!(a.query));
    res.insert("searched".into(), json!(if core["step2"].as_bool().unwrap_or(false) { "name+appearance" } else { "name" }));
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

/// 도구 두 개의 실행기: 공용 색인 + 기억 보기.
pub struct ObjectSearch {
    pub index: sys::Index,
    pub mem: ViewJson,
    pub mem_dir: PathBuf,
    last_query: String,
}

impl ObjectSearch {
    pub fn open(mem_dir: &Path, paths: &sys::Paths) -> Result<ObjectSearch, String> {
        let mem = ViewJson::load(mem_dir)?;
        let index = sys::Index::open(&mem_dir.to_string_lossy(), paths)?;
        Ok(ObjectSearch { index, mem, mem_dir: mem_dir.to_path_buf(), last_query: String::new() })
    }

    /// view.json 이 바뀌었으면 기억·색인을 다시 읽음
    fn refresh(&mut self) -> Result<(), String> {
        if self.mem.refresh()? {
            self.index.reload()?;
        }
        Ok(())
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
            _ => error_obs(name, &format!("unknown tool '{name}'; tools: {SEARCH}, {CONFIRM}")),
        }
    }

    pub fn search(&mut self, a: &SearchArgs) -> Value {
        if let Err(e) = self.refresh() {
            return json!({"status": "error", "message": e});
        }
        if let Some(r) = &a.room {
            if self.mem.find_rooms(r).is_empty() {
                let names: Vec<String> = self.mem.objects().iter().filter_map(|o| o.room).collect::<std::collections::BTreeSet<_>>()
                    .into_iter().filter_map(|r| self.mem.room_name(r).map(|n| format!("{n} (R{r})"))).collect();
                return error_obs(SEARCH, &format!("unknown room '{r}'; rooms: {}", names.join(", ")));
            }
        }
        if let Some(n) = a.near {
            if self.mem.get(n).is_none() {
                return error_obs(SEARCH, &format!("unknown object id O{n} for near"));
            }
        }
        self.last_query = a.query.clone();
        // 필터는 여기서 하므로 색인에서는 기준을 넘는 것 전부(k = 0)
        let core = match self.index.search(&a.query, 0, false).and_then(|s| serde_json::from_str::<Value>(&s).map_err(|e| e.to_string())) {
            Ok(v) => v,
            Err(e) => return json!({"status": "error", "message": e}),
        };
        let idx = &self.index;
        let name_of = |id: u32| idx.object(id).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()).and_then(|v| v["name"].as_str().map(String::from));
        format_search(&core, &self.mem, a, &name_of)
    }

    pub fn confirm(&mut self, a: &ConfirmArgs) -> Value {
        if let Err(e) = self.refresh() {
            return json!({"status": "error", "message": e});
        }
        if self.mem.get(a.id).is_none() {
            return error_obs(CONFIRM, &format!("unknown object id O{}", a.id));
        }
        match self.index.confirm(a.id, &a.name, &a.source, &self.last_query).and_then(|s| serde_json::from_str::<Value>(&s).map_err(|e| e.to_string())) {
            Ok(mut v) => {
                if v["status"] == "ok" {
                    v["id"] = json!(format!("O{}", a.id));
                    if let Some(l) = v.as_object_mut() {
                        l.remove("logged");
                    }
                }
                v
            }
            Err(e) => json!({"status": "error", "message": e}),
        }
    }
}

#[cfg(test)]
mod tests;

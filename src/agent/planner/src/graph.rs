//! 씬그래프 조회와 에이전트 쪽 물체 기억.
//!
//! 물체 지도는 scenemap(2D SLAM + YOLOE 검출 물체 지도, C++/CUDA, 같은 프로세스 C ABI — docs/scenemap_설계.md)이 만든다.
//! 계획기는 [`SceneQuery`] 로 묻고, [`ScenemapGraph`] 가 그것을 [`SceneGraph`] 자리에 끼운다.
//! 좌표계는 이번 판 출발점 기준 오도메트리("map") — 중계기 [`crate::odom`] 과 같은 식이다.
//!
//! 그래프가 아직 없으면 [`StaticGraph`](JSON 파일/가짜 세계)로 대신한다.
//!
//! [`ObjectMemory`] 는 에이전트가 그래프 위에 얹는 기억이다: 처음 본 자리(→ `back`), 다룬 적 있는지(→ `the other`).

use crate::odom::Pose;
use crate::util::norm_category;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Node {
    pub id: String,
    /// 종류 이름(표시용). scenemap 은 검출기 프롬프트 표 이름.
    pub label: String,
    #[serde(default)]
    pub score: f64,
    /// 출발점 기준 좌표(m)
    pub center: [f64; 3],
    #[serde(default)]
    pub extent: [f64; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub room: Option<String>,
    /// 받침 물체 id(알면)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on: Option<String>,
    #[serde(default)]
    pub num_observations: u32,
    /// 처음 등록 때 중심(scenemap first_position) — "back"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original: Option<[f64; 3]>,
    /// 그래프가 본 다룸 여부(scenemap: 표시 || 옮겨짐 || 들고 있음) — "the other"
    #[serde(default)]
    pub handled: bool,
    /// 바닥·벽 같은 구조물
    #[serde(default)]
    pub structural: bool,
}

impl Node {
    /// 물체 JSON → Node. scenemap 질의 결과를 JSON 으로 저장한 파일(`{"objects": [...]}`)과 외부 물체 표 JSON 파일을
    /// 읽는다(필드: id, name|category, score|category_score, structural, position, extent, room, handled, original_position,
    /// n_obs|num_observations).
    pub fn from_json(v: &serde_json::Value) -> Option<Node> {
        let arr3 = |k: &str| -> Option<[f64; 3]> {
            let a = v.get(k)?.as_array()?;
            Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?])
        };
        let id = match v.get("id")? {
            serde_json::Value::String(s) => s.clone(),
            x => x.to_string(),
        };
        Some(Node {
            label: v.get("name").or_else(|| v.get("category")).and_then(|c| c.as_str()).unwrap_or("object").to_string(),
            score: v.get("score").or_else(|| v.get("category_score")).and_then(|s| s.as_f64()).unwrap_or(0.0),
            center: arr3("position")?,
            extent: arr3("extent").unwrap_or_default(),
            room: v.get("room").and_then(|r| r.as_str()).map(|s| s.to_string()),
            on: None,
            num_observations: v.get("n_obs").or_else(|| v.get("num_observations")).and_then(|n| n.as_u64()).unwrap_or(0) as u32,
            original: arr3("original_position"),
            handled: v.get("handled").and_then(|h| h.as_bool()).unwrap_or(false),
            structural: v.get("structural").and_then(|h| h.as_bool()).unwrap_or(false),
            id,
        })
    }

    /// scenemap 물체 → Node
    pub fn from_scene(o: &SceneObject) -> Node {
        Node {
            id: o.id.to_string(),
            label: o.name.clone(),
            score: o.score,
            center: o.position,
            extent: o.extent,
            room: o.room.clone(),
            on: None,
            num_observations: o.n_obs,
            original: Some(o.first_position),
            handled: o.handled || o.state == ObjState::Moved || o.state == ObjState::Held,
            structural: o.structural,
        }
    }
}

pub trait SceneGraph: Send {
    /// 문장으로 물체 찾기(점수 높은 순)
    fn query(&mut self, text: &str, top_k: usize) -> Result<Vec<Node>, String>;
    /// 지금 그래프의 모든 물체
    fn all(&mut self) -> Result<Vec<Node>, String>;
    /// 새 판: 이 과제에서 쓸 이름표 목록 알려 주기(그래프 쪽이 쓰면)
    fn reset(&mut self, _labels: &[String]) -> Result<(), String> {
        Ok(())
    }
    /// 에이전트가 이 물체를 다뤘다고 알린다(scenemap mark_handled)
    fn mark_handled(&mut self, _id: &str) -> Result<(), String> {
        Ok(())
    }
    /// 한 점 둘레 물체(map 좌표)
    fn near(&mut self, p: [f64; 3], radius: f64) -> Result<Vec<Node>, String> {
        Ok(self.all()?.into_iter().filter(|n| (0..3).map(|i| (n.center[i] - p[i]).powi(2)).sum::<f64>().sqrt() <= radius).collect())
    }
    fn describe(&self) -> String;
}

// ---------------- scenemap(같은 프로세스) ----------------

/// scenemap 물체 상태(docs/scenemap_설계.md 3.2): 보임·사라짐·옮겨짐·들고 있음.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ObjState {
    #[default]
    Seen,
    Gone,
    Moved,
    Held,
}

/// scenemap 물체 한 개(3.3 `objects` 표). 좌표는 map(판 시작 = 원점).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SceneObject {
    pub id: u32,
    /// 프롬프트 표 이름(BDDL 이름을 정규화한 것)
    pub name: String,
    pub score: f64,
    pub position: [f64; 3],
    pub extent: [f64; 3],
    /// 처음 자리("back")
    pub first_position: [f64; 3],
    pub n_obs: u32,
    /// 마지막으로 본 시뮬 시각 [s]
    pub last_seen: f64,
    pub state: ObjState,
    /// 계획기가 다뤘다고 표시(mark_handled)
    pub handled: bool,
    pub structural: bool,
    pub room: Option<String>,
}

/// 2D 점유 격자(3.1 출력). cells: 행 우선, -1 모름, 0~100 점유 확률.
#[derive(Debug, Clone, Default)]
pub struct Grid2 {
    pub resolution: f64,
    pub origin: [f64; 2],
    pub width: u32,
    pub height: u32,
    pub cells: Vec<i8>,
}

/// 계획기 ↔ scenemap 질의(docs/scenemap_설계.md 3.3·4.3). 같은 프로세스에서 부른다(스냅숏이라 계획기 스레드가 안 막힘).
/// 구현: simlink 의 C ABI 래퍼(libscenemap 또는 가짜 구현), 시험용 [`MemScene`].
pub trait SceneQuery: Send + Sync {
    fn objects(&self) -> Result<Vec<SceneObject>, String>;
    fn object(&self, id: u32) -> Result<Option<SceneObject>, String> {
        Ok(self.objects()?.into_iter().find(|o| o.id == id))
    }
    /// 한 점(map) 둘레 r 안, 가까운 순
    fn near(&self, p: [f64; 3], r: f64) -> Result<Vec<SceneObject>, String>;
    /// 이름으로(점수 높은 순)
    fn find(&self, name: &str) -> Result<Vec<SceneObject>, String>;
    fn map(&self) -> Result<Grid2, String>;
    /// 격자 위 최단 경로 길이 [m], 못 가면 None
    fn reachable(&self, from: [f64; 2], to: [f64; 2]) -> Result<Option<f64>, String>;
    /// 지금 로봇 자세(map)와 그 시뮬 시각 [s]
    fn pose(&self) -> Result<(f64, crate::odom::Pose), String>;
    fn mark_handled(&self, id: u32) -> Result<(), String>;
    /// 새 판의 이름표(프롬프트 표) — 검출기·scenemap 이 같은 순서를 쓴다
    fn set_labels(&self, labels: &[String]) -> Result<(), String>;
    fn describe(&self) -> String;
}

/// 계획기 [`SceneGraph`] 자리를 scenemap 질의로 채운다(계획기 코드는 그대로).
pub struct ScenemapGraph {
    pub q: std::sync::Arc<dyn SceneQuery>,
}

fn nodes_of(v: Vec<SceneObject>) -> Vec<Node> {
    v.iter().map(Node::from_scene).collect()
}

impl SceneGraph for ScenemapGraph {
    fn query(&mut self, text: &str, top_k: usize) -> Result<Vec<Node>, String> {
        let mut v = nodes_of(self.q.find(text)?);
        v.retain(|n| !n.structural);
        if top_k > 0 {
            v.truncate(top_k);
        }
        Ok(v)
    }
    fn all(&mut self) -> Result<Vec<Node>, String> {
        Ok(nodes_of(self.q.objects()?))
    }
    fn reset(&mut self, labels: &[String]) -> Result<(), String> {
        self.q.set_labels(labels)
    }
    fn mark_handled(&mut self, id: &str) -> Result<(), String> {
        let n: u32 = id.parse().map_err(|_| format!("scenemap id 는 숫자: {id}"))?;
        self.q.mark_handled(n)
    }
    fn near(&mut self, p: [f64; 3], radius: f64) -> Result<Vec<Node>, String> {
        Ok(nodes_of(self.q.near(p, radius)?))
    }
    fn describe(&self) -> String {
        format!("scenemap:{}", self.q.describe())
    }
}

/// 메모리 안 scenemap 흉내(시험). 이름 찾기는 [`text_score`].
#[derive(Default)]
pub struct MemScene {
    /// (물체, 로봇 자세, 이름표)
    pub inner: std::sync::Mutex<(Vec<SceneObject>, crate::odom::Pose, Vec<String>)>,
}

impl SceneQuery for MemScene {
    fn objects(&self) -> Result<Vec<SceneObject>, String> {
        Ok(self.inner.lock().unwrap().0.clone())
    }
    fn near(&self, p: [f64; 3], r: f64) -> Result<Vec<SceneObject>, String> {
        let d = |o: &SceneObject| (0..3).map(|i| (o.position[i] - p[i]).powi(2)).sum::<f64>().sqrt();
        let mut v: Vec<SceneObject> = self.objects()?.into_iter().filter(|o| d(o) <= r).collect();
        v.sort_by(|a, b| d(a).partial_cmp(&d(b)).unwrap());
        Ok(v)
    }
    fn find(&self, name: &str) -> Result<Vec<SceneObject>, String> {
        let mut v: Vec<(f64, SceneObject)> =
            self.objects()?.into_iter().map(|o| (text_score(name, &o.name), o)).filter(|x| x.0 > 0.0).collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.id.cmp(&b.1.id)));
        Ok(v.into_iter().map(|(s, o)| SceneObject { score: s, ..o }).collect())
    }
    fn map(&self) -> Result<Grid2, String> {
        Ok(Grid2::default())
    }
    fn reachable(&self, from: [f64; 2], to: [f64; 2]) -> Result<Option<f64>, String> {
        Ok(Some((from[0] - to[0]).hypot(from[1] - to[1])))
    }
    fn pose(&self) -> Result<(f64, crate::odom::Pose), String> {
        Ok((0.0, self.inner.lock().unwrap().1))
    }
    fn mark_handled(&self, id: u32) -> Result<(), String> {
        let mut g = self.inner.lock().unwrap();
        g.0.iter_mut().filter(|o| o.id == id).for_each(|o| o.handled = true);
        Ok(())
    }
    fn set_labels(&self, labels: &[String]) -> Result<(), String> {
        self.inner.lock().unwrap().2 = labels.to_vec();
        Ok(())
    }
    fn describe(&self) -> String {
        "mem".into()
    }
}

fn words(s: &str) -> Vec<String> {
    norm_category(s).split_whitespace().map(|w| w.trim_end_matches('s').to_string()).filter(|w| !w.is_empty()).collect()
}

/// 이름표 문자열 유사도(0~1): 단어 겹침 + 포함 관계. CLIP 이 없을 때의 대체.
pub fn text_score(query: &str, label: &str) -> f64 {
    let q = words(query);
    let l = words(label);
    if q.is_empty() || l.is_empty() {
        return 0.0;
    }
    let inter = q.iter().filter(|w| l.contains(w)).count() as f64;
    let jac = inter / (q.len() + l.len()) as f64 * 2.0;
    let contains = if norm_category(label).contains(&norm_category(query)) || norm_category(query).contains(&norm_category(label)) {
        0.3
    } else {
        0.0
    };
    (jac + contains).min(1.0)
}

/// 고정 목록 그래프(가짜 세계·JSON 파일·재생용).
#[derive(Debug, Clone, Default)]
pub struct StaticGraph {
    pub nodes: Vec<Node>,
    pub name: String,
}

impl StaticGraph {
    pub fn load(path: &str) -> Result<StaticGraph, String> {
        let t = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let v: serde_json::Value = serde_json::from_str(&t).map_err(|e| format!("{path}: {e}"))?;
        // 물체 표 JSON({"objects": [...]}: scenemap 질의 저장·외부 물체 표)도 그대로 읽는다
        if let Some(objs) = v.get("objects").and_then(|o| o.as_array()) {
            return Ok(StaticGraph { nodes: objs.iter().filter_map(Node::from_json).collect(), name: format!("file:{path}") });
        }
        let nodes: Vec<Node> = serde_json::from_value(v.get("nodes").cloned().unwrap_or(v)).map_err(|e| format!("{path}: {e}"))?;
        Ok(StaticGraph { nodes, name: format!("file:{path}") })
    }
}

impl SceneGraph for StaticGraph {
    fn query(&mut self, text: &str, top_k: usize) -> Result<Vec<Node>, String> {
        let mut v: Vec<Node> = self
            .nodes
            .iter()
            .filter_map(|n| {
                let s = text_score(text, &n.label).max(if n.id == text { 1.0 } else { 0.0 });
                (s > 0.0).then(|| Node { score: s, ..n.clone() })
            })
            .collect();
        v.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().then(a.id.cmp(&b.id)));
        v.truncate(top_k);
        Ok(v)
    }
    fn all(&mut self) -> Result<Vec<Node>, String> {
        Ok(self.nodes.clone())
    }
    fn describe(&self) -> String {
        if self.name.is_empty() {
            "static".into()
        } else {
            self.name.clone()
        }
    }
}

/// 여러 스레드가 같이 보는 그래프(가짜 세계가 매 경계 전에 갱신).
#[derive(Clone, Default)]
pub struct SharedGraph(pub std::sync::Arc<std::sync::Mutex<StaticGraph>>);

impl SceneGraph for SharedGraph {
    fn query(&mut self, text: &str, top_k: usize) -> Result<Vec<Node>, String> {
        self.0.lock().unwrap().query(text, top_k)
    }
    fn all(&mut self) -> Result<Vec<Node>, String> {
        self.0.lock().unwrap().all()
    }
    fn describe(&self) -> String {
        "shared".into()
    }
}

/// 그래프 없음(탐색만 가능).
pub struct NullGraph;
impl SceneGraph for NullGraph {
    fn query(&mut self, _t: &str, _k: usize) -> Result<Vec<Node>, String> {
        Ok(vec![])
    }
    fn all(&mut self) -> Result<Vec<Node>, String> {
        Ok(vec![])
    }
    fn describe(&self) -> String {
        "none".into()
    }
}

/// 바깥 그래프 서비스(HTTP JSON) 계약(도구·시험용):
/// - `POST {base}/reset`  `{"labels": ["radio", "coffee table", ...]}` → 200. 새 판, CLIP 이름표 후보.
/// - `POST {base}/query`  `{"text": "radio", "top_k": 5}` → `{"nodes": [Node...]}` (score = CLIP cosine)
/// - `GET  {base}/objects` → `{"nodes": [Node...]}` (label = 후보 중 CLIP 최고점)
///   좌표는 중계기 오도메트리와 같은 "map" 좌표(m).
pub struct HttpGraph {
    pub base: String,
    pub timeout: std::time::Duration,
}

impl HttpGraph {
    pub fn new(base: &str, timeout_s: u64) -> HttpGraph {
        HttpGraph { base: base.trim_end_matches('/').to_string(), timeout: std::time::Duration::from_secs(timeout_s) }
    }
    fn nodes(r: crate::http::Resp) -> Result<Vec<Node>, String> {
        if r.status != 200 {
            return Err(format!("HTTP {}: {}", r.status, r.text().chars().take(200).collect::<String>()));
        }
        let v = r.json()?;
        serde_json::from_value(v.get("nodes").cloned().unwrap_or(serde_json::Value::Array(vec![]))).map_err(|e| e.to_string())
    }
}

impl SceneGraph for HttpGraph {
    fn query(&mut self, text: &str, top_k: usize) -> Result<Vec<Node>, String> {
        Self::nodes(crate::http::post_json(&format!("{}/query", self.base), &serde_json::json!({"text": text, "top_k": top_k}), &[], self.timeout)?)
    }
    fn all(&mut self) -> Result<Vec<Node>, String> {
        Self::nodes(crate::http::get(&format!("{}/objects", self.base), self.timeout)?)
    }
    fn reset(&mut self, labels: &[String]) -> Result<(), String> {
        let r = crate::http::post_json(&format!("{}/reset", self.base), &serde_json::json!({ "labels": labels }), &[], self.timeout)?;
        if r.status == 200 {
            Ok(())
        } else {
            Err(format!("HTTP {}", r.status))
        }
    }
    fn describe(&self) -> String {
        format!("http:{}", self.base)
    }
}

// ---------------- 에이전트 쪽 물체 기억 ----------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Known {
    pub id: String,
    pub label: String,
    pub first_center: [f64; 3],
    pub last_center: [f64; 3],
    pub first_step: u64,
    #[serde(default)]
    pub first_on: Option<String>,
    #[serde(default)]
    pub room: Option<String>,
    /// 이 물체를 대상으로 한 단계가 성공한 횟수(→ the other 에서 뺌)
    pub handled: u32,
    /// 그래프가 본 다룸 여부(scenemap: 표시·옮겨짐·들고 있음)
    #[serde(default)]
    pub graph_handled: bool,
    #[serde(default)]
    pub last_skill: Option<String>,
}

impl Known {
    pub fn is_handled(&self) -> bool {
        self.handled > 0 || self.graph_handled
    }
    pub fn moved(&self) -> f64 {
        let d: f64 = (0..3).map(|i| (self.last_center[i] - self.first_center[i]).powi(2)).sum();
        d.sqrt()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObjectMemory {
    pub known: BTreeMap<String, Known>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Resolved {
    pub id: String,
    pub name: String,
    pub center: Option<[f64; 3]>,
    pub how: String,
}

impl ObjectMemory {
    pub fn observe(&mut self, nodes: &[Node], step: u64) {
        for n in nodes {
            let e = self.known.entry(n.id.clone()).or_insert_with(|| Known {
                id: n.id.clone(),
                label: n.label.clone(),
                first_center: n.original.unwrap_or(n.center),
                last_center: n.center,
                first_step: step,
                first_on: n.on.clone(),
                room: n.room.clone(),
                handled: 0,
                graph_handled: false,
                last_skill: None,
            });
            e.last_center = n.center;
            // scenemap 은 처음 등록 때 중심을 따로 준다(first_position) → 그것이 "처음 자리"
            if let Some(o) = n.original {
                e.first_center = o;
            }
            e.graph_handled = n.handled;
            if e.label.is_empty() {
                e.label = n.label.clone();
            }
            if e.room.is_none() {
                e.room = n.room.clone();
            }
        }
    }

    pub fn mark_handled(&mut self, id: &str, skill: &str) {
        if let Some(k) = self.known.get_mut(id) {
            k.handled += 1;
            k.last_skill = Some(skill.to_string());
        }
    }

    pub fn name_of(&self, id: &str) -> Option<String> {
        self.known.get(id).map(|k| crate::util::display_name(if k.label.is_empty() { &k.id } else { &k.label }))
    }

    /// 참조(그래프 id / 이름 / 종류)를 구체 물체로. 종류만 주면 가장 가까운 것(점수 동률 시).
    pub fn resolve(&self, r: &str, pose: &Pose) -> Option<Resolved> {
        let r = r.trim();
        if let Some(k) = self.known.get(r) {
            return Some(Resolved { id: k.id.clone(), name: self.name_of(&k.id).unwrap_or_default(), center: Some(k.last_center), how: "id".into() });
        }
        let mut best: Option<(f64, f64, &Known)> = None;
        for k in self.known.values() {
            let s = text_score(r, &k.label).max(text_score(r, &k.id));
            if s < 0.5 {
                continue;
            }
            let d = dist2(pose, k.last_center);
            if best.map(|(bs, bd, _)| s > bs + 1e-9 || ((s - bs).abs() < 1e-9 && d < bd)).unwrap_or(true) {
                best = Some((s, d, k));
            }
        }
        best.map(|(_, _, k)| Resolved { id: k.id.clone(), name: self.name_of(&k.id).unwrap_or_default(), center: Some(k.last_center), how: "label".into() })
    }

    /// 맥락을 보는 참조 풀기: 놓기 단계의 첫 물체면 손에 든 것 먼저, 아니면 아직 안 다룬 것 먼저, 그다음 가까운 순.
    pub fn resolve_ctx(&self, r: &str, pose: &Pose, held: &[String], placing: bool) -> Option<Resolved> {
        let r = r.trim();
        if self.known.contains_key(r) {
            return self.resolve(r, pose);
        }
        let mut c: Vec<(f64, &Known)> = self
            .known
            .values()
            .filter_map(|k| {
                let s = text_score(r, &k.label).max(text_score(r, &k.id));
                (s >= 0.5).then_some((s, k))
            })
            .collect();
        if c.is_empty() {
            return None;
        }
        let best_s = c.iter().map(|x| x.0).fold(0.0, f64::max);
        c.retain(|x| x.0 >= best_s - 1e-9);
        c.sort_by(|(_, a), (_, b)| {
            let ka = if placing { !held.contains(&a.id) } else { a.is_handled() };
            let kb = if placing { !held.contains(&b.id) } else { b.is_handled() };
            ka.cmp(&kb).then(dist2(pose, a.last_center).partial_cmp(&dist2(pose, b.last_center)).unwrap())
        });
        let k = c[0].1;
        Some(Resolved {
            id: k.id.clone(),
            name: self.name_of(&k.id).unwrap_or_default(),
            center: Some(k.last_center),
            how: if placing { "label: in hand first".into() } else { "label: not handled first, then nearest".into() },
        })
    }

    /// "the other <종류>": 같은 종류 중 아직 안 다룬 것. `prefer`(지금 단계의 물체, 예: 방금 다가간 캔)가 조건에 맞으면 그것,
    /// 아니면 가까운 순. `exclude` 는 다룬 것들.
    pub fn resolve_other(&self, category: &str, exclude: &[String], pose: &Pose, prefer: Option<&str>) -> Option<Resolved> {
        let mut c: Vec<&Known> = self
            .known
            .values()
            .filter(|k| text_score(category, &k.label) >= 0.5 && !exclude.contains(&k.id))
            .collect();
        c.sort_by(|a, b| {
            let pa = Some(a.id.as_str()) != prefer;
            let pb = Some(b.id.as_str()) != prefer;
            pa.cmp(&pb)
                .then(a.is_handled().cmp(&b.is_handled()))
                .then(a.handled.cmp(&b.handled))
                .then(dist2(pose, a.last_center).partial_cmp(&dist2(pose, b.last_center)).unwrap())
        });
        c.first().map(|k| Resolved {
            id: k.id.clone(),
            name: self.name_of(&k.id).unwrap_or_default(),
            center: Some(k.last_center),
            how: format!("the other: handled {} times, nearest unhandled {}", k.handled, category),
        })
    }

    /// "back": 물체가 처음 있던 받침(알면) 또는 처음 자리 좌표.
    pub fn resolve_back(&self, id: &str, bddl_support: Option<&str>, pose: &Pose) -> Option<Resolved> {
        let k = self.known.get(id)?;
        if let Some(on) = &k.first_on {
            if let Some(s) = self.known.get(on) {
                return Some(Resolved { id: s.id.clone(), name: self.name_of(&s.id).unwrap_or_default(), center: Some(s.last_center), how: "back: first support in graph".into() });
            }
        }
        if let Some(sup) = bddl_support {
            if let Some(r) = self.resolve(sup, pose) {
                return Some(Resolved { how: "back: initial support from BDDL :init".into(), ..r });
            }
        }
        // 처음 자리에 가장 가까운 다른 물체를 받침으로 본다
        let near = self
            .known
            .values()
            .filter(|o| o.id != k.id)
            .min_by(|a, b| {
                let da: f64 = (0..2).map(|i| (a.last_center[i] - k.first_center[i]).powi(2)).sum();
                let db: f64 = (0..2).map(|i| (b.last_center[i] - k.first_center[i]).powi(2)).sum();
                da.partial_cmp(&db).unwrap()
            });
        match near {
            Some(n) => Some(Resolved { id: n.id.clone(), name: self.name_of(&n.id).unwrap_or_default(), center: Some(k.first_center), how: "back: object nearest to first-seen position".into() }),
            None => Some(Resolved { id: format!("pos:{:.2},{:.2}", k.first_center[0], k.first_center[1]), name: "its original place".into(), center: Some(k.first_center), how: "back: first-seen position".into() }),
        }
    }
}

fn dist2(p: &Pose, c: [f64; 3]) -> f64 {
    (c[0] - p.x).powi(2) + (c[1] - p.y).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: &str, label: &str, x: f64, y: f64, on: Option<&str>) -> Node {
        Node { id: id.into(), label: label.into(), center: [x, y, 0.8], on: on.map(|s| s.into()), ..Default::default() }
    }

    #[test]
    fn other_and_back() {
        let mut m = ObjectMemory::default();
        m.observe(&[n("c1", "can of soda", 1.0, 0.0, None), n("c2", "can of soda", 3.0, 0.0, None), n("t", "trash can", 5.0, 0.0, None), n("r", "radio", 2.0, 2.0, Some("tbl")), n("tbl", "coffee table", 2.0, 2.0, None)], 0);
        let p = Pose::default();
        assert_eq!(m.resolve("can of soda", &p).unwrap().id, "c1");
        m.mark_handled("c1", "place in");
        assert_eq!(m.resolve_other("can of soda", &[], &p, None).unwrap().id, "c2");
        // 방금 다가간 것(prefer)이 안 다룬 같은 종류면 그것
        m.observe(&[n("c3", "can of soda", 9.0, 0.0, None)], 1);
        assert_eq!(m.resolve_other("can of soda", &["c1".into()], &p, Some("c3")).unwrap().id, "c3");
        // 그래프가 다뤘다고 본 것(handled)은 뺀다
        let mut moved = n("c2", "can of soda", 3.0, 0.0, None);
        moved.handled = true;
        m.observe(&[moved], 2);
        assert_eq!(m.resolve_other("can of soda", &[], &p, None).unwrap().id, "c3");
        // 라디오를 옮긴 뒤에도 back 은 처음 받침
        m.observe(&[n("r", "radio", 0.0, 0.0, None)], 10);
        assert_eq!(m.resolve_back("r", None, &p).unwrap().id, "tbl");
        assert!(m.known["r"].moved() > 2.0);
    }

    #[test]
    fn static_query() {
        let mut g = StaticGraph { nodes: vec![n("a", "radio", 0.0, 0.0, None), n("b", "coffee table", 0.0, 0.0, None)], name: String::new() };
        assert_eq!(g.query("the radio", 3).unwrap()[0].id, "a");
        assert_eq!(g.query("table", 3).unwrap()[0].id, "b");
    }
}

#[cfg(test)]
mod scenemap_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn scenemap_graph_adapts_queries() {
        let m = Arc::new(MemScene::default());
        {
            let mut g = m.inner.lock().unwrap();
            g.0 = vec![
                SceneObject { id: 3, name: "radio receiver".into(), position: [1.0, 0.5, 0.6], first_position: [1.0, 0.5, 0.6], n_obs: 4, ..Default::default() },
                SceneObject { id: 7, name: "coffee table".into(), position: [1.2, 0.4, 0.3], first_position: [1.0, 0.0, 0.3], state: ObjState::Moved, ..Default::default() },
                SceneObject { id: 9, name: "wall".into(), position: [3.0, 0.0, 1.0], structural: true, ..Default::default() },
            ];
        }
        let mut g = ScenemapGraph { q: m.clone() };
        let r = g.query("radio", 5).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!((r[0].id.as_str(), r[0].label.as_str()), ("3", "radio receiver"));
        assert_eq!(r[0].original, Some([1.0, 0.5, 0.6]));
        let all = g.all().unwrap();
        assert_eq!(all.len(), 3);
        assert!(all.iter().find(|n| n.id == "7").unwrap().handled, "옮겨짐 = 다룸");
        assert!(all.iter().find(|n| n.id == "9").unwrap().structural);
        let near = g.near([1.0, 0.5, 0.5], 0.5).unwrap();
        assert_eq!(near.iter().map(|n| n.id.clone()).collect::<Vec<_>>(), vec!["3", "7"]);
        g.mark_handled("3").unwrap();
        assert!(g.all().unwrap().iter().find(|n| n.id == "3").unwrap().handled);
        g.reset(&["radio".into(), "table".into()]).unwrap();
        assert_eq!(m.inner.lock().unwrap().2.len(), 2);
        // 저장한 물체 표 JSON 도 같은 뜻으로 읽힌다
        let v = serde_json::json!({"id": 3, "name": "radio receiver", "position": [1.0, 0.5, 0.6], "original_position": [1.0, 0.5, 0.6], "n_obs": 4});
        let n = Node::from_json(&v).unwrap();
        assert_eq!((n.id.as_str(), n.label.as_str(), n.num_observations), ("3", "radio receiver", 4));
    }
}

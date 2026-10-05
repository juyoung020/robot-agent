//! 스킬 설정 `<skill>/skill.json` (코드 아님, 데이터). 스킬마다 다른 것 — 쓰는 도구, 첫 관측, LLM 없는 기준선 호출,
//! 끝 조건·되묻기 문장, 지표 칸 — 을 여기서 읽고, 루프([`crate::run`])는 스킬을 모른다.
//!
//! ```json
//! {"skill": "explore",
//!  "tools": [{"name": "move_robot", "desc": "tool.md", "modes": ["go_to", "probe", "delta", "absolute"]}],
//!  "observe": {"tool": "move_robot", "args": {...}},                 첫 관측(기록 안 함, 호출 수에 안 셈)
//!  "baseline": {"name": "frontier", "tool": "move_robot", "args": {...}},   --policy baseline: LLM 대신 매번 이 호출
//!  "end": {"name": "no_frontier", "when_empty": "map.frontiers", "nudge": "... {n} ...", "max_nudges": 1},
//!  "metrics": {"coverage": {"key": "gt_cov", "marks": [0.5, 0.8], "fields": ["sim_s", "path_m"]},
//!              "timeline": {"<칸>": "<_m 키>"}, "timeline_raw": {...}, "summary": {...}, "decision": ["contacts", ...]},
//!  "limits": {"max_calls": 80, "max_sim_s": 900, "max_wall_s": 3600, "keep_pairs": 3, "max_tokens": 400, "temperature": 0}}
//! ```
//! 측정값은 도구 결과의 `_m`(LLM 에게 안 보임)에서 읽는다. 시뮬 시각은 `_m.sim_s`.

use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ToolUse {
    pub name: String,
    /// 스킬 폴더 안 설명 파일(기본 tool.md)
    pub desc: String,
    /// LLM 에게 보일 mode enum(없으면 도구 기본)
    pub modes: Option<Vec<String>>,
}

#[derive(Clone, Debug)]
pub struct Call {
    pub tool: String,
    pub args: Value,
}

#[derive(Clone, Debug)]
pub struct Baseline {
    pub name: String,
    pub call: Call,
}

#[derive(Clone, Debug)]
pub struct EndRule {
    pub name: String,
    /// 관측(LLM 에게 보이는 결과)에서 점으로 이은 경로. 비었거나 없으면 끝 조건 참
    pub when_empty: String,
    pub nudge: Option<String>,
    pub max_nudges: usize,
}

#[derive(Clone, Debug)]
pub struct Coverage {
    pub key: String,
    pub marks: Vec<f64>,
    pub fields: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Limits {
    pub max_calls: usize,
    pub max_sim_s: f64,
    pub max_wall_s: f64,
    pub keep_pairs: usize,
    pub max_tokens: u32,
    pub temperature: f64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_calls: 80, max_sim_s: 900.0, max_wall_s: 3600.0, keep_pairs: 3, max_tokens: 400, temperature: 0.0 }
    }
}

#[derive(Clone, Debug)]
pub struct SkillSpec {
    pub name: String,
    pub dir: PathBuf,
    pub tools: Vec<ToolUse>,
    pub observe: Option<Call>,
    pub baseline: Option<Baseline>,
    pub end: Option<EndRule>,
    pub coverage: Option<Coverage>,
    /// 기록 칸 이름 → `_m` 키 (실수로)
    pub timeline: BTreeMap<String, String>,
    /// 기록 칸 이름 → `_m` 키 (값 그대로)
    pub timeline_raw: BTreeMap<String, String>,
    pub summary: BTreeMap<String, String>,
    /// decisions.jsonl `outcome` 에 더할 `_m` 키
    pub decision: Vec<String>,
    pub limits: Limits,
}

fn call(v: &Value) -> Option<Call> {
    Some(Call { tool: v["tool"].as_str()?.to_string(), args: v["args"].clone() })
}

fn str_map(v: &Value) -> BTreeMap<String, String> {
    v.as_object().map(|o| o.iter().filter_map(|(k, x)| Some((k.clone(), x.as_str()?.to_string()))).collect()).unwrap_or_default()
}

impl SkillSpec {
    pub fn load(dir: &Path) -> Result<SkillSpec, String> {
        let p = dir.join("skill.json");
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        Self::from_json(dir, &v).map_err(|e| format!("{}: {e}", p.display()))
    }

    pub fn from_json(dir: &Path, v: &Value) -> Result<SkillSpec, String> {
        let name = v["skill"].as_str().ok_or("skill: name missing")?.to_string();
        let tools: Vec<ToolUse> = v["tools"]
            .as_array()
            .ok_or("tools: list missing")?
            .iter()
            .map(|t| {
                Ok(ToolUse {
                    name: t["name"].as_str().ok_or("tools[].name missing")?.to_string(),
                    desc: t["desc"].as_str().unwrap_or("tool.md").to_string(),
                    modes: t["modes"].as_array().map(|a| a.iter().filter_map(|m| m.as_str().map(String::from)).collect()),
                })
            })
            .collect::<Result<_, &str>>()?;
        if tools.is_empty() {
            return Err("tools: at least one".into());
        }
        let m = &v["metrics"];
        let l = &v["limits"];
        let d = Limits::default();
        Ok(SkillSpec {
            name,
            dir: dir.to_path_buf(),
            tools,
            observe: call(&v["observe"]),
            baseline: v.get("baseline").and_then(|b| Some(Baseline { name: b["name"].as_str().unwrap_or("baseline").to_string(), call: call(b)? })),
            end: v.get("end").and_then(|e| {
                Some(EndRule {
                    name: e["name"].as_str()?.to_string(),
                    when_empty: e["when_empty"].as_str()?.to_string(),
                    nudge: e["nudge"].as_str().map(String::from),
                    max_nudges: e["max_nudges"].as_u64().unwrap_or(1) as usize,
                })
            }),
            coverage: m.get("coverage").and_then(|c| {
                Some(Coverage {
                    key: c["key"].as_str()?.to_string(),
                    marks: c["marks"].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default(),
                    fields: c["fields"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default(),
                })
            }),
            timeline: str_map(&m["timeline"]),
            timeline_raw: str_map(&m["timeline_raw"]),
            summary: str_map(&m["summary"]),
            decision: m["decision"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default(),
            limits: Limits {
                max_calls: l["max_calls"].as_u64().map_or(d.max_calls, |x| x as usize),
                max_sim_s: l["max_sim_s"].as_f64().unwrap_or(d.max_sim_s),
                max_wall_s: l["max_wall_s"].as_f64().unwrap_or(d.max_wall_s),
                keep_pairs: l["keep_pairs"].as_u64().map_or(d.keep_pairs, |x| x as usize),
                max_tokens: l["max_tokens"].as_u64().map_or(d.max_tokens, |x| x as u32),
                temperature: l["temperature"].as_f64().unwrap_or(d.temperature),
            },
        })
    }

    pub fn tool_files(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.desc.clone()).collect()
    }
}

/// 점으로 이은 경로로 값 찾기("map.frontiers")
pub fn path<'a>(v: &'a Value, p: &str) -> &'a Value {
    p.split('.').fold(v, |acc, k| &acc[k])
}

/// 끝 조건: 경로의 값이 빈 배열이거나 배열이 아니면 참
pub fn end_met(rule: &EndRule, obs: &Value) -> bool {
    path(obs, &rule.when_empty).as_array().map_or(true, |a| a.is_empty())
}

//! 과제 카드 모음(assets/tasks.json): 과제 번호·이름·지시문·제한시간·방·BDDL·대표 단계 순서·단계별 스텝 예산.
//!
//! `bagent build-assets` 가 저장소 파일에서 만든다(평가 때는 이 JSON 하나만 있으면 된다):
//! - 번호·방: `BEHAVIOR-1K/datasets/2026-challenge-task-instances/metadata/B100_task_misc.csv` (평가기 TASK_NAMES_TO_INDICES 와 같은 파일)
//! - 사람 평균 길이: 같은 폴더 `task.jsonl` `length` → 제한시간 = int(length × 1.5) (평가기 evaluator.py:279, EVAL_TIMEOUT_MULTIPLIER)
//! - 지시문: `data/2026-challenge-demos/meta/tasks.jsonl` `task`
//! - BDDL: `BEHAVIOR-1K/bddl3/bddl/activity_definitions/<과제>/problem0.bddl`
//! - 대표 순서·스텝 예산: `data/2026-challenge-demos/annotations/task-*/episode_*.json` `skill_annotation`

use crate::bddl::{self, Problem};
use crate::util::{display_name, percentile};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PriorStep {
    pub skill: String,
    /// 물체 종류(표시 이름). 묶음은 "a, b" 로.
    pub objects: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spatial: Vec<String>,
}

impl PriorStep {
    pub fn text(&self) -> String {
        match crate::vocab::lookup(&self.skill) {
            Some(k) => k.render(&self.objects, &self.spatial, self.memory.as_deref()),
            None => format!("{} {}", self.skill, self.objects.join(" ")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SeqPrior {
    /// "top" (정확히 같은 순서의 빈도 순위) 또는 "medoid" (다른 판들과 편집거리 합이 가장 작은 판)
    pub kind: String,
    pub count: usize,
    pub freq: f64,
    pub steps: Vec<PriorStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkillStat {
    pub name: String,
    pub n: usize,
    pub kind: String,
    pub p50_steps: u64,
    pub p90_steps: u64,
    pub max_steps: u64,
    /// 에이전트 기본 스텝 예산 = p90 × 1.5 (최소 90)
    pub budget_steps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskCard {
    pub index: u32,
    pub name: String,
    pub prompt: String,
    pub human_steps: f64,
    pub max_steps: u64,
    pub rooms: Vec<String>,
    pub problem: Problem,
    pub n_episodes: usize,
    pub median_stages: usize,
    /// 단계 이름만 볼 때 가장 흔한 순서의 비율
    pub skill_seq_top_freq: f64,
    pub priors: Vec<SeqPrior>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Catalog {
    pub version: u32,
    pub note: String,
    pub skills: Vec<SkillStat>,
    pub tasks: Vec<TaskCard>,
}

impl Catalog {
    pub fn load(path: &Path) -> Result<Catalog, String> {
        let t = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&t).map_err(|e| format!("{}: {e}", path.display()))
    }
    pub fn default_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets").join("tasks.json")
    }
    pub fn task_by_index(&self, i: i64) -> Option<&TaskCard> {
        self.tasks.iter().find(|t| t.index as i64 == i)
    }
    pub fn task_by_name(&self, n: &str) -> Option<&TaskCard> {
        self.tasks.iter().find(|t| t.name == n)
    }
    pub fn budget(&self, skill: &str) -> u64 {
        self.skills
            .iter()
            .find(|s| s.name == skill)
            .map(|s| s.budget_steps)
            .or_else(|| crate::vocab::lookup(skill).map(|k| k.default_budget()))
            .unwrap_or(600)
    }
}

// ---------------- 만들기 ----------------

fn obj_name(v: &Value) -> String {
    match v {
        Value::String(s) => display_name(s),
        Value::Array(a) => {
            let mut names: Vec<String> = a.iter().map(obj_name).collect();
            names.dedup();
            let mut uniq: Vec<String> = Vec::new();
            for n in names {
                if !uniq.contains(&n) {
                    uniq.push(n);
                }
            }
            uniq.join(", ")
        }
        _ => String::new(),
    }
}

struct Episode {
    steps: Vec<PriorStep>,
    durations: Vec<(String, String, u64)>, // (skill, kind, frames)
}

fn read_episode(path: &Path) -> Option<Episode> {
    let t = std::fs::read_to_string(path).ok()?;
    let v: Value = serde_json::from_str(&t).ok()?;
    let mut ep = Episode { steps: Vec::new(), durations: Vec::new() };
    for s in v.get("skill_annotation")?.as_array()? {
        let descs = s.get("skill_description").and_then(|d| d.as_array()).cloned().unwrap_or_default();
        let fd = s.get("frame_duration").and_then(|d| d.as_array());
        let frames = fd.and_then(|f| Some(f.get(1)?.as_u64()?.saturating_sub(f.first()?.as_u64()?))).unwrap_or(0);
        for (i, d) in descs.iter().enumerate() {
            let skill = d.as_str().unwrap_or("").to_string();
            let objs = s
                .get("object_id")
                .and_then(|o| o.get(i))
                .and_then(|o| o.as_array())
                .map(|a| a.iter().map(obj_name).collect())
                .unwrap_or_default();
            let memory = s
                .get("memory_prefix")
                .and_then(|m| m.get(i))
                .and_then(|m| m.as_str())
                .filter(|m| !m.is_empty())
                .map(|m| m.to_string());
            let spatial: Vec<String> = match s.get("spatial_prefix").and_then(|m| m.get(i)) {
                Some(Value::Array(a)) => a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect(),
                Some(Value::String(x)) => vec![x.clone()],
                _ => vec![],
            };
            let spatial = if spatial.iter().all(|x| x.is_empty()) { vec![] } else { spatial };
            let kind = s
                .get("skill_type")
                .and_then(|m| m.get(i))
                .and_then(|m| m.as_str())
                .unwrap_or("")
                .to_string();
            ep.durations.push((skill.clone(), kind, frames));
            ep.steps.push(PriorStep { skill, objects: objs, memory, spatial });
        }
    }
    Some(ep)
}

fn edit_distance(a: &[u32], b: &[u32]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn step_key(s: &PriorStep) -> String {
    format!("{}|{}|{}", s.skill, s.objects.join("/"), s.memory.as_deref().unwrap_or(""))
}

fn priors_for(eps: &[Episode]) -> (Vec<SeqPrior>, f64, usize) {
    let n = eps.len().max(1);
    // 단계 이름만
    let mut by_skill: HashMap<Vec<&str>, usize> = HashMap::new();
    for e in eps {
        *by_skill.entry(e.steps.iter().map(|s| s.skill.as_str()).collect()).or_default() += 1;
    }
    let skill_top = by_skill.values().copied().max().unwrap_or(0) as f64 / n as f64;
    // 물체 종류까지
    let mut by_obj: HashMap<Vec<String>, (usize, usize)> = HashMap::new();
    for (i, e) in eps.iter().enumerate() {
        let k: Vec<String> = e.steps.iter().map(step_key).collect();
        by_obj.entry(k).or_insert((0, i)).0 += 1;
    }
    let mut ranked: Vec<(usize, usize)> = by_obj.values().copied().collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut out: Vec<SeqPrior> = ranked
        .iter()
        .take(3)
        .filter(|(c, _)| *c >= 2 || n == 1)
        .map(|&(c, i)| SeqPrior { kind: "top".into(), count: c, freq: c as f64 / n as f64, steps: eps[i].steps.clone() })
        .collect();
    let top_freq = out.first().map(|p| p.freq).unwrap_or(0.0);
    if top_freq < 0.2 && eps.len() >= 3 {
        // 메도이드: 표본(최대 80판)끼리 편집거리 합 최소
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let step = (eps.len() / 80).max(1);
        let sample: Vec<(usize, Vec<u32>)> = eps
            .iter()
            .enumerate()
            .step_by(step)
            .map(|(i, e)| {
                let codes = e
                    .steps
                    .iter()
                    .map(|s| {
                        let l = vocab.len() as u32;
                        *vocab.entry(step_key(s)).or_insert(l)
                    })
                    .collect();
                (i, codes)
            })
            .collect();
        let best = sample
            .iter()
            .map(|(i, a)| (sample.iter().map(|(_, b)| edit_distance(a, b)).sum::<usize>(), *i))
            .min()
            .map(|(_, i)| i);
        if let Some(i) = best {
            let k: Vec<String> = eps[i].steps.iter().map(step_key).collect();
            let c = by_obj.get(&k).map(|v| v.0).unwrap_or(1);
            out.push(SeqPrior { kind: "medoid".into(), count: c, freq: c as f64 / n as f64, steps: eps[i].steps.clone() });
        }
    }
    let mut lens: Vec<usize> = eps.iter().map(|e| e.steps.len()).collect();
    lens.sort();
    let med = lens.get(lens.len() / 2).copied().unwrap_or(0);
    (out, skill_top, med)
}

/// 저장소 루트(옛 Windows·WSL 시절 경로 기준 설명)에서 카탈로그를 만든다.
pub fn build(root: &Path, threads: usize) -> Result<Catalog, String> {
    let meta = root.join("BEHAVIOR-1K/datasets/2026-challenge-task-instances/metadata");
    let misc = std::fs::read_to_string(meta.join("B100_task_misc.csv")).map_err(|e| format!("B100_task_misc.csv: {e}"))?;
    let tasks_csv = parse_csv(&misc);
    let mut length: HashMap<String, f64> = HashMap::new();
    for l in std::fs::read_to_string(meta.join("task.jsonl")).unwrap_or_default().lines() {
        if let Ok(v) = serde_json::from_str::<Value>(l) {
            if let (Some(n), Some(len)) = (v["task_name"].as_str(), v["length"].as_f64()) {
                length.insert(n.to_string(), len);
            }
        }
    }
    let demos = root.join("data/2026-challenge-demos");
    let mut prompt: HashMap<String, String> = HashMap::new();
    for l in std::fs::read_to_string(demos.join("meta/tasks.jsonl")).unwrap_or_default().lines() {
        if let Ok(v) = serde_json::from_str::<Value>(l) {
            if let (Some(n), Some(t)) = (v["task_name"].as_str(), v["task"].as_str()) {
                prompt.insert(n.to_string(), t.to_string());
            }
        }
    }
    // 주석: 과제 폴더별로 병렬 읽기
    let ann = demos.join("annotations");
    let mut dirs: Vec<(u32, PathBuf)> = std::fs::read_dir(&ann)
        .map_err(|e| format!("{}: {e}", ann.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.strip_prefix("task-").and_then(|x| x.parse::<u32>().ok()).map(|i| (i, e.path()))
        })
        .collect();
    dirs.sort();
    let work = std::sync::Mutex::new(dirs.clone());
    let results: std::sync::Mutex<HashMap<u32, Vec<Episode>>> = std::sync::Mutex::new(HashMap::new());
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| loop {
                let job = work.lock().unwrap().pop();
                let Some((idx, dir)) = job else { break };
                let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
                    .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map(|x| x == "json").unwrap_or(false)).collect())
                    .unwrap_or_default();
                files.sort();
                let eps: Vec<Episode> = files.iter().filter_map(|f| read_episode(f)).collect();
                results.lock().unwrap().insert(idx, eps);
            });
        }
    });
    let results = results.into_inner().unwrap();

    // 단계별 길이 통계
    let mut dur: HashMap<String, (Vec<f64>, HashMap<String, usize>)> = HashMap::new();
    for eps in results.values() {
        for e in eps {
            for (s, k, f) in &e.durations {
                let d = dur.entry(s.clone()).or_default();
                d.0.push(*f as f64);
                *d.1.entry(k.clone()).or_default() += 1;
            }
        }
    }
    let mut skills: Vec<SkillStat> = dur
        .into_iter()
        .map(|(name, (mut v, kinds))| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p90 = percentile(&v, 0.9) as u64;
            SkillStat {
                n: v.len(),
                kind: kinds.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k).unwrap_or_default(),
                p50_steps: percentile(&v, 0.5) as u64,
                p90_steps: p90,
                max_steps: v.last().copied().unwrap_or(0.0) as u64,
                budget_steps: ((p90 as f64 * 1.5) as u64).max(90),
                name,
            }
        })
        .collect();
    skills.sort_by(|a, b| b.n.cmp(&a.n));

    let bddl_dir = root.join("BEHAVIOR-1K/bddl3/bddl/activity_definitions");
    let mut tasks = Vec::new();
    for (idx, name, rooms) in tasks_csv {
        let problem = std::fs::read_to_string(bddl_dir.join(&name).join("problem0.bddl"))
            .ok()
            .and_then(|t| bddl::problem(&t).ok())
            .unwrap_or_default();
        let eps = results.get(&idx).map(|v| v.as_slice()).unwrap_or(&[]);
        let (priors, skill_top, med) = if eps.is_empty() { (vec![], 0.0, 0) } else { priors_for(eps) };
        let human = length.get(&name).copied().unwrap_or(0.0);
        tasks.push(TaskCard {
            index: idx,
            prompt: prompt.get(&name).cloned().unwrap_or_default(),
            human_steps: human,
            max_steps: (human * 1.5) as u64,
            rooms,
            problem,
            n_episodes: eps.len(),
            median_stages: med,
            skill_seq_top_freq: skill_top,
            priors,
            name,
        });
    }
    tasks.sort_by_key(|t| t.index);
    Ok(Catalog {
        version: 1,
        note: "bagent build-assets 로 생성. 출처는 src/agent/planner/src/catalog.rs 머리말".into(),
        skills,
        tasks,
    })
}

/// 따옴표 안 줄바꿈이 있는 CSV(B100_task_misc.csv) → (번호, 이름, 방 목록)
fn parse_csv(text: &str) -> Vec<(u32, String, Vec<String>)> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut cell = String::new();
    let mut q = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, q) {
            ('"', true) if chars.peek() == Some(&'"') => {
                cell.push('"');
                chars.next();
            }
            ('"', _) => q = !q,
            (',', false) => row.push(std::mem::take(&mut cell)),
            ('\n', false) => {
                row.push(std::mem::take(&mut cell));
                rows.push(std::mem::take(&mut row));
            }
            ('\r', false) => {}
            (c, _) => cell.push(c),
        }
    }
    if !cell.is_empty() || !row.is_empty() {
        row.push(cell);
        rows.push(row);
    }
    rows.into_iter()
        .skip(1)
        .filter_map(|r| {
            let idx = r.first()?.trim().parse().ok()?;
            let name = r.get(1)?.trim().to_string();
            let rooms = r.get(2).map(|s| s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()).unwrap_or_default();
            Some((idx, name, rooms))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_multiline() {
        let t = "Task ID,Task,Rooms to inlcude\n0,turning_on_radio,\"corridor_0\nkitchen_0\"\n5,setting_mousetraps,\"bathroom_0\"\n";
        let r = parse_csv(t);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].2, vec!["corridor_0", "kitchen_0"]);
        assert_eq!(r[1].0, 5);
    }

    #[test]
    fn edit() {
        assert_eq!(edit_distance(&[1, 2, 3], &[1, 3]), 1);
        assert_eq!(edit_distance(&[], &[1, 2]), 2);
    }
}

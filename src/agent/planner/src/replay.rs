//! 기록 재생 검증: 기록(JSONL)에 남은 경계 사건·그래프 노드·LLM 응답을 다시 먹여 같은 결정이 나오는지 본다.
//! LLM 응답은 목적(plan/summary)별 순서대로 돌려주고, 요청 지문이 기록과 다르면 따로 알린다(맥락 조립이 바뀐 것).

use crate::catalog::Catalog;
use crate::graph::{SharedGraph, StaticGraph};
use crate::llm::{ChatResult, ReplayLlm};
use crate::planner::{Agent, BoundaryEvent, Core, Decider, Decision, LlmDecider, PlannerCfg, PriorDecider};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;

#[derive(Debug, Default)]
pub struct VerifyReport {
    pub total: usize,
    pub same: usize,
    pub diffs: Vec<String>,
    pub fingerprint_mismatches: usize,
    pub mismatch_detail: Vec<String>,
}

pub fn verify(events: &[Value], catalog: &Catalog, dir: &Path) -> Result<VerifyReport, String> {
    let mut rep = VerifyReport::default();
    let mut envs: BTreeMap<u64, Vec<&Value>> = BTreeMap::new();
    for e in events {
        envs.entry(e["env"].as_u64().unwrap_or(0)).or_default().push(e);
    }
    for (env, evs) in envs {
        let Some(agent_ev) = evs.iter().rev().find(|e| e["type"] == "agent" && e.get("task").is_some()) else { continue };
        let task_name = agent_ev["task"].as_str().unwrap_or("");
        let task = catalog.task_by_name(task_name).ok_or_else(|| format!("카탈로그에 {task_name} 없음"))?;
        let cfg: PlannerCfg = serde_json::from_value(agent_ev["cfg"].clone()).map_err(|e| e.to_string())?;
        let mut queues: HashMap<String, VecDeque<(String, ChatResult, Value)>> = HashMap::new();
        for e in evs.iter().filter(|e| e["type"] == "llm") {
            let r: ChatResult = serde_json::from_value(e["result"].clone()).map_err(|e| e.to_string())?;
            queues
                .entry(e["purpose"].as_str().unwrap_or("plan").to_string())
                .or_default()
                .push_back((e["fingerprint"].as_str().unwrap_or("").to_string(), r, e["request"].clone()));
        }
        let graph = SharedGraph::default();
        let is_prior = agent_ev["decider"].as_str() == Some("prior");
        let mism = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let decider: Box<dyn Decider> =
            if is_prior { Box::new(PriorDecider) } else { Box::new(LlmDecider { llm: Box::new(ReplayLlm { queues, mismatches: mism.clone() }) }) };
        let core = Core::new(cfg, task.clone(), catalog, Box::new(graph.clone()), env as usize);
        let mut agent = Agent::new(core, decider);
        let decisions: Vec<&&Value> = evs.iter().filter(|e| e["type"] == "decision").collect();
        let mut di = 0;
        let mut first_ep = true;
        for e in &evs {
            match e["type"].as_str() {
                Some("episode_start") => {
                    if !first_ep {
                        agent.reset_episode(e["episode"].as_u64().unwrap_or(0) as u32);
                    }
                    first_ep = false;
                }
                Some("boundary") => {
                    let mut bev: BoundaryEvent = serde_json::from_value(e["event"].clone()).map_err(|e| e.to_string())?;
                    bev.images = bev
                        .image_files
                        .iter()
                        .filter_map(|f| {
                            let stem = f.trim_end_matches(".jpg");
                            let cam = if stem.ends_with("left_wrist") {
                                "left_wrist"
                            } else if stem.ends_with("right_wrist") {
                                "right_wrist"
                            } else {
                                "head"
                            };
                            Some((cam.to_string(), std::fs::read(dir.join(f)).ok()?))
                        })
                        .collect();
                    if bev.image_files.is_empty() && agent.core.cfg.send_images {
                        // 가짜 세계 기록: 같은 무늬 영상
                        bev.images = vec![("head".into(), crate::image::Rgb::pattern(64, 64, 7).jpeg(70))];
                    }
                    let nodes: Vec<crate::graph::Node> = serde_json::from_value(e["nodes"].clone()).map_err(|er| format!("step {} 노드 해석 실패: {er}", bev.step))?;
                    *graph.0.lock().unwrap() = StaticGraph { nodes, name: "replay".into() };
                    let d = agent.decide(&bev);
                    agent.maintain();
                    let rec: Option<Decision> = decisions.get(di).and_then(|x| serde_json::from_value(x["decision"].clone()).ok());
                    di += 1;
                    rep.total += 1;
                    let ok = rec.as_ref().map(|r| r.key() == d.key()).unwrap_or(false);
                    if ok {
                        rep.same += 1;
                    } else {
                        rep.diffs.push(format!("env{env} step {}: 기록 {:?} / 재생 {}", bev.step, rec.map(|r| r.key()), d.key()));
                    }
                }
                _ => {}
            }
        }
        let m = mism.lock().unwrap();
        rep.fingerprint_mismatches += m.len();
        rep.mismatch_detail.extend(m.iter().cloned());
    }
    Ok(rep)
}

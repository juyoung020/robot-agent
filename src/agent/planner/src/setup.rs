//! 명령줄 인자 → 계획기 부품(과제 카드·설정·LLM·그래프·결정기·에이전트 공장).
//! `bagent`(relay·link·sim …)와 WSL 통합 노드(`src/sim/integ/simlink`)가 같은 코드로 에이전트를 만든다.

use crate::catalog::{Catalog, TaskCard};
use crate::fakes;
use crate::graph::{HttpGraph, NullGraph, SceneGraph, SceneQuery, ScenemapGraph, StaticGraph};
use crate::instruction::Format;
use crate::llm::{HttpLlm, Llm, ManagedLlm};
use crate::planner::{Agent, Core, Decider, LlmDecider, PlannerCfg, PriorDecider};
use crate::util::Args;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// 환경 번호·과제 카드 → 에이전트 (배치 평가면 환경마다 하나).
pub type AgentFactory = Arc<dyn Fn(usize, &TaskCard) -> Agent + Send + Sync>;

pub fn load_catalog(a: &Args) -> Result<Catalog, String> {
    let p = a.get("catalog").map(PathBuf::from).unwrap_or_else(Catalog::default_path);
    Catalog::load(&p)
}

pub fn planner_cfg(a: &Args) -> Result<PlannerCfg, String> {
    let mut c = PlannerCfg::default();
    if let Some(f) = a.get("format") {
        c.format = Format::parse(f).ok_or_else(|| format!("--format 은 task|subtask|purpose|metric: {f}"))?;
        if c.format == Format::Metric {
            eprintln!("경고: --format metric 은 실험 전용이다(VLA 에 숫자 명령을 쓰지 않기로 함, docs/에이전트_설계.md 1.5)");
        }
    }
    c.send_images = !a.flag("no-images");
    c.max_prompt_tokens = a.num("max-prompt-tokens", c.max_prompt_tokens);
    c.decision_budget_s = a.num("decision-budget-s", c.decision_budget_s);
    c.ctx_budget_tokens = a.num("ctx-budget", c.ctx_budget_tokens);
    c.thinking = a.flag("thinking");
    if let Some(s) = a.get("seed") {
        c.sampling.seed = s.parse().ok();
    }
    Ok(c)
}

pub fn build_llm(a: &Args) -> Result<Box<dyn Llm>, String> {
    let timeout = a.num("llm-timeout-s", 60u64);
    let spec = a.str_or("llm", "oracle");
    let inner: Box<dyn Llm> = match spec.as_str() {
        "oracle" | "mock" => Box::new(fakes::oracle_llm()),
        "kau" => Box::new(HttpLlm::kau(timeout)?),
        url if url.starts_with("http") => Box::new(HttpLlm::new(url, &a.str_or("model", "qwen3.5-9b"), a.get("key-env"), timeout)),
        other => return Err(format!("--llm 을 모름: {other}")),
    };
    if a.get("llm-up-cmd").is_some() || a.get("llm-down-cmd").is_some() {
        if spec == "kau" {
            return Err("API 모드(kau)에서는 서버 올리고 내리기 훅을 쓰지 않는다".into());
        }
        return Ok(Box::new(ManagedLlm {
            inner,
            up_cmd: a.get("llm-up-cmd").map(|s| s.to_string()),
            down_cmd: a.get("llm-down-cmd").map(|s| s.to_string()),
            health_url: a.get("llm-health").map(|s| s.to_string()),
            ready_timeout: Duration::from_secs(a.num("llm-ready-s", 120)),
            up: false,
        }));
    }
    Ok(inner)
}

/// 같은 프로세스 scenemap 질의(simlink 가 C ABI 래퍼를 시작 때 등록한다). `--graph scenemap` 이 이것을 쓴다.
static SCENE: std::sync::OnceLock<Arc<dyn SceneQuery>> = std::sync::OnceLock::new();

pub fn register_scene(q: Arc<dyn SceneQuery>) -> Result<(), String> {
    SCENE.set(q).map_err(|_| "scenemap 질의가 이미 등록됨".to_string())
}

pub fn scene() -> Option<Arc<dyn SceneQuery>> {
    SCENE.get().cloned()
}

pub fn build_graph(a: &Args) -> Box<dyn SceneGraph> {
    match a.str_or("graph", "none").as_str() {
        "none" => Box::new(NullGraph),
        "scenemap" => match scene() {
            Some(q) => Box::new(ScenemapGraph { q }),
            None => {
                eprintln!("--graph scenemap: 등록된 scenemap 이 없다(simlink 안에서만 된다) — 그래프 없이");
                Box::new(NullGraph)
            }
        },
        u if u.starts_with("http") => Box::new(HttpGraph::new(u, a.num("graph-timeout-s", 5))),
        path => match StaticGraph::load(path) {
            Ok(g) => Box::new(g),
            Err(e) => {
                eprintln!("그래프 파일 못 읽음({e}) — 그래프 없이");
                Box::new(NullGraph)
            }
        },
    }
}

pub fn build_decider(a: &Args) -> Result<Box<dyn Decider>, String> {
    match a.str_or("decider", "llm").as_str() {
        "prior" => Ok(Box::new(PriorDecider)),
        "llm" => Ok(Box::new(LlmDecider { llm: build_llm(a)? })),
        o => Err(format!("--decider 는 llm|prior: {o}")),
    }
}

/// 에이전트 공장. 시작할 때 결정기를 한 번 만들어 보아 설정 오류(키 없음 등)를 먼저 잡는다.
pub fn agent_factory(a: &Args, catalog: Arc<Catalog>) -> Result<AgentFactory, String> {
    let pcfg = planner_cfg(a)?;
    build_decider(a)?;
    let a2 = a.clone();
    Ok(Arc::new(move |env, task| {
        let core = Core::new(pcfg.clone(), task.clone(), &catalog, build_graph(&a2), env);
        let decider = build_decider(&a2).unwrap_or_else(|_| Box::new(PriorDecider));
        Agent::new(core, decider)
    }))
}

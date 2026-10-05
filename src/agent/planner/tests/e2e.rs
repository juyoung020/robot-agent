//! 끝에서 끝까지 시험: 가짜 평가기 ↔ 중계기 ↔ 가짜 VLA, 가짜 세계 에이전트 반복문, 기록 재생, 가짜 LLM HTTP 서버.

use bagent::catalog::Catalog;
use bagent::fakes::{self, FakePiLog, ObsSpec};
use bagent::graph::{NullGraph, SharedGraph};
use bagent::llm::{FnCall, FnLlm, HttpLlm, Msg, ToolCall};
use bagent::mockworld::{run_episode, World};
use bagent::planner::{Agent, Core, LlmDecider, PlannerCfg, PriorDecider};
use bagent::relay::{self, Mode, RelayCfg};
use bagent::trace::{self, Tracer};
use serde_json::json;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

fn catalog() -> Catalog {
    Catalog::load(&Catalog::default_path()).expect("assets/tasks.json (bagent build-assets 로 생성)")
}

fn tmpdir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("bagent_test_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// 중계기를 띄우고(연결 하나), 가짜 VLA 기록과 함께 돌려준다.
fn start_relay(mode: Mode, with_agent: bool) -> (String, Arc<Mutex<FakePiLog>>, std::thread::JoinHandle<()>) {
    let log = Arc::new(Mutex::new(FakePiLog::default()));
    let pl = TcpListener::bind("127.0.0.1:0").unwrap();
    let paddr = pl.local_addr().unwrap().to_string();
    let l2 = log.clone();
    std::thread::spawn(move || fakes::serve_fake_pi(pl, Some(l2), true));
    let rl = TcpListener::bind("127.0.0.1:0").unwrap();
    let raddr = rl.local_addr().unwrap().to_string();
    let cat = Arc::new(catalog());
    let mut cfg = RelayCfg::new(&raddr, &paddr, mode, cat.clone());
    cfg.once = true;
    cfg.latency_every = 0;
    cfg.trace_dir = Some(tmpdir(&format!("relay_{}", raddr.replace([':', '.'], "_"))));
    if with_agent {
        let cat2 = cat.clone();
        cfg.factory = Some(Arc::new(move |env, task| {
            let core = Core::new(PlannerCfg::default(), task.clone(), &cat2, Box::new(NullGraph), env);
            Agent::new(core, Box::new(LlmDecider { llm: Box::new(fakes::oracle_llm()) }))
        }));
    }
    let h = std::thread::spawn(move || {
        relay::run_listener(rl, cfg).expect("relay");
    });
    (raddr, log, h)
}

#[test]
fn relay_agent_mode_keeps_bytes_and_injects_prompts() {
    let (raddr, log, h) = start_relay(Mode::Agent, true);
    // 평가기처럼 먼저 /healthz
    let r = bagent::http::get(&format!("http://{raddr}/healthz"), std::time::Duration::from_secs(5)).unwrap();
    assert_eq!(r.status, 200);
    let spec = ObsSpec { batch: 1, rgbd: false, robot: "robot_r1".into(), task_id: 0 };
    let ev = fakes::run_fake_eval(&raddr, &spec, 300, 2, true).unwrap();
    h.join().unwrap();
    let g = log.lock().unwrap();
    assert_eq!(g.resets, 2, "reset 은 그대로 넘어가고 응답이 없어야 함");
    assert_eq!(g.steps, 600);
    // 원래 관측 바이트 == 주입 키를 뺀 바이트
    assert_eq!(g.obs_hashes, ev.obs_hashes, "관측 값이 바뀌었다");
    // VLA 행동 바이트 그대로
    assert_eq!(g.resp_hashes, ev.resp_hashes, "행동 값이 바뀌었다");
    // 첫 스텝부터 에이전트 지시(배치 1 → 문자열 배열)
    assert_eq!(g.prompts[0], json!(["move to radio"]));
    assert!(g.flushes >= 1);
    let distinct: std::collections::BTreeSet<String> = g.prompts.iter().map(|p| p.to_string()).collect();
    assert!(distinct.len() >= 2, "단계가 바뀌어야 함: {distinct:?}");
}

#[test]
fn relay_passthrough_is_byte_identical() {
    let (raddr, log, h) = start_relay(Mode::Passthrough, false);
    let spec = ObsSpec { batch: 2, rgbd: false, robot: "robot".into(), task_id: 0 };
    let ev = fakes::run_fake_eval(&raddr, &spec, 50, 1, true).unwrap();
    h.join().unwrap();
    let g = log.lock().unwrap();
    assert_eq!(g.obs_hashes, ev.obs_hashes);
    assert_eq!(g.resp_hashes, ev.resp_hashes);
    assert!(g.prompts.iter().all(|p| p.is_null()), "passthrough 는 주입하지 않는다");
}

#[test]
fn relay_fixed_prompt_batched() {
    let (raddr, log, h) = start_relay(Mode::Fixed("press radio".into()), false);
    let spec = ObsSpec { batch: 3, rgbd: false, robot: "robot_r1".into(), task_id: 0 };
    let ev = fakes::run_fake_eval(&raddr, &spec, 20, 1, true).unwrap();
    h.join().unwrap();
    let g = log.lock().unwrap();
    assert_eq!(g.obs_hashes, ev.obs_hashes);
    assert_eq!(g.prompts[5], json!(["press radio", "press radio", "press radio"]));
}

fn sim(scn: &str, decider: &str, p: f64, dir: Option<&std::path::Path>) -> bagent::mockworld::EpisodeResult {
    let cat = catalog();
    let task = cat.task_by_name(World::task_name(scn)).unwrap().clone();
    let graph = SharedGraph::default();
    let core = Core::new(PlannerCfg::default(), task.clone(), &cat, Box::new(graph.clone()), 0);
    let d: Box<dyn bagent::planner::Decider> = match decider {
        "prior" => Box::new(PriorDecider),
        _ => Box::new(LlmDecider { llm: Box::new(fakes::oracle_llm()) }),
    };
    let mut agent = Agent::new(core, d);
    if let Some(dir) = dir {
        agent.set_tracer(Some(Tracer::create(dir).unwrap()));
    }
    let mut w = World::scenario(scn, 0, p).unwrap();
    let r = run_episode(&mut w, &mut agent, &graph, Default::default(), task.max_steps, 0, true);
    if let Some(t) = &agent.core.tracer {
        t.flush();
    }
    r
}

#[test]
fn mock_world_radio_and_trash() {
    let r = sim("radio", "llm", 1.0, None);
    assert!(r.success, "{r:?}");
    assert_eq!(r.prompts[0], "move to radio");
    let t = sim("trash", "llm", 1.0, None);
    assert!(t.success, "{t:?}");
    // the other 가 매번 다른 캔으로 풀려야 세 개가 다 들어간다
    let placed: Vec<&String> = t.world_log.iter().filter(|l| l.contains("placed can_of_soda")).collect();
    assert_eq!(placed.len(), 3, "{:?}", t.world_log);
    let p = sim("radio", "prior", 1.0, None);
    assert!(p.success, "{p:?}");
    assert_eq!(p.llm_calls, 0);
}

#[test]
fn llm_errors_fall_back_without_stalling() {
    let cat = catalog();
    let task = cat.task_by_name("turning_on_radio").unwrap().clone();
    let graph = SharedGraph::default();
    let core = Core::new(PlannerCfg::default(), task.clone(), &cat, Box::new(graph.clone()), 0);
    let bad = FnLlm { f: |_r: &bagent::llm::ChatRequest| Err::<Msg, String>("connection refused".into()), name: "down".into() };
    let mut agent = Agent::new(core, Box::new(LlmDecider { llm: Box::new(bad) }));
    let mut w = World::scenario("radio", 0, 1.0).unwrap();
    let r = run_episode(&mut w, &mut agent, &graph, Default::default(), task.max_steps, 0, false);
    assert!(r.fallbacks >= 1);
    assert!(r.prompts.len() >= 2, "대체 결정으로 단계가 진행돼야 함: {:?}", r.prompts);
    assert!(r.success, "대체만으로도 참고 순서를 따라가야 함: {r:?}");
}

#[test]
fn bad_tool_arguments_become_observations() {
    let cat = catalog();
    let task = cat.task_by_name("turning_on_radio").unwrap().clone();
    let graph = SharedGraph::default();
    let core = Core::new(PlannerCfg::default(), task.clone(), &cat, Box::new(graph.clone()), 0);
    let mut n = 0;
    let llm = FnLlm {
        f: move |r: &bagent::llm::ChatRequest| {
            n += 1;
            Ok(match n {
                1 => Msg::assistant_calls(vec![ToolCall { id: "a".into(), kind: "function".into(), function: FnCall { name: "issue_command".into(), arguments: "{not json".into() } }]),
                2 => Msg::assistant_calls(vec![ToolCall {
                    id: "b".into(),
                    kind: "function".into(),
                    function: FnCall { name: "issue_command".into(), arguments: json!({"previous": "none", "skill": "fly to", "objects": ["radio"], "purpose": "x", "expected": "y"}).to_string() },
                }]),
                3 => Msg { role: "assistant".into(), content: Some(bagent::llm::Content::Text("I think we should move.".into())), tool_calls: None, tool_call_id: None },
                _ => fakes::oracle_reply(&r.messages, true),
            })
        },
        name: "flaky".into(),
    };
    let mut agent = Agent::new(core, Box::new(LlmDecider { llm: Box::new(llm) }));
    let mut w = World::scenario("radio", 0, 1.0).unwrap();
    let r = run_episode(&mut w, &mut agent, &graph, Default::default(), task.max_steps, 0, false);
    assert!(agent.core.stats.tool_errors >= 2, "{:?}", agent.core.stats);
    assert_eq!(agent.core.stats.fallbacks, 0);
    assert!(r.success);
}

#[test]
fn trace_replay_reproduces_decisions() {
    let dir = tmpdir("replay");
    let r = sim("trash", "llm", 1.0, Some(&dir));
    assert!(r.success);
    let events = trace::read(&dir.join("trace.jsonl")).unwrap();
    assert!(events.iter().any(|e| e["type"] == "llm"));
    let rep = bagent::replay::verify(&events, &catalog(), &dir).unwrap();
    assert_eq!(rep.same, rep.total, "{:?}", rep.diffs);
    assert!(rep.total >= 10);
    assert_eq!(rep.fingerprint_mismatches, 0, "{:?}", rep.mismatch_detail);
    let html = trace::export_html(&events, "t");
    assert!(html.contains("\"type\":\"decision\""));
    assert!(!trace::timeline(&events).is_empty());
}

#[test]
fn mock_llm_over_http() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    std::thread::spawn(move || fakes::serve_mock_llm(l, 0));
    let cat = catalog();
    let task = cat.task_by_name("turning_on_radio").unwrap().clone();
    let graph = SharedGraph::default();
    let core = Core::new(PlannerCfg::default(), task.clone(), &cat, Box::new(graph.clone()), 0);
    let llm = HttpLlm::new(&format!("http://{addr}/v1"), "mock-oracle", None, 10);
    let mut agent = Agent::new(core, Box::new(LlmDecider { llm: Box::new(llm) }));
    let mut w = World::scenario("radio", 0, 1.0).unwrap();
    let r = run_episode(&mut w, &mut agent, &graph, Default::default(), task.max_steps, 0, true);
    assert!(r.success, "{r:?}");
    assert_eq!(agent.core.stats.llm_errors, 0);
}

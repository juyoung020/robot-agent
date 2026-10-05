//! `bagent` 명령. 설명과 예는 `bagent help` 또는 docs/에이전트_설계.md "실행 방법".

use bagent::catalog::Catalog;
use bagent::fakes;
use bagent::graph::{NullGraph, SharedGraph, StaticGraph};
use bagent::instruction::Format;
use bagent::planner::{Agent, Core, LlmDecider, PriorDecider};
use bagent::relay::{self, Mode, RelayCfg};
use bagent::setup::{build_decider, build_graph, build_llm, load_catalog, planner_cfg};
use bagent::trace::{self, Tracer};
use bagent::util::Args;
use serde_json::{json, Value};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const HELP: &str = r#"bagent — BEHAVIOR 2026 상위 계획 에이전트 + 평가기↔VLA 중계기

  relay        중계기. --listen 0.0.0.0:8000 --upstream 127.0.0.1:8100 --mode agent|passthrough|fixed|schedule
               [--prompt 문장(fixed)] [--schedule 표.json] [--format subtask|task|purpose|metric]
               [--llm kau|oracle|http://…/v1] [--decider llm|prior] [--graph none|http://…|노드.json]
               [--task 과제이름] [--max-steps N] [--no-pause] [--no-images] [--trace-dir 폴더] [--once]
               [--llm-up-cmd …] [--llm-down-cmd …] [--llm-health http://…/health]  (로컬 LLM 서버 올리고 내리기)
  sim          가짜 세계에서 에이전트 한 판. --scenario radio|trash [--llm oracle|kau|URL] [--decider llm|prior]
               [--format …] [--p-success 0.9] [--seed 0] [--episodes 1] [--trace-dir 폴더] [--no-images]
  replay       기록 재생. <trace.jsonl> [--verify] [--html 파일]
  build-assets 과제 카드 만들기. [--root 저장소 뿌리(기본: 이 크레이트의 ../../..)] [--out assets/tasks.json] [--threads 16]
  render       과제 참고 순서를 형식 4가지로 찍기(토큰 추정 포함). --task turning_on_radio
  schedule     시연 주석 한 판 → 스텝별 문장 표. --episode 주석.json [--format subtask] [--out 표.json]
  mock-llm     가짜 OpenAI 호환 서버. --listen 127.0.0.1:8091 [--delay-ms 0]
  fake-pi      가짜 VLA 서버. --listen 127.0.0.1:8100
  bench        가짜 평가기로 왕복 시간 재기. --target 127.0.0.1:8000 [--n 300] [--rgbd] [--batch 1]
  bench-local  한 프로세스 안에서 직접 연결 vs 중계기 비교. [--n 300] [--rgbd] [--mode passthrough|fixed|agent]
  llm-check    같은 계획 요청을 여러 번 보내 지연·결정론 확인. [--llm kau] [--n 3] [--scenario radio]
  link         평가기 연결(평가기 안 VLA ↔ 계획기, 관측은 세기만 — scenemap 에 넣는 판은 src/sim/integ/simlink).
               --listen 0.0.0.0:7801 [--no-planner] [--prompt-mode task|subtask] [--stage external|vote|off]
               [--pose integrate|corrected] [--head-gap 6] [--wrist-every 0] [--settle-ms 400] [--once]
               + relay 와 같은 계획기 인자(--llm --graph --decider --format --task --max-steps --trace-dir)
  link-bench   가짜 접착부(Rust)로 link 지연 재기. --target 127.0.0.1:7801 [--n 600] [--task turning_on_radio]
공통: --catalog assets/tasks.json (기본: 크레이트 assets)  키는 환경변수로만(KAU_API_KEY)"#;

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let cmd = raw.first().cloned().unwrap_or_else(|| "help".into());
    let a = Args::parse(raw.into_iter().skip(1));
    let r = match cmd.as_str() {
        "relay" => cmd_relay(&a),
        "sim" => cmd_sim(&a),
        "replay" => cmd_replay(&a),
        "build-assets" => cmd_build_assets(&a),
        "render" => cmd_render(&a),
        "schedule" => cmd_schedule(&a),
        "mock-llm" => {
            let l = TcpListener::bind(a.str_or("listen", "127.0.0.1:8091")).map_err(|e| e.to_string());
            l.map(|l| {
                eprintln!("[mock-llm] {}", l.local_addr().unwrap());
                fakes::serve_mock_llm(l, a.num("delay-ms", 0))
            })
        }
        "fake-pi" => {
            let l = TcpListener::bind(a.str_or("listen", "127.0.0.1:8100")).map_err(|e| e.to_string());
            l.map(|l| {
                eprintln!("[fake-pi] {}", l.local_addr().unwrap());
                fakes::serve_fake_pi(l, None, false)
            })
        }
        "bench" => cmd_bench(&a),
        "bench-local" => cmd_bench_local(&a),
        "llm-check" => cmd_llm_check(&a),
        "link" => cmd_link(&a),
        "link-bench" => cmd_link_bench(&a),
        "bench-image" => {
            // 경계 스냅숏 비용: 머리 720²(RGBA) → RGB 복사 → 448² 축소 → JPEG, 손목 480² → 224²
            let n = a.num("n", 50usize);
            let mut rgba = vec![0u8; 720 * 720 * 4];
            let mut rng = bagent::util::Rng::new(3);
            for p in rgba.iter_mut() {
                *p = rng.next_u64() as u8;
            }
            let mut t = Vec::new();
            for _ in 0..n {
                let t0 = std::time::Instant::now();
                let px: Vec<u8> = rgba.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect();
                let img = bagent::image::Rgb { w: 720, h: 720, px };
                let j = img.downscale(448).jpeg(80);
                let w = bagent::image::Rgb { w: 480, h: 480, px: vec![128; 480 * 480 * 3] }.downscale(224).jpeg(80);
                t.push(t0.elapsed().as_micros() as u32);
                std::hint::black_box((j, w));
            }
            println!("머리+손목1 스냅숏(무작위 영상, 최악) {}", summarize_us(&t));
            Ok(())
        }
        _ => {
            println!("{HELP}");
            Ok(())
        }
    };
    if let Err(e) = r {
        eprintln!("오류: {e}");
        std::process::exit(1);
    }
}

fn default_trace_dir(kind: &str) -> PathBuf {
    let t = bagent::util::unix_ms();
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runs").join(format!("{kind}_{t}"))
}

fn cmd_relay(a: &Args) -> Result<(), String> {
    let catalog = Arc::new(load_catalog(a)?);
    let mode = match a.str_or("mode", "agent").as_str() {
        "passthrough" => Mode::Passthrough,
        "fixed" => Mode::Fixed(a.get("prompt").ok_or("--mode fixed 는 --prompt 필요")?.to_string()),
        "schedule" => {
            let p = a.get("schedule").ok_or("--mode schedule 은 --schedule 표.json 필요")?;
            let v: Vec<(u64, String)> = serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            Mode::Schedule(v)
        }
        "agent" => Mode::Agent,
        o => return Err(format!("--mode 를 모름: {o}")),
    };
    let mut cfg = RelayCfg::new(&a.str_or("listen", "0.0.0.0:8000"), &a.str_or("upstream", "127.0.0.1:8100"), mode.clone(), catalog.clone());
    cfg.task_override = a.get("task").map(|s| s.to_string());
    cfg.max_steps_override = a.get("max-steps").and_then(|s| s.parse().ok());
    cfg.pause = !a.flag("no-pause");
    cfg.stream = !a.flag("no-stream");
    cfg.images = !a.flag("no-images");
    cfg.once = a.flag("once");
    cfg.latency_every = a.num("latency-every", 1000);
    cfg.trace_dir = if a.flag("no-trace") { None } else { Some(a.get("trace-dir").map(PathBuf::from).unwrap_or_else(|| default_trace_dir("relay"))) };
    if matches!(mode, Mode::Agent) {
        let pcfg = planner_cfg(a)?;
        // 사용자 결정(09-29): VLA 에 숫자 명령을 쓰지 않는다. 실제 평가 경로(중계기 agent)에서는 받지 않는다.
        if pcfg.format == Format::Metric && !a.flag("allow-metric-experiment") {
            return Err("relay agent 모드에서 --format metric 은 쓰지 않는다(숫자 명령 금지). 실험이면 --allow-metric-experiment 를 같이 줘라".into());
        }
        cfg.image_side = pcfg.image_side;
        cfg.jpeg_quality = pcfg.jpeg_quality;
        // 시작할 때 한 번 만들어 보아 설정 오류를 먼저 잡는다
        build_decider(a)?;
        let a2 = a.clone();
        let cat2 = catalog.clone();
        cfg.factory = Some(Arc::new(move |env, task| {
            let core = Core::new(pcfg.clone(), task.clone(), &cat2, build_graph(&a2), env);
            let decider = build_decider(&a2).unwrap_or_else(|_| Box::new(PriorDecider));
            Agent::new(core, decider)
        }));
    }
    if let Some(d) = &cfg.trace_dir {
        eprintln!("[relay] 기록: {}", d.display());
    }
    relay::run(cfg).map_err(|e| e.to_string())
}

fn cmd_sim(a: &Args) -> Result<(), String> {
    let catalog = load_catalog(a)?;
    let scn = a.str_or("scenario", "radio");
    let seed: u64 = a.num("seed", 0);
    let task_name = a.get("task").map(|s| s.to_string()).unwrap_or_else(|| bagent::mockworld::World::task_name(&scn).to_string());
    let task = catalog.task_by_name(&task_name).ok_or_else(|| format!("카탈로그에 과제 없음: {task_name}"))?.clone();
    let pcfg = planner_cfg(a)?;
    let graph = SharedGraph::default();
    let core = Core::new(pcfg.clone(), task.clone(), &catalog, Box::new(graph.clone()), 0);
    let mut agent = Agent::new(core, build_decider(a)?);
    let dir = a.get("trace-dir").map(PathBuf::from).unwrap_or_else(|| default_trace_dir(&format!("sim_{scn}")));
    let tracer = Tracer::create(&dir).map_err(|e| e.to_string())?;
    agent.set_tracer(Some(tracer.clone()));
    let episodes: u32 = a.num("episodes", 1);
    let mut results = Vec::new();
    for ep in 0..episodes {
        let mut w = bagent::mockworld::World::scenario(&scn, seed + ep as u64, a.num("p-success", 0.9)).ok_or("--scenario 는 radio|trash")?;
        let max = a.num("max-steps", task.max_steps.max(3000));
        let r = bagent::mockworld::run_episode(&mut w, &mut agent, &graph, Default::default(), max, ep, pcfg.send_images);
        tracer.event(0, "episode_end", json!({"episode": ep, "result": r}));
        println!("판 {ep}: 성공 {} / {} 스텝 / 결정 {} / LLM {}회 / 대체 {}", r.success, r.steps, r.decisions, r.llm_calls, r.fallbacks);
        for p in &r.prompts {
            println!("   → {p}");
        }
        for l in &r.world_log {
            println!("   · {l}");
        }
        results.push(r);
    }
    tracer.event(0, "agent_stats", json!(agent.core.stats));
    tracer.flush();
    println!("기록: {}", dir.join("trace.jsonl").display());
    Ok(())
}

fn cmd_replay(a: &Args) -> Result<(), String> {
    let path = PathBuf::from(a.pos.first().ok_or("기록 파일(trace.jsonl)을 주세요")?);
    let events = trace::read(&path)?;
    if let Some(h) = a.get("html") {
        let out = if h == "1" || h == "true" { path.with_file_name("player.html") } else { PathBuf::from(h) };
        std::fs::write(&out, trace::export_html(&events, &format!("에이전트 재생 — {}", path.display()))).map_err(|e| e.to_string())?;
        println!("재생기: {}", out.display());
    }
    if a.flag("verify") {
        let catalog = load_catalog(a)?;
        let rep = bagent::replay::verify(&events, &catalog, path.parent().unwrap_or(Path::new(".")))?;
        for d in &rep.diffs {
            println!("다름: {d}");
        }
        println!("재생 결정 {}/{} 일치, 요청 지문 다름 {}건", rep.same, rep.total, rep.fingerprint_mismatches);
        return if rep.same == rep.total { Ok(()) } else { Err("재생 결과가 기록과 다름".into()) };
    }
    if a.get("html").is_none() {
        println!("{}", trace::timeline(&events));
    }
    Ok(())
}

// 저장소 뿌리(BEHAVIOR-1K/·data/ 가 있는 곳) = 이 크레이트(src/agent/planner)의 ../../..
fn repo_root() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    p.canonicalize().unwrap_or(p)
}

fn cmd_build_assets(a: &Args) -> Result<(), String> {
    let root = a.get("root").map(PathBuf::from).unwrap_or_else(repo_root);
    let t0 = std::time::Instant::now();
    let c = bagent::catalog::build(&root, a.num("threads", 16))?;
    let out = a.get("out").map(PathBuf::from).unwrap_or_else(Catalog::default_path);
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    std::fs::write(&out, serde_json::to_string_pretty(&c).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let with = c.tasks.iter().filter(|t| t.n_episodes > 0).count();
    let mut freqs: Vec<f64> = c.tasks.iter().filter(|t| t.n_episodes > 0).map(|t| t.skill_seq_top_freq).collect();
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let objf: Vec<f64> = {
        let mut v: Vec<f64> = c.tasks.iter().filter(|t| t.n_episodes > 0).map(|t| t.priors.iter().find(|p| p.kind == "top").map(|p| p.freq).unwrap_or(0.0)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    println!(
        "과제 {}개(주석 있는 것 {with}개), 단계 종류 {}개, {:.1} s → {}",
        c.tasks.len(),
        c.skills.len(),
        t0.elapsed().as_secs_f64(),
        out.display()
    );
    println!(
        "가장 흔한 순서 비율 중앙: 단계 이름만 {:.2}, 물체 종류까지 {:.2}",
        bagent::util::percentile(&freqs, 0.5),
        bagent::util::percentile(&objf, 0.5)
    );
    Ok(())
}

fn cmd_render(a: &Args) -> Result<(), String> {
    let c = load_catalog(a)?;
    let t = c.task_by_name(&a.str_or("task", "turning_on_radio")).ok_or("과제 없음")?;
    println!("과제 {} — {}", t.name, t.prompt);
    let Some(p) = t.priors.first() else { return Ok(()) };
    for s in &p.steps {
        let ins = bagent::instruction::Instruction {
            skill: s.skill.clone(),
            objects: s.objects.clone(),
            names: s.objects.clone(),
            spatial: s.spatial.clone(),
            memory: s.memory.clone(),
            purpose: format!("progress the task '{}'", t.name.replace('_', " ")),
            expected: String::new(),
            metric: Some(bagent::instruction::Metric { forward_m: 1.5, left_m: -0.3, turn_deg: 20.0 }),
            budget_steps: c.budget(&s.skill),
        };
        for f in [Format::Task, Format::Subtask, Format::Purpose, Format::Metric] {
            let r = ins.render(f, &t.prompt, 90);
            println!("{:>8} [{:>3}] {}", f.as_str(), bagent::util::estimate_tokens(&r), r);
        }
    }
    Ok(())
}

fn cmd_schedule(a: &Args) -> Result<(), String> {
    let p = a.get("episode").ok_or("--episode 주석.json")?;
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let fmt = Format::parse(&a.str_or("format", "subtask")).ok_or("--format")?;
    let mut out: Vec<(u64, String)> = Vec::new();
    for s in v["skill_annotation"].as_array().ok_or("skill_annotation 없음")? {
        let start = s["frame_duration"][0].as_u64().unwrap_or(0);
        let skill = s["skill_description"][0].as_str().unwrap_or("").to_string();
        let objs: Vec<String> = s["object_id"][0].as_array().map(|x| x.iter().map(|o| o.as_str().map(bagent::util::display_name).unwrap_or_default()).collect()).unwrap_or_default();
        let memory = s["memory_prefix"][0].as_str().map(|m| m.to_string());
        let spatial: Vec<String> = s["spatial_prefix"][0].as_array().map(|x| x.iter().map(|o| o.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
        let ins = bagent::instruction::Instruction { skill, names: objs.clone(), objects: objs, spatial, memory, budget_steps: 0, ..Default::default() };
        out.push((start, ins.render(fmt, v["task_name"].as_str().unwrap_or(""), 90)));
    }
    let js = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?;
    match a.get("out") {
        Some(o) => std::fs::write(o, js).map_err(|e| e.to_string())?,
        None => println!("{js}"),
    }
    Ok(())
}

fn summarize_us(v: &[u32]) -> String {
    if v.is_empty() {
        return "-".into();
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    let p = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize] as f64 / 1000.0;
    format!("p50 {:.3} ms, p90 {:.3} ms, p99 {:.3} ms, max {:.3} ms (n={})", p(0.5), p(0.9), p(0.99), s[s.len() - 1] as f64 / 1000.0, s.len())
}

fn spec_from(a: &Args) -> fakes::ObsSpec {
    fakes::ObsSpec { batch: a.num("batch", 1), rgbd: a.flag("rgbd"), robot: a.str_or("robot", "robot_r1"), task_id: a.num("task-id", 0) }
}

fn cmd_bench(a: &Args) -> Result<(), String> {
    let spec = spec_from(a);
    let log = fakes::run_fake_eval(&a.str_or("target", "127.0.0.1:8000"), &spec, a.num("n", 300), a.num("episodes", 1), false).map_err(|e| e.to_string())?;
    let warm = a.num("warmup", 20usize).min(log.rtt_us.len());
    println!("관측 {} 바이트, 왕복 {}", log.bytes_per_obs, summarize_us(&log.rtt_us[warm..]));
    Ok(())
}

fn cmd_bench_local(a: &Args) -> Result<(), String> {
    let spec = spec_from(a);
    let n: u64 = a.num("n", 300);
    let catalog = Arc::new(load_catalog(a).unwrap_or_default());
    // 가짜 VLA
    let pl = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let paddr = pl.local_addr().unwrap().to_string();
    std::thread::spawn(move || fakes::serve_fake_pi(pl, None, false));
    // 중계기
    let rl = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let raddr = rl.local_addr().unwrap().to_string();
    let mode = match a.str_or("mode", "passthrough").as_str() {
        "fixed" => Mode::Fixed("pick up radio from coffee table".into()),
        "agent" => Mode::Agent,
        _ => Mode::Passthrough,
    };
    let mut cfg = RelayCfg::new(&raddr, &paddr, mode.clone(), catalog.clone());
    cfg.latency_every = 0;
    cfg.stream = !a.flag("no-stream");
    if matches!(mode, Mode::Agent) {
        let a2 = a.clone();
        let cat2 = catalog.clone();
        let pcfg = planner_cfg(a)?;
        cfg.factory = Some(Arc::new(move |env, task| {
            let core = Core::new(pcfg.clone(), task.clone(), &cat2, Box::new(NullGraph), env);
            Agent::new(core, build_decider(&a2).unwrap_or_else(|_| Box::new(PriorDecider)))
        }));
    }
    let cfg2 = cfg.clone();
    std::thread::spawn(move || relay::run_listener(rl, cfg2));
    let warm = 20usize;
    let rounds: usize = a.num("rounds", 3);
    let mut direct = Vec::new();
    let mut via = Vec::new();
    let mut bytes = 0;
    for _ in 0..rounds {
        let d = fakes::run_fake_eval(&paddr, &spec, n, 1, false).map_err(|e| e.to_string())?;
        bytes = d.bytes_per_obs;
        direct.extend_from_slice(&d.rtt_us[warm.min(d.rtt_us.len())..]);
        let v = fakes::run_fake_eval(&raddr, &spec, n, 1, false).map_err(|e| e.to_string())?;
        via.extend_from_slice(&v.rtt_us[warm.min(v.rtt_us.len())..]);
    }
    println!("관측 {bytes} 바이트, 모드 {}", a.str_or("mode", "passthrough"));
    println!("직접   : {}", summarize_us(&direct));
    println!("중계기 : {}", summarize_us(&via));
    let med = |v: &Vec<u32>| {
        let mut s = v.clone();
        s.sort_unstable();
        s[s.len() / 2] as f64 / 1000.0
    };
    println!("중앙값 차이(중계기 추가 지연): {:.3} ms", med(&via) - med(&direct));
    Ok(())
}

fn cmd_llm_check(a: &Args) -> Result<(), String> {
    let catalog = load_catalog(a)?;
    let scn = a.str_or("scenario", "radio");
    let task = catalog.task_by_name(bagent::mockworld::World::task_name(&scn)).ok_or("과제 없음")?.clone();
    let pcfg = planner_cfg(a)?;
    let n: usize = a.num("n", 3);
    let mut outs = Vec::new();
    for i in 0..n {
        let graph = SharedGraph::default();
        let w = bagent::mockworld::World::scenario(&scn, 0, 1.0).unwrap();
        *graph.0.lock().unwrap() = StaticGraph { nodes: w.graph_nodes(), name: "mock".into() };
        let core = Core::new(pcfg.clone(), task.clone(), &catalog, Box::new(graph), 0);
        let mut agent = Agent::new(core, Box::new(LlmDecider { llm: build_llm(a)? }));
        let dir = default_trace_dir("llmcheck");
        let tr = Tracer::create(&dir).map_err(|e| e.to_string())?;
        agent.set_tracer(Some(tr.clone()));
        let mut sess = bagent::session::EnvSession::new(0, Default::default(), 30.0, task.max_steps, &task.prompt);
        let t = sess.on_obs([0.0; 3], [0.1, 0.1]).unwrap();
        let images = if pcfg.send_images { vec![("head".to_string(), bagent::image::Rgb::pattern(64, 64, 7).jpeg(70))] } else { vec![] };
        let ev = sess.event(t, [0.1, 0.1], images);
        let t0 = std::time::Instant::now();
        let d = agent.decide(&ev);
        let ms = t0.elapsed().as_millis();
        tr.flush();
        println!("#{i}: {ms} ms, LLM {}회, 도구 {}회(오류 {}), 대체 {} → {}", agent.core.stats.llm_calls, agent.core.stats.tool_calls, agent.core.stats.tool_errors, agent.core.stats.fallbacks, d.key());
        outs.push(d.key());
    }
    let same = outs.iter().all(|o| *o == outs[0]);
    println!("{n}번 결정 모두 같음: {same}");
    Ok(())
}

fn cmd_link(a: &Args) -> Result<(), String> {
    let catalog = Arc::new(load_catalog(a)?);
    let cfg = bagent::link::cfg_from_args(a, catalog)?;
    if let Some(d) = &cfg.trace_dir {
        eprintln!("[link] 기록: {}", d.display());
    }
    bagent::link::run(cfg, &|| Box::new(bagent::link::NullSink::default())).map_err(|e| e.to_string())
}

/// 가짜 접착부: 평가기와 같은 크기의 관측(머리 720² RGBA + 깊이 f32, 손목 480²)을 link 가 요청하는 대로 보낸다.
fn cmd_link_bench(a: &Args) -> Result<(), String> {
    use bagent::link::{self, Client, Frame, Hello, CamSpec};
    let n: u64 = a.num("n", 600);
    let hello = Hello {
        task: a.str_or("task", "turning_on_radio"),
        num_envs: 1,
        hz: 30.0,
        max_steps: Some(n),
        cams: vec![
            CamSpec { name: "head".into(), w: 720, h: 720, k: [306.0, 306.0, 360.0, 360.0] },
            CamSpec { name: "left_wrist".into(), w: 480, h: 480, k: [388.6639, 388.6639, 240.0, 240.0] },
            CamSpec { name: "right_wrist".into(), w: 480, h: 480, k: [388.6639, 388.6639, 240.0, 240.0] },
        ],
        crp_order: vec!["left_wrist".into(), "right_wrist".into(), "head".into()],
        stage_count: a.num("stage-count", 5),
        client: "bagent link-bench".into(),
        ..Default::default()
    };
    let (mut c, ack) = Client::connect(&a.str_or("target", "127.0.0.1:7801"), &hello).map_err(|e| e.to_string())?;
    eprintln!("[link-bench] HELLO_ACK {ack}");
    c.reset().map_err(|e| e.to_string())?;
    let head_rgba = vec![90u8; 720 * 720 * 4];
    let head_d: Vec<u8> = (0..720 * 720).flat_map(|i| (1.0f32 + (i % 720) as f32 * 0.001).to_le_bytes()).collect();
    let wr_rgba = vec![60u8; 480 * 480 * 4];
    let wr_d: Vec<u8> = (0..480 * 480).flat_map(|_| 0.5f32.to_le_bytes()).collect();
    let frames = |want: u16| -> Vec<Frame> {
        let mut v = Vec::new();
        for cam in 0..3u8 {
            let (s, rgba, d) = if cam == 0 { (720, &head_rgba, &head_d) } else { (480, &wr_rgba, &wr_d) };
            if want & link::want_rgb(cam) != 0 {
                v.push(Frame { cam, kind: link::KIND_RGBA8, h: s, w: s, data: rgba.clone() });
            }
            if want & link::want_depth(cam) != 0 {
                v.push(Frame { cam, kind: link::KIND_DEPTH_F32, h: s, w: s, data: d.clone() });
            }
        }
        v
    };
    let mut t_small = Vec::new();
    let mut t_frames = Vec::new();
    let mut t_hold = Vec::new();
    let mut decisions = 0;
    let mut frames_sent = 0u64;
    for step in 0..n {
        let mut p = vec![0f32; 61];
        // 앞으로 천천히, 가끔 돌기, 250 스텝마다 그리퍼 닫기/열기
        p[0] = if (step / 100) % 2 == 0 { 0.3 } else { 0.0 };
        p[2] = if step % 200 > 150 { 0.2 } else { 0.0 };
        let g = if (step / 250) % 2 == 0 { 0.045 } else { 0.01 };
        p[24] = g;
        p[25] = g;
        p[49] = 0.045;
        p[50] = 0.045;
        let crp: Vec<f32> = [[0.1f32, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0], [0.1, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0], [0.05, 0.0, 1.6, 0.0, 0.0, 0.0, 1.0]].concat();
        let t0 = std::time::Instant::now();
        let (bits, wait, decs) = c.step(0, step, &p, &crp, &frames).map_err(|e| e.to_string())?;
        let us = t0.elapsed().as_micros() as u32;
        if wait {
            t_hold.push(us);
        } else if bits != 0 {
            t_frames.push(us);
        } else {
            t_small.push(us);
        }
        frames_sent += bits.count_ones() as u64;
        for d in decs {
            if d.get("kind").and_then(|k| k.as_str()) != Some("stage") {
                decisions += 1;
            }
            eprintln!("[link-bench] step {step}: {d}");
        }
    }
    c.bye().map_err(|e| e.to_string())?;
    println!("스텝 {n}, 결정 {decisions}, 보낸 영상 {frames_sent}");
    println!("  영상 없는 스텝 {}", summarize_us(&t_small));
    println!("  영상 스텝     {}", summarize_us(&t_frames));
    println!("  경계(대기) 스텝 {}", summarize_us(&t_hold));
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn repo_root_is_this_checkout() {
        let r = super::repo_root();
        assert!(r.join("src/agent/planner/Cargo.toml").is_file(), "{}", r.display());
    }
}

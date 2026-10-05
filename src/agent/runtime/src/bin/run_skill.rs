//! `run-skill` — 스킬 한 판(스킬과 무관한 실행 명령).
//!
//! ```text
//! run-skill --skill explore --policy llm|baseline [--addr 127.0.0.1:8771 | --mock <gt.pgm> --gt <gt.json> [--start x,y,yaw_deg]] --out <dir>
//!           [--max-calls N] [--max-sim-s S] [--max-wall-s S] [--keep 3] [--temperature T] [--thinking] [--no-nudge]
//!           [--style full|compact] [--run-id ID] [--task NAME] [--skill-dir DIR] [--common FILE]
//! ```
//! - `--skill <이름>` = `src/agent/skills/<이름>/`(또는 `--skill-dir`). 프롬프트(system.md·task.md·도구 설명)와 skill.json 을 실행할 때 읽는다.
//!   공통 조각은 `<skill dir>/../../prompts/common.md`(또는 `--common`).
//! - `--policy baseline` 은 skill.json `baseline` 호출(LLM 없음). 그 이름(explore 는 `frontier`)도 받는다.
//! - LLM: KAU (KAU_BASE_URL / KAU_MODEL / KAU_API_KEY 환경변수 — `set -a; . ~/.config/behavior-2026/kau.env; set +a`).
//! - `--mock`: 가짜 집(LLM 에게 안 보이는 덮음 기준은 `move_robot::mock_eval`).

use agent_runtime::{run, Config, Policy, Prompts, SkillSpec};
use move_robot::link::{Backend, TcpSim};
use std::path::PathBuf;

fn flag(a: &[String], n: &str) -> Option<String> {
    a.iter().position(|x| x == n).and_then(|i| a.get(i + 1).cloned())
}

fn die(msg: String) -> ! {
    eprintln!("{msg}");
    std::process::exit(2)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let agent = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let skill_dir = match (flag(&a, "--skill-dir"), flag(&a, "--skill")) {
        (Some(d), _) => PathBuf::from(d),
        (None, Some(s)) => agent.join("skills").join(s),
        (None, None) => die("--skill <name> (folder src/agent/skills/<name>) or --skill-dir <dir>".into()),
    };
    let skill_dir = skill_dir.canonicalize().unwrap_or_else(|e| die(format!("{}: {e}", skill_dir.display())));
    let spec = SkillSpec::load(&skill_dir).unwrap_or_else(|e| die(e));
    let common = flag(&a, "--common").map(PathBuf::from).unwrap_or(skill_dir.join("../../prompts/common.md"));
    let prompts = Prompts::load(&skill_dir, &common, &spec.tool_files()).unwrap_or_else(|e| die(format!("prompts: {e}")));
    let mut cfg = Config::new(&spec);
    let bname = spec.baseline.as_ref().map(|b| b.name.clone());
    cfg.policy = match flag(&a, "--policy").as_deref() {
        Some("llm") => Policy::Llm { thinking: a.iter().any(|x| x == "--thinking") },
        Some("baseline") | None => Policy::Baseline,
        Some(p) if Some(p) == bname.as_deref() => Policy::Baseline,
        Some(p) => die(format!("--policy {p}: llm | baseline{}", bname.map(|b| format!(" | {b}")).unwrap_or_default())),
    };
    cfg.out_dir = PathBuf::from(flag(&a, "--out").unwrap_or_else(|| format!("{}_out", spec.name)));
    let num = |n: &str| flag(&a, n).map(|v| v.parse::<f64>().unwrap_or_else(|e| die(format!("{n} {v}: {e}"))));
    if let Some(v) = num("--max-calls") {
        cfg.limits.max_calls = v as usize;
    }
    if let Some(v) = num("--max-sim-s") {
        cfg.limits.max_sim_s = v;
    }
    if let Some(v) = num("--max-wall-s") {
        cfg.limits.max_wall_s = v;
    }
    if let Some(v) = num("--keep") {
        cfg.limits.keep_pairs = v as usize;
    }
    if let Some(v) = num("--temperature") {
        cfg.limits.temperature = v;
    }
    if a.iter().any(|x| x == "--no-nudge") {
        cfg.nudge = false;
    }
    cfg.run_id = flag(&a, "--run-id").unwrap_or_else(|| cfg.out_dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    cfg.task = flag(&a, "--task").unwrap_or_else(|| spec.name.clone());
    let style = flag(&a, "--style");
    let mut backend: Box<dyn Backend> = if let Some(pgm) = flag(&a, "--mock") {
        let gt = flag(&a, "--gt").unwrap_or_else(|| die("--gt <gt.json> with --mock".into()));
        let start = flag(&a, "--start").map(|st| {
            // x,y,yaw_deg (world)
            let v: Vec<f64> = st.split(',').map(|x| x.trim().parse().unwrap_or_else(|e| die(format!("--start {st}: {e}")))).collect();
            [v[0], v[1], v[2].to_radians()]
        });
        let mut m = move_robot::mock_eval::mock_from_gt(&pgm, &gt, start).unwrap_or_else(|e| die(e));
        if style.as_deref() == Some("compact") {
            m.robot.nav.style = move_robot::robot_nav_style_compact();
        }
        Box::new(m)
    } else {
        let mut t = TcpSim::new(&flag(&a, "--addr").unwrap_or_else(|| move_robot::link::DEFAULT_ADDR.into()));
        t.timeout = std::time::Duration::from_secs(600);
        Box::new(t)
    };
    let mut llm_box: Option<Box<dyn bagent::llm::Llm>> = match cfg.policy {
        Policy::Llm { .. } => Some(Box::new(bagent::llm::HttpLlm::kau(180).unwrap_or_else(|e| die(e)))),
        _ => None,
    };
    eprintln!("[run-skill] skill {} policy {:?} prompts {} ({}) out {}", spec.name, cfg.policy, prompts.version, prompts.sha, cfg.out_dir.display());
    let s = run(&spec, &cfg, backend.as_mut(), llm_box.as_deref_mut().map(|l| l as &mut dyn bagent::llm::Llm), &prompts);
    println!("{}", serde_json::to_string_pretty(&s).unwrap());
}

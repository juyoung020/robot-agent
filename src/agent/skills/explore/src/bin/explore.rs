//! `explore` — 스킬 explore 한 판.
//!
//! ```text
//! explore --policy llm|frontier [--addr 127.0.0.1:8771 | --mock <gt.pgm> --gt <gt.json>] --out <dir>
//!         [--max-calls 80] [--max-sim-s 900] [--keep 3] [--thinking] [--style full|compact] [--run-id ID]
//! ```
//! 프롬프트: <skill dir>/system.md·task.md·tool.md + src/agent/prompts/common.md (실행할 때 읽음; --skill-dir, --common 으로 바꿈).
//! LLM: KAU (KAU_BASE_URL / KAU_MODEL / KAU_API_KEY 환경변수 — `set -a; . ~/.config/behavior-2026/kau.env; set +a`).

use explore::{run, Config, Policy, Prompts};
use move_robot::link::{Backend, Mock, MockWorld, TcpSim};
use serde_json::Value;
use std::path::PathBuf;

fn flag(a: &[String], n: &str) -> Option<String> {
    a.iter().position(|x| x == n).and_then(|i| a.get(i + 1).cloned())
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let skill_dir = flag(&a, "--skill-dir").map(PathBuf::from).unwrap_or(here.clone());
    let common = flag(&a, "--common").map(PathBuf::from).unwrap_or(here.join("../../prompts/common.md"));
    let prompts = Prompts::load(&skill_dir, &common).unwrap_or_else(|e| {
        eprintln!("prompts: {e}");
        std::process::exit(2)
    });
    let mut cfg = Config::default();
    cfg.policy = match flag(&a, "--policy").as_deref() {
        Some("llm") => Policy::Llm { thinking: a.iter().any(|x| x == "--thinking") },
        _ => Policy::Frontier,
    };
    cfg.out_dir = PathBuf::from(flag(&a, "--out").unwrap_or_else(|| "explore_out".into()));
    if let Some(v) = flag(&a, "--max-calls") {
        cfg.max_calls = v.parse().unwrap();
    }
    if let Some(v) = flag(&a, "--max-sim-s") {
        cfg.max_sim_s = v.parse().unwrap();
    }
    if let Some(v) = flag(&a, "--max-wall-s") {
        cfg.max_wall_s = v.parse().unwrap();
    }
    if let Some(v) = flag(&a, "--keep") {
        cfg.keep_pairs = v.parse().unwrap();
    }
    if let Some(v) = flag(&a, "--temperature") {
        cfg.temperature = v.parse().unwrap();
    }
    if a.iter().any(|x| x == "--no-nudge") {
        cfg.nudge = false;
    }
    cfg.run_id = flag(&a, "--run-id").unwrap_or_else(|| cfg.out_dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    cfg.task = flag(&a, "--task").unwrap_or_else(|| "explore".into());
    let style = flag(&a, "--style");
    let mut backend: Box<dyn Backend> = if let Some(pgm) = flag(&a, "--mock") {
        let gt: Value = serde_json::from_str(&std::fs::read_to_string(flag(&a, "--gt").expect("--gt <gt.json> with --mock")).unwrap()).unwrap();
        let res = gt["res"].as_f64().unwrap();
        let (ox, oy) = (gt["origin"][0].as_f64().unwrap(), gt["origin"][1].as_f64().unwrap());
        let world = MockWorld::from_pgm(&pgm, res, ox, oy).unwrap();
        let sp = &gt["start_pose_world"];
        let mut start = [sp[0].as_f64().unwrap(), sp[1].as_f64().unwrap(), sp[2].as_f64().unwrap()];
        if let Some(st) = flag(&a, "--start") {
            // x,y,yaw_deg (world)
            let v: Vec<f64> = st.split(',').map(|x| x.trim().parse().unwrap()).collect();
            start = [v[0], v[1], v[2].to_radians()];
        }
        let reference = reachable_reference(&world.floor, start);
        let mut m = Mock::with_world(world, start);
        m.robot.set_reference(reference);
        if style.as_deref() == Some("compact") {
            m.robot.nav.style = move_robot::robot_nav_style_compact();
        }
        Box::new(m)
    } else {
        let mut t = TcpSim::new(&flag(&a, "--addr").unwrap_or_else(|| move_robot::link::DEFAULT_ADDR.into()));
        t.timeout = std::time::Duration::from_secs(120);
        Box::new(t)
    };
    let mut llm_box: Option<Box<dyn bagent::llm::Llm>> = match cfg.policy {
        Policy::Llm { .. } => Some(Box::new(bagent::llm::HttpLlm::kau(180).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        }))),
        _ => None,
    };
    eprintln!("[explore] policy {:?} prompts {} ({}) out {}", cfg.policy, prompts.version, prompts.sha, cfg.out_dir.display());
    let s = run(&cfg, backend.as_mut(), llm_box.as_deref_mut().map(|l| l as &mut dyn bagent::llm::Llm), &prompts);
    println!("{}", serde_json::to_string_pretty(&s).unwrap());
}

/// 정답 기준: 바닥 칸 중 출발점에서 몸통 원(0.28 m)이 닿는 곳(벽에서 0.28 m 떨어진 칸으로 이어진 연결 성분)과 그 둘레 0.6 m
/// (카메라로 볼 수 있는 바닥). 값 1 = 기준 칸.
fn reachable_reference(floor: &move_robot::Grid, start: [f64; 3]) -> move_robot::Grid {
    let mut obst = floor.clone();
    for v in obst.cells.iter_mut() {
        *v = if *v == 1 { 0 } else { 100 };
    }
    let d = move_robot::map::obstacle_distance(&obst, 1.0);
    let (w, h) = (floor.w as i64, floor.h as i64);
    let mut reach = vec![false; floor.cells.len()];
    let (sx, sy) = floor.cell_of(start[0], start[1]);
    let mut q = std::collections::VecDeque::new();
    if let Some(i) = floor.idx(sx, sy) {
        reach[i] = true;
        q.push_back(i);
    }
    while let Some(i) = q.pop_front() {
        let (x, y) = ((i % floor.w) as i64, (i / floor.w) as i64);
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                continue;
            }
            let j = (ny * w + nx) as usize;
            if !reach[j] && floor.cells[j] == 1 && (d[j] >= 0.28 || (nx - sx).pow(2) + (ny - sy).pow(2) < 64) {
                reach[j] = true;
                q.push_back(j);
            }
        }
    }
    // 둘레 0.6 m 의 바닥(로봇이 서서 볼 수 있는 곳)
    let mut out = floor.clone();
    let k = (0.6 / floor.res) as i64;
    let mut dist = vec![i64::MAX; floor.cells.len()];
    let mut q = std::collections::VecDeque::new();
    for i in 0..reach.len() {
        if reach[i] {
            dist[i] = 0;
            q.push_back(i);
        }
    }
    while let Some(i) = q.pop_front() {
        if dist[i] >= k {
            continue;
        }
        let (x, y) = ((i % floor.w) as i64, (i / floor.w) as i64);
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                continue;
            }
            let j = (ny * w + nx) as usize;
            if floor.cells[j] == 1 && dist[j] == i64::MAX {
                dist[j] = dist[i] + 1;
                q.push_back(j);
            }
        }
    }
    for i in 0..out.cells.len() {
        out.cells[i] = (dist[i] != i64::MAX) as i8;
    }
    out
}

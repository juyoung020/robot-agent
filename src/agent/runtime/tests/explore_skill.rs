//! 런타임 + 스킬 explore 설정(skill.json) — Isaac Sim·LLM 없이.
use agent_runtime::{run, tools, Config, Policy, Prompts, SkillSpec};
use move_robot::link::{Mock, MockWorld};
use std::path::PathBuf;

fn skill_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../skills/explore").canonicalize().unwrap()
}

fn two_rooms() -> MockWorld {
    let res = 0.05;
    let (w, h) = (200usize, 120usize);
    let mut g = move_robot::Grid::new(res, 0.0, 0.0, w, h, 0);
    for y in 0..h {
        for x in 0..w {
            let (px, py) = ((x as f64 + 0.5) * res, (y as f64 + 0.5) * res);
            let inside = px > 0.2 && px < 9.8 && py > 0.2 && py < 5.8;
            let wall = (px - 5.0).abs() < 0.1 && !(2.5..3.5).contains(&py);
            g.cells[y * w + x] = (inside && !wall) as i8;
        }
    }
    MockWorld::new(g)
}

#[test]
fn explore_skill_loads_and_llm_sees_the_same_tool() {
    let d = skill_dir();
    let spec = SkillSpec::load(&d).unwrap();
    assert_eq!(spec.name, "explore");
    assert_eq!(spec.baseline.as_ref().unwrap().name, "frontier");
    let p = Prompts::load(&d, &d.join("../../prompts/common.md"), &spec.tool_files()).unwrap();
    assert!(p.version.starts_with("common-v"), "{}", p.version);
    let t = tools::by_name(&spec.tools[0].name).unwrap();
    // LLM 에게 보이는 도구 정의는 옛 스킬 실행기(definition_with)와 같다 — 모드 explore 는 안 보임
    assert_eq!(t.definition(&p.tool_descs[0], spec.tools[0].modes.as_deref()), move_robot::definition_with(&p.tool_descs[0]));
}

#[test]
fn baseline_explores_two_rooms_until_no_frontier() {
    let d = skill_dir();
    let spec = SkillSpec::load(&d).unwrap();
    let p = Prompts::load(&d, &d.join("../../prompts/common.md"), &spec.tool_files()).unwrap();
    let out = std::env::temp_dir().join(format!("agent_runtime_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let mut cfg = Config::new(&spec);
    cfg.policy = Policy::Baseline;
    cfg.out_dir = out.clone();
    let mut m = Mock::with_world(two_rooms(), [1.5, 3.0, 0.0]);
    let s = run(&spec, &cfg, &mut m, None, &p);
    assert_eq!(s["end"], "no_frontier", "{s}");
    assert_eq!(s["policy"], "frontier");
    assert!(s["calls"].as_u64().unwrap() >= 1);
    assert_eq!(m.world.as_ref().unwrap().contacts, 0);
    let dec = std::fs::read_to_string(out.join("decisions.jsonl")).unwrap();
    assert_eq!(dec.lines().count() as u64, s["calls"].as_u64().unwrap());
    let first: serde_json::Value = serde_json::from_str(dec.lines().next().unwrap()).unwrap();
    assert_eq!(first["mode"], "base:go_to");
    let _ = std::fs::remove_dir_all(&out);
}

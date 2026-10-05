//! 평가기 연결(link) 끝에서 끝까지: Rust 가짜 접착부 ↔ link(계획기 = 규칙 LLM, 관측 = 세기만).

use bagent::catalog::Catalog;
use bagent::graph::NullGraph;
use bagent::link::{self, Client, Frame, Hello, LinkCfg, ObsPacket, ObsSink, PromptMode, StageMode, CamSpec};
use bagent::planner::{Agent, Core, LlmDecider, PlannerCfg};
use serde_json::Value;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

fn catalog() -> Catalog {
    Catalog::load(&Catalog::default_path()).expect("assets/tasks.json")
}

/// 받은 것을 기록만 하는 내보내기(시험용)
#[derive(Clone, Default)]
struct Rec(Arc<Mutex<Vec<(u64, bool, usize, [f64; 3], bool)>>>);

impl ObsSink for Rec {
    fn push(&mut self, p: ObsPacket) {
        self.0.lock().unwrap().push((p.step, p.boundary, p.frames.len(), [p.base.x, p.base.y, p.base.yaw], p.cam_rel[0].is_some()));
    }
}

fn hello(stage_count: u32) -> Hello {
    Hello {
        task: "turning_on_radio".into(),
        num_envs: 1,
        hz: 30.0,
        max_steps: Some(3000),
        cams: vec![
            CamSpec { name: "head".into(), w: 8, h: 8, k: [3.4, 3.4, 4.0, 4.0] },
            CamSpec { name: "left_wrist".into(), w: 4, h: 4, k: [3.0, 3.0, 2.0, 2.0] },
            CamSpec { name: "right_wrist".into(), w: 4, h: 4, k: [3.0, 3.0, 2.0, 2.0] },
        ],
        crp_order: vec!["left_wrist".into(), "right_wrist".into(), "head".into()],
        stage_count,
        client: "test".into(),
        ..Default::default()
    }
}

fn start(prompt_mode: PromptMode, stage: StageMode, rec: Rec) -> (String, std::thread::JoinHandle<()>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    let cat = Arc::new(catalog());
    let mut cfg = LinkCfg::new(&addr, cat.clone());
    cfg.once = true;
    cfg.prompt_mode = prompt_mode;
    cfg.stage = stage;
    cfg.stats_every = 0;
    cfg.trace_dir = Some(std::env::temp_dir().join(format!("bagent_link_{}_{}", addr.replace([':', '.'], "_"), std::process::id())));
    let cat2 = cat.clone();
    cfg.factory = Some(Arc::new(move |env, task| {
        let core = Core::new(PlannerCfg::default(), task.clone(), &cat2, Box::new(NullGraph), env);
        Agent::new(core, Box::new(LlmDecider { llm: Box::new(bagent::fakes::oracle_llm()) }))
    }));
    let h = std::thread::spawn(move || {
        link::run_listener(l, cfg, &move || Box::new(rec.clone())).unwrap();
    });
    (addr, h)
}

fn frames(want: u16) -> Vec<Frame> {
    let mut v = Vec::new();
    for cam in 0..3u8 {
        let s = if cam == 0 { 8 } else { 4 };
        if want & link::want_rgb(cam) != 0 {
            v.push(Frame { cam, kind: link::KIND_RGBA8, h: s, w: s, data: vec![100; (s * s * 4) as usize] });
        }
        if want & link::want_depth(cam) != 0 {
            v.push(Frame { cam, kind: link::KIND_DEPTH_F32, h: s, w: s, data: 1.0f32.to_le_bytes().repeat((s * s) as usize) });
        }
    }
    v
}

/// 스텝 n 의 proprio: 앞 100 스텝 전진 후 정지(이동 멈춤), 400 스텝에서 왼 그리퍼 닫기.
fn proprio(step: u64) -> Vec<f32> {
    let mut p = vec![0f32; 61];
    p[0] = if step < 100 { 0.5 } else { 0.0 };
    let g = if step < 400 { 0.045 } else { 0.01 };
    p[24] = g;
    p[25] = g;
    p[49] = 0.045;
    p[50] = 0.045;
    p
}

fn crp() -> Vec<f32> {
    [[0.1f32, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0], [0.1, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0], [0.05, 0.0, 1.6, 0.0, 0.0, 0.0, 1.0]].concat()
}

fn run(prompt_mode: PromptMode, stage: StageMode, n: u64) -> (Vec<(u64, bool, Vec<Value>, u16)>, Vec<(u64, bool, usize, [f64; 3], bool)>) {
    let rec = Rec::default();
    let (addr, h) = start(prompt_mode, stage, rec.clone());
    let (mut c, ack) = Client::connect(&addr, &hello(10)).unwrap();
    assert_eq!(ack["planner"], true);
    assert_eq!(ack["task"], "turning_on_radio");
    c.reset().unwrap();
    let mut log = Vec::new();
    for s in 0..n {
        let (bits, wait, decs) = c.step(0, s, &proprio(s), &crp(), &frames).unwrap();
        log.push((s, wait, decs, bits));
    }
    c.bye().unwrap();
    h.join().unwrap();
    let r = rec.0.lock().unwrap().clone();
    (log, r)
}

#[test]
fn link_task_mode_holds_decides_and_sends_keyframes() {
    let (log, rec) = run(PromptMode::Task, StageMode::External, 700);
    // 첫 스텝은 판 시작 경계: 기다리고, 모든 카메라를 보냈고, 결정이 왔다
    assert!(log[0].1, "첫 스텝 hold");
    assert_eq!(log[0].3, link::WANT_ALL);
    let d0 = log[0].2.iter().find(|d| d["kind"] != "stage").expect("판 시작 결정");
    assert_eq!(d0["trigger"], "episode_start");
    // 과제 단위 모드: VLA 문장은 바꾸지 않는다(prompt null), 단계 번호는 고정 입력(mode 1)
    assert!(d0["prompt"].is_null());
    assert_eq!(d0["stage_mode"], 1);
    assert_eq!(d0["stage"], 0);
    assert!(d0["text"].as_str().unwrap().contains("radio"));
    // 이동 멈춤(100 스텝 뒤)과 그리퍼 닫힘(400 스텝 + 10)에서 경계가 나고, 그 스텝만 기다린다
    let holds: Vec<u64> = log.iter().filter(|x| x.1).map(|x| x.0).collect();
    let decided: Vec<(u64, String)> = log
        .iter()
        .flat_map(|x| x.2.iter().filter(|d| d["kind"] != "stage").map(move |d| (x.0, d["trigger"].as_str().unwrap_or("").to_string())))
        .collect();
    assert!(decided.iter().any(|(_, t)| t == "settled"), "이동 멈춤 경계: {decided:?}");
    assert!(decided.iter().any(|(s, t)| t == "gripper_change" && *s >= 410 && *s <= 412), "그리퍼 경계: {decided:?}");
    assert!(holds.len() < 30, "기다린 스텝은 경계 근처만: {}", holds.len());
    // 영상은 매 스텝이 아니다(keyframe): 머리는 이동 중엔 자주, 멈추면 6 스텝마다
    let with_frames = log.iter().filter(|x| x.3 != 0).count();
    assert!(with_frames < 700 / 3, "영상 보낸 스텝 {with_frames}");
    let still: Vec<&(u64, bool, Vec<Value>, u16)> = log.iter().filter(|x| x.0 > 200 && x.0 < 380 && !x.1).collect();
    let still_frames = still.iter().filter(|x| x.3 != 0).count();
    assert!(still_frames >= still.len() / 7 && still_frames <= still.len() / 5 + 2, "멈춤 구간 영상 {still_frames}/{}", still.len());
    // 내보내기는 매 스텝(자세) 받고, 경계 표시와 카메라 자세가 있다
    assert_eq!(rec.len(), 700);
    assert!(rec[0].1 && rec[0].4);
    // 자세: 0.5 m/s × 99 스텝 / 30 Hz ≈ 1.65 m 전진(첫 스텝은 속도만 기억)
    let last = rec.last().unwrap().3;
    assert!((last[0] - 0.5 * 100.0 / 30.0).abs() < 0.02, "x = {}", last[0]);
    // 단계 번호: 경계 사이에도 진행률이 넘어가면 보냄(0 부터 오름)
    let stages: Vec<u64> = log.iter().flat_map(|x| x.2.iter().filter_map(|d| d["stage"].as_u64())).collect();
    assert!(stages.windows(2).all(|w| w[1] >= w[0]) || stages.len() < 2, "단계 {stages:?}");
}

#[test]
fn link_subtask_mode_sends_prompts() {
    let (log, _) = run(PromptMode::Subtask, StageMode::Off, 150);
    let d0 = log[0].2.iter().find(|d| d["kind"] != "stage").unwrap();
    let p = d0["prompt"].as_str().expect("문장");
    assert!(p.contains("radio"), "{p}");
    assert!(d0["stage"].is_null());
    assert!(log.iter().all(|x| x.2.iter().all(|d| d["kind"] != "stage")));
}

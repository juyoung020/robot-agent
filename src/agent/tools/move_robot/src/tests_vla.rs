//! VLA 실행기(POLICY 1.2·1.3·7.1·7.2) 시험: 가짜 LIMO([`crate::limo_mock`]) 위에서 끝까지.

use crate::limo::{self, LimoState};
use crate::limo_mock::{demo_scene, LimoMock, MockObj};
use crate::link::MockWorld;
use crate::map::Grid;
use crate::verify::{MemObject, ObjState};
use crate::vla::{self, route, Route, SkillKind, VlaParams};
use serde_json::{json, Value};

fn st(v: &Value) -> &str {
    v["status"].as_str().unwrap_or("?")
}

fn has_result_fields(v: &Value) {
    for k in ["status", "reason", "evidence", "steps", "min_clear_m", "contacts"] {
        assert!(v.get(k).is_some(), "result has no '{k}': {v}");
    }
}

#[test]
fn limo_limits_match_real_limits_json() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../robot/real_limits.json");
    let j: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for (i, (lo, hi)) in limo::ARM_LIM.iter().enumerate() {
        let r = &j[format!("omx_joint{}", i + 1)];
        assert!((r[0].as_f64().unwrap() - lo).abs() < 1e-6 && (r[1].as_f64().unwrap() - hi).abs() < 1e-6, "joint{} {r}", i + 1);
    }
    let g = &j["omx_gripper_joint_1"];
    assert!((g[0].as_f64().unwrap() - limo::GRIPPER_LIM.0).abs() < 1e-6 && (g[1].as_f64().unwrap() - limo::GRIPPER_LIM.1).abs() < 1e-6);
}

#[test]
fn fk_zero_pose_matches_rviz_tf() {
    // SIM_PORTING M0: base_footprint → omx_end_effector_link = (0.273, −0.002, 0.361)
    let p = limo::fk_eef(&[0.0; 5]);
    assert!((p[0] - 0.273).abs() < 0.002 && (p[1] + 0.002).abs() < 0.002 && (p[2] - 0.361).abs() < 0.002, "{p:?}");
    // 역기구학이 닿는 점으로 돌아온다
    let goal = [0.30, 0.10, 0.15];
    let q = vla::ik(goal, &limo::ARM_HOME);
    let f = limo::fk_eef(&q);
    assert!(((f[0] - goal[0]).powi(2) + (f[1] - goal[1]).powi(2) + (f[2] - goal[2]).powi(2)).sqrt() < 0.005, "{f:?}");
}

#[test]
fn routing_table_policy_1_2() {
    for (s, r) in [
        ("move to dining table", Route::MoveRobot),
        ("explore", Route::MoveRobot),
        ("pick up cup from dining table", Route::Vla),
        ("pick up cup", Route::Vla),
        ("place cup in the trash can", Route::Vla),
        ("place cup on kitchen counter", Route::Vla),
        ("open the left door of cabinet", Route::Vla),
        ("close the drawer of desk", Route::Vla),
        ("turn on light switch", Route::Vla),
        ("press button", Route::Vla),
        ("approach cup", Route::Vla),
        ("open the door", Route::Undecided),
    ] {
        assert_eq!(route(s).0, r, "{s}");
    }
    assert_eq!(SkillKind::from_sentence("place cup in the trash can"), SkillKind::Place);
    // move to 는 VLA 가 거절하고 move_robot 으로 보낸다
    let mut m = LimoMock::new(demo_scene());
    let r = m.run(&json!({"executor": "vla", "skill": "move to dining table", "objects": ["O1"]}));
    assert_eq!(st(&r), "error");
    assert_eq!(r["route"], "move_robot");
    // objects 없으면 고치는 방법
    let r = m.run(&json!({"executor": "vla", "skill": "pick up cup", "max_s": 10}));
    assert_eq!(st(&r), "error");
    assert!(r["message"].as_str().unwrap().contains("objects"));
}

#[test]
fn slots_goal_flag_comes_from_ids_not_order() {
    let mut objs: Vec<MemObject> = (0..20)
        .map(|k| MemObject { id: format!("O{k}"), name: "x".into(), score: 1.0, pos: [0.5 + k as f64 * 0.3, 0.0, 0.1], extent: [0.1; 3], first_pos: [0.0; 3], n_obs: 1, last_seen: 0.0, state: ObjState::Seen })
        .collect();
    objs[19].first_pos = objs[19].pos;
    let goals = vec!["O19".to_string()];
    let s = vla::build_slots(&objs, [0.0; 3], &goals, 0.0, &|_| 0.02, 1.0);
    assert_eq!(s.len(), vla::MAX_SLOTS);
    assert!(s[0].is_goal && s[0].id == "O19", "the farthest object is a goal and still gets a slot");
    assert_eq!(s.iter().filter(|x| x.is_goal).count(), 1);
    assert!(s[1..].windows(2).all(|w| w[0].dist <= w[1].dist), "rest nearest first");
    // 정규화: 숫자 id
    let mut m = LimoMock::new(demo_scene());
    let sl = m.robot.vla_slots(&["2".into()]);
    assert!(sl.iter().any(|x| x.id == "O2" && x.is_goal));
    m.robot.vla.objects.clear();
}

#[test]
fn pick_done_with_end_signal_and_verification() {
    let mut m = LimoMock::new(demo_scene());
    let r = m.run(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"], "max_s": 40}));
    has_result_fields(&r);
    assert_eq!(st(&r), "done", "{r}");
    assert_eq!(r["reason"], "verified");
    assert!(r["evidence"]["why"].as_str().unwrap().contains("closed on O1"), "{r}");
    assert!(r["evidence"]["m"]["lifted_m"].as_f64().unwrap() > 0.1);
    assert_eq!(r["contacts"], 0);
    assert!(r["min_clear_m"].is_null(), "no map in this mock");
    assert!(r["steps"].as_u64().unwrap() > 10);
    assert_eq!(r["policy"], "scripted");
    // 끝난 뒤 실행기는 쥔 채 유지(그리퍼 지령이 열리지 않음), 베이스 0
    assert_eq!(m.last_out[0], 0.0);
    assert!(m.held == Some(0));
    for _ in 0..30 {
        m.step(None);
    }
    assert!(m.held == Some(0), "the hold after done keeps the object");
}

#[test]
fn pick_of_phantom_object_fails_as_grasp_missed() {
    let mut s = demo_scene();
    s[0].phantom = true; // 기억에는 있는데 실제로는 없음 → 끝까지 닫힘
    let mut m = LimoMock::new(s);
    let r = m.run(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"], "max_s": 40}));
    assert_eq!(st(&r), "failed", "{r}");
    assert_eq!(r["reason"], "grasp_missed");
}

#[test]
fn place_done_then_handback_vs_verify() {
    let mut m = LimoMock::new(demo_scene());
    // 집고 → 놓기 (두 단계, 같은 실행기)
    let r = m.run(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"]}));
    assert_eq!(st(&r), "done", "{r}");
    let r = m.run(&json!({"executor": "vla", "skill": "place cup on box", "objects": ["O1", "O2"], "max_s": 40}));
    has_result_fields(&r);
    assert_eq!(st(&r), "done", "{r}");
    assert!(r["evidence"]["m"]["to_support_m"].as_f64().unwrap() < 0.6);
    assert!(m.held.is_none());
    // 놓기에는 받침이 꼭 있어야
    let r = m.run(&json!({"executor": "vla", "skill": "place cup on box", "objects": ["O1"]}));
    assert_eq!(st(&r), "error");
}

#[test]
fn start_conditions_hand_back() {
    let mut m = LimoMock::new(demo_scene());
    let r = m.run(&json!({"executor": "vla", "skill": "pick up chair", "objects": ["O3"]}));
    has_result_fields(&r);
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("handback", "too_far"), "{r}");
    assert_eq!(r["steps"], 0);
    assert_eq!(r["evidence"]["next"], "move_robot go_to");
    let r = m.run(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O99"]}));
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("handback", "not_in_map"));
    // 오래 못 본 + 불확실도 큼
    m.robot.vla.params.unc_max_m = 0.0;
    m.robot.vla.objects[0].last_seen = -100.0;
    let r = m.run(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"]}));
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("handback", "not_visible"), "{r}");
}

#[test]
fn timeout_when_no_end_signal() {
    let dir = std::env::temp_dir().join(format!("mr_vla_replay_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("spin.jsonl");
    // 제자리에서 천천히 돌기만(진척은 있음), 끝 신호 없음
    let home = limo::ARM_HOME;
    std::fs::write(&f, format!("{}\n", json!({"a": [0.0, 0.2, home[0], home[1], home[2], home[3], home[4], 0.0], "end": 0.0}))).unwrap();
    let mut m = LimoMock::new(demo_scene());
    let r = m.run(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"], "max_s": 6, "policy": format!("replay:{}", f.display())}));
    has_result_fields(&r);
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("timeout", "budget"), "{r}");
    assert!((r["time_s"].as_f64().unwrap() - 6.0).abs() < 0.1);
    assert!(r["evidence"]["verify"].is_object());
}

#[test]
fn stalled_policy_is_handed_back() {
    let mut m = LimoMock::new(demo_scene());
    let r = m.run(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"], "max_s": 9, "policy": "scripted:idle"}));
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("handback", "stalled"), "{r}");
    let t = r["time_s"].as_f64().unwrap();
    assert!((3.0..3.3).contains(&t), "budget/3 = 3 s, got {t}");
}

#[test]
fn unsafe_body_contact_fails_and_speed_is_capped() {
    let mut m = LimoMock::new(demo_scene());
    m.wall_x = Some(0.8);
    let r = m.run(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"], "max_s": 30, "policy": "scripted:ram"}));
    has_result_fields(&r);
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("failed", "unsafe"), "{r}");
    assert!(r["contacts"].as_u64().unwrap() >= 1);
    assert!(r["filter"]["clipped_steps"].as_u64().unwrap() > 0, "1 m/s asked, capped to 0.3");
    assert!(m.body.base_v[0] <= 0.3 + 1e-6);
}

fn room_with_wall(wall_x: f64) -> MockWorld {
    // 6 × 6 m 방(0.05 m), x ≥ wall_x 는 벽
    let res = 0.05;
    let (w, h) = (120usize, 120usize);
    let (ox, oy) = (-3.0, -3.0);
    let mut cells = vec![1i8; w * h];
    for y in 0..h {
        for x in 0..w {
            let wx = ox + (x as f64 + 0.5) * res;
            if wx >= wall_x || x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                cells[y * w + x] = 0;
            }
        }
    }
    let mut mw = MockWorld::new(Grid { res, ox, oy, w, h, cells });
    mw.body_r = 0.2;
    mw
}

#[test]
fn filter_stops_base_before_mapped_wall_then_unsafe() {
    let mut m = LimoMock::new(demo_scene());
    m.world = Some(room_with_wall(1.2));
    m.feed();
    let r = m.run(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"], "max_s": 30, "policy": "scripted:ram"}));
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("failed", "unsafe"), "{r}");
    assert!(r["evidence"]["why"].as_str().unwrap().contains("base at an obstacle"), "{r}");
    assert_eq!(r["contacts"], 0, "the filter stopped it before touching: {r}");
    assert!(r["filter"]["base_stops"].as_u64().unwrap() > 0);
    let mc = r["min_clear_m"].as_f64().expect("map present → min_clear_m");
    assert!(mc >= 0.0, "{mc}");
    assert!(m.pose[0] < 1.2 - 0.18);
}

#[test]
fn target_vanishes_hands_back_not_visible() {
    let mut s = demo_scene();
    s[0].vanish_step = Some(60);
    let mut m = LimoMock::new(s);
    let r = m.run(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"], "max_s": 40}));
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("handback", "not_visible"), "{r}");
}

#[test]
fn unsure_policy_hands_back() {
    let mut m = LimoMock::new(demo_scene());
    let home = limo::ARM_HOME;
    let r = m.run_ext(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"], "max_s": 20, "policy": "external"}), |mk| {
        let a = [0.0, 0.1 * (mk.sim_steps as f64 * 0.1).sin(), home[0], home[1], home[2], home[3], home[4], 0.0];
        (a, 0.0, 0.9)
    });
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("handback", "unsure"), "{r}");
}

#[test]
fn external_policy_end_signal_needs_verification() {
    // 대상이 팔 닿는 곳에 있음: approach 확인 = 닿는 거리 + 멈춤
    let mut s = demo_scene();
    s[0].mem.pos = [0.3, 0.0, 0.1];
    s[0].mem.first_pos = s[0].mem.pos;
    let mut m = LimoMock::new(s);
    let home = limo::ARM_HOME;
    let a = [0.0, 0.0, home[0], home[1], home[2], home[3], home[4], 0.0];
    let r = m.run_ext(&json!({"executor": "vla", "skill": "approach cup", "objects": ["O1"], "policy": "external"}), |_| (a, 0.95, 0.0));
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("done", "verified"), "{r}");
    let t = r["time_s"].as_f64().unwrap();
    assert!((0.45..0.7).contains(&t), "end signal held 0.5 s: {t}");
    // 끝 신호만 있고 확인 못 하는 스킬(서랍) → handback(unverified), done 이 아님
    let r = m.run_ext(&json!({"executor": "vla", "skill": "open the drawer of desk", "objects": ["O1"], "policy": "external"}), |_| (a, 0.95, 0.0));
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("handback", "unverified"), "{r}");
}

#[test]
fn safety_filter_limits() {
    let p = VlaParams::default();
    let mut f = vla::SafetyFilter::default();
    let mut s = LimoState::default();
    s.arm = limo::ARM_HOME;
    let none = |_: bool| None;
    let dt = 1.0 / 30.0;
    // 관절 한계 밖 + 큰 점프 → 한 스텝 몫만, 결국 한계 안쪽
    let mut a = [1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0];
    let mut last = [0.0; 8];
    for i in 0..400 {
        let (o, info) = f.filter(&a, &s, dt, &p, &none, 0.1);
        if i == 0 {
            assert!(info.clipped);
            assert!((o[3] - limo::ARM_HOME[1]).abs() <= p.arm_vmax * dt + 1e-9, "rate limited");
        }
        s.arm.copy_from_slice(&o[2..7]); // 완벽히 따라감
        s.grip = o[7];
        last = o;
    }
    assert!(last[0] <= p.base_vmax + 1e-9 && last[1] <= p.base_wmax + 1e-9);
    assert!((last[3] - (limo::ARM_LIM[1].1 - limo::JOINT_MARGIN)).abs() < 1e-6, "j2 at its real upper limit minus margin: {}", last[3]);
    assert!((last[7] - 1.0).abs() < 1e-9);
    // NaN → 유지, 베이스 0
    a[0] = f64::NAN;
    a[4] = f64::NAN;
    let (o, info) = f.filter(&a, &s, dt, &p, &none, 0.1);
    assert!(info.bad_input && o.iter().all(|x| x.is_finite()));
    // 장애물 0.1 m 앞(멈춤 거리 0.1) → 앞으로 0
    let wall = |fwd: bool| if fwd { Some((0.1, "map obstacle")) } else { None };
    let (o, info) = f.filter(&[0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], &s, dt, &p, &wall, 0.1);
    assert_eq!(o[0], 0.0);
    assert!(info.heavy());
    // 뒤로는 막히지 않음
    let (o, _) = f.filter(&[-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], &s, dt, &p, &wall, 0.1);
    assert!(o[0] < 0.0);
}

#[test]
fn arm_blocked_is_detected_and_holds_measured() {
    let p = VlaParams::default();
    let mut f = vla::SafetyFilter::default();
    let mut s = LimoState::default();
    s.arm = limo::ARM_HOME;
    let none = |_: bool| None;
    let mut a = [0.0; 8];
    a[2..7].copy_from_slice(&limo::ARM_HOME);
    a[3] = 1.0; // j2 를 크게, 그런데 팔은 안 움직임(막힘)
    let mut blocked = false;
    for _ in 0..60 {
        let (o, info) = f.filter(&a, &s, 1.0 / 30.0, &p, &none, 0.1);
        if info.arm_blocked {
            blocked = true;
            assert!((o[3] - s.arm[1]).abs() < 1e-9, "holds where it is");
            break;
        }
    }
    assert!(blocked);
}

#[test]
fn move_robot_waits_while_vla_runs_and_r1_tick_untouched() {
    let mut m = LimoMock::new(vec![MockObj::new("O1", [0.9, 0.3, 0.12], [0.06, 0.06, 0.1])]);
    assert!(!m.robot.vla_start(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"]})));
    assert!(m.robot.vla_busy());
    assert!(m.robot.command(&json!({"part": "base", "mode": "delta", "values": [0.1, 0, 0]})));
    let r = m.robot.take_result().unwrap();
    assert_eq!(r["status"], "error");
    assert!(m.robot.vla_stop("cancelled"));
    let r = m.robot.take_result().unwrap();
    assert_eq!((st(&r), r["reason"].as_str().unwrap()), ("handback", "cancelled"));
    has_result_fields(&r);
    assert!(!m.robot.vla_busy());
}

#[test]
fn vla_call_through_link_backend() {
    use crate::link::run_tool;
    let mut b = crate::link::LimoMockBackend(LimoMock::new(demo_scene()));
    let r = run_tool(&json!({"executor": "vla", "skill": "pick up cup", "objects": ["O1"]}), &mut b);
    assert_eq!(st(&r), "done", "{r}");
    let r = run_tool(&json!({"executor": "vla", "skill": "move to table", "objects": ["O1"]}), &mut b);
    assert_eq!(r["route"], "move_robot");
}

//! 단위 시험 — Isaac Sim 없이. 스키마·인자 읽기·한계 자르기·행동 벡터 배치·보간·닫힌 고리(가짜 로봇).

use super::*;
use crate::link::{run_tool, Backend, Mock};

fn args(s: &str) -> Value {
    serde_json::from_str(s).unwrap()
}

#[test]
fn schema_is_small_and_strict() {
    let d = definition();
    let f = &d["function"];
    assert_eq!(f["name"], "move_robot");
    assert_eq!(f["parameters"]["required"], json!(["part", "mode"]));
    assert_eq!(f["parameters"]["properties"]["part"]["enum"].as_array().unwrap().len(), 6);
    assert_eq!(f["parameters"]["properties"]["mode"]["enum"], json!(["go_to", "probe", "delta", "absolute"]));
    // 9B 맥락 16k: 도구 정의 한 개는 작게(약 450 토큰 이하)
    assert!(d.to_string().len() < 1800, "definition is {} bytes", d.to_string().len());
}

#[test]
fn parse_ok_and_tolerant() {
    let c = parse(&args(r#"{"part":"left_arm","mode":"delta","values":[0,10,0,0,0,0,-5]}"#)).unwrap();
    assert_eq!((c.part, c.mode, c.values.len(), c.duration_s), (Part::LeftArm, Mode::Delta, 7, None));
    // OpenAI arguments 문자열, 대소문자·공백, 그리퍼 숫자 하나, 숫자 문자열
    let c = parse(&Value::String(r#"{"part":"Right Gripper","mode":"ABSOLUTE","values":0.5,"duration_s":"1.5"}"#.into())).unwrap();
    assert_eq!((c.part, c.mode, c.values.clone(), c.duration_s), (Part::RightGripper, Mode::Absolute, vec![0.5], Some(1.5)));
    let c = parse(&args(r#"{"part":"base","mode":"delta","values":["0.5",0,"90"]}"#)).unwrap();
    assert_eq!(c.values, vec![0.5, 0.0, 90.0]);
}

#[test]
fn parse_errors_are_fixable_sentences() {
    let e = parse(&args(r#"{"part":"head","mode":"delta","values":[1]}"#)).unwrap_err();
    assert!(e.contains("part must be one of") && e.contains("left_arm"), "{e}");
    let e = parse(&args(r#"{"part":"left_arm","mode":"delta","values":[1,2]}"#)).unwrap_err();
    assert!(e.contains("exactly 7"), "{e}");
    let e = parse(&args(r#"{"part":"base","mode":"absolute","values":[1,0,0]}"#)).unwrap_err();
    assert!(e.contains("no absolute mode") && e.contains("go_to"), "{e}");
    assert!(parse(&args(r#"{"part":"left_arm","mode":"go_to","target":"F1"}"#)).unwrap_err().contains("base modes"));
    assert!(parse(&args(r#"{"part":"base","mode":"go_to","target":"kitchen"}"#)).unwrap_err().contains("F1 or R2"));
    assert!(parse(&args(r#"{"part":"base","mode":"go_to"}"#)).unwrap_err().contains("target"));
    assert!(parse(&args(r#"{"part":"base","mode":"probe","values":[30]}"#)).unwrap_err().contains("[turn_left_deg, forward_m]"));
    let c = parse(&args(r#"{"part":"base","mode":"go-to","target":"f2"}"#)).unwrap();
    assert_eq!((c.mode, c.target.as_deref()), (Mode::GoTo, Some("F2")));
    assert!(parse(&args(r#"{"part":"torso","values":[0,0,0,0]}"#)).unwrap_err().contains("mode is required"));
    assert!(parse(&args(r#"{"part":"torso","mode":"delta"}"#)).unwrap_err().contains("4 numbers"));
    assert!(parse(&args(r#"{"part":"torso","mode":"delta","values":[0,0,"x",0]}"#)).unwrap_err().contains("values[2]"));
    assert!(parse(&args(r#"{"part":"torso","mode":"delta","values":[0,0,0,0],"duration_s":-1}"#)).is_err());
    assert!(parse(&Value::String("not json".into())).unwrap_err().contains("not JSON"));
}

#[test]
fn clamping_to_joint_limits() {
    let lim = joint_limits(Part::LeftArm);
    // URDF 한계 안쪽 2°
    assert!((lim[1].0 - (-0.1745 + JOINT_MARGIN)).abs() < 1e-12);
    let mut t = vec![3.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let c = clamp_target(&mut t, &lim);
    assert_eq!(c, vec![0, 1, 3]);
    assert!((t[0] - (1.3090 - JOINT_MARGIN)).abs() < 1e-12);
    assert!((t[3] - (0.3491 - JOINT_MARGIN)).abs() < 1e-12);
    // 오른팔 2번은 거울
    assert!(joint_limits(Part::RightArm)[1].1 < 0.2);
    assert_eq!(joint_limits(Part::LeftGripper), vec![(0.0, 1.0)]);
}

#[test]
fn action_vector_layout() {
    assert_eq!(action_slots(Part::Base), 0..3);
    assert_eq!(action_slots(Part::Torso), 3..7);
    assert_eq!(action_slots(Part::LeftArm), 7..14);
    assert_eq!(action_slots(Part::LeftGripper), 14..15);
    assert_eq!(action_slots(Part::RightArm), 15..22);
    assert_eq!(action_slots(Part::RightGripper), 22..23);
    let total: usize = PARTS.iter().map(|n| action_slots(Part::from_name(n).unwrap()).len()).sum();
    assert_eq!(total, ACTION_DIM);
    for n in PARTS {
        let p = Part::from_name(n).unwrap();
        assert_eq!(action_slots(p).len(), p.dof());
    }
    assert_eq!(gripper_to_action(0.0), -1.0);
    assert_eq!(gripper_to_action(1.0), 1.0);
    assert_eq!(gripper_to_action(0.5), 0.0);
}

#[test]
fn hold_action_mirrors_proprio() {
    let mut m = Mock::default();
    m.plant.q[act::LEFT_ARM.start + 2] = 0.3;
    let mut r = Robot::default();
    let mut a = [0f32; ACTION_DIM];
    assert_eq!(r.tick(&m.plant.proprio(), &mut a), Tick::Idle);
    assert_eq!(&a[0..3], &[0.0, 0.0, 0.0]);
    assert!((a[3] - 1.025).abs() < 1e-6 && (a[4] + 1.45).abs() < 1e-6);
    assert!((a[act::LEFT_ARM.start + 2] - 0.3).abs() < 1e-6);
    assert_eq!(a[act::LEFT_GRIPPER], 1.0);
    // 짧은 proprio 는 BadObs, 행동은 유지값 그대로
    assert_eq!(r.tick(&[0.0; 10], &mut a), Tick::BadObs);
    assert!((a[3] - 1.025).abs() < 1e-6);
}

#[test]
fn min_jerk_profile() {
    assert_eq!(min_jerk(0.0), 0.0);
    assert_eq!(min_jerk(1.0), 1.0);
    assert!((min_jerk(0.5) - 0.5).abs() < 1e-12);
    assert_eq!(min_jerk(2.0), 1.0);
    // 단조 증가, 최고 기울기 1.875 (τ = 0.5)
    let n = 1000;
    let mut peak = 0.0f64;
    for i in 0..n {
        let a = min_jerk(i as f64 / n as f64);
        let b = min_jerk((i + 1) as f64 / n as f64);
        assert!(b >= a);
        peak = peak.max((b - a) * n as f64);
    }
    assert!((peak - MIN_JERK_PEAK).abs() < 1e-3, "{peak}");
}

#[test]
fn duration_respects_safe_speed() {
    let s = Safety::default();
    // 90° 를 45°/s 로: 1.875 · 2 s = 3.75 s
    let (t, slowed) = plan_duration(&[0.0, std::f64::consts::FRAC_PI_2], s.arm_vmax, None, &s);
    assert!((t - 3.75).abs() < 1e-9 && !slowed);
    let (t, slowed) = plan_duration(&[std::f64::consts::FRAC_PI_2], s.arm_vmax, Some(0.5), &s);
    assert!((t - 3.75).abs() < 1e-9 && slowed);
    let (t, slowed) = plan_duration(&[0.1], s.arm_vmax, Some(5.0), &s);
    assert!((t - 5.0).abs() < 1e-9 && !slowed);
    let (t, _) = plan_duration(&[0.0], s.arm_vmax, None, &s);
    assert_eq!(t, s.min_duration);
}

/// 보간 중 지령 속도가 안전 속도를 넘지 않는다(실제 행동 벡터에서 잰다)
#[test]
fn commanded_speed_never_exceeds_limit() {
    let mut m = Mock::default();
    assert!(!m.robot.command(&args(r#"{"part":"right_arm","mode":"delta","values":[-60,-30,0,-45,0,0,0],"duration_s":0.2}"#)));
    let mut prev = m.last_action;
    let mut vmax = 0.0f64;
    loop {
        let t = m.step();
        for i in act::RIGHT_ARM {
            vmax = vmax.max(((m.last_action[i] - prev[i]) as f64).abs() * HZ);
        }
        prev = m.last_action;
        if t == Tick::Done {
            break;
        }
    }
    assert!(vmax <= Safety::default().arm_vmax * 1.001, "{} deg/s", vmax.to_degrees());
    let r = m.robot.take_result().unwrap();
    assert_eq!(r["status"], "reached", "{r}");
    assert_eq!(r["slowed"], true);
}

#[test]
fn arm_delta_reaches_and_reports() {
    let mut m = Mock::default();
    let r = run_tool(&args(r#"{"part":"left_arm","mode":"delta","values":[0,30,0,-20,0,0,0]}"#), &mut m);
    assert_eq!(r["status"], "reached", "{r}");
    let st = r["state"].as_array().unwrap();
    assert!((st[1].as_f64().unwrap() - 30.0).abs() < 1.6 && (st[3].as_f64().unwrap() + 20.0).abs() < 1.6, "{r}");
    assert_eq!(r["units"], "deg");
    assert!(r.get("clamped").is_none() && r.get("hint").is_none());
    // 끝난 뒤에도 목표를 유지한다
    for _ in 0..30 {
        m.step();
    }
    assert!((m.plant.q[act::LEFT_ARM.start + 1] - 30f64.to_radians()).abs() < 0.01);
}

#[test]
fn absolute_beyond_limit_is_clamped() {
    let mut m = Mock::default();
    let r = m.exec(&args(r#"{"part":"torso","mode":"absolute","values":[200,-83,0,0]}"#));
    assert_eq!(r["status"], "reached", "{r}");
    assert_eq!(r["clamped"], json!([0]));
    let t0 = r["target"][0].as_f64().unwrap();
    assert!((t0 - (1.8326f64.to_degrees() - 2.0)).abs() < 0.11, "{r}");
}

#[test]
fn read_state_with_zero_delta() {
    let mut m = Mock::default();
    let steps = m.sim_steps;
    let r = m.exec(&args(r#"{"part":"torso","mode":"delta","values":[0,0,0,0]}"#));
    assert_eq!(r["status"], "reached");
    assert_eq!(r["steps"], 0);
    assert_eq!(m.sim_steps, steps, "reading must not step the sim");
    assert!((r["state"][0].as_f64().unwrap() - 1.025f64.to_degrees()).abs() < 0.1, "{r}");
}

#[test]
fn arm_contact_is_blocked_and_stops_pushing() {
    let mut m = Mock::default();
    m.plant.stops.push((act::LEFT_ARM.start + 1, 0.3)); // 2번 관절이 0.3 rad 에서 막힘
    let r = m.exec(&args(r#"{"part":"left_arm","mode":"absolute","values":[0,80,0,0,0,0,0]}"#));
    assert_eq!(r["status"], "blocked", "{r}");
    assert!(r["hint"].as_str().unwrap().contains("contact"));
    // 멈춘 자리에서 버틴다(계속 밀지 않는다)
    assert!((m.robot.hold()[act::LEFT_ARM.start + 1] - 0.3).abs() < 0.05, "{:?}", m.robot.hold());
}

#[test]
fn gripper_close_on_object_is_blocked_but_keeps_squeezing() {
    let mut m = Mock::default();
    m.plant.object_in[1] = Some(0.4);
    let r = m.exec(&args(r#"{"part":"right_gripper","mode":"absolute","values":[0]}"#));
    assert_eq!(r["status"], "blocked", "{r}");
    assert!(r["hint"].as_str().unwrap().contains("holding an object"));
    assert!((r["state"][0].as_f64().unwrap() - 0.4).abs() < 0.02);
    assert_eq!(m.robot.hold()[act::RIGHT_GRIPPER], -1.0); // 계속 쥔다
    // 물체 없이 열고 닫기
    let r = m.exec(&args(r#"{"part":"left_gripper","mode":"absolute","values":[0]}"#));
    assert_eq!(r["status"], "reached", "{r}");
}

#[test]
fn base_closed_loop_moves_and_turns() {
    let mut m = Mock::default();
    let r = m.exec(&args(r#"{"part":"base","mode":"delta","values":[1.0,0.3,0]}"#));
    assert_eq!(r["status"], "reached", "{r}");
    assert!((m.plant.pose[0] - 1.0).abs() < 0.04 && (m.plant.pose[1] - 0.3).abs() < 0.04, "{:?}", m.plant.pose);
    // 속도 지령은 안전 속도 이하, 끝나면 0
    assert_eq!(&m.last_action[0..3], &[0.0, 0.0, 0.0]);
    let start = m.plant.pose;
    let r = m.exec(&args(r#"{"part":"base","mode":"delta","values":[0,0,90]}"#));
    assert_eq!(r["status"], "reached", "{r}");
    assert!(((m.plant.pose[2] - start[2]).to_degrees() - 90.0).abs() < 2.5, "{:?}", m.plant.pose);
    // 로봇 기준: 90° 돈 뒤 전진 0.5 m 는 세계 +y
    let p0 = m.plant.pose;
    let r = m.exec(&args(r#"{"part":"base","mode":"delta","values":[0.5,0,0]}"#));
    assert_eq!(r["status"], "reached", "{r}");
    assert!((m.plant.pose[1] - p0[1] - 0.5).abs() < 0.05 && (m.plant.pose[0] - p0[0]).abs() < 0.05, "{:?}", m.plant.pose);
}

#[test]
fn base_speed_limit_and_wall() {
    let mut m = Mock::default();
    assert!(!m.robot.command(&args(r#"{"part":"base","mode":"delta","values":[5,0,0]}"#)));
    let mut vmax = 0.0f64;
    m.plant.wall_x = Some(0.6);
    loop {
        let t = m.step();
        vmax = vmax.max(m.last_action[0] as f64 * act::BASE_OUT[0]);
        if t == Tick::Done {
            break;
        }
    }
    assert!(vmax <= Safety::default().base_vmax + 1e-6, "{vmax}");
    let r = m.robot.take_result().unwrap();
    assert_eq!(r["status"], "blocked", "{r}");
    assert_eq!(r["clamped"], json!([0])); // 5 m → 2 m
    assert!((r["state"][0].as_f64().unwrap() - 0.6).abs() < 0.05, "{r}");
}

#[test]
fn errors_are_observations() {
    let mut m = Mock::default();
    let r = run_tool(&args(r#"{"part":"left_arm","mode":"delta","values":[1]}"#), &mut m);
    assert_eq!(r["status"], "error");
    assert!(r["message"].as_str().unwrap().contains("exactly 7"));
    // 관측 전 명령
    let mut rb = Robot::default();
    assert!(rb.command(&args(r#"{"part":"torso","mode":"delta","values":[1,0,0,0]}"#)));
    assert_eq!(rb.take_result().unwrap()["status"], "error");
    // 움직이는 중 또 명령
    assert!(!m.robot.command(&args(r#"{"part":"torso","mode":"delta","values":[5,0,0,0]}"#)));
    assert!(m.robot.command(&args(r#"{"part":"torso","mode":"delta","values":[5,0,0,0]}"#)));
    assert!(m.robot.take_result().unwrap()["message"].as_str().unwrap().contains("still executing"));
}

#[test]
fn other_parts_hold_while_one_moves() {
    let mut m = Mock::default();
    let before = *m.robot.hold();
    m.exec(&args(r#"{"part":"right_arm","mode":"delta","values":[10,0,0,0,0,0,0]}"#));
    let after = m.robot.hold();
    for i in act::TORSO.chain(act::LEFT_ARM).chain([act::LEFT_GRIPPER, act::RIGHT_GRIPPER]) {
        assert_eq!(before[i], after[i], "slot {i}");
    }
    assert!((after[act::RIGHT_ARM.start] - 10f64.to_radians()).abs() < 1e-9);
}

#[test]
fn ffi_roundtrip() {
    use std::ffi::CString;
    let m = Mock::default();
    unsafe {
        let r = ffi::mr_new(30.0);
        let p = m.plant.proprio();
        let mut a = [0f32; ACTION_DIM];
        assert_eq!(ffi::mr_tick(r, p.as_ptr(), p.len(), a.as_mut_ptr()), 0);
        let c = CString::new(r#"{"part":"torso","mode":"delta","values":[0,0,0,0]}"#).unwrap();
        assert_eq!(ffi::mr_command(r, c.as_ptr()), 1);
        let mut small = [0 as std::ffi::c_char; 4];
        let need = ffi::mr_take_result(r, small.as_mut_ptr(), small.len());
        assert!(need < 0);
        let mut buf = vec![0 as std::ffi::c_char; (-need) as usize];
        let n = ffi::mr_take_result(r, buf.as_mut_ptr(), buf.len());
        assert_eq!(n, -need - 1);
        let s = std::ffi::CStr::from_ptr(buf.as_ptr()).to_str().unwrap();
        assert!(s.contains("\"reached\""), "{s}");
        let def = std::ffi::CStr::from_ptr(ffi::mr_tool_definition()).to_str().unwrap();
        assert!(def.contains("move_robot"));
        ffi::mr_free(r);
    }
}

// ---------------------------------------------------------------- 지도·주행(가짜 집)

use crate::link::MockWorld;

/// 10 × 6 m 두 방, x = 5 벽에 문(y 2.5..3.5), 바깥은 벽
fn two_rooms() -> MockWorld {
    let res = 0.05;
    let (w, h) = (200usize, 120usize);
    let mut g = crate::map::Grid::new(res, 0.0, 0.0, w, h, 0);
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

fn exec(m: &mut Mock, s: &str) -> Value {
    run_tool(&args(s), m)
}

#[test]
fn read_base_returns_map_summary() {
    let mut m = Mock::with_world(two_rooms(), [1.5, 3.0, 0.0]);
    let r = exec(&mut m, r#"{"part":"base","mode":"delta","values":[0,0,0]}"#);
    let map = &r["map"];
    assert!(map["free_m2"].as_f64().unwrap() > 3.0, "{r}");
    assert!(map["around"]["F"].as_str().unwrap().contains("depth"), "{r}");
    assert!(!map["frontiers"].as_array().unwrap().is_empty(), "{r}");
    assert!(r["_m"]["obs_us"].as_u64().is_some());
}

#[test]
fn go_to_frontier_without_contact() {
    let mut m = Mock::with_world(two_rooms(), [1.5, 3.0, 0.0]);
    exec(&mut m, r#"{"part":"base","mode":"probe","values":[180,0]}"#);
    let r = exec(&mut m, r#"{"part":"base","mode":"go_to","target":"F1"}"#);
    assert_eq!(r["status"], "reached", "{r}");
    assert!(r["moved_m"].as_f64().unwrap() > 0.3, "{r}");
    assert_eq!(m.world.as_ref().unwrap().contacts, 0);
}

#[test]
fn probe_stops_before_wall() {
    let mut m = Mock::with_world(two_rooms(), [3.5, 1.5, 0.0]);
    // 앞 1.4 m 에 벽(x = 4.9): 1.5 m 가라 해도 벽 앞에서 멈춤
    let r = exec(&mut m, r#"{"part":"base","mode":"probe","values":[0,1.5]}"#);
    assert_eq!(r["status"], "blocked", "{r}");
    let x = m.plant.pose[0];
    assert!(x < 4.9 - 0.28 && x > 4.0, "x {x} {r}");
    assert_eq!(m.world.as_ref().unwrap().contacts, 0);
}

#[test]
fn go_to_unknown_point_is_rejected_with_hint() {
    let mut m = Mock::with_world(two_rooms(), [1.5, 3.0, 0.0]);
    let r = exec(&mut m, r#"{"part":"base","mode":"go_to","values":[-1.0,0]}"#);
    assert_eq!(r["status"], "error", "{r}");
    assert!(r["message"].as_str().unwrap().contains("UNKNOWN"), "{r}");
    let r = exec(&mut m, r#"{"part":"base","mode":"go_to","target":"F9"}"#);
    assert!(r["message"].as_str().unwrap().contains("unknown target"), "{r}");
}

#[test]
fn delta_into_unseen_side_is_stopped() {
    // 카메라는 앞만 본다: 옆(모르는 곳)으로 1 m 가라 하면 아는 빈칸 끝에서 멈춤
    let mut m = Mock::with_world(two_rooms(), [2.5, 3.0, 0.0]);
    let r = exec(&mut m, r#"{"part":"base","mode":"delta","values":[0,1.0,0]}"#);
    assert_eq!(r["status"], "blocked", "{r}");
    assert!(r["stopped_by"].as_str().unwrap().contains("unknown"), "{r}");
}

#[test]
fn base_final_approach_no_undershoot() {
    let mut m = Mock::default();
    let r = m.exec(&args(r#"{"part":"base","mode":"delta","values":[0.2,0,0]}"#));
    assert_eq!(r["status"], "reached", "{r}");
    assert!((m.plant.pose[0] - 0.2).abs() < 0.012, "{:?}", m.plant.pose);
    let r = m.exec(&args(r#"{"part":"base","mode":"delta","values":[0,0,30]}"#));
    assert_eq!(r["status"], "reached", "{r}");
    assert!((m.plant.pose[2].to_degrees() - 30.0).abs() < 1.2, "{:?}", m.plant.pose);
}

#[test]
fn obstacle_appearing_on_path_is_avoided_or_reported() {
    let mut m = Mock::with_world(two_rooms(), [1.0, 3.0, 0.0]);
    // 문 쪽으로 보며 지도 쌓기
    exec(&mut m, r#"{"part":"base","mode":"probe","values":[0,0.5]}"#);
    // 가는 길(3.4, 3.0)에 0.25 m 장애물이 0.5 s 뒤 나타남(로봇 앞 약 1.5 m — 멈출 수 있는 거리)
    let now = m.sim_steps;
    m.world.as_mut().unwrap().events.push((now + 15, 3.4, 3.0, 0.25, true));
    let r = exec(&mut m, r#"{"part":"base","mode":"go_to","values":[2.5,0]}"#);
    assert!(r["status"] == "reached" || r["status"] == "blocked" || r["status"] == "timeout", "{r}");
    assert_eq!(m.world.as_ref().unwrap().contacts, 0, "{r}");
    // 치우면 다시 열림
    let now = m.sim_steps;
    m.world.as_mut().unwrap().events.push((now + 1, 3.4, 3.0, 0.25, false));
    for _ in 0..3 {
        exec(&mut m, r#"{"part":"base","mode":"probe","values":[0,0]}"#);
    }
}

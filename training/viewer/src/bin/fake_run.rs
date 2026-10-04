#![recursion_limit = "512"]
// fake_run — 가짜 실행 생성기(TRAIN_VIEWER.md 11절 V0). 학습 없이 규약대로 실행 폴더를 만든다: 뷰어 시험·화면 개발용.
// 흉내 내는 것: 재개 되감기(trim 안 한 줄), 끝의 반쪽 줄, 줄기(s_eval, s_teacher), labs, 리플레이(.trp: 프레임·슬롯·자라는 지도), 평가 표,
// 씨앗 묶음(group), FP8 키, 안 잰 키 빼기(그 업데이트에 그 스킬 판이 없으면 키 없음).
// 모든 run.json 에 "synthetic": true 를 적는다 — 화면이 "synthetic" 배지를 단다.
//
//   fake_run --root DIR                 정적 실행 여러 개
//   fake_run --root DIR --live SECONDS  + 살아 있는 실행 하나(pid 있음)를 초마다 덧붙임(실시간 시험)
use serde_json::{json, Value};
use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use trainfmt::trp::{TrpWriter, EV_CONTACT, EV_GRASP, EV_RESET, EV_SUCCESS};
use trainfmt::{now_ts, write_atomic, RunWriter};

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        let (a, b) = (self.u().max(1e-12), self.u());
        (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos()
    }
    fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T {
        &v[(self.u() * v.len() as f64) as usize % v.len()]
    }
}

const SKILLS: &[&str] = &["approach", "pick", "place"];
const HOMES: &[&str] = &["Rs_int", "Beechwood_0_int", "Wainscott_1_int", "Pomaria_2_int"];
const TERMS: &[(&str, &[&str])] = &[
    ("approach", &["progress", "aim", "visible", "success", "collision", "time"]),
    ("pick", &["reach", "align", "grasp", "lift", "success", "contact", "time"]),
    ("place", &["carry", "align", "release", "success", "drop", "time"]),
];
const T_MAX: &[(&str, f64)] = &[("approach", 30.0), ("pick", 25.0), ("place", 30.0)];
const OBJECTS: &[(&str, &str, f64, f64, f64)] = &[
    // 이름, 범주, 크기 x y z
    ("cup_0", "cup", 0.08, 0.08, 0.10),
    ("table_0", "table", 1.2, 0.8, 0.75),
    ("chair_0", "chair", 0.5, 0.5, 0.9),
    ("sofa_0", "sofa", 2.0, 0.9, 0.8),
    ("fridge_0", "fridge", 0.7, 0.7, 1.8),
    ("bowl_0", "bowl", 0.15, 0.15, 0.07),
];

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn terms(skill: &str) -> &'static [&'static str] {
    TERMS.iter().find(|t| t.0 == skill).unwrap().1
}
fn t_max(skill: &str) -> f64 {
    T_MAX.iter().find(|t| t.0 == skill).unwrap().1
}

struct Spec {
    name: String,
    kind: &'static str,
    group: String,
    seed: u64,
    fp8: bool,
    rows: usize,
    rewind_at: Option<usize>,
    half_line: bool,
    eval_stream: bool,
    teacher_stream: bool,
    replays: usize,
    lab: bool,
    speed: f64,
}

fn meta(sp: &Spec) -> Value {
    let reward_terms: serde_json::Map<String, Value> = TERMS
        .iter()
        .map(|(s, ts)| (s.to_string(), json!(ts.iter().map(|t| json!({"name": t, "weight": 1.0, "unit": "per step"})).collect::<Vec<_>>())))
        .collect();
    let mut jm = serde_json::Map::new();
    for i in 1..=5 {
        jm.insert(format!("q{}", i), json!([[format!("omx_joint{}", i), 1.0, 0.0]]));
    }
    jm.insert("qg".into(), json!([["omx_gripper_joint_1", 1.0, 0.0], ["omx_gripper_joint_2", -1.0, 0.0]]));
    json!({
        "kind": sp.kind,
        "name": sp.name,
        "group": sp.group,
        "seed": sp.seed,
        "synthetic": true,
        "trainer": "fake_run",
        "teacher": if matches!(sp.kind, "dagger" | "bc") { json!("fake_teacher_bf16_s1") } else { Value::Null },
        "ctrl_hz": 10, "phys_hz": 100, "n_envs": 4096, "rollout_T": 64, "env_steps_per_iter": 262144, "log_envs": 8,
        "t_max": T_MAX.iter().map(|(s, t)| (s.to_string(), json!(t))).collect::<serde_json::Map<_, _>>(),
        "budget_s": {"approach": 20.0, "pick": 18.0, "place": 22.0},
        "homes": HOMES,
        "skills": SKILLS,
        "reward_terms": reward_terms,
        "success": {"lift_m": 0.1, "gripper_m": 0.005, "hold_s": 1.0},
        "grid_res": 0.05, "grid_wh": [160, 120], "slots": 16,
        "completion_bins": [0.0, 0.3, 0.7, 1.0],
        "map_modes": ["gt", "slam", "partial", "none"],
        "curriculum": {"stages": [{"name": "C0", "first_map": "full"}, {"name": "C1", "first_map": "partial"}, {"name": "C2", "first_map": "empty"}]},
        "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "target_kl": 0.01, "lr": 3e-4, "n_epochs": 5, "n_minibatch": 4, "max_grad_norm": 1.0, "log_std_init": -0.5,
        "precision": {"fp8": if sp.fp8 { 7 } else { 0 }, "default": "bf16"},
        "action_names": ["vx", "wz", "q1", "q2", "q3", "q4", "q5", "g"],
        "joint_map": jm,
        "frame_cols": 1,
        "refs": {"train/approx_kl": 0.01, "train/grad_norm": 1.0},
        "logged": {"progress": true, "episodes": true, "replays": sp.replays > 0},
    })
}

/// 진행도(0..1)에서 스킬 성공 확률
fn p_succ(skill: &str, prog: f64, fp8: bool, seed: u64) -> f64 {
    let (mid, top) = match skill {
        "approach" => (0.15, 0.97),
        "pick" => (0.35, 0.85),
        _ => (0.55, 0.78),
    };
    let jitter = ((seed as f64) * 0.37).sin() * 0.02 - if fp8 { 0.01 } else { 0.0 };
    (top + jitter) * sig((prog - mid) * 14.0)
}

// ------------------------------------------------------------------------------------------
// 판 하나 흉내 + 리플레이

struct Room {
    w: i32,
    h: i32,
    res: f64,
    ox: f64,
    oy: f64,
    occ: Vec<i8>, // 참 격자: 0 빈, 100 막힘
}

fn room(rng: &mut Rng) -> Room {
    let (w, h, res) = (160, 120, 0.05);
    let mut occ = vec![0i8; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            if x < 3 || y < 3 || x >= w - 3 || y >= h - 3 {
                occ[(y * w + x) as usize] = 100;
            }
        }
    }
    // 가구 덩이 몇 개 + 칸막이 벽
    for _ in 0..4 {
        let (cx, cy) = (20 + (rng.u() * 120.0) as i32, 15 + (rng.u() * 90.0) as i32);
        let (rw, rh) = (6 + (rng.u() * 14.0) as i32, 6 + (rng.u() * 10.0) as i32);
        for y in cy..(cy + rh).min(h - 3) {
            for x in cx..(cx + rw).min(w - 3) {
                occ[(y * w + x) as usize] = 100;
            }
        }
    }
    for y in 3..70 {
        occ[(y * w + 80) as usize] = 100;
        occ[(y * w + 81) as usize] = 100;
    }
    Room { w, h, res, ox: -4.0, oy: -3.0, occ }
}

struct EpOut {
    line: Value,
    trp: Option<TrpWriter>,
}

#[allow(clippy::too_many_arguments)]
fn episode(rng: &mut Rng, skill: &str, home: &str, stage: &str, comp0: f64, p: f64, iter: f64, steps: f64, ep_no: u64, env: u32, driver: &str, with_replay: bool) -> EpOut {
    let success = rng.u() < p;
    let collided = !success && rng.u() < 0.35;
    let timeout = !success && !collided;
    let tm = t_max(skill);
    let t = if success { tm * (0.25 + 0.35 * rng.u()) * (1.15 - 0.3 * comp0) } else if collided { tm * rng.u() * 0.6 } else { tm };
    let n = (t * 10.0).round().max(5.0) as usize;
    let ts = terms(skill);
    // 항목별 합(성공 20, 충돌 −10, 시간 −0.01/스텝 …)
    let mut r = serde_json::Map::new();
    let mut per_step: Vec<Vec<f64>> = vec![vec![0.0; ts.len()]; n];
    for (j, term) in ts.iter().enumerate() {
        for (k, row) in per_step.iter_mut().enumerate() {
            let ph = k as f64 / n as f64;
            row[j] = match *term {
                "progress" | "reach" | "carry" => 0.08 * (1.0 - ph) + 0.01 * rng.n(),
                "aim" | "align" | "visible" => 0.02 * ph + 0.005 * rng.n(),
                "grasp" | "release" => if k == n * 2 / 3 && success { 2.0 } else { 0.0 },
                "lift" => if success && ph > 0.7 { 0.05 } else { 0.0 },
                "success" => if success && k == n - 1 { 20.0 } else { 0.0 },
                "collision" | "contact" | "drop" => if collided && k == n - 1 { -10.0 } else if rng.u() < 0.01 { -0.5 } else { 0.0 },
                "time" => -0.01,
                _ => 0.0,
            };
        }
        let s: f64 = per_step.iter().map(|row| row[j]).sum();
        r.insert(term.to_string(), json!((s * 1e4).round() / 1e4));
    }
    let ret: f64 = r.values().filter_map(|v| v.as_f64()).sum();
    let contacts = per_step.iter().filter(|row| ts.iter().enumerate().any(|(j, t)| matches!(*t, "contact" | "collision" | "drop") && row[j] < 0.0)).count();
    let path_opt = 1.5 + 3.0 * rng.u();
    let path = path_opt * (1.05 + 0.6 * rng.u() * (1.0 - comp0));
    let outcome = if success { "success" } else if collided { "collision" } else { "timeout" };
    let mode = if comp0 >= 1.0 { "gt" } else if comp0 > 0.0 { "partial" } else { "slam" };
    let file = format!("ep_{:06}_{}_{}.trp", ep_no, skill, outcome);
    let mut line = json!({
        "ts": (now_ts() * 10.0).round() / 10.0, "iter": iter, "env_steps": steps, "env": env, "ep": ep_no,
        "skill": skill, "home": home, "stage": stage, "curriculum": stage, "map_mode": mode, "completion0": (comp0 * 1000.0).round() / 1000.0,
        "driver": driver, "success": success, "outcome": outcome, "collided": collided, "timeout": timeout,
        "t": (t * 10.0).round() / 10.0, "steps": n, "ret": (ret * 1e4).round() / 1e4, "r": r,
        "contacts": contacts, "drops": 0, "min_clear_m": ((0.02 + 0.3 * rng.u()) * 1000.0).round() / 1000.0,
        "path_len": (path * 100.0).round() / 100.0, "path_len_opt": (path_opt * 100.0).round() / 100.0,
        "final_dist": if success { 0.05 + 0.05 * rng.u() } else { 0.3 + rng.u() }, "final_aim_deg": 15.0 * rng.u(),
    });
    if !with_replay {
        return EpOut { line, trp: None };
    }
    line["replay"] = json!(file);
    // ----- 궤적 -----
    let rm = room(rng);
    let free = |x: f64, y: f64| {
        let cx = ((x - rm.ox) / rm.res) as i32;
        let cy = ((y - rm.oy) / rm.res) as i32;
        cx > 0 && cy > 0 && cx < rm.w && cy < rm.h && rm.occ[(cy * rm.w + cx) as usize] == 0
    };
    let sample_free = |rng: &mut Rng| loop {
        let (x, y) = (rm.ox + 0.4 + rng.u() * (rm.w as f64 * rm.res - 0.8), rm.oy + 0.4 + rng.u() * (rm.h as f64 * rm.res - 0.8));
        if free(x, y) && free(x + 0.25, y) && free(x - 0.25, y) && free(x, y + 0.25) && free(x, y - 0.25) {
            return (x, y);
        }
    };
    let (mut x, mut y) = sample_free(rng);
    let mut yaw = rng.u() * 2.0 * PI - PI;
    let (gx, gy) = sample_free(rng);
    let gz = 0.75;
    let mut obj_pos: Vec<(f64, f64, f64)> = OBJECTS.iter().map(|_| { let p = sample_free(rng); (p.0, p.1, 0.0) }).collect();
    obj_pos[0] = (gx, gy, gz);
    obj_pos[1] = (gx + 0.3, gy, 0.0);
    let mut cols: Vec<String> = trainfmt::trp::FRAME_COLS_V1.iter().map(|s| s.to_string()).collect();
    for t in ts {
        cols.push(format!("r_{}", t));
    }
    for c in ["ret_cum", "value", "end_p", "ev", "tgt_slot", "tgt_x", "tgt_y", "tgt_z", "completion"] {
        cols.push(c.to_string());
    }
    let slot_cols: Vec<String> = trainfmt::trp::SLOT_COLS_V1.iter().map(|s| s.to_string()).collect();
    let head = json!({
        "meta": line, "dt": 0.1, "stride": 1,
        "objects": OBJECTS.iter().enumerate().map(|(i, o)| json!({"slot": i, "id": i, "name": o.0, "cat": o.1})).collect::<Vec<_>>(),
        "joint_map": meta(&Spec { name: String::new(), kind: "teacher", group: String::new(), seed: 0, fp8: false, rows: 0, rewind_at: None, half_line: false, eval_stream: false, teacher_stream: false, replays: 0, lab: false, speed: 1.0 })["joint_map"],
        "grid": {"res": rm.res, "ox": rm.ox, "oy": rm.oy, "w": rm.w, "h": rm.h},
        "synthetic": true,
    });
    let mut w = TrpWriter::new(&cols, &slot_cols, 16, head);
    // 아는 지도: 처음 완성도만큼 미리 공개
    let mut known = vec![-1i8; (rm.w * rm.h) as usize];
    for (i, k) in known.iter_mut().enumerate() {
        if rng.u() < comp0 * 0.9 || (comp0 >= 1.0) {
            *k = rm.occ[i];
        }
    }
    w.map_rect(0, rm.w, rm.h, rm.res, rm.ox, rm.oy, 0, 0, rm.w - 1, rm.h - 1, &known);
    let (mut sx, mut sy, mut syaw);
    let mut ret_cum = 0.0;
    let mut q = [0.0f64, -1.0, 0.8, 0.6, 0.0, 0.0];
    let mut seen = vec![comp0 >= 1.0; OBJECTS.len()];
    let mut age = vec![0.0f64; OBJECTS.len()];
    let mut belief: Vec<(f64, f64, f64)> = obj_pos.iter().map(|p| (p.0 + 0.3 * rng.n(), p.1 + 0.3 * rng.n(), p.2)).collect();
    let mut drift = (0.0, 0.0, 0.0);
    let (mut min_ee, mut last_base) = (f64::INFINITY, f64::INFINITY);
    for k in 0..n {
        let ph = k as f64 / n as f64;
        // 목표로 가는 조종(성공 판은 닿음, 실패 판은 헤맴)
        let (dx, dy) = (gx - x, gy - y);
        let dist = (dx * dx + dy * dy).sqrt();
        let want = dy.atan2(dx);
        let mut e = want - yaw;
        e = e.sin().atan2(e.cos());
        let wander = if success { 0.0 } else { 0.8 * (k as f64 * 0.07).sin() };
        let wz = (2.0 * e + wander).clamp(-0.9, 0.9);
        let vx = if dist > 0.45 { (0.45 * (1.0 - e.abs() / PI)).max(0.05) } else { 0.0 };
        let (nx, ny) = (x + vx * yaw.cos() * 0.1, y + vx * yaw.sin() * 0.1);
        if free(nx, ny) {
            x = nx;
            y = ny;
        }
        yaw += wz * 0.1;
        drift = (drift.0 + 0.002 * rng.n(), drift.1 + 0.002 * rng.n(), drift.2 + 0.001 * rng.n());
        sx = x + drift.0;
        sy = y + drift.1;
        syaw = yaw + drift.2;
        // 팔: 가까우면 뻗고 쥐고 들어 올림
        let near = dist < 0.6 && skill != "approach";
        let target_q = if near { [e.clamp(-1.0, 1.0), 0.4, -0.2, 0.3, 0.0, if ph > 0.66 && success { 0.0 } else { 0.9 }] } else { [0.0, -1.0, 0.8, 0.6, 0.0, 0.9] };
        for j in 0..6 {
            q[j] += (target_q[j] - q[j]) * 0.15;
        }
        let reach = 0.12 + 0.22 * (1.0 + q[1].sin()) / 2.0;
        let (ex, ey, ez) = (x + (0.1 + reach) * (yaw + q[0]).cos(), y + (0.1 + reach) * (yaw + q[0]).sin(), 0.25 + 0.25 * (-q[2]).sin().max(-0.5));
        min_ee = min_ee.min(((ex - gx).powi(2) + (ey - gy).powi(2) + (ez - gz).powi(2)).sqrt());
        last_base = dist;
        // 보상
        let row_r = &per_step[k];
        let rs: f64 = row_r.iter().sum();
        ret_cum += rs;
        let mut ev = 0u32;
        if k == 0 {
            ev |= EV_RESET;
        }
        if row_r.iter().any(|v| *v < -0.4) {
            ev |= EV_CONTACT;
        }
        if success && k == n * 2 / 3 && skill != "approach" {
            ev |= EV_GRASP;
        }
        if success && k == n - 1 {
            ev |= EV_SUCCESS;
        }
        // 지도 자람: keyframe 5 프레임마다, 로봇 둘레 2.5 m 원 공개
        let mut known_frac = 0.0;
        if k % 5 == 0 && k > 0 {
            let (cx, cy) = (((x - rm.ox) / rm.res) as i32, ((y - rm.oy) / rm.res) as i32);
            let rr = (2.5 / rm.res) as i32;
            let (x0, y0, x1, y1) = ((cx - rr).max(0), (cy - rr).max(0), (cx + rr).min(rm.w - 1), (cy + rr).min(rm.h - 1));
            let mut ch = false;
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let i = (yy * rm.w + xx) as usize;
                    if (xx - cx).pow(2) + (yy - cy).pow(2) <= rr * rr && known[i] < 0 {
                        known[i] = if rm.occ[i] > 0 { 90 } else { 5 };
                        ch = true;
                    }
                }
            }
            if ch {
                let mut cells = Vec::with_capacity(((x1 - x0 + 1) * (y1 - y0 + 1)) as usize);
                for yy in y0..=y1 {
                    for xx in x0..=x1 {
                        cells.push(known[(yy * rm.w + xx) as usize]);
                    }
                }
                w.map_rect(k as u32, rm.w, rm.h, rm.res, rm.ox, rm.oy, x0, y0, x1, y1, &cells);
            }
        }
        known_frac = if known_frac == 0.0 { known.iter().filter(|v| **v >= 0).count() as f64 / known.len() as f64 } else { known_frac };
        // 슬롯
        let mut sl = vec![0f32; 16 * 15];
        for (i, o) in OBJECTS.iter().enumerate() {
            let (px, py, pz) = obj_pos[i];
            let d = ((px - x).powi(2) + (py - y).powi(2)).sqrt();
            let rel = (py - y).atan2(px - x) - yaw;
            let in_view = d < 3.0 && rel.sin().atan2(rel.cos()).abs() < 0.8;
            if in_view {
                seen[i] = true;
                age[i] = 0.0;
                belief[i] = (belief[i].0 + (px - belief[i].0) * 0.3, belief[i].1 + (py - belief[i].1) * 0.3, pz);
            } else {
                age[i] += 0.1;
            }
            if !seen[i] {
                continue;
            }
            let held = i == 0 && success && skill == "pick" && ph > 0.7;
            let (bx, by, bz) = if held { (ex, ey, ez) } else { belief[i] };
            let unc = ((belief[i].0 - px).powi(2) + (belief[i].1 - py).powi(2)).sqrt() + 0.02;
            let row = [i as f64, bx, by, bz, if held { ex } else { px }, if held { ey } else { py }, if held { ez } else { pz }, o.2, o.3, o.4, if in_view { 1.0 } else { 2.0 }, unc, if held { 3.0 } else { 0.0 }, (i == 0) as i32 as f64, age[i]];
            for (j, v) in row.iter().enumerate() {
                sl[i * 15 + j] = *v as f32;
            }
        }
        let mut row: Vec<f32> = vec![
            (k as f64 * 0.1) as f32, x as f32, y as f32, yaw as f32, sx as f32, sy as f32, syaw as f32, vx as f32, wz as f32,
            q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32, q[4] as f32, q[5] as f32, ex as f32, ey as f32, ez as f32,
            (vx / 0.5) as f32, (wz / 0.9) as f32, q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32, q[4] as f32, q[5] as f32,
        ];
        for v in row_r {
            row.push(*v as f32);
        }
        let value = (if success { 20.0 } else { 2.0 }) * (0.3 + 0.7 * ph) + 0.5 * rng.n();
        row.extend_from_slice(&[ret_cum as f32, value as f32, (ph.powi(4)) as f32, ev as f32, 0.0, gx as f32, gy as f32, gz as f32, known_frac as f32]);
        w.frame(&row, &sl);
    }
    // 시험 자료가 스스로 모순되지 않게: 성공이라 해 놓고 손끝이 컵에 닿지 않았으면(집기·놓기 0.15 m) 또는 로봇이 다가가지 않았으면(다가가기 1 m)
    // 결과를 "n/a" 로(실제로 이룬 일이 아님)
    if success && ((skill != "approach" && min_ee > 0.15) || (skill == "approach" && last_base > 1.0)) {
        line["success"] = Value::Null;
        line["outcome"] = json!("n/a");
        let f2 = format!("ep_{:06}_{}_na.trp", ep_no, skill);
        line["replay"] = json!(f2);
        w.head["meta"] = line.clone();
    }
    w.head["meta"] = line.clone();
    EpOut { line, trp: Some(w) }
}

// ------------------------------------------------------------------------------------------
// 실행 하나

fn progress_row(w: &mut RunWriter, sp: &Spec, rng: &mut Rng, it: usize, iters_per_row: usize, wall: f64, eps_by_skill: &[(String, usize, usize)]) {
    let prog = it as f64 / (sp.rows * iters_per_row) as f64 * sp.speed;
    let steps = (it * 262_144) as f64;
    let a = &mut w.agg;
    a.last("time/iterations", it as f64);
    a.last("time/total_timesteps", steps);
    a.last("time/time_elapsed", wall);
    a.last("time/fps", 262_144.0 * iters_per_row as f64 / 1.0 * (1.0 + 0.03 * rng.n()) / if sp.fp8 { 0.8 } else { 1.0 });
    a.last("time/rollout_ms", 27.0 + 2.0 * rng.n());
    a.last("time/update_ms", if sp.fp8 { 70.0 } else { 86.0 } + 3.0 * rng.n());
    a.last("time/iter_ms", 113.0 + 3.0 * rng.n());
    let n_tot: usize = eps_by_skill.iter().map(|e| e.1).sum::<usize>() * 400;
    a.last("rollout/n_episodes", n_tot as f64);
    let mut ret = 0.0;
    let (mut sw, mut sn) = (0.0, 0.0);
    for (s, n, _) in eps_by_skill {
        if *n > 0 {
            sw += p_succ(s, prog, sp.fp8, sp.seed) * *n as f64;
            sn += *n as f64;
        }
    }
    if sn > 0.0 {
        a.last("rollout/success_rate", (sw / sn + 0.01 * rng.n()).clamp(0.0, 1.0));
    }
    for (s, n, _) in eps_by_skill {
        let ps = p_succ(s, prog, sp.fp8, sp.seed);
        if *n > 0 {
            a.last(&format!("rollout/success_rate/{}", s), (ps + 0.01 * rng.n()).clamp(0.0, 1.0));
            a.last(&format!("rollout/n_episodes/{}", s), (*n * 400) as f64);
            a.last(&format!("rollout/ep_len_mean/{}", s), t_max(s) * 10.0 * (1.0 - 0.6 * ps) + 3.0 * rng.n());
        }
        for t in terms(s) {
            let v = match *t {
                "success" => 20.0 * ps,
                "time" => -0.01 * t_max(s) * 10.0 * (1.0 - 0.6 * ps),
                "collision" | "contact" | "drop" => -10.0 * (1.0 - ps) * 0.3 * (1.0 + 0.5 * (prog * 6.0).sin()),
                _ => 2.0 * sig((prog - 0.1) * 8.0) + 0.1 * rng.n(),
            };
            a.last(&format!("reward/{}/{}", s, t), v);
            ret += v / 3.0;
        }
    }
    a.last("rollout/ep_rew_mean", ret + 0.3 * rng.n());
    a.last("rollout/ep_len_mean", 150.0 * (1.0 - 0.5 * prog.min(1.0)) + 4.0 * rng.n());
    a.last("rollout/timeout_rate", (0.6 * (1.0 - prog).max(0.0) + 0.02 * rng.u()).min(1.0));
    a.last("rollout/collision_rate", 0.05 * (1.0 - prog).max(0.0) + 0.01 + 0.004 * rng.n().abs());
    a.last("rollout/contacts_per_ep", 0.8 * (1.0 - prog * 0.7) + 0.05 * rng.n());
    a.last("rollout/joint_limit_steps", 3.0 * (1.0 - prog).max(0.0) + 0.2 * rng.u());
    a.last("train/policy_gradient_loss", -0.01 + 0.004 * rng.n());
    a.last("train/value_loss", 0.5 * (-prog * 3.0).exp() + 0.05 + 0.01 * rng.n().abs());
    a.last("train/entropy_loss", -(2.0 - 1.5 * prog.min(1.0) + 0.02 * rng.n()));
    a.last("train/log_std", -0.5 - 1.6 * prog.min(1.0) + 0.01 * rng.n());
    a.last("train/approx_kl", (0.008 + 0.003 * rng.n()).abs() * if sp.fp8 { 1.15 } else { 1.0 });
    a.last("train/clip_fraction", 0.12 + 0.02 * rng.n());
    a.last("train/explained_variance", (0.3 + 0.65 * prog.min(1.0) + 0.02 * rng.n()).min(0.995));
    a.last("train/grad_norm", 0.6 + 0.2 * rng.n().abs());
    a.last("train/learning_rate", 3e-4 * (1.0 - 0.7 * prog.min(1.0)));
    a.last("train/advantage_std", 1.0 + 0.1 * rng.n());
    a.last("curriculum/start_completion_mean", 0.3 + 0.2 * prog.min(1.0));
    for (i, st) in ["C0", "C1", "C2"].iter().enumerate() {
        let f = [0.6 - 0.5 * prog.min(1.0), 0.3, 0.1 + 0.5 * prog.min(1.0)][i];
        a.last(&format!("curriculum/start_map_fraction/{}", st), f.max(0.0));
    }
    a.last("gpu/mem_used_mb", 1930.0 + if sp.fp8 { -300.0 } else { 0.0 });
    a.last("log/ring_dropped", 0.0);
    a.sum("log/eps_dropped", if it % 997 == 0 { 3.0 } else { 0.0 });
    if sp.fp8 {
        for l in ["A1", "A2", "A3", "E2"] {
            a.last(&format!("fp8/amax/{}", l), 4.0 + rng.n().abs());
            a.last(&format!("fp8/overflow/{}", l), (1e-5 * rng.u()).max(0.0));
            a.last(&format!("fp8/underflow/{}", l), 0.01 + 0.002 * rng.n().abs());
        }
        a.last("fp8/grad_cos", (0.995 + 0.002 * rng.n()).min(1.0));
    }
    if matches!(sp.kind, "dagger" | "bc") {
        a.last("train/flow_loss", 0.4 * (-prog * 4.0).exp() + 0.05 + 0.005 * rng.n());
        a.last("train/end_bce", 0.2 * (-prog * 3.0).exp() + 0.02);
        a.last("val/flow_loss", 0.45 * (-prog * 3.5).exp() + 0.07 + 0.005 * rng.n());
        a.last("val/action_mse", 0.1 * (-prog * 3.0).exp() + 0.01);
        a.last("dagger/round", (prog * 8.0).floor());
        a.last("dagger/beta", (1.0 - prog * 1.5).max(0.0));
        a.last("dagger/action_mse", 0.3 * (-prog * 3.0).exp() + 0.02);
        a.last("dagger/new_samples", 262144.0);
    }
    for _ in 0..iters_per_row {
        a.tick();
    }
}

fn make_run(root: &Path, sp: &Spec, live_secs: f64) {
    let dir = if sp.lab { root.join("labs").join(&sp.name) } else { root.join(&sp.name) };
    let _ = fs::remove_dir_all(&dir);
    let mut m = meta(sp);
    let live = live_secs > 0.0;
    let mut w = RunWriter::create(&dir, m.clone(), false, !sp.lab && sp.kind == "teacher" && sp.seed == 1 && !sp.fp8).expect("run dir");
    if !live {
        // 정적 실행: 프로세스가 끝나므로 pid 를 빼고 지난 시각으로(목록의 "training" 배지가 안 뜨게)
        let started = now_ts() - sp.rows as f64 - 600.0;
        m["started"] = json!(started);
        m["segments"] = json!([{"started": started}]);
        if sp.rewind_at.is_some() {
            m["segments"] = json!([{"started": started}, {"started": started + 900.0, "from_iter": 1200, "note": "resumed without trim (viewer must drop rewound lines)"}]);
        }
        m["ended"] = json!(now_ts() - 300.0);
        m["schema"] = json!(1);
        w.meta = m;
        w.write_meta().unwrap();
    }
    w.every_s = 0.0;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (sp.seed * 7919 + sp.name.len() as u64 * 104_729));
    let iters_per_row = 6;
    let mut ep_no: u64 = 0;
    let mut replays_left = sp.replays;
    let mut eval_k = 0;
    let rows = if live { usize::MAX } else { sp.rows };
    let t0 = std::time::Instant::now();
    let mut r = 0usize;
    let mut it = 0usize;
    while r < rows {
        if live && t0.elapsed().as_secs_f64() > live_secs {
            break;
        }
        // 되감기 흉내: rewind_at 줄에서 이터를 200 줄 전으로 돌리고(trim 없이) 계속
        if Some(r) == sp.rewind_at {
            it -= 200 * iters_per_row;
        }
        it += iters_per_row;
        let wall = r as f64;
        let prog = it as f64 / (sp.rows * iters_per_row) as f64 * sp.speed;
        // 이 줄 동안 끝난 표본 판(log_envs 8 개 환경의 모든 판)
        let mut eps_by: Vec<(String, usize, usize)> = SKILLS.iter().map(|s| (s.to_string(), 0, 0)).collect();
        let n_eps = 2 + (rng.u() * 4.0) as usize;
        for _ in 0..n_eps {
            let si = (rng.u() * if prog < 0.2 { 1.0 } else if prog < 0.45 { 2.0 } else { 3.0 }) as usize;
            let skill = SKILLS[si];
            let home = *rng.pick(HOMES);
            let stage = if rng.u() < 0.2 { "C0" } else if rng.u() < 0.5 { "C1" } else { "C2" };
            let comp0 = match stage {
                "C0" => 1.0,
                "C1" => (rng.u() * 0.95 + 0.02).min(0.99),
                _ => 0.0,
            };
            let p = (p_succ(skill, prog, sp.fp8, sp.seed) * (0.85 + 0.15 * comp0) * if home == "Pomaria_2_int" { 0.85 } else { 1.0 }).min(1.0);
            ep_no += 1;
            let every = if live { 10 } else { (sp.rows / sp.replays.max(1)).max(1) };
            let want_replay = replays_left > 0 && r % every == every / 2 && eps_by.iter().all(|e| e.1 == 0);
            let driver = if sp.kind == "dagger" { "student" } else { "teacher" };
            let env_no = (rng.u() * 8.0) as u32;
            let e = episode(&mut rng, skill, home, stage, comp0, p, it as f64, (it * 262_144) as f64, ep_no, env_no, driver, want_replay);
            eps_by[si].1 += 1;
            if e.line["success"].as_bool() == Some(true) {
                eps_by[si].2 += 1;
            }
            if let Some(t) = e.trp {
                let f = e.line["replay"].as_str().unwrap().to_string();
                t.finish(&w.replay_dir(None).join(f)).unwrap();
                replays_left -= 1;
            }
            w.episode(None, &e.line);
            // DAgger: 교사가 몬 판은 따로 줄기(3.1)
            if sp.teacher_stream && rng.u() < (1.0 - prog * 1.5).max(0.1) {
                ep_no += 1;
                let e2 = episode(&mut rng, skill, home, stage, comp0, (p + 0.15).min(0.98), it as f64, (it * 262_144) as f64, ep_no, 0, "teacher", false);
                w.episode(Some("teacher"), &e2.line);
            }
        }
        progress_row(&mut w, sp, &mut rng, it, iters_per_row, wall, &eps_by);
        w.flush_row(wall);
        // 평가: 200 줄마다(평균 행동, 줄기 s_eval) + 평가 표
        if sp.eval_stream && r % 200 == 199 {
            eval_k += 1;
            let mut rows_tab = vec![];
            for skill in SKILLS {
                let mut ns = 0;
                for j in 0..20 {
                    ep_no += 1;
                    let home = HOMES[j % HOMES.len()];
                    let comp0 = [0.0, 0.15, 0.5, 0.85, 1.0][j % 5];
                    let e = episode(&mut rng, skill, home, "eval", comp0, (p_succ(skill, prog, sp.fp8, sp.seed) + 0.03).min(1.0), it as f64, (it * 262_144) as f64, ep_no, 0, "teacher", j == 0);
                    if e.line["success"].as_bool() == Some(true) {
                        ns += 1;
                    }
                    if let Some(t) = e.trp {
                        let f = e.line["replay"].as_str().unwrap().replace(".trp", "_pin.trp");
                        let mut l2 = e.line.clone();
                        l2["replay"] = json!(f);
                        t.finish(&w.replay_dir(Some("eval")).join(&f)).unwrap();
                        w.episode(Some("eval"), &l2);
                    } else {
                        w.episode(Some("eval"), &e.line);
                    }
                }
                let sr = ns as f64 / 20.0;
                rows_tab.push(json!({"eval": "skill", "split": skill, "n": 20, "success": sr, "ref": 0.8, "pass": sr >= 0.8}));
                rows_tab.push(json!({"eval": "unseen_homes", "split": skill, "n": 20, "success": (sr - 0.08).max(0.0), "ref": 0.7, "pass": sr - 0.08 >= 0.7}));
            }
            w.eval(&format!("it{:07}", it), &json!({"iter": it, "env_steps": it * 262_144, "ckpt": format!("ckpt_{:06}.bin", it), "rows": rows_tab, "synthetic": true}));
            let _ = eval_k;
        }
        r += 1;
        if live {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
    if sp.half_line {
        // 끝의 반쪽 줄(죽은 학습기 흉내): 뷰어는 줄끝까지 온 줄만 받아야 한다
        let mut f = fs::OpenOptions::new().append(true).open(dir.join("progress.jsonl")).unwrap();
        write!(f, "{{\"time/iter\":{},\"time/env_ste", it + 6).unwrap();
    }
    if live {
        w.finish(r as f64, json!({}));
    }
}

fn main() {
    let mut root: Option<PathBuf> = None;
    let mut live = 0.0;
    let mut only_live = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--root" => root = it.next().map(PathBuf::from),
            "--live" => live = it.next().and_then(|v| v.parse().ok()).unwrap_or(120.0),
            "--only-live" => only_live = true,
            _ => {}
        }
    }
    let root = root.unwrap_or_else(|| {
        eprintln!("usage: fake_run --root DIR [--live SECONDS] [--only-live]");
        std::process::exit(2)
    });
    fs::create_dir_all(&root).unwrap();
    let base = |name: &str, kind: &'static str, group: &str, seed: u64, fp8: bool| Spec {
        name: name.to_string(),
        kind,
        group: group.to_string(),
        seed,
        fp8,
        rows: 1200,
        rewind_at: None,
        half_line: false,
        eval_stream: false,
        teacher_stream: false,
        replays: 0,
        lab: false,
        speed: 1.0,
    };
    if !only_live {
        let mut specs = vec![];
        let mut s = base("fake_teacher_bf16_s1", "teacher", "fake_g6_bf16", 1, false);
        s.eval_stream = true;
        s.replays = 14;
        specs.push(s);
        let mut s = base("fake_teacher_bf16_s2", "teacher", "fake_g6_bf16", 2, false);
        s.rewind_at = Some(900);
        s.half_line = true;
        s.replays = 4;
        specs.push(s);
        let mut s = base("fake_teacher_bf16_s3", "teacher", "fake_g6_bf16", 3, false);
        s.speed = 0.95;
        specs.push(s);
        for seed in 1..=3 {
            let mut s = base(&format!("fake_teacher_fp8_s{}", seed), "teacher", "fake_g6_fp8", seed, true);
            s.speed = 0.97;
            specs.push(s);
        }
        let mut s = base("fake_dagger_s1", "dagger", "fake_dagger", 1, false);
        s.teacher_stream = true;
        s.eval_stream = true;
        s.replays = 6;
        s.rows = 600;
        specs.push(s);
        let mut s = base("try_gru", "lab", "fake_lab", 1, false);
        s.lab = true;
        s.rows = 300;
        s.replays = 2;
        specs.push(s);
        for sp in &specs {
            make_run(&root, sp, 0.0);
            eprintln!("fake_run: {}", sp.name);
        }
        let _ = write_atomic(&root.join("latest.txt"), b"fake_teacher_bf16_s1\n");
    }
    if live > 0.0 {
        let mut s = base("fake_live_teacher", "teacher", "fake_live", 7, false);
        s.replays = 1000;
        s.eval_stream = true;
        s.rows = 600;
        eprintln!("fake_run: live run for {} s (pid {})", live, std::process::id());
        make_run(&root, &s, live);
    }
}

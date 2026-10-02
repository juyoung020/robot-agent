//! 에이전트 쪽 실행 경로. LLM 루프는 [`run_tool`] 하나만 부른다.
//!
//! - [`TcpSim`]: 평가기 안 접착부(`behavior-2026/src/sim/move_robot/move_robot_sim.py --listen`)에 도구 인자 한 줄을 보내고
//!   결과 한 줄을 받는다(줄 단위 JSON, 호출마다 연결 하나). 실행기(닫힌 고리)는 저쪽 프로세스 안 같은 Rust 코드다.
//! - [`Mock`]: 같은 실행기 + 가짜 로봇([`MockPlant`]). Isaac Sim 없이 시험·시연.

use crate::{act, error_obs, parse, Part, Robot, Tick, ACTION_DIM, FINGER_MAX, PROPRIO_DIM};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

pub const DEFAULT_ADDR: &str = "127.0.0.1:8771";

pub trait Backend {
    fn exec(&mut self, args: &Value) -> Value;
}

/// LLM 도구 호출 하나 실행. 인자는 여기서 먼저 검사해(왕복 없이) 틀리면 오류 관찰값을 바로 돌려준다.
pub fn run_tool(args: &Value, backend: &mut dyn Backend) -> Value {
    match parse(args) {
        Err(e) => error_obs(&e),
        Ok(_) => backend.exec(args),
    }
}

pub struct TcpSim {
    pub addr: String,
    pub timeout: Duration,
}

impl TcpSim {
    pub fn new(addr: &str) -> TcpSim {
        TcpSim { addr: addr.into(), timeout: Duration::from_secs(60) }
    }
    fn call(&self, args: &Value) -> Result<Value, String> {
        let a: Vec<std::net::SocketAddr> =
            std::net::ToSocketAddrs::to_socket_addrs(&self.addr).map_err(|e| format!("address {}: {e}", self.addr))?.collect();
        let addr = a.first().ok_or("no address")?;
        let mut s = TcpStream::connect_timeout(addr, Duration::from_secs(3)).map_err(|e| format!("simulator not reachable at {}: {e}", self.addr))?;
        s.set_read_timeout(Some(self.timeout)).ok();
        let args = match args {
            Value::String(t) => serde_json::from_str(t).unwrap_or(Value::Null),
            v => v.clone(),
        };
        let mut line = args.to_string();
        line.push('\n');
        s.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        let mut out = String::new();
        BufReader::new(s).read_line(&mut out).map_err(|e| format!("no answer from simulator: {e}"))?;
        serde_json::from_str(out.trim()).map_err(|e| format!("bad answer from simulator: {e}"))
    }
}

impl Backend for TcpSim {
    fn exec(&mut self, args: &Value) -> Value {
        self.call(args).unwrap_or_else(|e| json!({"status": "error", "message": e, "hint": "the simulator link is down; report it"}))
    }
}

/// 가짜 R1Pro: 관절은 1차 지연(위치 제어기 흉내)·속도 한도, 그리퍼는 손가락 속도 0.25 m/s, 베이스는 지령 속도에 1차 지연.
/// 막힘 시험용으로 관절 멈춤·그리퍼 사이 물체·벽을 넣을 수 있다.
#[derive(Clone, Debug)]
pub struct MockPlant {
    pub dt: f64,
    pub q: [f64; ACTION_DIM],
    pub qd: [f64; ACTION_DIM],
    /// 그리퍼 벌림 비율
    pub grip: [f64; 2],
    pub grip_v: [f64; 2],
    pub base_v: [f64; 3],
    /// 출발점 기준 실제 자세(x, y, yaw)
    pub pose: [f64; 3],
    /// (행동 칸, 이 값 넘어 못 감) — 접촉 흉내
    pub stops: Vec<(usize, f64)>,
    /// 그리퍼 사이 물체: 이 비율 아래로 못 닫음
    pub object_in: [Option<f64>; 2],
    /// 출발점 기준 x 가 이 값 넘으면 베이스가 못 감
    pub wall_x: Option<f64>,
}

impl Default for MockPlant {
    fn default() -> Self {
        let mut q = [0.0; ACTION_DIM];
        q[act::TORSO].copy_from_slice(&[1.025, -1.45, -0.47, 0.0]); // r1pro.yaml reset_joint_pos
        MockPlant {
            dt: 1.0 / crate::HZ,
            q,
            qd: [0.0; ACTION_DIM],
            grip: [1.0, 1.0],
            grip_v: [0.0; 2],
            base_v: [0.0; 3],
            pose: [0.0; 3],
            stops: vec![],
            object_in: [None, None],
            wall_x: None,
        }
    }
}

impl MockPlant {
    pub fn proprio(&self) -> Vec<f32> {
        let mut p = vec![0f32; PROPRIO_DIM];
        let put = |p: &mut Vec<f32>, r: std::ops::Range<usize>, v: &[f64]| {
            for (i, x) in r.zip(v) {
                p[i] = *x as f32;
            }
        };
        put(&mut p, crate::prop::BASE_QVEL, &self.base_v);
        put(&mut p, crate::prop::ARM_L_QPOS, &self.q[act::LEFT_ARM]);
        put(&mut p, crate::prop::ARM_L_QVEL, &self.qd[act::LEFT_ARM]);
        put(&mut p, crate::prop::ARM_R_QPOS, &self.q[act::RIGHT_ARM]);
        put(&mut p, crate::prop::ARM_R_QVEL, &self.qd[act::RIGHT_ARM]);
        put(&mut p, crate::prop::TRUNK_QPOS, &self.q[act::TORSO]);
        put(&mut p, crate::prop::TRUNK_QVEL, &self.qd[act::TORSO]);
        let w = |f: f64| [f * FINGER_MAX, f * FINGER_MAX];
        put(&mut p, crate::prop::GRIP_L_QPOS, &w(self.grip[0]));
        put(&mut p, crate::prop::GRIP_L_QVEL, &w(self.grip_v[0]));
        put(&mut p, crate::prop::GRIP_R_QPOS, &w(self.grip[1]));
        put(&mut p, crate::prop::GRIP_R_QVEL, &w(self.grip_v[1]));
        p
    }

    pub fn step(&mut self, a: &[f32]) {
        let dt = self.dt;
        let k = 1.0 - (-25.0 * dt).exp();
        for i in act::TORSO.chain(act::LEFT_ARM).chain(act::RIGHT_ARM) {
            let old = self.q[i];
            let mut n = old + (a[i] as f64 - old) * k;
            for &(j, lim) in &self.stops {
                if j == i {
                    n = if lim >= 0.0 { n.min(lim) } else { n.max(lim) };
                }
            }
            self.q[i] = n;
            self.qd[i] = (n - old) / dt;
        }
        for (g, idx) in [act::LEFT_GRIPPER, act::RIGHT_GRIPPER].into_iter().enumerate() {
            let target = (a[idx] as f64 + 1.0) / 2.0;
            let rate = 0.25 / FINGER_MAX * dt;
            let old = self.grip[g];
            let mut n = old + (target - old).clamp(-rate, rate);
            if let Some(obj) = self.object_in[g] {
                n = n.max(obj.min(old));
            }
            self.grip[g] = n;
            self.grip_v[g] = (n - old) / dt;
        }
        let kb = 1.0 - (-15.0 * dt).exp();
        for i in 0..3 {
            let cmd = (a[i] as f64).clamp(-1.0, 1.0) * act::BASE_OUT[i];
            self.base_v[i] += (cmd - self.base_v[i]) * kb;
        }
        let (s, c) = self.pose[2].sin_cos();
        let (mut wx, wy) = (c * self.base_v[0] - s * self.base_v[1], s * self.base_v[0] + c * self.base_v[1]);
        if let Some(w) = self.wall_x {
            if self.pose[0] >= w && wx > 0.0 {
                wx = 0.0;
                // 벽에 밀리면 로봇 기준 속도도 0 으로 보인다
                self.base_v = [0.0, 0.0, self.base_v[2]];
            }
        }
        self.pose[0] += wx * dt;
        self.pose[1] += wy * dt;
        self.pose[2] += self.base_v[2] * dt;
    }
}

/// 가짜 집: 정답 바닥 격자(world, 0.05 m) + 머리 카메라 흉내(시야 ±49.6°, 0.4–6 m 광선) → scenemap 처럼 로그 오즈 점유 격자와
/// 가상 스캔을 만든다. 몸통 원(0.28 m)이 장애물 칸에 닿으면 움직이지 않고 접촉을 센다.
pub struct MockWorld {
    /// true = 다닐 수 있는 바닥
    pub floor: crate::map::Grid,
    pub logodds: Vec<f32>,
    pub seen: Vec<bool>,
    pub body_r: f64,
    pub fov: f64,
    pub range: f64,
    pub min_range: f64,
    pub kf_every: u64,
    pub contacts: u64,
    /// 장애물 칸 추가/제거(움직이는 장애물 시험): (스텝, x, y, 반지름, 넣기)
    pub events: Vec<(u64, f64, f64, f64, bool)>,
    pub last_scan_hits: usize,
    pub map_us: u64,
    /// 닿은 자리(world x, y, yaw)와 그때 시각 — 시험·진단
    pub contact_log: Vec<[f64; 4]>,
    /// 지난 스텝에 닿아 있었나(접촉 사건은 닿기 시작할 때 한 번 센다)
    pub in_contact: bool,
    /// 닿아 있던 스텝 수
    pub contact_steps: u64,
}

impl MockWorld {
    /// P5 PGM(255 = 바닥, 행 = y 증가 순) + 원점·해상도
    pub fn from_pgm(path: &str, res: f64, ox: f64, oy: f64) -> Result<MockWorld, String> {
        let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let mut it = 0usize;
        let mut tok = || -> Result<String, String> {
            while it < b.len() && (b[it] as char).is_whitespace() {
                it += 1;
            }
            let st = it;
            while it < b.len() && !(b[it] as char).is_whitespace() {
                it += 1;
            }
            Ok(String::from_utf8_lossy(&b[st..it]).into_owned())
        };
        if tok()? != "P5" {
            return Err("not a P5 pgm".into());
        }
        let w: usize = tok()?.parse().map_err(|_| "w")?;
        let h: usize = tok()?.parse().map_err(|_| "h")?;
        let _mx = tok()?;
        let data = &b[it + 1..it + 1 + w * h];
        let floor = crate::map::Grid { res, ox, oy, w, h, cells: data.iter().map(|&v| if v > 127 { 1 } else { 0 }).collect() };
        Ok(MockWorld::new(floor))
    }
    pub fn new(floor: crate::map::Grid) -> MockWorld {
        let n = floor.w * floor.h;
        MockWorld { floor, logodds: vec![0.0; n], seen: vec![false; n], body_r: 0.28, fov: 49.6f64.to_radians(), range: 6.0, min_range: 0.4, kf_every: 6, contacts: 0, events: vec![], last_scan_hits: 0, map_us: 0, contact_log: vec![], in_contact: false, contact_steps: 0 }
    }
    pub fn is_floor(&self, x: f64, y: f64) -> bool {
        self.floor.at(x, y) == 1
    }
    /// 몸통 사각형(0.55 × 0.52 m, R1Pro 베이스)이 바닥만 덮나. body_r 은 둘레 여유(0 = 딱 맞음)
    pub fn body_ok(&self, x: f64, y: f64, yaw: f64) -> bool {
        let (hl, hw) = (0.275 + self.body_r * 0.0, 0.26);
        let (s, c) = yaw.sin_cos();
        let step = self.floor.res * 0.5;
        let nx = (2.0 * hl / step).ceil() as i64;
        let ny = (2.0 * hw / step).ceil() as i64;
        for i in 0..=nx {
            for j in 0..=ny {
                if i != 0 && i != nx && j != 0 && j != ny && (i % 4 != 0 || j % 4 != 0) {
                    continue;
                }
                let (lx, ly) = (-hl + 2.0 * hl * i as f64 / nx as f64, -hw + 2.0 * hw * j as f64 / ny as f64);
                if !self.is_floor(x + c * lx - s * ly, y + s * lx + c * ly) {
                    return false;
                }
            }
        }
        true
    }
    /// 광선을 쏴 지도에 넣고 MapIn 을 만든다
    pub fn sense(&mut self, pose: [f64; 3], stamp: f64) -> crate::map::MapIn {
        let t0 = std::time::Instant::now();
        let g = &self.floor;
        let mut hits_b = vec![];
        let mut free_b = vec![];
        let n_rays = (2.0 * self.fov / 0.5f64.to_radians()) as usize;
        let mut touched = vec![];
        for k in 0..=n_rays {
            let a = -self.fov + 2.0 * self.fov * k as f64 / n_rays as f64;
            let yaw = pose[2] + a;
            let (s, c) = yaw.sin_cos();
            let step = g.res * 0.5;
            let mut t = 0.0;
            let mut hit = None;
            while t < self.range {
                let (x, y) = (pose[0] + c * t, pose[1] + s * t);
                if g.at(x, y) != 1 {
                    hit = Some(t);
                    break;
                }
                if t >= self.min_range {
                    let (cx, cy) = g.cell_of(x, y);
                    if let Some(i) = g.idx(cx, cy) {
                        touched.push((i, false));
                    }
                }
                t += step;
            }
            match hit {
                Some(t) => {
                    let (x, y) = (pose[0] + c * t, pose[1] + s * t);
                    let (cx, cy) = g.cell_of(x, y);
                    if let Some(i) = g.idx(cx, cy) {
                        touched.push((i, true));
                    }
                    hits_b.push([t * a.cos(), t * a.sin()]);
                }
                None => free_b.push([self.range * a.cos(), self.range * a.sin()]),
            }
        }
        // 한 스캔에서 칸마다 한 번, 맞음 우선
        touched.sort_by_key(|(i, h)| (*i, !*h));
        touched.dedup_by_key(|(i, _)| *i);
        for (i, h) in touched {
            self.seen[i] = true;
            self.logodds[i] = (self.logodds[i] + if h { 0.85 } else { -0.4 }).clamp(-4.0, 4.0);
        }
        self.last_scan_hits = hits_b.len();
        let cells = self
            .logodds
            .iter()
            .zip(&self.seen)
            .map(|(l, s)| if *s { (100.0 / (1.0 + (-l).exp())).round() as i8 } else { -1 })
            .collect();
        let grid = crate::map::Grid { res: g.res, ox: g.ox, oy: g.oy, w: g.w, h: g.h, cells };
        let scan = crate::map::Scan::from_base(pose, [0.0, 0.0], &hits_b, &free_b, stamp);
        self.map_us = t0.elapsed().as_micros() as u64;
        crate::map::MapIn { stamp, pose, grid, rooms: None, n_rooms: 0, scan: Some(scan) }
    }
    fn apply_events(&mut self, step: u64) {
        let mut rest = vec![];
        for e in std::mem::take(&mut self.events) {
            if e.0 > step {
                rest.push(e);
                continue;
            }
            let (_, x, y, r, put) = e;
            let k = (r / self.floor.res).ceil() as i64;
            let (cx, cy) = self.floor.cell_of(x, y);
            for dy in -k..=k {
                for dx in -k..=k {
                    if let Some(i) = self.floor.idx(cx + dx, cy + dy) {
                        let (px, py) = self.floor.center(i);
                        if (px - x).hypot(py - y) <= r {
                            self.floor.cells[i] = if put { 0 } else { 1 };
                        }
                    }
                }
            }
        }
        self.events = rest;
    }
}

/// 실행기 + 가짜 로봇
pub struct Mock {
    pub robot: Robot,
    pub plant: MockPlant,
    pub max_steps: u64,
    pub last_action: [f32; ACTION_DIM],
    /// 실행한 스텝 수(누적)
    pub sim_steps: u64,
    /// 있으면 지도·충돌까지 흉내(plant.pose = world 자세)
    pub world: Option<MockWorld>,
}

impl Default for Mock {
    fn default() -> Self {
        let mut m = Mock { robot: Robot::default(), plant: MockPlant::default(), max_steps: 3000, last_action: [0.0; ACTION_DIM], sim_steps: 0, world: None };
        m.step(); // 첫 관측(유지값 잡기)
        m
    }
}

impl Mock {
    /// 가짜 집에서 start(world x, y, yaw) 로 시작
    pub fn with_world(world: MockWorld, start: [f64; 3]) -> Mock {
        let mut m = Mock::default();
        m.plant.pose = start;
        m.world = Some(world);
        m.max_steps = 30 * 120;
        m.feed_map();
        m
    }
    fn feed_map(&mut self) {
        let pose = self.plant.pose;
        let stamp = self.sim_steps as f64 / crate::HZ;
        if let Some(w) = self.world.as_mut() {
            let mi = w.sense(pose, stamp);
            self.robot.set_map(mi, &crate::nav::MapDelta { unknown_dirty: true, ..Default::default() });
            self.robot.nav.contacts = w.contacts;
        }
    }
    pub fn step(&mut self) -> Tick {
        let p = self.plant.proprio();
        let t = self.robot.tick(&p, &mut self.last_action);
        let before = self.plant.pose;
        self.plant.step(&self.last_action);
        self.sim_steps += 1;
        if let Some(w) = self.world.as_mut() {
            w.apply_events(self.sim_steps);
            if !w.body_ok(self.plant.pose[0], self.plant.pose[1], self.plant.pose[2]) {
                // 닿음: 자세는 그대로, 속도 0 으로 보임
                w.contact_steps += 1;
                if !w.in_contact {
                    w.in_contact = true;
                    w.contacts += 1;
                    w.contact_log.push([before[0], before[1], before[2], self.sim_steps as f64 / crate::HZ]);
                    if let Ok(dir) = std::env::var("MR_DEBUG_DIR") {
                        let _ = std::fs::create_dir_all(&dir);
                        self.robot.debug_dump_now(&dir);
                        let _ = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/contacts.txt")).and_then(|mut f| {
                            use std::io::Write;
                            writeln!(f, "{:.3} {:.3} {:.3} {:.2} cmd {:?}", before[0], before[1], before[2], self.sim_steps as f64 / crate::HZ, &self.last_action[0..3])
                        });
                    }
                }
                self.plant.pose = before;
                self.plant.base_v = [0.0; 3];
            } else {
                w.in_contact = false;
            }
            if self.sim_steps % w.kf_every == 0 {
                self.feed_map();
            }
        }
        t
    }
}

impl Backend for Mock {
    fn exec(&mut self, args: &Value) -> Value {
        if !self.robot.command(args) {
            for _ in 0..self.max_steps {
                if self.step() == Tick::Done {
                    break;
                }
            }
        }
        self.robot.take_result().unwrap_or_else(|| error_obs("mock: no result"))
    }
}

/// 부분 이름 목록(도움말용)
pub fn part_help() -> String {
    crate::PARTS
        .iter()
        .map(|n| {
            let p = Part::from_name(n).unwrap();
            format!("{n}: {} values, {}", p.dof(), p.units())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

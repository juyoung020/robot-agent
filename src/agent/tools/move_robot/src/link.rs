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

/// 실행기 + 가짜 로봇
pub struct Mock {
    pub robot: Robot,
    pub plant: MockPlant,
    pub max_steps: u64,
    pub last_action: [f32; ACTION_DIM],
    /// 실행한 스텝 수(누적)
    pub sim_steps: u64,
}

impl Default for Mock {
    fn default() -> Self {
        let mut m = Mock { robot: Robot::default(), plant: MockPlant::default(), max_steps: 3000, last_action: [0.0; ACTION_DIM], sim_steps: 0 };
        m.step(); // 첫 관측(유지값 잡기)
        m
    }
}

impl Mock {
    pub fn step(&mut self) -> Tick {
        let p = self.plant.proprio();
        let t = self.robot.tick(&p, &mut self.last_action);
        self.plant.step(&self.last_action);
        self.sim_steps += 1;
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

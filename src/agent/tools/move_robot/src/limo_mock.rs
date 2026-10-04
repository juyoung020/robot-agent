//! 가짜 LIMO + OMX-F 와 가짜 물체 기억 — VLA 실행기 배관(끝 신호·확인·결과·안전 거르기)을 Isaac Sim 없이 끝까지 시험한다.
//!
//! - 베이스: 지령 (vx, wz) 에 1차 지연, 팔: 관절 목표에 1차 지연, 그리퍼: 1 /s 로 따라감.
//! - 물체: 손끝(순기구학 [`crate::limo::fk_eef`])이 물체 중심 5 cm 안에서 그리퍼가 닫히면 쥠(벌림 0.3 에서 멈춤), 쥔 물체는 손끝을 따라감,
//!   벌림 > 0.4 면 놓음(받침 상자 위면 그 윗면, 아니면 바닥으로). `phantom` 물체는 기억에만 있고 잡히지 않는다.
//! - 기억: 6 스텝(keyframe)마다 시야(±40°, 3 m) 안이거나 쥔 물체는 지금 위치·시각으로, 아니면 옛 값 그대로. `vanish_step` 뒤엔 `gone`.
//! - 몸통 접촉: `wall_x` 를 넘으려 하면 멈추고 몸통 접촉 1. `world` 를 주면 지도(가짜 깊이 카메라)도 만들어 거르개의 장애물 정지를 시험한다.

use crate::limo::{self, LimoState, ACTION_DIM as A8};
use crate::link::MockWorld;
use crate::verify::{MemObject, ObjState};
use crate::{Robot, Tick};
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct MockObj {
    pub mem: MemObject,
    pub phantom: bool,
    pub vanish_step: Option<u64>,
}

impl MockObj {
    pub fn new(id: &str, pos: [f64; 3], extent: [f64; 3]) -> MockObj {
        MockObj {
            mem: MemObject { id: id.into(), name: id.into(), score: 0.9, pos, extent, first_pos: pos, n_obs: 5, last_seen: 0.0, state: ObjState::Seen },
            phantom: false,
            vanish_step: None,
        }
    }
}

pub struct LimoMock {
    pub robot: Robot,
    pub body: LimoState,
    /// world 자세(= map, 출발 때 같게 둠)
    pub pose: [f64; 3],
    pub objs: Vec<MockObj>,
    /// 실제 물체 위치(기억과 따로)
    pub truth: Vec<[f64; 3]>,
    pub held: Option<usize>,
    pub grasp_frac: f64,
    pub wall_x: Option<f64>,
    pub world: Option<MockWorld>,
    pub contacts_body: u64,
    pub contacts_arm: u64,
    pub sim_steps: u64,
    pub max_steps: u64,
    pub last_out: [f32; A8],
    pub dt: f64,
}

impl LimoMock {
    pub fn new(objs: Vec<MockObj>) -> LimoMock {
        let mut body = LimoState::default();
        body.arm = limo::ARM_HOME;
        body.grip = 0.0;
        body.eef = limo::fk_eef(&body.arm);
        let truth = objs.iter().map(|o| o.mem.pos).collect();
        let mut m = LimoMock {
            robot: Robot::new(30.0),
            body,
            pose: [0.0; 3],
            objs,
            truth,
            held: None,
            grasp_frac: 0.3,
            wall_x: None,
            world: None,
            contacts_body: 0,
            contacts_arm: 0,
            sim_steps: 0,
            max_steps: 30 * 200,
            last_out: [0.0; A8],
            dt: 1.0 / 30.0,
        };
        m.feed();
        m.idle_step();
        m
    }

    /// 물체 하나를 쥔 채로 시작(놓기 시험)
    pub fn start_holding(&mut self, i: usize) {
        self.held = Some(i);
        self.body.grip = self.grasp_frac;
        self.follow();
        self.objs[i].mem.first_pos = self.truth[i];
        self.feed();
        self.robot.vla.filter.init = false;
        self.idle_step();
    }

    pub fn now(&self) -> f64 {
        self.sim_steps as f64 * self.dt
    }

    fn eef_world(&self) -> [f64; 3] {
        let e = limo::fk_eef(&self.body.arm);
        let (s, c) = self.pose[2].sin_cos();
        [self.pose[0] + c * e[0] - s * e[1], self.pose[1] + s * e[0] + c * e[1], e[2]]
    }

    fn follow(&mut self) {
        if let Some(i) = self.held {
            self.truth[i] = self.eef_world();
        }
    }

    /// 기억 갱신 + 지도 + 접촉 → 실행기
    pub fn feed(&mut self) {
        let now = self.now();
        let mut mem = vec![];
        for (i, o) in self.objs.iter_mut().enumerate() {
            if o.vanish_step.map_or(false, |s| self.sim_steps >= s) {
                o.mem.state = ObjState::Gone;
                mem.push(o.mem.clone());
                continue;
            }
            let p = self.truth[i];
            let (dx, dy) = (p[0] - self.pose[0], p[1] - self.pose[1]);
            let bearing = (dy.atan2(dx) - self.pose[2] + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
            let in_view = bearing.abs() < 40f64.to_radians() && dx.hypot(dy) < 3.0;
            if in_view || self.held == Some(i) {
                o.mem.pos = p;
                o.mem.last_seen = now;
                o.mem.n_obs += 1;
            }
            o.mem.state = if self.held == Some(i) { ObjState::Held } else { ObjState::Seen };
            mem.push(o.mem.clone());
        }
        self.robot.vla_set_objects(mem, now);
        self.robot.vla_set_contacts(self.contacts_body, self.contacts_arm);
        if let Some(w) = self.world.as_mut() {
            let mi = w.sense(self.pose, now);
            self.robot.set_map(mi, &crate::nav::MapDelta { unknown_dirty: true, ..Default::default() });
        }
    }

    fn idle_step(&mut self) {
        let p = self.body.to_proprio();
        let mut out = [0f32; A8];
        self.robot.vla_tick(&p, &mut out);
    }

    /// 실행기 한 스텝 + 가짜 몸 한 스텝
    pub fn step(&mut self, ext: Option<(&[f64; A8], f64, f64)>) -> Tick {
        let p = self.body.to_proprio();
        let mut out = [0f32; A8];
        let t = match ext {
            Some((a, e, u)) => self.robot.vla_tick_ext(&p, a, e, u, &mut out),
            None => self.robot.vla_tick(&p, &mut out),
        };
        self.last_out = out;
        self.plant(&out);
        self.sim_steps += 1;
        if self.sim_steps % 6 == 0 {
            self.feed();
        }
        t
    }

    fn plant(&mut self, a: &[f32; A8]) {
        let dt = self.dt;
        let kb = 1.0 - (-15.0 * dt).exp();
        self.body.base_v[0] += (a[0] as f64 - self.body.base_v[0]) * kb;
        self.body.base_v[1] = 0.0;
        self.body.base_v[2] += (a[1] as f64 - self.body.base_v[2]) * kb;
        let (s, c) = self.pose[2].sin_cos();
        let mut nx = self.pose[0] + c * self.body.base_v[0] * dt;
        let ny = self.pose[1] + s * self.body.base_v[0] * dt;
        let mut blocked = false;
        if let Some(w) = self.wall_x {
            if nx + 0.18 > w {
                blocked = true;
            }
        }
        if let Some(w) = self.world.as_ref() {
            if !w.body_ok(nx, ny, self.pose[2]) {
                blocked = true;
            }
        }
        if blocked {
            nx = self.pose[0];
            if self.body.base_v[0].abs() > 1e-3 {
                self.contacts_body += 1;
            }
            self.body.base_v[0] = 0.0;
        } else {
            self.pose[1] = ny;
        }
        self.pose[0] = nx;
        self.pose[2] += self.body.base_v[2] * dt;
        let ka = 1.0 - (-25.0 * dt).exp();
        for j in 0..5 {
            let old = self.body.arm[j];
            self.body.arm[j] = old + (a[2 + j] as f64 - old) * ka;
            self.body.arm_v[j] = (self.body.arm[j] - old) / dt;
        }
        self.body.eef = limo::fk_eef(&self.body.arm);
        let old = self.body.grip;
        let mut g = old + (a[7] as f64 - old).clamp(-dt, dt);
        let ew = self.eef_world();
        if self.held.is_none() && g < old {
            // 닫는 중: 손끝 5 cm 안 물체(진짜만)를 쥔다
            for (i, p) in self.truth.iter().enumerate() {
                let d = ((p[0] - ew[0]).powi(2) + (p[1] - ew[1]).powi(2) + (p[2] - ew[2]).powi(2)).sqrt();
                if d < 0.05 && !self.objs[i].phantom && g <= self.grasp_frac {
                    self.held = Some(i);
                    break;
                }
            }
        }
        if self.held.is_some() {
            if g < self.grasp_frac {
                g = self.grasp_frac;
            }
            if g > self.grasp_frac + 0.1 {
                let i = self.held.take().unwrap();
                let mut p = self.truth[i];
                let ext = self.objs[i].mem.extent;
                let mut z = ext[2] * 0.5;
                for (k, o) in self.objs.iter().enumerate() {
                    if k == i {
                        continue;
                    }
                    let (q, e) = (self.truth[k], o.mem.extent);
                    if (p[0] - q[0]).abs() < e[0] * 0.5 && (p[1] - q[1]).abs() < e[1] * 0.5 {
                        z = q[2] + e[2] * 0.5 + ext[2] * 0.5;
                    }
                }
                p[2] = z;
                self.truth[i] = p;
            }
        }
        self.body.grip_v = (g - old) / dt;
        self.body.grip = g;
        self.follow();
    }

    /// VLA 호출 하나를 끝까지(결과 JSON)
    pub fn run(&mut self, call: &Value) -> Value {
        if self.robot.vla_start(call) {
            return self.robot.take_result().unwrap_or(json!({"status": "error", "message": "no result"}));
        }
        for _ in 0..self.max_steps {
            if self.step(None) == Tick::Done {
                break;
            }
        }
        self.robot.take_result().unwrap_or(json!({"status": "error", "message": "mock: no result in max_steps"}))
    }

    /// 밖의 정책 함수로 끝까지
    pub fn run_ext(&mut self, call: &Value, mut f: impl FnMut(&LimoMock) -> ([f64; A8], f64, f64)) -> Value {
        if self.robot.vla_start(call) {
            return self.robot.take_result().unwrap_or(json!({"status": "error"}));
        }
        for _ in 0..self.max_steps {
            let (a, e, u) = f(self);
            if self.step(Some((&a, e, u))) == Tick::Done {
                break;
            }
        }
        self.robot.take_result().unwrap_or(json!({"status": "error", "message": "mock: no result in max_steps"}))
    }
}

/// 시험·시연용 장면: 컵(O1) 이 앞 0.9 m 오른쪽 0.3 m 낮은 곳, 상자(O2) 가 오른쪽 뒤, 먼 의자(O3) 3 m
pub fn demo_scene() -> Vec<MockObj> {
    vec![
        MockObj::new("O1", [0.9, 0.3, 0.12], [0.06, 0.06, 0.1]),
        MockObj::new("O2", [0.2, -0.9, 0.05], [0.3, 0.3, 0.1]),
        MockObj::new("O3", [3.0, 0.5, 0.4], [0.5, 0.5, 0.8]),
    ]
}

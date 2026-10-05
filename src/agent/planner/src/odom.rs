//! 오도메트리: base_qvel(로봇 기준 vx, vy, wz) 을 30 Hz 로 적분한 이번 판 출발점 기준 위치.
//! `src/scene_graph/scenemap/eval/demo_data.py` `camera_poses()` 와 **같은 식**(전진 오일러, 프레임 i 는 i-1 의 속도로 갱신)
//! 을 써서 씬그래프 좌표와 맞춘다. 로봇 전역 위치(시뮬레이터 정답)는 평가 때 금지라 쓰지 않는다.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

impl Pose {
    /// 세계(출발점) 좌표 점 → 로봇 기준 (앞 +, 왼쪽 +)
    pub fn to_robot(&self, px: f64, py: f64) -> (f64, f64) {
        let (dx, dy) = (px - self.x, py - self.y);
        let (s, c) = self.yaw.sin_cos();
        (c * dx + s * dy, -s * dx + c * dy)
    }
    /// 로봇 기준 점 → 세계
    pub fn to_world(&self, fwd: f64, left: f64) -> (f64, f64) {
        let (s, c) = self.yaw.sin_cos();
        (self.x + c * fwd - s * left, self.y + s * fwd + c * left)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Odom {
    pub pose: Pose,
    pub dist: f64,
    pub turned: f64,
    last: Option<[f64; 3]>,
    pub dt: f64,
}

impl Odom {
    pub fn new(hz: f64) -> Odom {
        Odom { dt: 1.0 / hz, ..Default::default() }
    }
    pub fn reset(&mut self) {
        let dt = self.dt;
        *self = Odom { dt, ..Default::default() };
    }
    /// 새 관측의 base_qvel 을 받는다. 위치는 직전 관측의 속도로 한 스텝 전진.
    pub fn step(&mut self, qvel: [f64; 3]) {
        if let Some([vx, vy, wz]) = self.last {
            let (s, c) = self.pose.yaw.sin_cos();
            self.pose.x += (c * vx - s * vy) * self.dt;
            self.pose.y += (s * vx + c * vy) * self.dt;
            self.pose.yaw += wz * self.dt;
            self.dist += vx.hypot(vy) * self.dt;
            self.turned += wz.abs() * self.dt;
        }
        self.last = Some(qvel);
    }
    pub fn speed(&self) -> (f64, f64) {
        self.last.map(|[vx, vy, wz]| (vx.hypot(vy), wz.abs())).unwrap_or((0.0, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn square_path() {
        let mut o = Odom::new(30.0);
        o.step([1.0, 0.0, 0.0]);
        for _ in 0..30 {
            o.step([1.0, 0.0, 0.0]);
        }
        assert!((o.pose.x - 1.0).abs() < 1e-9);
        let p = o.pose;
        let (f, l) = p.to_robot(2.0, 1.0);
        assert!((f - 1.0).abs() < 1e-9 && (l - 1.0).abs() < 1e-9);
        let (wx, wy) = p.to_world(f, l);
        assert!((wx - 2.0).abs() < 1e-9 && (wy - 1.0).abs() < 1e-9);
    }
}

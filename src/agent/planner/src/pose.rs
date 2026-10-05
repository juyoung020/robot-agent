//! 위치 추정기 자리(교체 가능) + 카메라 자세 계산.
//!
//! 평가 규칙상 로봇 전역 위치(시뮬레이터 정답)는 쓰지 않는다. 쓸 수 있는 것은 관측(proprio)과 영상뿐이다.
//! 추정기는 [`PoseEstimator`] 하나로 감싸서 계획기(`BoundaryEvent.pose`)와 관측 내보내기가 **같은 값**을 쓰게 한다(`link.rs`).
//!
//! - [`QvelIntegrator`]: base_qvel 30 Hz 전진 오일러 적분. [`crate::odom::Odom`] 과 같은 식(프레임 i 는 i-1 의 속도로 갱신).
//! - [`Corrected`]: 바깥 보정 `T_map_odom` 을 앞에 곱한다. scenemap slam2d(깊이 가상 스캔 + 2D 격자 매칭)가 낸 자세를
//!   받는 자리다. 적분만으로는 10 m 이동에 중앙 1.16 m 틀어진다(원본 HDF5 GT 13판 실측) — 기록된 각속도가 실제 회전보다 3~12% 크다.
//!
//! 카메라 외부 자세(베이스 기준 xyz + xyzw, 카메라 prim = OpenGL 축)는 proprio 관절값 + 순기구학([`crate::fk`])으로 만든다
//! (평가기 `cam_rel_poses` 는 쓰지 않음). 광학 프레임 = prim · Rx(π) (OmniGibson `obs_utils.depth_to_pcd` 와 같은 보정).
//! 바깥 추정기(scenemap 의 `pose()`)는 [`correction_from_fix`] 로 한 시점의 절대 자세를 보정으로 바꿔 넣는다.

use crate::odom::{Odom, Pose};

pub type Mat3 = [[f64; 3]; 3];

/// 교체 가능한 위치 추정기. 관측 한 스텝마다 `step` 한 번, 필요할 때 `pose`.
pub trait PoseEstimator: Send {
    fn name(&self) -> String;
    fn reset(&mut self);
    /// 새 관측의 base_qvel(로봇 기준 vx, vy, wz). 위치는 직전 관측의 속도로 한 스텝 전진.
    fn step(&mut self, qvel: [f64; 3]);
    /// 지금 추정(map = 이번 판 출발점, 베이스 x·y·yaw)
    fn pose(&self) -> Pose;
    /// 바깥 보정 `T_map_odom`(평면). 보정을 모르는 추정기는 무시한다.
    fn set_correction(&mut self, _c: Pose) {}
    /// 보정 전 자세(바깥 추정기의 한 시점 자세 → 보정 계산에 씀)
    fn raw_pose(&self) -> Pose {
        self.pose()
    }
}

/// base_qvel 적분(기본).
pub struct QvelIntegrator {
    odom: Odom,
}

impl QvelIntegrator {
    pub fn new(hz: f64) -> QvelIntegrator {
        QvelIntegrator { odom: Odom::new(hz) }
    }
}

impl PoseEstimator for QvelIntegrator {
    fn name(&self) -> String {
        "qvel_integrator".into()
    }
    fn reset(&mut self) {
        self.odom.reset();
    }
    fn step(&mut self, qvel: [f64; 3]) {
        self.odom.step(qvel);
    }
    fn pose(&self) -> Pose {
        self.odom.pose
    }
}

/// `T_map_base = T_map_odom ∘ T_odom_base`. 안쪽 추정기(보통 적분)에 바깥 보정을 곱한다.
pub struct Corrected<E: PoseEstimator> {
    pub inner: E,
    pub correction: Pose,
}

impl<E: PoseEstimator> Corrected<E> {
    pub fn new(inner: E) -> Corrected<E> {
        Corrected { inner, correction: Pose::default() }
    }
}

impl<E: PoseEstimator> PoseEstimator for Corrected<E> {
    fn name(&self) -> String {
        format!("corrected({})", self.inner.name())
    }
    fn reset(&mut self) {
        self.inner.reset();
        self.correction = Pose::default();
    }
    fn step(&mut self, qvel: [f64; 3]) {
        self.inner.step(qvel);
    }
    fn pose(&self) -> Pose {
        compose(&self.correction, &self.inner.pose())
    }
    fn set_correction(&mut self, c: Pose) {
        self.correction = c;
    }
    fn raw_pose(&self) -> Pose {
        self.inner.pose()
    }
}

/// 평면 자세 합성 a ∘ b.
pub fn compose(a: &Pose, b: &Pose) -> Pose {
    let (s, c) = a.yaw.sin_cos();
    Pose { x: a.x + c * b.x - s * b.y, y: a.y + s * b.x + c * b.y, yaw: a.yaw + b.yaw }
}

/// 평면 자세의 역.
pub fn inverse(p: &Pose) -> Pose {
    let (s, c) = p.yaw.sin_cos();
    Pose { x: -(c * p.x + s * p.y), y: -(-s * p.x + c * p.y), yaw: -p.yaw }
}

/// 바깥 추정기가 시각 t 에 준 절대 자세 `fix` 와, 같은 시각의 보정 전 자세 `raw_t` → 보정 `T_map_odom = fix ∘ raw_t⁻¹`.
pub fn correction_from_fix(fix: &Pose, raw_t: &Pose) -> Pose {
    compose(fix, &inverse(raw_t))
}

/// 이름으로 추정기 만들기(`--pose integrate|corrected`).
pub fn make(kind: &str, hz: f64) -> Result<Box<dyn PoseEstimator>, String> {
    match kind {
        "integrate" | "qvel" => Ok(Box::new(QvelIntegrator::new(hz))),
        "corrected" => Ok(Box::new(Corrected::new(QvelIntegrator::new(hz)))),
        o => Err(format!("--pose 는 integrate|corrected: {o}")),
    }
}

pub fn quat_to_mat(x: f64, y: f64, z: f64, w: f64) -> Mat3 {
    let n = x * x + y * y + z * z + w * w;
    let s = if n > 1e-12 { 2.0 / n } else { 0.0 };
    [
        [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
    ]
}

/// 회전 행렬 → 쿼터니언 xyzw (obs_player `mat_to_quat` 와 같은 식).
pub fn mat_to_quat(r: &Mat3) -> [f64; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        return [(r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s, 0.25 * s];
    }
    let i = if r[0][0] >= r[1][1] && r[0][0] >= r[2][2] {
        0
    } else if r[1][1] >= r[2][2] {
        1
    } else {
        2
    };
    let (j, k) = ((i + 1) % 3, (i + 2) % 3);
    let s = (1.0 + r[i][i] - r[j][j] - r[k][k]).sqrt() * 2.0;
    let mut q = [0.0; 4];
    q[i] = 0.25 * s;
    q[j] = (r[j][i] + r[i][j]) / s;
    q[k] = (r[k][i] + r[i][k]) / s;
    q[3] = (r[k][j] - r[j][k]) / s;
    q
}

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

/// map 기준 베이스 자세(z = 0 평면) → (R, t).
pub fn base_rt(p: &Pose) -> (Mat3, [f64; 3]) {
    let (s, c) = p.yaw.sin_cos();
    ([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], [p.x, p.y, 0.0])
}

/// `T_map_cam(광학) = T_map_base × T_base_cam(cam_rel_pose: xyz + xyzw) × Rx(π)`.
pub fn cam_optical(base: &Pose, rel: &[f64; 7]) -> (Mat3, [f64; 3]) {
    const RX_PI: Mat3 = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    let (rwb, twb) = base_rt(base);
    let rbc = mul(&quat_to_mat(rel[3], rel[4], rel[5], rel[6]), &RX_PI);
    let rwc = mul(&rwb, &rbc);
    let twc: [f64; 3] = std::array::from_fn(|r| rwb[r][0] * rel[0] + rwb[r][1] * rel[1] + rwb[r][2] * rel[2] + twb[r]);
    (rwc, twc)
}

/// 두 회전 사이 각(rad).
pub fn rot_angle(a: &Mat3, b: &Mat3) -> f64 {
    // trace(Aᵀ B) = 1 + 2 cos θ
    let mut tr = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            tr += a[k][i] * b[k][i];
        }
    }
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrator_matches_odom() {
        let mut e = QvelIntegrator::new(30.0);
        let mut o = Odom::new(30.0);
        let mut r = crate::util::Rng::new(5);
        for _ in 0..500 {
            let q = [r.f64() - 0.3, r.f64() - 0.5, (r.f64() - 0.5) * 0.8];
            e.step(q);
            o.step(q);
        }
        assert_eq!(e.pose(), o.pose);
    }

    #[test]
    fn correction_composes() {
        let mut e = Corrected::new(QvelIntegrator::new(30.0));
        e.step([1.0, 0.0, 0.0]);
        for _ in 0..30 {
            e.step([1.0, 0.0, 0.0]);
        }
        let raw = e.pose();
        assert!((raw.x - 1.0).abs() < 1e-9);
        e.set_correction(Pose { x: 0.5, y: -0.2, yaw: std::f64::consts::FRAC_PI_2 });
        let p = e.pose();
        assert!((p.x - 0.5).abs() < 1e-9 && (p.y - 0.8).abs() < 1e-9 && (p.yaw - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        e.reset();
        assert_eq!(e.pose(), Pose::default());
        // 바깥 고정점: 시각 t 의 보정 전 자세 raw_t 에서 fix 를 받으면, 그 뒤 적분분을 fix 기준으로 잇는다
        let mut e = Corrected::new(QvelIntegrator::new(30.0));
        for _ in 0..31 {
            e.step([1.0, 0.0, 0.2]);
        }
        let raw_t = e.raw_pose();
        let fix = Pose { x: 5.0, y: 1.0, yaw: 0.3 };
        for _ in 0..30 {
            e.step([0.5, 0.1, -0.1]);
        }
        let raw_now = e.raw_pose();
        e.set_correction(correction_from_fix(&fix, &raw_t));
        let p = e.pose();
        let want = compose(&fix, &compose(&inverse(&raw_t), &raw_now));
        assert!((p.x - want.x).abs() < 1e-9 && (p.y - want.y).abs() < 1e-9 && (p.yaw - want.yaw).abs() < 1e-9);
        let back = compose(&inverse(&fix), &fix);
        assert!(back.x.abs() < 1e-12 && back.y.abs() < 1e-12 && back.yaw.abs() < 1e-12);
    }

    #[test]
    fn camera_optical_frame() {
        // 카메라 prim 이 베이스 앞 0.1 m, 높이 1.2 m, 베이스와 같은 방향(쿼터니언 단위).
        // prim 은 OpenGL 축(-Z 가 앞), 광학은 +Z 가 앞: Rx(π) 뒤 광학 z 축 = prim 의 -z.
        let base = Pose { x: 2.0, y: 1.0, yaw: std::f64::consts::FRAC_PI_2 };
        let (r, t) = cam_optical(&base, &[0.1, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0]);
        // 베이스가 +y 를 봄 → 카메라 위치 = (2, 1.1, 1.2)
        assert!((t[0] - 2.0).abs() < 1e-9 && (t[1] - 1.1).abs() < 1e-9 && (t[2] - 1.2).abs() < 1e-9);
        // 광학 z 축(열 2) = 베이스 기준 -z 를 map 으로 = (0, 0, -1)
        assert!((r[2][2] + 1.0).abs() < 1e-9);
        let q = mat_to_quat(&r);
        let back = quat_to_mat(q[0], q[1], q[2], q[3]);
        assert!(rot_angle(&r, &back) < 1e-9);
        let (r2, _) = cam_optical(&Pose { yaw: base.yaw + 0.1, ..base }, &[0.1, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0]);
        assert!((rot_angle(&r, &r2) - 0.1).abs() < 1e-9);
    }
}

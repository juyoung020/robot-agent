//! LIMO + OMX-F 몸(부분·한계·관측·순기구학). VLA 실행기([`crate::vla`])와 그 안전 거르개가 쓴다.
//!
//! - 부분: `base`(vx m/s, wz rad/s — 차동, POLICY 9절 기본) · `arm`(omx_joint1..5, rad, 관절 목표 위치) · `gripper`(벌림 0 = 닫힘 … 1 = 열림)
//! - 행동 8 = [vx, wz, j1, j2, j3, j4, j5, gripper] (VLA_INPUT 5절, 물리 단위). 시뮬 접착부(move_robot_limo.py)가 평가기 행동 9 로 바꾼다.
//! - 관절 한계: `src/robot/real_limits.json`(실제 OMX-F 범위; 업스트림 URDF 의 ±2π 는 자리표시). 시험이 이 파일과 같은지 본다.
//! - 순기구학: `src/robot/map_vla_description`(map_vla.urdf.xacro + omx_f_arm.urdf.xacro) 의 관절 원점 그대로, base_footprint 기준.
//!   영점 자세 손끝 = (0.273, −0.002, 0.361) m (SIM_PORTING M0, rviz TF) 과 시험으로 맞춘다.

use std::f64::consts::PI;

pub const ACTION_DIM: usize = 8;
/// 평가기 proprio(limo_omx_eval.yaml proprio_obs): base_qvel 3, arm_0_qpos 5, arm_0_qvel 5, eef_0_pos 3, eef_0_quat 4, gripper_0_qpos 2, gripper_0_qvel 2
pub const PROPRIO_DIM: usize = 24;

pub mod act {
    use std::ops::Range;
    pub const BASE: Range<usize> = 0..2;
    pub const ARM: Range<usize> = 2..7;
    pub const GRIPPER: usize = 7;
}

pub mod prop {
    use std::ops::Range;
    pub const BASE_QVEL: Range<usize> = 0..3;
    pub const ARM_QPOS: Range<usize> = 3..8;
    pub const ARM_QVEL: Range<usize> = 8..13;
    pub const EEF_POS: Range<usize> = 13..16;
    pub const EEF_QUAT: Range<usize> = 16..20;
    pub const GRIP_QPOS: Range<usize> = 20..22;
    pub const GRIP_QVEL: Range<usize> = 22..24;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LimoPart {
    Base,
    Arm,
    Gripper,
}

pub const LIMO_PARTS: [&str; 3] = ["base", "arm", "gripper"];

impl LimoPart {
    pub fn from_name(s: &str) -> Option<LimoPart> {
        match s.trim().to_ascii_lowercase().as_str() {
            "base" => Some(LimoPart::Base),
            "arm" | "omx" | "arm_0" => Some(LimoPart::Arm),
            "gripper" | "gripper_0" | "hand" => Some(LimoPart::Gripper),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        LIMO_PARTS[self as usize]
    }
    pub fn dof(self) -> usize {
        match self {
            LimoPart::Base => 2,
            LimoPart::Arm => 5,
            LimoPart::Gripper => 1,
        }
    }
    pub fn slots(self) -> std::ops::Range<usize> {
        match self {
            LimoPart::Base => act::BASE,
            LimoPart::Arm => act::ARM,
            LimoPart::Gripper => act::GRIPPER..act::GRIPPER + 1,
        }
    }
}

/// omx_joint1..5 실제 범위(rad) — `src/robot/real_limits.json` 과 같아야 한다(시험 `limo_limits_match_real_limits_json`)
pub const ARM_LIM: [(f64, f64); 5] = [(-4.712389, 6.283185), (-2.094395, 1.570796), (-2.094395, 1.570796), (-1.745329, 1.745329), (-4.712389, 4.712389)];
/// omx_gripper_joint_1 (rad): 0 = 닫힘, 1.745329 = 100° 열림. gripper_joint_2 는 거울(−joint_1)
pub const GRIPPER_LIM: (f64, f64) = (0.0, 1.745329);
/// 평가기 홈 자세(limo_omx_eval.yaml reset_joint_pos)
pub const ARM_HOME: [f64; 5] = [0.0, -1.6, 1.45, 0.15, 0.0];
/// 한계 안쪽 여유(rad)
pub const JOINT_MARGIN: f64 = 2.0 * PI / 180.0;

pub fn arm_limits() -> [(f64, f64); 5] {
    let mut l = ARM_LIM;
    for x in l.iter_mut() {
        *x = (x.0 + JOINT_MARGIN, x.1 - JOINT_MARGIN);
    }
    l
}

/// 그리퍼 벌림 비율 ↔ joint_1 각
pub fn grip_frac(angle: f64) -> f64 {
    (angle / GRIPPER_LIM.1).clamp(0.0, 1.0)
}

/// proprio 에서 읽은 몸 상태(안 단위)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LimoState {
    /// 로봇 기준 vx, vy, wz
    pub base_v: [f64; 3],
    pub arm: [f64; 5],
    pub arm_v: [f64; 5],
    /// 손끝 위치(로봇 기준, m)
    pub eef: [f64; 3],
    /// 그리퍼 벌림 비율 0..1, 속도(비율/s)
    pub grip: f64,
    pub grip_v: f64,
}

impl LimoState {
    pub fn from_proprio(p: &[f32]) -> Result<LimoState, String> {
        if p.len() < PROPRIO_DIM {
            return Err(format!("proprio has {} values, need {PROPRIO_DIM} (limo_omx)", p.len()));
        }
        if p[..PROPRIO_DIM].iter().any(|x| !x.is_finite()) {
            return Err("proprio has NaN/inf".into());
        }
        let g = |r: std::ops::Range<usize>| -> Vec<f64> { p[r].iter().map(|x| *x as f64).collect() };
        let mut s = LimoState::default();
        s.base_v.copy_from_slice(&g(prop::BASE_QVEL));
        s.arm.copy_from_slice(&g(prop::ARM_QPOS));
        s.arm_v.copy_from_slice(&g(prop::ARM_QVEL));
        s.eef.copy_from_slice(&g(prop::EEF_POS));
        s.grip = grip_frac(p[prop::GRIP_QPOS.start] as f64);
        s.grip_v = p[prop::GRIP_QVEL.start] as f64 / GRIPPER_LIM.1;
        Ok(s)
    }
    pub fn to_proprio(&self) -> Vec<f32> {
        let mut p = vec![0f32; PROPRIO_DIM];
        let mut put = |r: std::ops::Range<usize>, v: &[f64]| {
            for (i, x) in r.zip(v) {
                p[i] = *x as f32;
            }
        };
        put(prop::BASE_QVEL, &self.base_v);
        put(prop::ARM_QPOS, &self.arm);
        put(prop::ARM_QVEL, &self.arm_v);
        put(prop::EEF_POS, &self.eef);
        put(prop::EEF_QUAT, &[0.0, 0.0, 0.0, 1.0]);
        let a = self.grip * GRIPPER_LIM.1;
        put(prop::GRIP_QPOS, &[a, -a]);
        let v = self.grip_v * GRIPPER_LIM.1;
        put(prop::GRIP_QVEL, &[v, -v]);
        p
    }
}

/// 순기구학: omx_joint1..5 → 손끝(omx_end_effector_link) 위치, base_footprint 기준(m).
/// 관절 원점은 URDF 그대로(map_vla.urdf.xacro: base_joint z 0.15, omx_mount x −0.04; omx_f_arm.urdf.xacro).
pub fn fk_eef(q: &[f64; 5]) -> [f64; 3] {
    type M = [[f64; 3]; 3];
    fn mul(a: &M, b: &M) -> M {
        let mut o = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
            }
        }
        o
    }
    fn app(a: &M, v: [f64; 3]) -> [f64; 3] {
        [0, 1, 2].map(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
    }
    let rz = |t: f64| -> M {
        let (s, c) = t.sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    };
    let ry = |t: f64| -> M {
        let (s, c) = t.sin_cos();
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    };
    let rx = |t: f64| -> M {
        let (s, c) = t.sin_cos();
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    };
    // (원점, 회전) 사슬: base_footprint → base_link → omx_link0 → joint1..5 → 손끝
    let chain: [([f64; 3], M); 7] = [
        ([0.0, 0.0, 0.15], rz(0.0)),
        ([-0.04, 0.0, 0.0], rz(0.0)),
        ([-0.01125, 0.0, 0.034], rz(q[0])),
        ([0.0, 0.0, 0.0635], ry(q[1])),
        ([0.0415, 0.0, 0.11315], ry(q[2])),
        ([0.162, 0.0, 0.0], ry(q[3])),
        ([0.0287, 0.0, 0.0], rx(q[4])),
    ];
    let mut r: M = rz(0.0);
    let mut p = [0.0; 3];
    for (o, rot) in chain.iter() {
        let d = app(&r, *o);
        p = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
        r = mul(&r, rot);
    }
    let d = app(&r, [0.09193, -0.0016, 0.0]);
    [p[0] + d[0], p[1] + d[1], p[2] + d[2]]
}

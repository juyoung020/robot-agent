//! R1Pro 카메라 외부 자세(베이스 기준 카메라 prim 자세 = robot2cam)를 proprio 관절값 + 순기구학으로 만든다.
//!
//! 평가기 관측의 `cam_rel_poses` 는 평가기가 시뮬레이터 카메라 자세·로봇 전역 자세 API 로 계산한 값이라 규칙 해석상 쓰지
//! 않는다(코디네이터 09-30: "robot2cam 과 팔 끝 위치는 proprio 관절값 + 로봇 모델 순기구학에서만"). 대신
//! - proprio `trunk_qpos`(53..57)·`arm_left_qpos`(3..10)·`arm_right_qpos`(28..35)
//! - URDF(`r1pro.urdf`) 관절 원점·축
//! - 링크 → 카메라 prim 고정 변환(시연 robot2cam 으로 한 번 뽑은 상수)
//!
//! 으로 계산한다. 표는 `assets/r1pro_cam_fk.json`(R1 Pro 전용 — LIMO 는 안 씀. 옛 시뮬 저장소의 `fit_cam_fk.py` 가 만듦). 시연 2판 482 표본에서 robot2cam 을 위치
//! 최대 0.002 mm·각 0.0001° 로 재현한다.

use crate::pose::{mat_to_quat, mul, quat_to_mat, Mat3};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
struct JointJ {
    #[allow(dead_code)]
    name: String,
    #[serde(rename = "type")]
    kind: String,
    xyz: [f64; 3],
    rpy: [f64; 3],
    axis: [f64; 3],
    q: i64,
}

#[derive(Debug, Clone, Deserialize)]
struct Off {
    xyz: [f64; 3],
    xyzw: [f64; 4],
}

#[derive(Debug, Clone, Deserialize)]
struct CamJ {
    chain: Vec<JointJ>,
    link_to_cam: Off,
}

#[derive(Debug, Clone, Deserialize)]
struct FileJ {
    cams: std::collections::BTreeMap<String, CamJ>,
}

#[derive(Debug, Clone)]
enum Kind {
    Fixed,
    Revolute([f64; 3]),
    Prismatic([f64; 3]),
}

#[derive(Debug, Clone)]
struct Joint {
    r: Mat3,
    t: [f64; 3],
    kind: Kind,
    q: usize,
}

#[derive(Debug, Clone)]
struct Chain {
    joints: Vec<Joint>,
    cam_r: Mat3,
    cam_t: [f64; 3],
}

/// 카메라 셋(0 머리, 1 왼손목, 2 오른손목)의 순기구학.
#[derive(Debug, Clone)]
pub struct CamFk {
    chains: [Option<Chain>; 3],
}

fn rpy(r: f64, p: f64, y: f64) -> Mat3 {
    let (sr, cr) = r.sin_cos();
    let (sp, cp) = p.sin_cos();
    let (sy, cy) = y.sin_cos();
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

fn axis_angle(a: &[f64; 3], th: f64) -> Mat3 {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-12);
    let (x, y, z) = (a[0] / n, a[1] / n, a[2] / n);
    let (s, c) = th.sin_cos();
    let v = 1.0 - c;
    [
        [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
    ]
}

fn mv(r: &Mat3, v: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2])
}

impl CamFk {
    pub fn default_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/r1pro_cam_fk.json")
    }

    pub fn load(path: &Path) -> Result<CamFk, String> {
        let t = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let f: FileJ = serde_json::from_str(&t).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut chains: [Option<Chain>; 3] = [None, None, None];
        for (name, c) in f.cams {
            let id = match name.as_str() {
                "head" => 0,
                "left_wrist" => 1,
                "right_wrist" => 2,
                _ => continue,
            };
            let joints = c
                .chain
                .iter()
                .map(|j| Joint {
                    r: rpy(j.rpy[0], j.rpy[1], j.rpy[2]),
                    t: j.xyz,
                    kind: match j.kind.as_str() {
                        "revolute" | "continuous" => Kind::Revolute(j.axis),
                        "prismatic" => Kind::Prismatic(j.axis),
                        _ => Kind::Fixed,
                    },
                    q: j.q.max(0) as usize,
                })
                .collect();
            let o = &c.link_to_cam;
            chains[id] = Some(Chain { joints, cam_r: quat_to_mat(o.xyzw[0], o.xyzw[1], o.xyzw[2], o.xyzw[3]), cam_t: o.xyz });
        }
        Ok(CamFk { chains })
    }

    /// 카메라 `cam` 의 베이스 기준 prim 자세(xyz + xyzw, OpenGL 축 — 평가기 cam_rel_poses·시연 robot2cam 과 같은 뜻).
    pub fn cam_rel(&self, cam: usize, proprio: &[f32]) -> Option<[f64; 7]> {
        let c = self.chains.get(cam)?.as_ref()?;
        let mut r: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut t = [0.0f64; 3];
        for j in &c.joints {
            let d = mv(&r, &j.t);
            t = [t[0] + d[0], t[1] + d[1], t[2] + d[2]];
            r = mul(&r, &j.r);
            match &j.kind {
                Kind::Fixed => {}
                Kind::Revolute(a) => r = mul(&r, &axis_angle(a, *proprio.get(j.q)? as f64)),
                Kind::Prismatic(a) => {
                    let q = *proprio.get(j.q)? as f64;
                    let d = mv(&r, &[a[0] * q, a[1] * q, a[2] * q]);
                    t = [t[0] + d[0], t[1] + d[1], t[2] + d[2]];
                }
            }
        }
        let d = mv(&r, &c.cam_t);
        let tc = [t[0] + d[0], t[1] + d[1], t[2] + d[2]];
        let q = mat_to_quat(&mul(&r, &c.cam_r));
        Some([tc[0], tc[1], tc[2], q[0], q[1], q[2], q[3]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 시연 200번 판 3000 프레임(observation.state 61 + robot2cam 3대) — fit_cam_fk.py 가 쓴 표본과 같은 데이터.
    const STATE: [f32; 61] = [
        -0.0003644, 0.0004747, 0.0005906, -0.5135943, 0.1073003, -0.0477751, -1.0079254, 0.3504786, 0.6210044, 0.5144721, -0.0518736,
        -0.0281572, -0.0266854, -0.2630615, 0.6464736, -0.2789042, 0.010686, 0.6501044, 0.3716516, 0.5129846, -0.1484585, 0.941596,
        0.1572233, 0.2581432, 0.0075572, 0.0003005, 0.0209931, -0.0171822, -0.4822987, 0.1745, 0.7338722, -1.5772073, -0.0225522,
        1.0406232, 0.0773677, -0.0002653, 0.0003976, 0.0011009, 0.0045551, 0.0169314, 0.0107928, 0.0021622, 0.5823845, 0.0940711,
        0.6938694, -0.3479472, 0.8557385, 0.1888288, 0.3331488, 0.05, 0.0244954, 0.008631, 0.0002444, 1.2696129, -1.896482,
        -0.9405322, -0.0004273, -0.0026646, -0.0017128, 0.0187404, -0.0033622,
    ];
    const R2C: [[f64; 7]; 3] = [
        [0.3747753, -2.79e-05, 1.2254213, 0.3100148, -0.3102268, -0.6355818, 0.6353628],
        [0.6619903, 0.3600194, 0.5937633, 0.1189633, 0.0504662, -0.5638065, 0.8157347],
        [0.5787872, 0.0836327, 0.7755998, 0.1755402, -0.0232762, -0.3736257, 0.9105206],
    ];

    #[test]
    fn reproduces_demo_robot2cam() {
        let fk = CamFk::load(&CamFk::default_path()).expect("r1pro_cam_fk.json");
        for cam in 0..3 {
            let p = fk.cam_rel(cam, &STATE).unwrap();
            let want = R2C[cam];
            let dp = ((p[0] - want[0]).powi(2) + (p[1] - want[1]).powi(2) + (p[2] - want[2]).powi(2)).sqrt();
            // 관절값이 f32·7자리 반올림이라 위치 0.1 mm, 각 0.01° 안
            assert!(dp < 1e-4, "cam {cam}: 위치 {dp} m");
            let ra = quat_to_mat(p[3], p[4], p[5], p[6]);
            let rb = quat_to_mat(want[3], want[4], want[5], want[6]);
            let ang = crate::pose::rot_angle(&ra, &rb).to_degrees();
            assert!(ang < 0.01, "cam {cam}: 각 {ang}°");
        }
    }
}

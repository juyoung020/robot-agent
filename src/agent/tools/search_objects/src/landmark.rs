//! 기준물(landmark) — 대상 둘레 1.5 m 안에서 가장 가까운 고정 가구 하나: id, 이름, 거리, 높이 차이.
//!
//! 물체 사이 관계말(on / in / next to …)은 **계산하지 않는다**(10-05 결정). LLM 이 자리·크기 숫자로 말을 고른다.
//! 고정 가구 = 기억에서 `movable == false`, 가장 긴 변 ≥ 0.3 m, `gone` 아님, 대상 자신이 아님.

use crate::memview::{Memory, ObjInfo};

pub const RADIUS_M: f64 = 1.5;
const MIN_SIZE_M: f64 = 0.3;

#[derive(Clone, Debug, PartialEq)]
pub struct Landmark {
    pub id: u32,
    /// 수평 거리: 대상 중심에서 가구 상자까지(m, 안이면 0)
    pub dist_m: f64,
    /// 대상 중심 높이 − 가구 윗면 높이(m)
    pub dz_m: f64,
}

pub fn nearest_fixed(mem: &dyn Memory, target: &ObjInfo) -> Option<Landmark> {
    mem.objects()
        .iter()
        .filter(|o| o.id != target.id && !o.movable && o.state != "gone")
        .filter(|o| o.extent.iter().cloned().fold(0.0, f64::max) >= MIN_SIZE_M)
        .map(|o| (o, o.xy_dist_to_box(target.pos[0], target.pos[1])))
        .filter(|(_, d)| *d <= RADIUS_M)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(o, d)| Landmark { id: o.id, dist_m: d, dz_m: target.pos[2] - o.top_z() })
}

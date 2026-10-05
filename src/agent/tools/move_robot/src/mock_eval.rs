//! 가짜 집 평가(LLM 에게 안 보임): 정답 바닥 PGM + `gt.json` 으로 [`Mock`] 을 만들고, 덮음(`_m.gt_cov`)을 잴 정답 기준을 넣는다.
//! 옛 스킬 explore `src/bin/explore.rs` 에서 옮김(10-06). 시뮬 판의 기준은 `src/sim/explore/gt_trav.py`·`run_explore.py` 가 만든다.

use crate::link::{Mock, MockWorld};
use crate::map::Grid;
use serde_json::Value;

/// `gt.json`(res, origin, start_pose_world) + PGM → 가짜 로봇. `start` = 월드 [x, y, yaw rad](없으면 gt.json 출발).
pub fn mock_from_gt(pgm: &str, gt_json: &str, start: Option<[f64; 3]>) -> Result<Mock, String> {
    let gt: Value = serde_json::from_str(&std::fs::read_to_string(gt_json).map_err(|e| format!("{gt_json}: {e}"))?).map_err(|e| format!("{gt_json}: {e}"))?;
    let res = gt["res"].as_f64().ok_or("gt.json: res")?;
    let (ox, oy) = (gt["origin"][0].as_f64().ok_or("gt.json: origin")?, gt["origin"][1].as_f64().ok_or("gt.json: origin")?);
    let world = MockWorld::from_pgm(pgm, res, ox, oy)?;
    let sp = &gt["start_pose_world"];
    let start = start.unwrap_or([sp[0].as_f64().unwrap_or(0.0), sp[1].as_f64().unwrap_or(0.0), sp[2].as_f64().unwrap_or(0.0)]);
    let reference = reachable_reference(&world.floor, start);
    let mut m = Mock::with_world(world, start);
    m.robot.set_reference(reference);
    Ok(m)
}

/// 정답 기준: 바닥 칸 중 출발점에서 몸통 부풀림 원(LIMO, [`crate::nav::Body::limo_omx`] robot_r 0.24 m)이 닿는 곳(벽에서 그만큼 떨어진 칸으로 이어진 연결 성분)과 그 둘레 0.6 m
/// (카메라로 볼 수 있는 바닥). 값 1 = 기준 칸.
pub fn reachable_reference(floor: &Grid, start: [f64; 3]) -> Grid {
    let mut obst = floor.clone();
    for v in obst.cells.iter_mut() {
        *v = if *v == 1 { 0 } else { 100 };
    }
    let d = crate::map::obstacle_distance(&obst, 1.0);
    let reach_r = crate::nav::Body::limo_omx().robot_r as f32;
    let (w, h) = (floor.w as i64, floor.h as i64);
    let mut reach = vec![false; floor.cells.len()];
    let (sx, sy) = floor.cell_of(start[0], start[1]);
    let mut q = std::collections::VecDeque::new();
    if let Some(i) = floor.idx(sx, sy) {
        reach[i] = true;
        q.push_back(i);
    }
    while let Some(i) = q.pop_front() {
        let (x, y) = ((i % floor.w) as i64, (i / floor.w) as i64);
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                continue;
            }
            let j = (ny * w + nx) as usize;
            if !reach[j] && floor.cells[j] == 1 && (d[j] >= reach_r || (nx - sx).pow(2) + (ny - sy).pow(2) < 64) {
                reach[j] = true;
                q.push_back(j);
            }
        }
    }
    // 둘레 0.6 m 의 바닥(로봇이 서서 볼 수 있는 곳)
    let mut out = floor.clone();
    let k = (0.6 / floor.res) as i64;
    let mut dist = vec![i64::MAX; floor.cells.len()];
    let mut q = std::collections::VecDeque::new();
    for i in 0..reach.len() {
        if reach[i] {
            dist[i] = 0;
            q.push_back(i);
        }
    }
    while let Some(i) = q.pop_front() {
        if dist[i] >= k {
            continue;
        }
        let (x, y) = ((i % floor.w) as i64, (i / floor.w) as i64);
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                continue;
            }
            let j = (ny * w + nx) as usize;
            if floor.cells[j] == 1 && dist[j] == i64::MAX {
                dist[j] = dist[i] + 1;
                q.push_back(j);
            }
        }
    }
    for i in 0..out.cells.len() {
        out.cells[i] = (dist[i] != i64::MAX) as i8;
    }
    out
}

//! 베이스 주행 층(Nav2 와 같은 층 구조, ROS 없이): 비용 지도 → 전역 경로(바뀐 칸이 경로 통로에 걸릴 때만 다시) →
//! 지역 제어(DWA 속도 표본, 몸통 원 충돌 검사 + 멈춤 거리) → 회복(늦춤 → 비켜 가기 → 멈추고 돌아보기 → 다시 계획 → 뒤로 →
//! `blocked` 보고와 대안).
//!
//! 지도는 하나: scenemap 2D 점유 격자(slam2d 로그 오즈 — 맞은 칸 +, 광선이 지난 칸 − 라 치운 물체는 비워진다)가 계획·비용의
//! 유일한 출처다. 바뀐 칸은 scenemap 이 주는 바뀐 영역(`sgrt_map_view.dirty_box`, `sm_take_dirty`) 안에서만 비교한다.
//! 물체 기억의 옮길 수 있는 물체는 "움직일 수 있음" 비용 힌트(둘레 벌점)로만 쓴다.
//! 지도 갱신 사이(keyframe 6 스텝)·카메라 발밑·움직이는 장애물은 지도를 쓰지 않는 얇은 안전 정지 [`depth_guard`]
//! (마지막 깊이 프레임 장애물 점, 앞 몸통 통로 + 멈춤 거리)가 막는다.

use crate::map::{ang_diff, is_free, is_occ, obstacle_distance, Analysis, Grid, MapIn, NavParams, Scan};

/// 비용 지도: scenemap 격자 복사(버퍼 다시 씀) + 장애물 거리장 + 바뀐 칸.
#[derive(Clone, Debug, Default)]
pub struct Costmap {
    pub grid: Grid,
    pub dobs: Vec<f32>,
    /// 옮길 수 있는 물체 둘레 벌점(0..1, 칸마다) — 계획 비용 힌트
    pub soft: Vec<f32>,
    /// 지난 갱신과 장애물 여부가 바뀐 칸. None = 격자 모양이 바뀜(전부 바뀐 셈)
    pub changed: Option<Vec<usize>>,
    pub stamp: f64,
    pub build_us: u64,
    pub scan: Option<Scan>,
    pub version: u64,
    pub map_version: u64,
    tmp: Grid,
}

/// 지도 밖에서 오는 갱신(sgrt_map_view 의 바뀐 영역 + 물체)
#[derive(Clone, Debug, Default)]
pub struct MapDelta {
    /// 바뀐 칸 경계 상자(이 격자 칸 좌표, 끝 포함). None = 안 바뀜, 또는 모름이면 `unknown_dirty`
    pub dirty: Option<[i64; 4]>,
    /// 바뀐 영역을 모름(가짜 로봇·옛 라이브러리): 전부 비교
    pub unknown_dirty: bool,
    pub map_version: u64,
    /// 옮길 수 있는 물체 (x, y, r)
    pub movable: Vec<[f64; 3]>,
}

impl Costmap {
    pub fn update(&mut self, m: &MapIn, d: &MapDelta, now: f64) {
        let t0 = std::time::Instant::now();
        let same = self.grid.w == m.grid.w && self.grid.h == m.grid.h && self.grid.ox == m.grid.ox && self.grid.oy == m.grid.oy && !self.grid.cells.is_empty();
        let n = m.grid.w * m.grid.h;
        let mut changed = None;
        if same {
            // 바뀐 영역 안에서만 장애물 여부 비교
            let (w, h) = (m.grid.w as i64, m.grid.h as i64);
            let bx = if d.unknown_dirty { Some([0, 0, w - 1, h - 1]) } else { d.dirty };
            let mut ch = vec![];
            if let Some([x0, y0, x1, y1]) = bx {
                for y in y0.max(0)..=y1.min(h - 1) {
                    for x in x0.max(0)..=x1.min(w - 1) {
                        let i = (y * w + x) as usize;
                        if is_occ(self.grid.cells[i]) != is_occ(m.grid.cells[i]) || is_free(self.grid.cells[i]) != is_free(m.grid.cells[i]) {
                            ch.push(i);
                        }
                    }
                }
            }
            changed = Some(ch);
        }
        let need_dt = !same || changed.as_ref().map_or(true, |c| c.iter().any(|&i| is_occ(self.grid.cells[i]) != is_occ(m.grid.cells[i])));
        self.grid.res = m.grid.res;
        self.grid.ox = m.grid.ox;
        self.grid.oy = m.grid.oy;
        self.grid.w = m.grid.w;
        self.grid.h = m.grid.h;
        self.grid.cells.clear();
        self.grid.cells.extend_from_slice(&m.grid.cells);
        if need_dt || self.dobs.len() != n {
            self.dobs = obstacle_distance(&self.grid, 2.5);
        }
        // 옮길 수 있는 물체 둘레 벌점(반지름 + 0.4 m 까지 1 → 0)
        self.soft.clear();
        self.soft.resize(n, 0.0);
        let g = &self.grid;
        for o in &d.movable {
            let rr = o[2] + 0.4;
            let (cx, cy) = g.cell_of(o[0], o[1]);
            let k = (rr / g.res).ceil() as i64;
            for y in cy - k..=cy + k {
                for x in cx - k..=cx + k {
                    if let Some(i) = g.idx(x, y) {
                        let (px, py) = g.center(i);
                        let dd = (px - o[0]).hypot(py - o[1]);
                        if dd < rr {
                            self.soft[i] = self.soft[i].max((1.0 - (dd - o[2]).max(0.0) / 0.4) as f32);
                        }
                    }
                }
            }
        }
        self.changed = changed;
        self.scan = m.scan.clone();
        self.stamp = now;
        self.version += 1;
        self.map_version = d.map_version;
        self.build_us = t0.elapsed().as_micros() as u64;
        let _ = &self.tmp;
    }

    /// 점 (x, y) 에서 가장 가까운 장애물 칸 "면"까지(m): 칸 중심 거리장을 쌍선형 보간하고 반 칸을 뺀다(접촉 판정과 같은 뜻)
    #[inline]
    pub fn clear_at(&self, x: f64, y: f64) -> f64 {
        let g = &self.grid;
        if g.w == 0 {
            return 2.5;
        }
        let fx = (x - g.ox) / g.res - 0.5;
        let fy = (y - g.oy) / g.res - 0.5;
        let (x0, y0) = (fx.floor() as i64, fy.floor() as i64);
        let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
        let d = |cx: i64, cy: i64| -> f64 {
            match g.idx(cx, cy) {
                Some(i) => self.dobs[i] as f64,
                None => 2.5,
            }
        };
        let v = (1.0 - ty) * ((1.0 - tx) * d(x0, y0) + tx * d(x0 + 1, y0)) + ty * ((1.0 - tx) * d(x0, y0 + 1) + tx * d(x0 + 1, y0 + 1));
        // 보간은 장애물 칸 바로 옆에서 실제보다 크게 나올 수 있다: 네 칸 중 가장 작은 것 + 한 칸을 넘지 않게
        let mn = d(x0, y0).min(d(x0 + 1, y0)).min(d(x0, y0 + 1)).min(d(x0 + 1, y0 + 1));
        v.min(mn + g.res) - 0.5 * g.res
    }
    #[inline]
    pub fn unknown_at(&self, x: f64, y: f64) -> bool {
        let (cx, cy) = self.grid.cell_of(x, y);
        match self.grid.idx(cx, cy) {
            Some(i) => self.grid.cells[i] < 0,
            None => true,
        }
    }

    pub fn as_map(&self, pose: [f64; 3], rooms: Option<crate::map::RoomGrid>, n_rooms: i32) -> MapIn {
        MapIn { stamp: self.stamp, pose, grid: self.grid.clone(), rooms, n_rooms, scan: self.scan.clone() }
    }
}

/// 비용 지도로 Analysis(도달 거리·프런티어)를 만든다(거리장·물체 벌점은 비용 지도 것을 씀)
pub fn analyze(cm: &Costmap, pose: [f64; 3], rooms: Option<crate::map::RoomGrid>, n_rooms: i32, p: &NavParams) -> (MapIn, Analysis) {
    let m = cm.as_map(pose, rooms, n_rooms);
    let a = Analysis::new_full(&m, cm.dobs.clone(), Some(&cm.soft), p);
    (m, a)
}

/// 얇은 안전 정지(지도 안 씀): 마지막 깊이 프레임의 장애물 점 중 진행 방향 몸통 통로(반폭 body_r) 안 가장 가까운 것까지
/// 몸통 원이 갈 수 있는 거리. 점들은 그 프레임 뒤 움직인 만큼 지금 자세로 옮겨 본다(점은 map 좌표라 그대로).
/// 진행 방향을 카메라가 못 보면(±50° 밖) None.
pub fn depth_guard(sc: &Scan, pose: [f64; 3], heading: f64, body_r: f64, max_m: f64) -> Option<f64> {
    depth_guard_f(sc, pose, heading, body_r, max_m, |_| true)
}

/// 지도에 이미 있는 장애물 점은 빼고(지역 제어가 이미 피함) — 지도에 없는(새로 나타난·움직이는·발밑) 것만
pub fn depth_guard_unmapped(cm: &Costmap, pose: [f64; 3], heading: f64, body_r: f64, max_m: f64) -> Option<f64> {
    let sc = cm.scan.as_ref()?;
    depth_guard_f(sc, pose, heading, body_r, max_m, |q| cm.clear_at(q[0], q[1]) > 0.08)
}

fn depth_guard_f(sc: &Scan, pose: [f64; 3], heading: f64, body_r: f64, max_m: f64, keep: impl Fn(&[f64; 2]) -> bool) -> Option<f64> {
    let rel = ang_diff(heading, pose[2]);
    if rel.abs() > 50f64.to_radians() {
        return None;
    }
    let (s, c) = heading.sin_cos();
    let mut best = max_m;
    for q in &sc.hits {
        let (fx, fy) = (q[0] - pose[0], q[1] - pose[1]);
        let lon = fx * c + fy * s;
        let lat = -fx * s + fy * c;
        if lon <= -0.05 || lat.abs() >= body_r || !keep(q) {
            continue;
        }
        let t = lon - (body_r * body_r - lat * lat).sqrt();
        if t < best {
            best = t;
        }
    }
    Some(best.max(0.0))
}

// ---------------------------------------------------------------- 지역 제어(DWA)

#[derive(Clone, Debug)]
pub struct DwaParams {
    pub body_r: f64,
    pub horizon_s: f64,
    pub dt: f64,
    pub nv: usize,
    pub nw: usize,
    pub acc: f64,
}

impl Default for DwaParams {
    fn default() -> Self {
        DwaParams { body_r: 0.30, horizon_s: 1.2, dt: 0.1, nv: 6, nw: 13, acc: 0.8 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DwaOut {
    pub v: f64,
    pub w: f64,
    /// 고른 궤적의 가장 작은 여유(장애물까지 − 몸통)
    pub clear: f64,
    pub ok: bool,
}

/// 몸통 원이 (x, y) 에서 안전한가: 장애물 거리 ≥ body_r + margin, 그리고 모르는 칸 아님(출발 둘레 빼고)
fn safe_at(cm: &Costmap, x: f64, y: f64, body: f64, start: [f64; 2], start_r: f64, allow_unknown: bool) -> Option<f64> {
    let c = cm.clear_at(x, y) - body;
    if c < 0.0 {
        return None;
    }
    if !allow_unknown && cm.unknown_at(x, y) && (x - start[0]).hypot(y - start[1]) > start_r {
        return None;
    }
    Some(c)
}

/// 속도 표본(v, w)마다 horizon 동안 굴려 보고 안전한 것 중 점수가 가장 좋은 것. 앞으로만(카메라가 앞을 봄).
pub fn dwa(cm: &Costmap, path: &[[f64; 2]], pose: [f64; 3], anchor: [f64; 2], carrot: [f64; 2], goal_dist: f64, vmax: f64, wmax: f64, margin: f64, dp: &DwaParams, start_r: f64, allow_unknown: bool) -> DwaOut {
    let mut best: Option<(f64, DwaOut)> = None;
    let start = anchor;
    // 좁은 곳에 이미 들어와 있으면(여유 < 몸통) 여유를 더 줄이지 않는 궤적만(빠져나오기), 바닥은 몸통 − 1.5 cm
    let cur = cm.clear_at(pose[0], pose[1]) - margin;
    let need = (cur - 0.005).min(dp.body_r).max(dp.body_r - 0.015);
    let steps = (dp.horizon_s / dp.dt).round() as usize;
    for iv in 0..dp.nv {
        let v = vmax * iv as f64 / (dp.nv - 1) as f64;
        // 멈춤 거리(v²/2a)는 이 궤적 위에서 안전해야 한다
        let brake = v * v / (2.0 * dp.acc);
        for iw in 0..dp.nw {
            let w = -wmax + 2.0 * wmax * iw as f64 / (dp.nw - 1) as f64;
            let (mut x, mut y, mut th) = (pose[0], pose[1], pose[2]);
            let mut minc = f64::INFINITY;
            let mut ok = true;
            let mut travelled = 0.0;
            for _ in 0..steps {
                th += w * dp.dt;
                x += v * th.cos() * dp.dt;
                y += v * th.sin() * dp.dt;
                travelled += v * dp.dt;
                // 모르는 칸: 전역 경로(아는 빈칸) 0.35 m 안이면 괜찮다(경로를 따라가는 작은 벗어남)
                let near_path = path.len() >= 2 && (0..path.len() - 1).any(|k| seg_dist(path[k], path[k + 1], [x, y]) < 0.35);
                match safe_at(cm, x, y, need + margin, start, start_r, allow_unknown || near_path) {
                    Some(c) => minc = minc.min(c),
                    None => {
                        // 멈춤 거리 안에서 막히면 이 표본은 못 씀
                        if travelled <= brake + 0.25 || v > 0.0 {
                            ok = false;
                        }
                        break;
                    }
                }
                if travelled > goal_dist {
                    break;
                }
            }
            if !ok {
                continue;
            }
            if minc.is_infinite() {
                minc = cm.clear_at(pose[0], pose[1]) - dp.body_r;
            }
            let d_end = (carrot[0] - x).hypot(carrot[1] - y);
            let h_err = ang_diff((carrot[1] - y).atan2(carrot[0] - x), th).abs();
            let score = -2.0 * d_end - 0.6 * h_err + 0.8 * minc.min(0.4) + 0.3 * v / vmax.max(1e-6);
            if best.map_or(true, |b| score > b.0) {
                best = Some((score, DwaOut { v, w, clear: minc, ok: true }));
            }
        }
    }
    match best {
        Some((_, o)) => o,
        None => DwaOut { v: 0.0, w: 0.0, clear: 0.0, ok: false },
    }
}

/// 경로(점들) 위 가장 가까운 곳에서 la m 앞의 점, 그리고 경로 끝까지 남은 길이
pub fn carrot_on(path: &[[f64; 2]], p: [f64; 2], la: f64) -> ([f64; 2], f64, usize) {
    if path.len() < 2 {
        let e = *path.last().unwrap_or(&p);
        return (e, (e[0] - p[0]).hypot(e[1] - p[1]), 0);
    }
    // 가장 가까운 선분
    let mut best = (f64::INFINITY, 0usize, 0.0f64);
    for k in 0..path.len() - 1 {
        let (a, b) = (path[k], path[k + 1]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l2 = (dx * dx + dy * dy).max(1e-12);
        let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0);
        let q = [a[0] + t * dx, a[1] + t * dy];
        let d = (q[0] - p[0]).hypot(q[1] - p[1]);
        if d < best.0 {
            best = (d, k, t);
        }
    }
    let (_, k0, t0) = best;
    let seg = |k: usize| (path[k + 1][0] - path[k][0]).hypot(path[k + 1][1] - path[k][1]);
    let mut remain = seg(k0) * (1.0 - t0);
    for k in k0 + 1..path.len() - 1 {
        remain += seg(k);
    }
    let mut need = la;
    let mut k = k0;
    let mut t = t0;
    loop {
        let l = seg(k);
        let left = l * (1.0 - t);
        if need <= left || k + 2 >= path.len() {
            let tt = if l > 1e-9 { (t + need / l).min(1.0) } else { 1.0 };
            let (a, b) = (path[k], path[k + 1]);
            return ([a[0] + tt * (b[0] - a[0]), a[1] + tt * (b[1] - a[1])], remain, k0);
        }
        need -= left;
        k += 1;
        t = 0.0;
    }
}

/// 경로 통로(경로 점 0.5 칸 간격)가 지금 비용 지도에서 막혔나: 몸통 반경(robot_r)보다 가까운 장애물
pub fn path_blocked(cm: &Costmap, path: &[[f64; 2]], from_k: usize, robot_r: f64) -> Option<[f64; 2]> {
    for k in from_k..path.len().saturating_sub(1) {
        let (a, b) = (path[k], path[k + 1]);
        let d = (b[0] - a[0]).hypot(b[1] - a[1]);
        let n = (d / (cm.grid.res * 0.5)).ceil().max(1.0) as usize;
        for s in 0..=n {
            let t = s as f64 / n as f64;
            let (x, y) = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
            if cm.clear_at(x, y) < robot_r - 0.05 {
                return Some([x, y]);
            }
        }
    }
    None
}

/// 바뀐 칸이 경로 통로(robot_r + 0.3 m 안)에 걸리나 — 이때만 다시 계획
pub fn change_hits_path(cm: &Costmap, path: &[[f64; 2]], robot_r: f64) -> bool {
    let Some(ch) = &cm.changed else { return true };
    if ch.is_empty() || path.len() < 2 {
        return false;
    }
    let lim = robot_r + 0.3;
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for q in path {
        x0 = x0.min(q[0]);
        y0 = y0.min(q[1]);
        x1 = x1.max(q[0]);
        y1 = y1.max(q[1]);
    }
    for &i in ch {
        let (x, y) = cm.grid.center(i);
        if x < x0 - lim || x > x1 + lim || y < y0 - lim || y > y1 + lim {
            continue;
        }
        for k in 0..path.len() - 1 {
            if seg_dist(path[k], path[k + 1], [x, y]) < lim {
                return true;
            }
        }
    }
    false
}

pub fn seg_dist(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = (dx * dx + dy * dy).max(1e-12);
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0);
    (a[0] + t * dx - p[0]).hypot(a[1] + t * dy - p[1])
}

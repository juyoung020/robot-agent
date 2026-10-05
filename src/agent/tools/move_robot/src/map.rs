//! 2D 지도 위 탐색 관측·경로 계획 — `move_robot` 이 LLM 에게 돌려주는 지도 요약과, 베이스 안전 정지·`go_to` 경로가 쓰는 것.
//!
//! 입력은 scenemap(mapper2d, 자세는 Cartographer) 2D 점유 격자(−1 모름, 0..100 점유 %)와 map 기준 자세, 방 격자(있으면), 마지막 keyframe 의
//! 가상 스캔(머리 깊이 → 베이스 기준 장애물 점·빈 광선 끝). 평가기 안에서는 `libsgrt.so` 의 `sgrt_map` 이 주고
//! (ctypes 가 포인터만 넘김, [`crate::ffi::mr_set_map`]), 가짜 로봇에서는 [`crate::link::MockWorld`] 가 만든다.
//!
//! 아는 곳 / 모르는 곳을 나눠 말한다(LLM 이 판단하게).
//! - 계획(`go_to`)은 **아는 빈칸**만 지난다: 점유 칸을 로봇 반경만큼 부풀린 C-공간에서 Dijkstra(벽에서 멀수록 쌈).
//! - 프런티어 = 계획으로 닿는 아는 빈칸 중 모르는 칸과 붙은 칸 덩어리. 목표점은 아는 공간 안이라 늘 계획할 수 있다.
//! - 주변 8 방향 여유: 지도(아는 빈칸이 어디까지 이어지나, 끝이 벽인가 모름인가)와 살아 있는 깊이(지금 카메라에 보이는
//!   장애물까지 로봇 몸통 원이 닿기 전까지 갈 수 있는 거리, 카메라 밖이면 null)를 따로.

use serde_json::{json, Value};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

/// 이 값 이하면 빈칸(점유 %). scenemap 은 광선이 한 번 지나면 40 %(로그 오즈 −0.4)라 한 번 본 빈칸도 빈칸
pub const FREE_MAX: i8 = 45;
/// 이 값 이상이면 장애물
pub const OCC_MIN: i8 = 60;

#[derive(Clone, Debug, Default)]
pub struct Grid {
    pub res: f64,
    pub ox: f64,
    pub oy: f64,
    pub w: usize,
    pub h: usize,
    pub cells: Vec<i8>,
}

impl Grid {
    pub fn new(res: f64, ox: f64, oy: f64, w: usize, h: usize, fill: i8) -> Grid {
        Grid { res, ox, oy, w, h, cells: vec![fill; w * h] }
    }
    #[inline]
    pub fn cell_of(&self, x: f64, y: f64) -> (i64, i64) {
        (((x - self.ox) / self.res).floor() as i64, ((y - self.oy) / self.res).floor() as i64)
    }
    #[inline]
    pub fn idx(&self, cx: i64, cy: i64) -> Option<usize> {
        if cx < 0 || cy < 0 || cx >= self.w as i64 || cy >= self.h as i64 {
            None
        } else {
            Some(cy as usize * self.w + cx as usize)
        }
    }
    #[inline]
    pub fn center(&self, i: usize) -> (f64, f64) {
        (self.ox + ((i % self.w) as f64 + 0.5) * self.res, self.oy + ((i / self.w) as f64 + 0.5) * self.res)
    }
    #[inline]
    pub fn at(&self, x: f64, y: f64) -> i8 {
        let (cx, cy) = self.cell_of(x, y);
        self.idx(cx, cy).map(|i| self.cells[i]).unwrap_or(-1)
    }
    pub fn free_area(&self) -> f64 {
        self.cells.iter().filter(|v| is_free(**v)).count() as f64 * self.res * self.res
    }
    pub fn known_area(&self) -> f64 {
        self.cells.iter().filter(|v| **v >= 0).count() as f64 * self.res * self.res
    }
}

#[inline]
pub fn is_free(v: i8) -> bool {
    (0..=FREE_MAX).contains(&v)
}
#[inline]
pub fn is_occ(v: i8) -> bool {
    v >= OCC_MIN
}

#[derive(Clone, Debug, Default)]
pub struct RoomGrid {
    pub res: f64,
    pub ox: f64,
    pub oy: f64,
    pub w: usize,
    pub h: usize,
    pub ids: Vec<u32>,
}

impl RoomGrid {
    pub fn at(&self, x: f64, y: f64) -> u32 {
        let cx = ((x - self.ox) / self.res).floor() as i64;
        let cy = ((y - self.oy) / self.res).floor() as i64;
        if cx < 0 || cy < 0 || cx >= self.w as i64 || cy >= self.h as i64 {
            0
        } else {
            self.ids[cy as usize * self.w + cx as usize]
        }
    }
}

/// 마지막 keyframe 가상 스캔. 점은 map 좌표로 바꿔 둔다.
#[derive(Clone, Debug, Default)]
pub struct Scan {
    /// 카메라 광선 시작(map)
    pub origin: [f64; 2],
    pub hits: Vec<[f64; 2]>,
    pub free_ends: Vec<[f64; 2]>,
    pub stamp: f64,
}

impl Scan {
    /// 베이스 기준 점들 + 그때 map 자세 → map 좌표
    pub fn from_base(pose: [f64; 3], origin_b: [f64; 2], hits_b: &[[f64; 2]], free_b: &[[f64; 2]], stamp: f64) -> Scan {
        let (s, c) = pose[2].sin_cos();
        let tf = |p: &[f64; 2]| [pose[0] + c * p[0] - s * p[1], pose[1] + s * p[0] + c * p[1]];
        Scan { origin: tf(&origin_b), hits: hits_b.iter().map(tf).collect(), free_ends: free_b.iter().map(tf).collect(), stamp }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MapIn {
    pub stamp: f64,
    /// map 기준 로봇 베이스 (x, y, yaw rad)
    pub pose: [f64; 3],
    pub grid: Grid,
    pub rooms: Option<RoomGrid>,
    pub n_rooms: i32,
    pub scan: Option<Scan>,
}

#[derive(Clone, Debug)]
pub struct NavParams {
    /// 부풀림 반경(m): 몸통 원 반경 + 여유. 이보다 장애물에 가까운 칸은 계획에서 막힘
    pub robot_r: f64,
    /// 이 거리까지는 벽에서 멀수록 비용이 쌈(가운데로 다님)
    pub safe_r: f64,
    /// 몸통 원이 장애물에 닿기 전 남길 여유(m) — 이보다 가까워지면 멈춤
    pub stop_margin: f64,
    /// 이보다 짧은 프런티어 덩어리는 버림(m)
    pub frontier_min_m: f64,
    /// 정보 이득: 목표점 둘레 이 반경 안 모르는 넓이
    pub gain_r: f64,
    pub max_frontiers: usize,
    /// 로봇이 서 있는 자리 둘레(m)는 모르는 칸이어도 지나갈 수 있다고 본다(카메라가 발밑을 못 봄)
    pub start_free_r: f64,
    /// 출발 둘레에서 지나갈 수 있는 최소 장애물 거리(몸통 반경 − 2 cm): 좁은 곳에 들어와 있으면 빠져나갈 길만 허용
    pub start_min_clear: f64,
}

impl Default for NavParams {
    fn default() -> Self {
        // 몸 크기 값은 LIMO + OMX-F([`crate::nav::Body::limo_omx`])
        let b = crate::nav::Body::limo_omx();
        NavParams { robot_r: b.robot_r, safe_r: 0.7, stop_margin: 0.10, frontier_min_m: 0.5, gain_r: 2.5, max_frontiers: 5, start_free_r: b.start_free_r, start_min_clear: b.start_min_clear }
    }
}

/// 한 번의 지도 분석(거리장·도달 거리·프런티어). 지도와 출발점이 같으면 다시 쓰고, 바뀌면 다시 만든다.
#[derive(Clone, Debug, Default)]
pub struct Analysis {
    /// 가장 가까운 장애물까지(m), 2.5 m 에서 자름
    pub dobs: Vec<f32>,
    /// 출발점에서 경로 비용(벽 벌점 포함) — f32::INFINITY = 못 감
    pub cost: Vec<f32>,
    /// 실제 경로 길이(m)
    pub len: Vec<f32>,
    pub parent: Vec<u32>,
    pub start: usize,
    pub start_xy: [f64; 2],
    pub frontiers: Vec<Frontier>,
}

#[derive(Clone, Debug)]
pub struct Frontier {
    pub goal: [f64; 2],
    pub goal_idx: usize,
    /// 목표점에서 모르는 쪽을 보는 방향(map, rad)
    pub look_yaw: f64,
    pub size_m: f64,
    pub path_m: f64,
    pub gain_m2: f64,
    pub room: u32,
}

const NB8: [(i64, i64, f32); 8] = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, std::f32::consts::SQRT_2),
    (1, -1, std::f32::consts::SQRT_2),
    (-1, 1, std::f32::consts::SQRT_2),
    (-1, -1, std::f32::consts::SQRT_2),
];

/// 장애물 거리장(가까운 장애물 칸 좌표를 퍼뜨리는 brushfire, 유클리드에 가까움)
pub fn obstacle_distance(g: &Grid, cap_m: f64) -> Vec<f32> {
    let n = g.w * g.h;
    let mut site = vec![u32::MAX; n];
    let mut d2 = vec![f32::INFINITY; n];
    let mut q = std::collections::VecDeque::new();
    for i in 0..n {
        if is_occ(g.cells[i]) {
            site[i] = i as u32;
            d2[i] = 0.0;
            q.push_back(i);
        }
    }
    let cap_c = (cap_m / g.res) as f32;
    let cap2 = cap_c * cap_c;
    let w = g.w as i64;
    while let Some(i) = q.pop_front() {
        let s = site[i] as usize;
        let (sx, sy) = ((s % g.w) as i64, (s / g.w) as i64);
        let (x, y) = ((i % g.w) as i64, (i / g.w) as i64);
        for (dx, dy, _) in NB8 {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= w || ny >= g.h as i64 {
                continue;
            }
            let j = (ny * w + nx) as usize;
            let (ex, ey) = ((nx - sx) as f32, (ny - sy) as f32);
            let nd = ex * ex + ey * ey;
            if nd < d2[j] && nd <= cap2 {
                d2[j] = nd;
                site[j] = s as u32;
                q.push_back(j);
            }
        }
    }
    let r = g.res as f32;
    d2.into_iter().map(|v| if v.is_finite() { v.sqrt() * r } else { cap_m as f32 }).collect()
}

impl Analysis {
    /// `pose` 에서 아는 빈칸 C-공간 Dijkstra + 프런티어
    pub fn new(m: &MapIn, p: &NavParams) -> Analysis {
        Analysis::new_with_dobs(m, obstacle_distance(&m.grid, 2.5), p)
    }

    /// 거리장을 받아서(비용 지도가 이미 만든 것)
    pub fn new_with_dobs(m: &MapIn, dobs: Vec<f32>, p: &NavParams) -> Analysis {
        Analysis::new_full(m, dobs, None, p)
    }

    /// + 칸마다 추가 벌점(0..1, 옮길 수 있는 물체 둘레 등)
    pub fn new_full(m: &MapIn, dobs: Vec<f32>, soft: Option<&[f32]>, p: &NavParams) -> Analysis {
        let g = &m.grid;
        let n = g.w * g.h;
        let mut a = Analysis { dobs, cost: vec![f32::INFINITY; n], len: vec![f32::INFINITY; n], parent: vec![u32::MAX; n], ..Default::default() };
        if n == 0 {
            return a;
        }
        let (sx, sy) = g.cell_of(m.pose[0], m.pose[1]);
        let Some(start) = g.idx(sx, sy) else { return a };
        a.start = start;
        a.start_xy = [m.pose[0], m.pose[1]];
        let sr2 = (p.start_free_r / g.res).powi(2) as i64;
        let rr = p.robot_r as f32;
        let safe = p.safe_r as f32;
        let pass = |i: usize, dobs: &[f32]| -> bool {
            let (x, y) = ((i % g.w) as i64, (i / g.w) as i64);
            let near = (x - sx).pow(2) + (y - sy).pow(2) <= sr2;
            let v = g.cells[i];
            if near {
                !is_occ(v) && dobs[i] as f64 >= p.start_min_clear
            } else {
                is_free(v) && dobs[i] >= rr
            }
        };
        let mut heap = BinaryHeap::new();
        a.cost[start] = 0.0;
        a.len[start] = 0.0;
        heap.push(Reverse((0u32, start as u32)));
        let w = g.w as i64;
        while let Some(Reverse((c, i))) = heap.pop() {
            let i = i as usize;
            let ci = c as f32 / 1000.0;
            if ci > a.cost[i] + 1e-3 {
                continue;
            }
            let (x, y) = ((i % g.w) as i64, (i / g.w) as i64);
            for (dx, dy, l) in NB8 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w || ny >= g.h as i64 {
                    continue;
                }
                let j = (ny * w + nx) as usize;
                if !pass(j, &a.dobs) {
                    continue;
                }
                // 벽에서 멀수록 쌈. 외접 원(0.39 m + 여유) 안쪽은 돌 수 없는 곳이라 크게 벌점(좁은 곳은 지나가기만)
                let q = (safe - a.dobs[j]).max(0.0) / safe;
                let qr = 0.0f32 * a.dobs[j];
                let pen = 1.0 + 4.0 * q * q + 25.0 * qr * qr + soft.map_or(0.0, |s| 2.0 * s[j]);
                let nc = ci + l * g.res as f32 * pen;
                if nc + 1e-4 < a.cost[j] {
                    a.cost[j] = nc;
                    a.len[j] = a.len[i] + l * g.res as f32;
                    a.parent[j] = i as u32;
                    heap.push(Reverse(((nc * 1000.0) as u32, j as u32)));
                }
            }
        }
        a.frontiers = find_frontiers(m, &a, p);
        a
    }

    pub fn reachable(&self, i: usize) -> bool {
        i < self.cost.len() && self.cost[i].is_finite()
    }

    /// 출발점 → i 칸 경로(map 점들, 시야 막힘 없는 점만 남겨 줄임)
    pub fn path_to(&self, g: &Grid, i: usize, p: &NavParams) -> Option<Vec<[f64; 2]>> {
        if !self.reachable(i) {
            return None;
        }
        let mut cells = vec![i];
        let mut k = i;
        while k != self.start {
            let pk = self.parent[k];
            if pk == u32::MAX {
                break;
            }
            k = pk as usize;
            cells.push(k);
        }
        cells.reverse();
        let pts: Vec<[f64; 2]> = cells.iter().map(|&c| {
            let (x, y) = g.center(c);
            [x, y]
        }).collect();
        let mut out = vec![self.start_xy];
        let mut anchor = self.start_xy;
        let mut last_ok = 0usize;
        let mut j = 1;
        while j < pts.len() {
            if self.line_clear(g, anchor, pts[j], p) {
                last_ok = j;
                j += 1;
            } else {
                let q = pts[last_ok.max(1).min(pts.len() - 1)];
                if q == anchor {
                    // 한 칸도 못 나감: 다음 칸을 그냥 넣는다
                    out.push(pts[j]);
                    anchor = pts[j];
                    last_ok = j;
                    j += 1;
                } else {
                    out.push(q);
                    anchor = q;
                }
            }
        }
        let end = *pts.last().unwrap();
        if out.last() != Some(&end) {
            out.push(end);
        }
        Some(out)
    }

    /// 두 점 사이 직선이 C-공간에서 막히지 않았나(출발점 둘레 풀림 포함)
    pub fn line_clear(&self, g: &Grid, a: [f64; 2], b: [f64; 2], _p: &NavParams) -> bool {
        let d = (b[0] - a[0]).hypot(b[1] - a[1]);
        let n = (d / (g.res * 0.5)).ceil().max(1.0) as usize;
        for k in 0..=n {
            let t = k as f64 / n as f64;
            let (x, y) = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
            let (cx, cy) = g.cell_of(x, y);
            match g.idx(cx, cy) {
                Some(i) if self.reachable(i) => {}
                _ => return false,
            }
        }
        true
    }

    /// 점 (x, y) 에서 가장 가까운 도달 가능 칸(r_m 안)
    pub fn nearest_reachable(&self, g: &Grid, x: f64, y: f64, r_m: f64) -> Option<usize> {
        let (cx, cy) = g.cell_of(x, y);
        let r = (r_m / g.res).ceil() as i64;
        let mut best: Option<(i64, usize)> = None;
        for dy in -r..=r {
            for dx in -r..=r {
                let d2 = dx * dx + dy * dy;
                if d2 > r * r {
                    continue;
                }
                if let Some(i) = g.idx(cx + dx, cy + dy) {
                    if self.reachable(i) && best.map_or(true, |(b, _)| d2 < b) {
                        best = Some((d2, i));
                    }
                }
            }
        }
        best.map(|b| b.1)
    }
}

fn find_frontiers(m: &MapIn, a: &Analysis, p: &NavParams) -> Vec<Frontier> {
    let g = &m.grid;
    let n = g.w * g.h;
    let w = g.w as i64;
    let mut cand = vec![false; n];
    let unknown_at = |x: i64, y: i64| -> bool {
        match g.idx(x, y) {
            None => true,
            Some(j) => g.cells[j] < 0,
        }
    };
    let (sx, sy) = g.cell_of(m.pose[0], m.pose[1]);
    let sr2 = ((p.start_free_r + 0.1) / g.res).powi(2) as i64;
    for i in 0..n {
        if !a.reachable(i) || !is_free(g.cells[i]) {
            continue;
        }
        let (x, y) = ((i % g.w) as i64, (i / g.w) as i64);
        if (x - sx).pow(2) + (y - sy).pow(2) <= sr2 {
            continue; // 발밑 모름은 프런티어가 아니다
        }
        if NB8.iter().any(|(dx, dy, _)| unknown_at(x + dx, y + dy)) {
            cand[i] = true;
        }
    }
    let mut seen = vec![false; n];
    let mut out = vec![];
    for i0 in 0..n {
        if !cand[i0] || seen[i0] {
            continue;
        }
        let mut members = vec![];
        let mut stack = vec![i0];
        seen[i0] = true;
        while let Some(i) = stack.pop() {
            members.push(i);
            let (x, y) = ((i % g.w) as i64, (i / g.w) as i64);
            for (dx, dy, _) in NB8 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w || ny >= g.h as i64 {
                    continue;
                }
                let j = (ny * w + nx) as usize;
                if cand[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        let size_m = members.len() as f64 * g.res;
        if size_m < p.frontier_min_m {
            continue;
        }
        let (mut mx, mut my) = (0.0, 0.0);
        let (mut ux, mut uy, mut un) = (0.0, 0.0, 0.0);
        for &i in &members {
            let (x, y) = g.center(i);
            mx += x;
            my += y;
            let (cx, cy) = ((i % g.w) as i64, (i / g.w) as i64);
            for (dx, dy, _) in NB8 {
                if unknown_at(cx + dx, cy + dy) {
                    ux += x + dx as f64 * g.res;
                    uy += y + dy as f64 * g.res;
                    un += 1.0;
                }
            }
        }
        let k = members.len() as f64;
        let (mx, my) = (mx / k, my / k);
        let near_c = *members
            .iter()
            .min_by(|&&i, &&j| {
                let (xi, yi) = g.center(i);
                let (xj, yj) = g.center(j);
                ((xi - mx).powi(2) + (yi - my).powi(2)).total_cmp(&((xj - mx).powi(2) + (yj - my).powi(2)))
            })
            .unwrap();
        // 목표: 덩어리 중심 가까운 칸 둘레 1 m 안 도달 칸 중 제자리에서 돌 수 있는(장애물 ≥ 0.45 m) 곳, 없으면 가장 넓은 곳
        let goal_idx = {
            let (cx0, cy0) = g.center(near_c);
            let r = (1.0 / g.res) as i64;
            let (gx0, gy0) = g.cell_of(cx0, cy0);
            let mut best: Option<(f64, usize)> = None;
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dy * dy > r * r {
                        continue;
                    }
                    if let Some(i) = g.idx(gx0 + dx, gy0 + dy) {
                        if !a.reachable(i) || !is_free(g.cells[i]) {
                            continue;
                        }
                        let d = (dx * dx + dy * dy) as f64 * g.res * g.res;
                        let room = a.dobs[i] as f64;
                        let score = if room >= 0.5 { -d.sqrt() } else { -10.0 + room * 10.0 - d.sqrt() };
                        if best.map_or(true, |b| score > b.0) {
                            best = Some((score, i));
                        }
                    }
                }
            }
            best.map(|b| b.1).unwrap_or(near_c)
        };
        let (gx, gy) = g.center(goal_idx);
        let look_yaw = if un > 0.0 { (uy / un - my).atan2(ux / un - mx) } else { (gy - m.pose[1]).atan2(gx - m.pose[0]) };
        // 정보 이득: 목표점 둘레 모르는 넓이(2 칸 간격 표본)
        let r = (p.gain_r / g.res) as i64;
        let (cx, cy) = g.cell_of(gx, gy);
        let mut cnt = 0usize;
        let mut dy = -r;
        while dy <= r {
            let mut dx = -r;
            while dx <= r {
                if dx * dx + dy * dy <= r * r && unknown_at(cx + dx, cy + dy) {
                    cnt += 1;
                }
                dx += 2;
            }
            dy += 2;
        }
        let gain_m2 = cnt as f64 * 4.0 * g.res * g.res;
        let room = m.rooms.as_ref().map(|r| r.at(gx, gy)).unwrap_or(0);
        out.push(Frontier { goal: [gx, gy], goal_idx, look_yaw, size_m, path_m: a.len[goal_idx] as f64, gain_m2, room });
    }
    // 보여 줄 순서: 정보 이득 / (경로 + 2 m)
    out.sort_by(|a, b| utility(b).total_cmp(&utility(a)));
    out
}

pub fn utility(f: &Frontier) -> f64 {
    f.gain_m2 / (f.path_m + 2.0)
}

/// 한 방향으로 몸통 원이 얼마나 갈 수 있나.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reach {
    /// 지도: 아는 빈칸으로 이어진 거리(m, 몸통 원 기준). 끝이 벽인가(true) 모름인가(false)
    pub known_m: f64,
    pub known_end_wall: bool,
    /// 살아 있는 깊이: 몸통 원이 보이는 장애물에 닿기 전까지(m). 그 방향이 카메라 밖이면 None
    pub depth_m: Option<f64>,
}

/// `from` 에서 map 방향 `yaw` 로: 지도·깊이 각각 갈 수 있는 거리(최대 max_m)
pub fn reach(m: &MapIn, dobs: &[f32], from: [f64; 2], yaw: f64, max_m: f64, p: &NavParams) -> Reach {
    let g = &m.grid;
    let (s, c) = yaw.sin_cos();
    let step = g.res * 0.5;
    let mut known_m = max_m;
    let mut wall = false;
    let mut t = 0.0;
    while t <= max_m {
        let (x, y) = (from[0] + c * t, from[1] + s * t);
        let (cx, cy) = g.cell_of(x, y);
        let (v, d) = match g.idx(cx, cy) {
            Some(i) => (g.cells[i], dobs[i] as f64),
            None => (-1, 9.0),
        };
        if v >= 0 && d < p.robot_r - 0.04 {
            known_m = (t - step).max(0.0);
            wall = true;
            break;
        }
        if v < 0 && t > p.start_free_r || (0..OCC_MIN).contains(&v) && v > FREE_MAX && t > p.start_free_r {
            known_m = (t - step).max(0.0);
            break;
        }
        t += step;
    }
    let depth_m = m.scan.as_ref().and_then(|sc| depth_reach(sc, from, yaw, max_m, p));
    Reach { known_m, known_end_wall: wall, depth_m }
}

/// 몸통 원(robot_r − 0.04)이 스캔 장애물 점에 닿기 전까지. 그 방향을 카메라가 안 보면 None.
pub fn depth_reach(sc: &Scan, from: [f64; 2], yaw: f64, max_m: f64, p: &NavParams) -> Option<f64> {
    let (s, c) = yaw.sin_cos();
    let r = p.robot_r - 0.04;
    // 이 방향을 카메라가 보나: 광선(장애물 점·빈 끝)의 방위가 ±6° 안에 있는가
    let mut seen_far: f64 = -1.0;
    let mut covered = false;
    for q in sc.hits.iter().chain(sc.free_ends.iter()) {
        let (dx, dy) = (q[0] - sc.origin[0], q[1] - sc.origin[1]);
        let a = dy.atan2(dx);
        // from 기준 방향(로봇 기준점이 카메라 원점과 가까우므로 대략)
        let (fx, fy) = (q[0] - from[0], q[1] - from[1]);
        let b = fy.atan2(fx);
        let _ = a;
        if ang_diff(b, yaw).abs() < 6f64.to_radians() {
            covered = true;
            seen_far = seen_far.max(fx * c + fy * s);
        }
    }
    if !covered {
        return None;
    }
    let mut best = max_m.min(seen_far.max(0.0) + 0.0);
    for q in &sc.hits {
        let (fx, fy) = (q[0] - from[0], q[1] - from[1]);
        let lon = fx * c + fy * s;
        let lat = -fx * s + fy * c;
        if lon <= 0.0 || lat.abs() >= r {
            continue;
        }
        let t = lon - (r * r - lat * lat).sqrt();
        if t < best {
            best = t.max(0.0);
        }
    }
    Some(best)
}

pub fn ang_diff(a: f64, b: f64) -> f64 {
    let mut d = (a - b) % (2.0 * PI);
    if d > PI {
        d -= 2.0 * PI;
    }
    if d < -PI {
        d += 2.0 * PI;
    }
    d
}

/// 로봇 기준 방위를 짧은 글로: 0 → "ahead", 40° 왼쪽 → "40L", 오른쪽 → "75R", 뒤 → "back"
pub fn dir_text(rel: f64) -> String {
    let d = rel.to_degrees().round() as i64;
    if d.abs() <= 10 {
        "ahead".into()
    } else if d.abs() >= 170 {
        "back".into()
    } else if d > 0 {
        format!("{d}L")
    } else {
        format!("{}R", -d)
    }
}

pub fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}
pub fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// 방 하나(방 격자에서)
#[derive(Clone, Debug)]
pub struct RoomInfo {
    pub id: u32,
    pub area_m2: f64,
    pub centroid: [f64; 2],
    /// 도달 가능한 그 방 칸 중 중심에 가장 가까운 것
    pub goal_idx: Option<usize>,
    pub path_m: Option<f64>,
}

pub fn rooms_info(m: &MapIn, a: &Analysis) -> Vec<RoomInfo> {
    let Some(rg) = &m.rooms else { return vec![] };
    let mut acc: std::collections::BTreeMap<u32, (f64, f64, f64)> = Default::default();
    for (k, &id) in rg.ids.iter().enumerate() {
        if id == 0 {
            continue;
        }
        let x = rg.ox + ((k % rg.w) as f64 + 0.5) * rg.res;
        let y = rg.oy + ((k / rg.w) as f64 + 0.5) * rg.res;
        let e = acc.entry(id).or_insert((0.0, 0.0, 0.0));
        e.0 += x;
        e.1 += y;
        e.2 += 1.0;
    }
    let g = &m.grid;
    let mut out = vec![];
    for (id, (sx, sy, n)) in acc {
        let c = [sx / n, sy / n];
        // 그 방에 속하는 도달 칸 중 중심에 가장 가까운 것
        let mut best: Option<(f64, usize)> = None;
        for i in 0..g.w * g.h {
            if !a.reachable(i) {
                continue;
            }
            let (x, y) = g.center(i);
            if rg.at(x, y) != id {
                continue;
            }
            let d = (x - c[0]).powi(2) + (y - c[1]).powi(2);
            if best.map_or(true, |b| d < b.0) {
                best = Some((d, i));
            }
        }
        out.push(RoomInfo {
            id,
            area_m2: n * rg.res * rg.res,
            centroid: c,
            goal_idx: best.map(|b| b.1),
            path_m: best.map(|b| a.len[b.1] as f64),
        });
    }
    out
}

/// 8 방향(로봇 기준 0°, 45° 왼쪽 …) 이름
pub const SECTORS: [(&str, f64); 8] = [("F", 0.0), ("FL", 45.0), ("L", 90.0), ("BL", 135.0), ("B", 180.0), ("BR", -135.0), ("R", -90.0), ("FR", -45.0)];

/// 주변 여유: 방향마다 "지도 m 벽|모름, 깊이 m|안 보임" 을 짧은 글로
pub fn ring(m: &MapIn, a: &Analysis, p: &NavParams) -> Value {
    let mut o = serde_json::Map::new();
    let from = [m.pose[0], m.pose[1]];
    for (name, deg) in SECTORS {
        let r = reach(m, &a.dobs, from, m.pose[2] + deg.to_radians(), 4.0, p);
        let mut s = format!("known {:.1}m then {}", r.known_m, if r.known_end_wall { "wall" } else { "unknown" });
        if r.known_m >= 3.95 {
            s = "known 4m+ free".into();
        }
        match r.depth_m {
            Some(d) if d >= 3.95 => s.push_str(", depth clear 4m+"),
            Some(d) => s.push_str(&format!(", depth clear {:.1}m", d)),
            None => {}
        }
        o.insert(name.into(), json!(s));
    }
    Value::Object(o)
}

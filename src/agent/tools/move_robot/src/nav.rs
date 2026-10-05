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

    /// 점 (x, y) 에서 가장 가까운 장애물 칸 "면"까지(m) 의 아래 한계: 둘레 네 칸 중심의 거리장 최솟값 − 0.71 칸.
    /// (거리장은 칸 중심 사이 거리라, 칸 면까지는 최대 반대각만큼 가깝다 — 접촉 판정을 낙관하지 않게)
    #[inline]
    pub fn clear_at(&self, x: f64, y: f64) -> f64 {
        let g = &self.grid;
        if g.w == 0 {
            return 2.5;
        }
        let fx = (x - g.ox) / g.res - 0.5;
        let fy = (y - g.oy) / g.res - 0.5;
        let (x0, y0) = (fx.floor() as i64, fy.floor() as i64);
        let d = |cx: i64, cy: i64| -> f64 {
            match g.idx(cx, cy) {
                Some(i) => self.dobs[i] as f64,
                None => 2.5,
            }
        };
        let mn = d(x0, y0).min(d(x0 + 1, y0)).min(d(x0, y0 + 1)).min(d(x0 + 1, y0 + 1));
        // 네 칸 중심 중 가장 가까운 칸까지 이 점이 떨어진 만큼은 더 가까울 수 있다: 그 칸 거리 − (점–칸 중심 거리)도 본다
        let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
        let (ix, iy) = (if tx < 0.5 { x0 } else { x0 + 1 }, if ty < 0.5 { y0 } else { y0 + 1 });
        let off = ((fx - ix as f64).hypot(fy - iy as f64)) * g.res;
        let near = d(ix, iy) - off;
        mn.max(near) - 0.71 * g.res
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

/// 얇은 안전 정지(지도 안 씀): 마지막 깊이 프레임의 장애물 점 중, 몸통 사각형(+margin)을 map 방향 `heading` 으로 밀었을 때
/// 처음 닿는 것까지의 거리. 진행 방향을 카메라가 못 보면(몸 앞 ±55° 밖) None.
pub fn depth_guard(sc: &Scan, fp: &Footprint, pose: [f64; 3], heading: f64, margin: f64, max_m: f64) -> Option<f64> {
    depth_guard_f(sc, fp, pose, heading, margin, max_m, |_| true)
}

/// 지도에 이미 있는 장애물 점은 빼고(지역 제어가 이미 피함) — 지도에 없는(새로 나타난·움직이는·발밑) 것만
pub fn depth_guard_unmapped(cm: &Costmap, fp: &Footprint, pose: [f64; 3], heading: f64, margin: f64, max_m: f64) -> Option<f64> {
    let sc = cm.scan.as_ref()?;
    depth_guard_f(sc, fp, pose, heading, margin, max_m, |q| cm.clear_at(q[0], q[1]) > 0.08)
}

fn depth_guard_f(sc: &Scan, fp: &Footprint, pose: [f64; 3], heading: f64, margin: f64, max_m: f64, keep: impl Fn(&[f64; 2]) -> bool) -> Option<f64> {
    let rel = ang_diff(heading, pose[2]);
    if rel.abs() > 55f64.to_radians() {
        return None;
    }
    let (s, c) = pose[2].sin_cos();
    let dir = [rel.cos(), rel.sin()];
    let mut best = max_m;
    for q in &sc.hits {
        let (fx, fy) = (q[0] - pose[0], q[1] - pose[1]);
        let p = [c * fx + s * fy, -s * fx + c * fy];
        if p[0] < -fp.hl - 0.05 || !keep(q) {
            continue;
        }
        if let Some(t) = fp.sweep_hit(p, dir, margin) {
            if t < best {
                best = t;
            }
        }
    }
    Some(best.max(0.0))
}

// ---------------------------------------------------------------- 몸통 모양(사각형)

/// 몸통 사각형(LIMO + OMX-F, 길이 × 폭). 원으로 보면 좁은 문을 못 지나므로 사각형 둘레 점으로 검사한다(Nav2 footprint 와 같은 뜻).
#[derive(Clone, Copy, Debug)]
pub struct Footprint {
    pub hl: f64,
    pub hw: f64,
    /// 둘레 점(로봇 기준), 약 5 cm 간격
    pts: [[f64; 2]; 44],
    pub round: bool,
}

impl Footprint {
    pub fn new(hl: f64, hw: f64) -> Footprint {
        let mut pts = [[0.0; 2]; 44];
        let mut k = 0;
        // 긴 변 12 칸, 짧은 변 10 칸
        for i in 0..12 {
            let t = -hl + 2.0 * hl * i as f64 / 12.0;
            pts[k] = [t, hw];
            pts[k + 1] = [-t, -hw];
            k += 2;
        }
        for i in 0..10 {
            let t = -hw + 2.0 * hw * i as f64 / 10.0;
            pts[k] = [hl, -t];
            pts[k + 1] = [-hl, t];
            k += 2;
        }
        Footprint { hl, hw, pts, round: false }
    }
    /// 원 몸통(둘레 44 점). sweep_hit 은 외접 정사각형으로(보수적)
    pub fn circle(r: f64) -> Footprint {
        let mut pts = [[0.0; 2]; 44];
        for (k, q) in pts.iter_mut().enumerate() {
            let a = k as f64 * 2.0 * std::f64::consts::PI / 44.0;
            *q = [r * a.cos(), r * a.sin()];
        }
        Footprint { hl: r, hw: r, pts, round: true }
    }
    pub fn circum(&self) -> f64 {
        if self.round {
            self.hl
        } else {
            self.hl.hypot(self.hw)
        }
    }
    /// 자세 (x, y, yaw) 에서 몸통 둘레의 가장 작은 장애물 거리(m). 음수 = 겹침
    pub fn clear(&self, cm: &Costmap, x: f64, y: f64, yaw: f64) -> f64 {
        let (s, c) = yaw.sin_cos();
        let mut m = f64::INFINITY;
        for q in &self.pts {
            let d = cm.clear_at(x + c * q[0] - s * q[1], y + s * q[0] + c * q[1]);
            if d < m {
                m = d;
            }
        }
        m
    }
    /// 둘레 점 중 모르는 칸에 있는 것이 있나
    pub fn touches_unknown(&self, cm: &Costmap, x: f64, y: f64, yaw: f64) -> bool {
        let (s, c) = yaw.sin_cos();
        self.pts.iter().step_by(3).any(|q| cm.unknown_at(x + c * q[0] - s * q[1], y + s * q[0] + c * q[1]))
    }
    /// 제자리에서 yaw0 → yaw1 로 돌 때 가장 작은 여유(5° 간격)
    pub fn turn_clear(&self, cm: &Costmap, x: f64, y: f64, yaw0: f64, yaw1: f64) -> f64 {
        let d = ang_diff(yaw1, yaw0);
        let n = (d.abs() / 5f64.to_radians()).ceil().max(1.0) as usize;
        (0..=n).map(|k| self.clear(cm, x, y, yaw0 + d * k as f64 / n as f64)).fold(f64::INFINITY, f64::min)
    }
    /// 로봇 기준 점 p 가 (로봇 기준) 방향 dir 으로 t 만큼 몸통이 밀려갈 때 몸통(+margin)에 들어오는 가장 작은 t ≥ 0. 안 들어오면 None
    pub fn sweep_hit(&self, p: [f64; 2], dir: [f64; 2], margin: f64) -> Option<f64> {
        let (hl, hw) = (self.hl + margin, self.hw + margin);
        // 몸통이 t·dir 만큼 가면 점의 상대 위치 = p − t·dir. |px − t dx| ≤ hl, |py − t dy| ≤ hw 인 t 구간
        let mut lo: f64 = 0.0;
        let mut hi: f64 = f64::INFINITY;
        for (pv, dv, h) in [(p[0], dir[0], hl), (p[1], dir[1], hw)] {
            if dv.abs() < 1e-9 {
                if pv.abs() > h {
                    return None;
                }
            } else {
                let (a, b) = ((pv - h) / dv, (pv + h) / dv);
                let (a, b) = if a < b { (a, b) } else { (b, a) };
                lo = lo.max(a);
                hi = hi.min(b);
            }
        }
        if lo <= hi {
            Some(lo)
        } else {
            None
        }
    }
}

impl Default for Footprint {
    /// LIMO + OMX-F 몸통([`LIMO_LEN`] × [`LIMO_WID`])
    fn default() -> Self {
        Footprint::new(LIMO_LEN / 2.0, LIMO_WID / 2.0)
    }
}

// ---------------------------------------------------------------- 로봇별 몸 크기

/// 몸 크기: DWA·회전·안전 정지의 몸통 모양([`Footprint`])과 전역 계획의 부풀림 원([`NavParams`] 의 robot_r,
/// start_free_r, start_min_clear). 우리 로봇 LIMO + OMX-F 하나([`Body::limo_omx`]).
#[derive(Clone, Debug)]
pub struct Body {
    pub name: String,
    pub fp: Footprint,
    /// 계획 부풀림 반경 = 몸통 외접원 + 여유(0.211 + 0.03)
    pub robot_r: f64,
    /// 출발 둘레(카메라가 못 보는 발밑) — 부풀림 + 0.05
    pub start_free_r: f64,
    /// 출발 둘레에서 지나갈 수 있는 최소 장애물 거리 — 부풀림 − 0.07
    pub start_min_clear: f64,
}

/// LIMO + OMX-F 몸통 길이 × 폭(m, base_link 중심 대칭 사각형). OmniGibson limo_omx 충돌 모양을 base_link 기준으로 잰 범위
/// (시뮬 headless, 10-04): base_link 껍질 x −0.163..0.159 · y ±0.095(사양 322 mm 길이), 바퀴 y ±0.109(사양 폭 220 mm),
/// 홈 자세로 접은 팔(omx_link2·3)이 뒤로 x −0.180 까지(바닥 위 0.23–0.30 m). 앞뒤 중 큰 쪽 0.18 로 대칭 → 0.36 × 0.22.
pub const LIMO_LEN: f64 = 0.36;
pub const LIMO_WID: f64 = 0.22;

impl Body {
    pub fn limo_omx() -> Body {
        let fp = Footprint::default();
        let rr = fp.circum() + 0.03;
        Body { name: "limo_omx".into(), fp, robot_r: rr, start_free_r: rr + 0.05, start_min_clear: rr - 0.07 }
    }
}

// ---------------------------------------------------------------- 지역 제어(DWA)

#[derive(Clone, Debug)]
pub struct DwaParams {
    pub body_r: f64,
    pub fp: Footprint,
    pub horizon_s: f64,
    pub dt: f64,
    pub nv: usize,
    pub nw: usize,
    pub acc: f64,
}

impl Default for DwaParams {
    fn default() -> Self {
        DwaParams { body_r: 0.30, fp: Footprint::default(), horizon_s: 1.2, dt: 0.1, nv: 6, nw: 13, acc: 0.8 }
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

/// 속도 표본(v, w)마다 horizon 동안 굴려 보고, 몸통 사각형 둘레가 장애물에서 margin 이상(좁은 곳에 이미 들어와 있으면 지금 여유
/// − 1 cm 이상) 떨어진 것 중 점수가 가장 좋은 것. 앞으로만(카메라가 앞을 봄). 모르는 칸은 전역 경로 0.35 m 안이면 허용.
pub fn dwa(cm: &Costmap, path: &[[f64; 2]], pose: [f64; 3], anchor: [f64; 2], carrot: [f64; 2], goal_dist: f64, vmax: f64, wmax: f64, margin: f64, dp: &DwaParams, start_r: f64, allow_unknown: bool) -> DwaOut {
    let mut best: Option<(f64, DwaOut)> = None;
    let fp = &dp.fp;
    let cur = fp.clear(cm, pose[0], pose[1], pose[2]);
    // 좁은 곳에 이미 들어와 있으면 여유를 더 줄이지 않는 궤적만(빠져나오기)
    let tight = cur < margin;
    let need = if tight { cur - 0.003 } else { margin };
    let steps = (dp.horizon_s / dp.dt).round() as usize;
    let dbg = std::env::var("MR_DEBUG_DWA2").is_ok();
    let mut rej = [0usize; 4];
    // 앞 nv 개 + 뒤 2 개(좁은 곳에서 돌 수 없을 때 빠져나오기: 아는 빈칸으로만, 느리게, 점수 깎음)
    for iv in 0..dp.nv + 2 {
        let v = if iv < dp.nv { vmax * iv as f64 / (dp.nv - 1) as f64 } else { -0.08 * (iv - dp.nv + 1) as f64 };
        for iw in 0..dp.nw {
            let w = -wmax + 2.0 * wmax * iw as f64 / (dp.nw - 1) as f64;
            let (mut x, mut y, mut th) = (pose[0], pose[1], pose[2]);
            let mut minc = f64::INFINITY;
            let mut ok = true;
            let mut travelled = 0.0;
            // 굴림 간격: 한 걸음 5 cm·5° 이하
            let sub = ((v.abs() * dp.dt / 0.05).max(w.abs() * dp.dt / 5f64.to_radians())).ceil().max(1.0) as usize;
            'roll: for _ in 0..steps {
                for _ in 0..sub {
                    let h = dp.dt / sub as f64;
                    th += w * h;
                    x += v * th.cos() * h;
                    y += v * th.sin() * h;
                    travelled += v.abs() * h;
                    let c = fp.clear(cm, x, y, th);
                    // 제자리 돌기·뒤로 빠지기는 느리고 clear_at 이 이미 반대각(3.5 cm)만큼 낙관하지 않으므로 여유 0 까지 허용
                    // 빠를수록 넓게: 0.5 m/s 면 +10 cm (지도에 늦게 들어오는 옆 장애물·멈춤 거리)
                    let need_here = need + 0.2 * v.max(0.0);
                    if c < need_here {
                        rej[0] += 1;
                        ok = false;
                        break 'roll;
                    }
                    // 뒤: 카메라가 못 본 곳. 이번 움직임 출발점 0.8 m 안(지나온 곳)이 아니면 모르는 칸 금지
                    if v < 0.0 && (x - anchor[0]).hypot(y - anchor[1]) > 0.8 && fp.touches_unknown(cm, x, y, th) {
                        rej[1] += 1;
                        ok = false;
                        break 'roll;
                    }
                    if !allow_unknown && v > 0.0 && (x - anchor[0]).hypot(y - anchor[1]) > start_r && cm.unknown_at(x, y) {
                        let near_path = path.len() >= 2 && (0..path.len() - 1).any(|k| seg_dist(path[k], path[k + 1], [x, y]) < 0.35);
                        if !near_path {
                            ok = false;
                            break 'roll;
                        }
                    }
                    minc = minc.min(c);
                }
                if travelled > goal_dist {
                    break;
                }
            }
            if !ok {
                continue;
            }
            // 좁은 곳: 끝 자세가 지금보다 넓어지는 궤적만(조금씩 더 들어가는 것 막기)
            if tight && (v != 0.0 || w != 0.0) && fp.clear(cm, x, y, th) < cur + 0.005 {
                rej[2] += 1;
                continue;
            }
            if minc.is_infinite() {
                minc = cur;
            }
            let d_end = (carrot[0] - x).hypot(carrot[1] - y);
            let h_err = ang_diff((carrot[1] - y).atan2(carrot[0] - x), th).abs();
            let score = -2.0 * d_end - 0.6 * h_err + 0.8 * minc.min(0.3) + 0.3 * v / vmax.max(1e-6) - if v < 0.0 { 0.3 } else { 0.0 };
            if best.map_or(true, |b| score > b.0) {
                best = Some((score, DwaOut { v, w, clear: minc, ok: true }));
            }
        }
    }
    // 좁은 곳(사각형 모서리가 거의 닿음): 곧게 앞·뒤로 0.5 s 밀어 여유가 커지는 쪽(카메라 밖이면 지나온 곳만). 옆으로는 못 간다
    if best.map_or(true, |b| b.1.v == 0.0 && b.1.w == 0.0) {
        let mut esc: Option<(f64, f64)> = None;
        for vx in [0.1, -0.1] {
            let mut okk = true;
            let mut end = cur;
            for j in 1..=10 {
                let t = 0.05 * j as f64;
                let (x, y) = (pose[0] + pose[2].cos() * vx * t, pose[1] + pose[2].sin() * vx * t);
                let c = fp.clear(cm, x, y, pose[2]);
                if c < cur.min(need) - 0.003 || ((x - anchor[0]).hypot(y - anchor[1]) > 0.8 && fp.touches_unknown(cm, x, y, pose[2])) {
                    okk = false;
                    break;
                }
                end = c;
            }
            if okk && end > cur + 0.01 && esc.map_or(true, |e| end > e.0) {
                esc = Some((end, vx));
            }
        }
        if let Some((end, vx)) = esc {
            best = Some((0.0, DwaOut { v: vx, w: 0.0, clear: end, ok: true }));
        }
    }
    if dbg {
        eprintln!("dwa2 cur={cur:.3} tight={tight} need={need:.3} rej(clear,unknown_back,tight_end)={:?} best={:?}", &rej[..3], best.map(|b| b.1));
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

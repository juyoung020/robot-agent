//! [`Robot`] 의 지도·주행 부분: 지도 받기, 베이스 `go_to`(아는 빈칸 경로 + DWA + 회복)·`probe`(모르는 곳 살피기,
//! 깊이 안전 정지), `delta` 안전 정지, 결과에 붙이는 지도 요약(LLM 관측)과 측정값(`_m`, 에이전트가 LLM 에게서 뺌).

use crate::map::{self, ang_diff, dir_text, r1, r2, Analysis, Frontier, Grid, MapIn, NavParams, RoomGrid};
use crate::nav::{self, carrot_on, change_hits_path, depth_guard, dwa, path_blocked, Costmap, DwaParams, MapDelta};
const TURN_MIN_CLEAR: f64 = 0.01;
/// DWA: 몸통 사각형 둘레가 장애물에서 이만큼 떨어져야(격자 0.05 m 근사 오차 포함)
const DWA_MARGIN: f64 = 0.02;
use crate::{Command, Robot, Safety};
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// 관측 모양(프롬프트 실험용 변형). 1 = 기본(프런티어·방·주변 글), 2 = 프런티어만(주변 여유 뺌)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObsStyle {
    Full,
    Compact,
}

#[derive(Clone, Debug)]
pub struct Target {
    pub id: String,
    pub goal: [f64; 2],
    pub look_yaw: Option<f64>,
    pub path_m: f64,
}

/// 지도·주행 상태(로봇 하나에 하나)
#[derive(Clone, Debug)]
pub struct NavState {
    pub params: NavParams,
    pub dwa: DwaParams,
    pub cm: Costmap,
    pub have_map: bool,
    pub rooms: Option<RoomGrid>,
    pub n_rooms: i32,
    /// 지금 map 자세 추정: 지도 자세 + 그 뒤 base_qvel 적분
    pub pose: [f64; 3],
    skip_integrate: bool,
    pub now: f64,
    /// 지난 관측에 보여 준 목표 id
    pub targets: Vec<Target>,
    pub visited: BTreeSet<u32>,
    pub last_free_m2: f64,
    pub last_frontiers: usize,
    /// 측정(LLM 에 안 보임)
    pub odo_m: f64,
    pub reference: Option<Grid>,
    pub ref_cells: usize,
    pub contacts: u64,
    pub n_blocked: u64,
    pub n_stall: u64,
    pub n_replans: u64,
    pub replan_us: u64,
    pub min_clear: f64,
    pub obs_us: u64,
    pub style: ObsStyle,
    /// 마지막 경로(뷰어·기록용, map 점)
    pub last_path: Vec<[f64; 2]>,
    pub path_log: Vec<[f64; 2]>,
    /// 실패한 go_to 목표(map x, y, 횟수) — 두 번 실패한 자리 둘레 0.6 m 프런티어는 관측에서 뺀다
    pub failed_goals: Vec<([f64; 2], u32)>,
}

impl Default for NavState {
    fn default() -> Self {
        NavState {
            params: NavParams::default(),
            dwa: DwaParams::default(),
            cm: Costmap::default(),
            have_map: false,
            rooms: None,
            n_rooms: 0,
            pose: [0.0; 3],
            skip_integrate: false,
            now: 0.0,
            targets: vec![],
            visited: BTreeSet::new(),
            last_free_m2: 0.0,
            last_frontiers: 0,
            odo_m: 0.0,
            reference: None,
            ref_cells: 0,
            contacts: 0,
            n_blocked: 0,
            n_stall: 0,
            n_replans: 0,
            replan_us: 0,
            min_clear: f64::INFINITY,
            obs_us: 0,
            style: ObsStyle::Full,
            last_path: vec![],
            path_log: vec![],
            failed_goals: vec![],
        }
    }
}

impl NavState {
    /// 매 스텝: base_qvel(로봇 기준)로 map 자세를 앞으로
    pub fn integrate(&mut self, v: [f64; 3], dt: f64) {
        self.now += dt;
        if self.skip_integrate {
            self.skip_integrate = false;
            return;
        }
        let (s, c) = self.pose[2].sin_cos();
        self.pose[0] += (c * v[0] - s * v[1]) * dt;
        self.pose[1] += (s * v[0] + c * v[1]) * dt;
        self.pose[2] += v[2] * dt;
        self.odo_m += v[0].hypot(v[1]) * dt;
        if self.have_map {
            let c = self.dwa.fp.clear(&self.cm, self.pose[0], self.pose[1], self.pose[2]);
            if v[0].hypot(v[1]) > 0.02 {
                self.min_clear = self.min_clear.min(c);
            }
            if self.path_log.last().map_or(true, |q| (q[0] - self.pose[0]).hypot(q[1] - self.pose[1]) > 0.05) {
                self.path_log.push([self.pose[0], self.pose[1]]);
            }
        }
    }

    pub fn set_map(&mut self, m: MapIn, d: &MapDelta) {
        self.cm.update(&m, d, self.now);
        self.pose = m.pose;
        self.skip_integrate = true;
        self.rooms = m.rooms;
        self.n_rooms = m.n_rooms;
        self.have_map = true;
        if let Some(r) = &self.rooms {
            let id = r.at(self.pose[0], self.pose[1]);
            if id != 0 {
                self.visited.insert(id);
            }
        }
    }

    pub fn analyze(&mut self) -> (MapIn, Analysis) {
        let t0 = std::time::Instant::now();
        let out = nav::analyze(&self.cm, self.pose, self.rooms.clone(), self.n_rooms, &self.params);
        self.replan_us = t0.elapsed().as_micros() as u64;
        out
    }

    /// 정답 기준 넓이 대비 지도 빈칸 비율(측정용)
    pub fn gt_coverage(&self) -> Option<f64> {
        let r = self.reference.as_ref()?;
        if self.ref_cells == 0 {
            return None;
        }
        let g = &self.cm.grid;
        let mut hit = 0usize;
        for (k, &v) in r.cells.iter().enumerate() {
            if v <= 0 {
                continue;
            }
            let (x, y) = r.center(k);
            if map::is_free(g.at(x, y)) {
                hit += 1;
            }
        }
        Some(hit as f64 / self.ref_cells as f64)
    }

    pub fn set_reference(&mut self, g: Grid) {
        self.ref_cells = g.cells.iter().filter(|v| **v > 0).count();
        self.reference = Some(g);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavKind {
    GoTo,
    Probe,
}

#[derive(Clone, Debug)]
pub struct NavMove {
    pub kind: NavKind,
    pub label: String,
    /// 0 따라가기/돌기, 1 앞으로(probe)·마지막 돌기(go_to), 2 멈춤 기다림, 3 둘러보기(회복), 4 뒤로(회복)
    pub phase: u8,
    pub goal: [f64; 2],
    pub look_yaw: Option<f64>,
    pub path: Vec<[f64; 2]>,
    pub path_m0: f64,
    pub turn_yaw: f64,
    pub fwd_m: f64,
    pub fwd_start: [f64; 2],
    pub v_cmd: [f64; 3],
    pub t_max: f64,
    pub stall: u32,
    pub stuck: u32,
    pub recover: u8,
    pub look_target: f64,
    pub back_start: [f64; 2],
    /// 되짚기 회복: 거꾸로 따라갈 지나온 점들
    pub retrace: Vec<[f64; 2]>,
    pub settle: u32,
    pub cm_ver: u64,
    pub last_replan: f64,
    pub replans: u32,
    pub start_xy: [f64; 2],
    pub start_yaw: f64,
    pub moved0: f64,
    pub stop: Option<(String, f64)>,
    pub guard_stops: u32,
    pub outcome: Option<&'static str>,
}

fn sat(x: f64, m: f64) -> f64 {
    x.clamp(-m, m)
}

impl Robot {
    /// go_to / probe 시작(오류는 LLM 이 고칠 수 있는 문장)
    pub(crate) fn start_nav(&mut self, cmd: &Command) -> Result<NavMove, String> {
        let s = self.safety.clone();
        let pose = self.nav.pose;
        let mut nm = NavMove {
            kind: NavKind::Probe,
            label: String::new(),
            phase: 0,
            goal: [pose[0], pose[1]],
            look_yaw: None,
            path: vec![],
            path_m0: 0.0,
            turn_yaw: pose[2],
            fwd_m: 0.0,
            fwd_start: [pose[0], pose[1]],
            v_cmd: [0.0; 3],
            t_max: 10.0,
            stall: 0,
            stuck: 0,
            recover: 0,
            look_target: 0.0,
            back_start: [0.0; 2],
            retrace: vec![],
            settle: 0,
            cm_ver: self.nav.cm.version,
            last_replan: self.nav.now,
            replans: 0,
            start_xy: [pose[0], pose[1]],
            start_yaw: pose[2],
            moved0: self.nav.odo_m,
            stop: None,
            guard_stops: 0,
            outcome: None,
        };
        match cmd.mode {
            crate::Mode::Probe => {
                let turn = cmd.values[0].clamp(-360.0, 360.0).to_radians();
                let fwd = cmd.values[1].min(s.probe_max_m);
                nm.label = format!("probe [{}, {}]", cmd.values[0], r2(fwd));
                nm.turn_yaw = pose[2] + turn;
                nm.fwd_m = fwd;
                nm.t_max = turn.abs() / s.nav_wmax * 1.5 + fwd / s.probe_vmax * 2.0 + 4.0;
                Ok(nm)
            }
            _ => {
                if !self.nav.have_map {
                    return Err("no map yet: use probe (e.g. [0, 0.5]) first".into());
                }
                nm.kind = NavKind::GoTo;
                let (m, a) = self.nav.analyze();
                let (goal, look, label) = if let Some(id) = &cmd.target {
                    let t = self.nav.targets.iter().find(|t| &t.id == id).cloned();
                    match t {
                        Some(t) => (t.goal, t.look_yaw, id.clone()),
                        None => {
                            let ids: Vec<&str> = self.nav.targets.iter().map(|t| t.id.as_str()).collect();
                            return Err(format!("unknown target '{id}'; ids in the last map summary: {}", if ids.is_empty() { "none".into() } else { ids.join(", ") }));
                        }
                    }
                } else {
                    let (sn, cs) = pose[2].sin_cos();
                    let (f, l) = (cmd.values[0], cmd.values[1]);
                    let p = [pose[0] + cs * f - sn * l, pose[1] + sn * f + cs * l];
                    let v = m.grid.at(p[0], p[1]);
                    if v < 0 {
                        return Err(format!("point [{f}, {l}] is in UNKNOWN space; go_to plans only through known free space. Use probe to look/move there, or go_to a frontier id"));
                    }
                    if map::is_occ(v) {
                        return Err(format!("point [{f}, {l}] is an obstacle on the map; pick another point or a frontier id"));
                    }
                    (p, None, format!("[{}, {}]", r2(f), r2(l)))
                };
                let gi = a.nearest_reachable(&m.grid, goal[0], goal[1], 0.6).ok_or_else(|| {
                    format!("no known free path to {label} now (it may be behind an obstacle or unknown space); pick another frontier or probe")
                })?;
                let path = a.path_to(&m.grid, gi, &self.nav.params).ok_or("no path")?;
                let (gx, gy) = m.grid.center(gi);
                nm.goal = [gx, gy];
                nm.look_yaw = look;
                nm.path_m0 = a.len[gi] as f64;
                nm.path = path;
                nm.label = label;
                nm.t_max = nm.path_m0 / (0.6 * s.nav_vmax) + 12.0;
                self.nav.last_path = nm.path.clone();
                Ok(nm)
            }
        }
    }

    /// Nav 한 스텝. 끝나면 Some(결과).
    pub(crate) fn tick_nav(&mut self, m: &mut NavMove, t: f64, meas_v: [f64; 3]) -> Option<Value> {
        let s: Safety = self.safety.clone();
        let p = self.nav.params.clone();
        let dp = self.nav.dwa.clone();
        let pose = self.nav.pose;
        let xy = [pose[0], pose[1]];
        let mut des = [0.0f64; 3]; // 로봇 기준 vx, vy, wz
        let margin = p.stop_margin;
        let turn_to = |target: f64, wmax: f64| -> (f64, f64) {
            let e = ang_diff(target, pose[2]);
            let w = (2.0 * 1.5 * e.abs()).sqrt().min(wmax).min(2.5 * e.abs()).max(s.base_wcreep.min(e.abs() * 10.0));
            (e, e.signum() * w)
        };
        let mut finish: Option<&'static str> = None;
        match (m.kind, m.phase) {
            // ---------------- probe: 돌기 → 앞으로(깊이 안전 정지)
            (NavKind::Probe, 0) => {
                if m.recover == 0 && self.nav.have_map {
                    // 돌기 전에 한 번 모서리 검사(사각형 몸통): 막히면 반대쪽으로 돌기, 그것도 막히면 보고
                    m.recover = 1;
                    let cm = &self.nav.cm;
                    if dp.fp.turn_clear(cm, xy[0], xy[1], pose[2], m.turn_yaw) < TURN_MIN_CLEAR && ang_diff(m.turn_yaw, pose[2]).abs() > 3f64.to_radians() {
                        let e = ang_diff(m.turn_yaw, pose[2]);
                        let other = pose[2] + e - 2.0 * std::f64::consts::PI * e.signum();
                        let mut ok_other = true;
                        let n = 36;
                        for k in 0..=n {
                            let yy = pose[2] + (other - pose[2]) * k as f64 / n as f64;
                            if dp.fp.clear(cm, xy[0], xy[1], yy) < TURN_MIN_CLEAR {
                                ok_other = false;
                                break;
                            }
                        }
                        if ok_other {
                            m.turn_yaw = other;
                        } else {
                            m.stop = Some(("cannot turn here: body corners would hit".into(), r2(dp.fp.clear(cm, xy[0], xy[1], pose[2]).max(0.0))));
                            m.outcome = Some("blocked");
                            m.phase = 2;
                        }
                    }
                    if m.phase == 2 {
                        return None;
                    }
                }
                let e_full = m.turn_yaw - pose[2];
                let (e, w) = turn_to(m.turn_yaw, s.nav_wmax);
                let (e, w) = if e_full.abs() > std::f64::consts::PI { (e_full, e_full.signum() * w.abs()) } else { (e, w) };
                des[2] = w;
                if e.abs() < 1.5f64.to_radians() {
                    m.phase = if m.fwd_m > 0.005 { 1 } else { 2 };
                    m.fwd_start = xy;
                }
            }
            (NavKind::Probe, 1) => {
                let done = (xy[0] - m.fwd_start[0]).hypot(xy[1] - m.fwd_start[1]);
                let remain = m.fwd_m - done;
                let (free, by) = self.free_ahead(pose, pose[2], true, m.fwd_start);
                let room = free - margin;
                if remain < 0.01 {
                    m.phase = 2;
                } else if room <= 0.01 {
                    m.stop = Some((by.into(), r2(free.max(0.0))));
                    m.phase = 2;
                    m.outcome = Some("blocked");
                } else {
                    let v = s.probe_vmax.min((2.0 * s.base_acc * remain.min(room)).sqrt()).max(s.base_creep.min(remain));
                    des[0] = v;
                    des[2] = sat(2.0 * ang_diff(m.turn_yaw, pose[2]), s.nav_wmax);
                }
            }
            // ---------------- go_to: 경로 따라가기(DWA)
            (NavKind::GoTo, 0) => {
                // 다시 계획: 지도가 바뀌고 바뀐 칸이 경로 통로에 걸리며 실제로 막혔을 때, 또는 3 s 마다(열린 지름길)
                if self.nav.cm.version != m.cm_ver {
                    m.cm_ver = self.nav.cm.version;
                    let hit = change_hits_path(&self.nav.cm, &m.path, p.robot_r) && path_blocked(&self.nav.cm, &m.path, 0, p.robot_r).is_some();
                    if hit || self.nav.now - m.last_replan > 3.0 {
                        if !self.replan(m) && hit {
                            m.stuck = 30; // 회복으로
                        }
                    }
                }
                let goal_d = (m.goal[0] - xy[0]).hypot(m.goal[1] - xy[1]);
                let (carrot, remain, _) = carrot_on(&m.path, xy, 0.8);
                if goal_d < 0.15 {
                    m.phase = 1;
                } else {
                    let herr = ang_diff((carrot[1] - xy[1]).atan2(carrot[0] - xy[0]), pose[2]);
                    let can_turn = |d: f64| dp.fp.turn_clear(&self.nav.cm, xy[0], xy[1], pose[2], pose[2] + d) >= TURN_MIN_CLEAR;
                    if herr.abs() > 30f64.to_radians() && can_turn(herr.signum() * 10f64.to_radians()) {
                        des[2] = sat(2.0 * herr, s.nav_wmax).abs().max(0.3) * herr.signum();
                        m.stuck = 0;
                    } else {
                        let vmax = s.nav_vmax.min((2.0 * s.base_acc * remain).sqrt() + s.base_creep);
                        let o = dwa(&self.nav.cm, &m.path, pose, m.start_xy, carrot, remain, vmax, s.nav_wmax, DWA_MARGIN, &dp, p.start_free_r, false);
                        // 얇은 깊이 정지
                        let g = nav::depth_guard_unmapped(&self.nav.cm, &dp.fp, pose, pose[2], 0.02, 3.0);
                        let v_ok = match g {
                            Some(d) => o.v.min((2.0 * s.base_acc * (d - margin).max(0.0)).sqrt()),
                            None => o.v,
                        };
                        if g.map_or(false, |d| d - margin <= 0.01) && o.v > 0.0 {
                            m.guard_stops += 1;
                        }
                        let guard_blocked = g.map_or(false, |d| d - margin <= 0.02);
                        if std::env::var("MR_DEBUG_DWA").is_ok() {
                            eprintln!("dwa t={:.2} pose=({:.2},{:.2},{:.0}) carrot=({:.2},{:.2}) herr={:.0} ok={} v={:.2} w={:.2} clear={:.2} guard={:?} cur_clear={:.2} stuck={}",
                                t, pose[0], pose[1], pose[2].to_degrees(), carrot[0], carrot[1], herr.to_degrees(), o.ok, o.v, o.w, o.clear, g, self.nav.cm.clear_at(xy[0], xy[1]), m.stuck);
                        }
                        // 지도에 아직 없는 장애물 점이 몸 둘레(외접 원 + 5 cm) 안: 돌면 모서리가 닿을 수 있다 → 돌지 말고 조금 물러남
                        let unmapped_near = self.nav.cm.scan.as_ref().map_or(false, |sc| {
                            let r = dp.fp.circum() + 0.05;
                            sc.hits.iter().any(|q| (q[0] - xy[0]).hypot(q[1] - xy[1]) < r && self.nav.cm.clear_at(q[0], q[1]) > 0.08)
                        });
                        if guard_blocked && unmapped_near {
                            let back = [xy[0] - 0.1 * pose[2].cos(), xy[1] - 0.1 * pose[2].sin()];
                            if dp.fp.clear(&self.nav.cm, back[0], back[1], pose[2]) > 0.0 && !self.nav.cm.unknown_at(back[0], back[1]) {
                                des[0] = -0.1;
                            }
                            m.guard_stops += 1;
                            m.stuck += 1;
                        } else if guard_blocked && herr.abs() > 8f64.to_radians() && can_turn(herr.signum() * 10f64.to_radians()) {
                            // 앞이 막혔지만 가야 할 쪽은 옆: 제자리에서 그쪽으로 돈다(둘레에 지도 밖 장애물 없음)
                            des[2] = sat(2.0 * herr, s.nav_wmax).abs().max(0.3) * herr.signum();
                            m.stuck = 0;
                        } else if o.ok && v_ok > 0.02 {
                            des[0] = v_ok;
                            des[1] = o.vy;
                            des[2] = o.w;
                            m.stuck = 0;
                        } else if o.ok && (o.v < -0.01 || o.w.abs() > 0.05 || o.vy.abs() > 0.01) {
                            // 뒤로 빠지기 / 옆으로 빠지기 / 제자리 돌기(DWA 가 굴려 보고 안전하다고 한 것)
                            des[0] = o.v;
                            des[1] = o.vy;
                            des[2] = o.w;
                            m.stuck += (o.v >= -0.01) as u32;
                        } else if herr.abs() > 8f64.to_radians() && can_turn(herr.signum() * 10f64.to_radians()) {
                            // 앞으로 갈 안전한 궤적이 없으면 먼저 가야 할 쪽으로 제자리 돌기
                            des[2] = sat(2.0 * herr, s.nav_wmax).abs().max(0.3) * herr.signum();
                            m.stuck += 1;
                        } else {
                            m.stuck += 1;
                        }
                    }
                }
                if m.stuck >= 10 {
                    // 회복: 1 다시 계획 → 2 둘러보기 → 3 뒤로 → 4 보고
                    m.stuck = 0;
                    m.recover += 1;
                    match m.recover {
                        1 => {
                            self.replan(m);
                        }
                        2 => {
                            m.phase = 3;
                            m.look_target = pose[2] + 35f64.to_radians();
                        }
                        3 => {
                            // 되짚기(Nav2 backup 과 같은 뜻): 지나온 자리(trail)를 거꾸로 0.7 m, 몸은 돌리지 않고 전방향으로
                            let trail = &self.nav.path_log;
                            let mut pts = vec![];
                            let mut acc = 0.0;
                            let mut last = xy;
                            for q in trail.iter().rev() {
                                let d = (q[0] - last[0]).hypot(q[1] - last[1]);
                                if d < 0.05 {
                                    continue;
                                }
                                acc += d;
                                pts.push(*q);
                                last = *q;
                                if acc > 0.7 {
                                    break;
                                }
                            }
                            if pts.is_empty() {
                                m.recover = 4;
                            } else {
                                m.retrace = pts;
                                m.phase = 4;
                                m.back_start = xy;
                                m.settle = 0;
                            }
                        }
                        _ => {}
                    }
                    if m.recover >= 4 {
                        let (free, by) = self.free_ahead(pose, pose[2], false, xy);
                        m.stop = Some((format!("path blocked ({by})"), r2(free.max(0.0))));
                        m.outcome = Some("blocked");
                        m.phase = 2;
                    }
                }
            }
            (NavKind::GoTo, 1) => match m.look_yaw {
                Some(ly) if dp.fp.turn_clear(&self.nav.cm, xy[0], xy[1], pose[2], ly) >= TURN_MIN_CLEAR || ang_diff(ly, pose[2]).abs() < 3f64.to_radians() => {
                    let (e, w) = turn_to(ly, s.nav_wmax);
                    des[2] = w;
                    if e.abs() < 3f64.to_radians() {
                        m.phase = 2;
                    }
                }
                _ => m.phase = 2,
            },
            (NavKind::GoTo, 3) => {
                if !can_turn_now(&dp, &self.nav.cm, xy, pose[2], m.look_target) {
                    m.phase = 0;
                    m.stuck = 10;
                }
                let (e, w) = turn_to(m.look_target, s.nav_wmax);
                des[2] = w;
                if e.abs() < 3f64.to_radians() {
                    m.phase = 0;
                    self.replan(m);
                }
            }
            (_, 5) => {
                des[0] = -0.12;
                if (xy[0] - m.back_start[0]).hypot(xy[1] - m.back_start[1]) > 0.15 || m.settle > 60 {
                    m.phase = 2;
                    m.settle = 0;
                }
                m.settle += 1;
            }
            (NavKind::GoTo, 4) => {
                // 되짚기: 다음 trail 점으로 0.12 m/s, 몸 방향 그대로(로봇 기준 속도로 바꿈)
                m.settle += 1;
                while let Some(q) = m.retrace.first() {
                    if (q[0] - xy[0]).hypot(q[1] - xy[1]) < 0.06 {
                        m.retrace.remove(0);
                    } else {
                        break;
                    }
                }
                match m.retrace.first() {
                    Some(q) if m.settle < 240 => {
                        let (dx, dy) = (q[0] - xy[0], q[1] - xy[1]);
                        let d = dx.hypot(dy).max(1e-6);
                        let (sn, cs) = pose[2].sin_cos();
                        let (wx, wy) = (dx / d * 0.12, dy / d * 0.12);
                        // 몸 방향이 그때와 다를 수 있다: 0.1 m 앞 자세가 지금보다 좁아지며 닿을 듯하면 멈춤
                        let ahead = dp.fp.clear(&self.nav.cm, xy[0] + dx / d * 0.1, xy[1] + dy / d * 0.1, pose[2]);
                        let now_c = dp.fp.clear(&self.nav.cm, xy[0], xy[1], pose[2]);
                        if ahead < 0.0 && ahead < now_c {
                            m.retrace.clear();
                        } else {
                            des[0] = cs * wx + sn * wy;
                            des[1] = -sn * wx + cs * wy;
                        }
                    }
                    _ => {
                        m.phase = 0;
                        m.settle = 0;
                        m.stuck = 0;
                        self.replan(m);
                    }
                }
            }
            (_, _) => {
                // 2: 0 지령으로 멈출 때까지
                m.settle += 1;
                let still = meas_v[0].hypot(meas_v[1]) < 0.01 && meas_v[2].abs() < 0.02;
                if (still && m.settle >= 3) || m.settle >= 12 {
                    finish = Some(m.outcome.unwrap_or("reached"));
                }
            }
        }
        // 가감속 한도
        let lim = [s.base_acc * self.dt, s.base_acc * self.dt, s.base_wacc * self.dt];
        for i in 0..3 {
            m.v_cmd[i] += (des[i] - m.v_cmd[i]).clamp(-lim[i], lim[i]);
        }
        if m.phase == 2 {
            m.v_cmd = [0.0; 3];
        }
        let _ = m.phase;
        // 막힘(접촉): 지령은 움직이는데 측정은 거의 0
        let cmd_sp = m.v_cmd[0].hypot(m.v_cmd[1]);
        let moving_cmd = cmd_sp > 0.05 || m.v_cmd[2].abs() > 0.1;
        let barely = meas_v[0].hypot(meas_v[1]) < 0.2 * cmd_sp.max(0.05) && meas_v[2].abs() < 0.2 * m.v_cmd[2].abs().max(0.1);
        m.stall = if moving_cmd && barely { m.stall + 1 } else { 0 };
        if m.stall >= 30 && finish.is_none() && m.phase != 5 {
            m.stop = Some(("contact: base pushed but did not move; backed off".into(), 0.0));
            m.outcome = Some("blocked");
            self.nav.n_stall += 1;
            // 닿았으면 왔던 쪽(뒤)으로 0.15 m 물러난 뒤 보고
            m.phase = 5;
            m.back_start = xy;
            m.stall = 0;
            m.v_cmd = [0.0; 3];
        }
        if t > m.t_max && finish.is_none() && m.phase != 2 && m.phase != 5 {
            m.outcome = Some("timeout");
            m.phase = 2;
            m.v_cmd = [0.0; 3];
        }
        let f = finish?;
        Some(self.nav_result(m, f, t))
    }

    /// 다시 계획(지금 자세 → 같은 목표). 성공하면 true.
    fn replan(&mut self, m: &mut NavMove) -> bool {
        let (mm, a) = self.nav.analyze();
        m.last_replan = self.nav.now;
        m.replans += 1;
        self.nav.n_replans += 1;
        let Some(gi) = a.nearest_reachable(&mm.grid, m.goal[0], m.goal[1], 0.6) else { return false };
        match a.path_to(&mm.grid, gi, &self.nav.params) {
            Some(pth) => {
                let (gx, gy) = mm.grid.center(gi);
                m.goal = [gx, gy];
                m.path = pth;
                self.nav.last_path = m.path.clone();
                true
            }
            None => false,
        }
    }

    /// 앞(heading)으로 몸통 원이 갈 수 있는 거리와 무엇이 막나: 비용 지도(장애물; probe 가 아니면 모르는 칸도)와 얇은 깊이 정지 중 작은 것
    pub(crate) fn free_ahead(&self, pose: [f64; 3], heading: f64, allow_unknown: bool, anchor: [f64; 2]) -> (f64, &'static str) {
        let cm = &self.nav.cm;
        let fp = self.nav.dwa.fp;
        let (sn, cs) = heading.sin_cos();
        let mut d_map = 3.0;
        let mut by = "none";
        if self.nav.have_map {
            let step = cm.grid.res * 0.5;
            let mut t = 0.0;
            while t < 3.0 {
                let (x, y) = (pose[0] + cs * t, pose[1] + sn * t);
                if fp.clear(cm, x, y, pose[2]) < 0.0 {
                    d_map = (t - step).max(0.0);
                    by = "map obstacle";
                    break;
                }
                if !allow_unknown && cm.unknown_at(x, y) && (x - anchor[0]).hypot(y - anchor[1]) > self.nav.params.start_free_r {
                    d_map = (t - step).max(0.0);
                    by = "unknown space (cannot see)";
                    break;
                }
                t += step;
            }
        }
        let g = cm.scan.as_ref().and_then(|sc| depth_guard(sc, &fp, pose, heading, 0.0, 3.0));
        match g {
            Some(d) if d < d_map => (d, "obstacle seen by camera"),
            None if allow_unknown && self.nav.have_map && !cm.scan.as_ref().map_or(false, |_| true) => (d_map, by),
            None if allow_unknown && self.nav.have_map => {
                // 카메라 밖으로 모르는 곳에 가려 함: 지도에서 아는 빈칸까지만
                let (d2, by2) = self.free_ahead(pose, heading, false, anchor);
                (d2, by2)
            }
            _ => (d_map, by),
        }
    }

    fn nav_result(&mut self, m: &NavMove, outcome: &'static str, t: f64) -> Value {
        let pose = self.nav.pose;
        if outcome != "reached" && m.kind == NavKind::GoTo {
            match self.nav.failed_goals.iter_mut().find(|(g, _)| (g[0] - m.goal[0]).hypot(g[1] - m.goal[1]) < 0.6) {
                Some(e) => e.1 += 1,
                None => self.nav.failed_goals.push((m.goal, 1)),
            }
        }
        if outcome != "reached" {
            if let Ok(dir) = std::env::var("MR_DEBUG_DIR") {
                self.debug_dump(&dir, m);
            }
        }
        let moved = self.nav.odo_m - m.moved0;
        let mut r = json!({"status": outcome, "part": "base", "mode": if m.kind == NavKind::GoTo { "go_to" } else { "probe" },
            "target": m.label, "moved_m": r2(moved), "turned_deg": r1(ang_diff(pose[2], m.start_yaw).to_degrees()),
            "time_s": r2(t)});
        if m.kind == NavKind::GoTo {
            r["planned_m"] = json!(r2(m.path_m0));
            if m.replans > 0 {
                r["replans"] = json!(m.replans);
            }
        }
        if let Some((why, d)) = &m.stop {
            r["stopped_by"] = json!(why);
            r["clear_m"] = json!(d);
        }
        if outcome != "reached" {
            self.nav.n_blocked += (outcome == "blocked") as u64;
            r["hint"] = json!(match (m.kind, outcome) {
                (NavKind::Probe, "blocked") => "stopped before an obstacle; turn to look elsewhere or go_to a frontier",
                (NavKind::GoTo, "blocked") => "could not get there safely; choose another frontier id from the map summary",
                _ => "did not finish in time; check the map summary and choose again",
            });
        }
        self.attach_map(&mut r);
        r
    }

    /// 결과에 지도 요약(LLM 관측)과 측정값을 붙인다. 지도가 없으면 그대로.
    pub(crate) fn attach_map(&mut self, r: &mut Value) {
        if !self.nav.have_map {
            return;
        }
        let t0 = std::time::Instant::now();
        let (m, a) = self.nav.analyze();
        let plan_us = self.nav.replan_us;
        let obs = self.observe(&m, &a);
        self.nav.obs_us = t0.elapsed().as_micros() as u64;
        r["map"] = obs;
        r["_m"] = json!({
            "gt_cov": self.nav.gt_coverage().map(|c| (c * 1000.0).round() / 1000.0),
            "free_m2": r1(m.grid.free_area()), "path_m": r2(self.nav.odo_m), "sim_s": r2(self.nav.now),
            "contacts": self.nav.contacts, "stalls": self.nav.n_stall, "blocked": self.nav.n_blocked, "replans": self.nav.n_replans,
            "min_clear_m": if self.nav.min_clear.is_finite() { json!(r2(self.nav.min_clear)) } else { Value::Null },
            "obs_us": self.nav.obs_us, "plan_us": plan_us, "costmap_us": self.nav.cm.build_us,
            "pose": [r2(self.nav.pose[0]), r2(self.nav.pose[1]), r1(self.nav.pose[2].to_degrees())],
            "frontiers_all": a.frontiers.len(),
        });
    }

    /// LLM 이 보는 지도 요약. 목표 id 를 새로 매기고 기억한다(다음 go_to 가 씀).
    fn observe(&mut self, m: &MapIn, a: &Analysis) -> Value {
        let pose = self.nav.pose;
        let p = self.nav.params.clone();
        let free = m.grid.free_area();
        let new_free = free - self.nav.last_free_m2;
        self.nav.last_free_m2 = free;
        let mut targets = vec![];
        let bad = |f: &Frontier| self.nav.failed_goals.iter().any(|(g, n)| *n >= 2 && (g[0] - f.goal[0]).hypot(g[1] - f.goal[1]) < 0.6);
        let n_skipped = a.frontiers.iter().filter(|f| bad(f)).count();
        let fr: Vec<&Frontier> = a.frontiers.iter().filter(|f| !bad(f)).take(p.max_frontiers).collect();
        let mut fl = vec![];
        for (k, f) in fr.iter().enumerate() {
            let id = format!("F{}", k + 1);
            let rel = ang_diff((f.goal[1] - pose[1]).atan2(f.goal[0] - pose[0]), pose[2]);
            let mut o = json!({"id": id, "path_m": r1(f.path_m), "dir": dir_text(rel), "new_area_m2": r1(f.gain_m2.min(40.0))});
            if f.room != 0 {
                o["room"] = json!(format!("R{}", f.room));
            }
            fl.push(o);
            targets.push((*f).clone());
        }
        let mut tg: Vec<Target> = targets
            .iter()
            .enumerate()
            .map(|(k, f)| Target { id: format!("F{}", k + 1), goal: f.goal, look_yaw: Some(f.look_yaw), path_m: f.path_m })
            .collect();
        let rooms = map::rooms_info(m, a);
        let cur_room = self.nav.rooms.as_ref().map(|r| r.at(pose[0], pose[1])).unwrap_or(0);
        let mut rl = vec![];
        for ri in &rooms {
            let id = format!("R{}", ri.id);
            let visited = self.nav.visited.contains(&ri.id);
            let mut o = json!({"id": id, "visited": visited});
            match (ri.goal_idx, ri.path_m) {
                (Some(gi), Some(pm)) => {
                    o["path_m"] = json!(r1(pm));
                    let (gx, gy) = m.grid.center(gi);
                    tg.push(Target { id: id.clone(), goal: [gx, gy], look_yaw: None, path_m: pm });
                }
                _ => {
                    o["path_m"] = Value::Null;
                }
            }
            rl.push(o);
        }
        self.nav.targets = tg;
        let nf = a.frontiers.len() - n_skipped;
        self.nav.last_frontiers = nf;
        let mut o = json!({
            "free_m2": r1(free),
            "new_free_m2": r1(new_free),
            "frontiers": fl,
        });
        if self.nav.style == ObsStyle::Full {
            o["around"] = map::ring(m, a, &p);
        }
        if !rl.is_empty() {
            o["rooms"] = json!(rl);
            o["in_room"] = if cur_room != 0 { json!(format!("R{cur_room}")) } else { Value::Null };
        }
        o["status"] = json!(if nf == 0 {
            "no reachable frontier left: the known map is closed (exploration complete unless a blocked area remains)".to_string()
        } else {
            format!("{nf} reachable frontier(s); all frontier goals are in KNOWN free space (go_to works)")
        });
        if n_skipped > 0 {
            o["skipped_failed"] = json!(n_skipped);
        }
        o
    }
}

impl Robot {
    /// 디버그: 비용 지도(회색 모름, 흰 빈칸, 검 장애물, 노랑 부풀림) + 경로(파랑) + 로봇(빨강) PPM
    pub fn debug_dump_now(&self, dir: &str) {
        let m = match &self.active {
            Some(crate::Active { motion: crate::Motion::Nav(m), .. }) => m.clone(),
            _ => return,
        };
        self.debug_dump(dir, &m);
    }

    pub(crate) fn debug_dump(&self, dir: &str, m: &NavMove) {
        let cm = &self.nav.cm;
        let g = &cm.grid;
        if g.w == 0 {
            return;
        }
        let mut img = vec![0u8; g.w * g.h * 3];
        for i in 0..g.w * g.h {
            let v = g.cells[i];
            let c: [u8; 3] = if v < 0 {
                [128, 128, 128]
            } else if map::is_occ(v) {
                [0, 0, 0]
            } else if (cm.dobs[i] as f64) < self.nav.params.robot_r {
                [230, 210, 120]
            } else if map::is_free(v) {
                [255, 255, 255]
            } else {
                [200, 160, 160]
            };
            let (x, y) = (i % g.w, i / g.w);
            let o = ((g.h - 1 - y) * g.w + x) * 3;
            img[o..o + 3].copy_from_slice(&c);
        }
        let mut put = |x: f64, y: f64, c: [u8; 3]| {
            let (cx, cy) = g.cell_of(x, y);
            if let Some(_) = g.idx(cx, cy) {
                let o = ((g.h - 1 - cy as usize) * g.w + cx as usize) * 3;
                img[o..o + 3].copy_from_slice(&c);
            }
        };
        for k in 0..m.path.len().saturating_sub(1) {
            let (a, b) = (m.path[k], m.path[k + 1]);
            let n = ((b[0] - a[0]).hypot(b[1] - a[1]) / 0.02).ceil() as usize + 1;
            for j in 0..=n {
                let t = j as f64 / n as f64;
                put(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), [0, 0, 255]);
            }
        }
        if let Some(sc) = &cm.scan {
            for h in &sc.hits {
                put(h[0], h[1], [0, 180, 0]);
            }
        }
        put(m.goal[0], m.goal[1], [255, 0, 255]);
        let p = self.nav.pose;
        for k in 0..12 {
            put(p[0] + 0.02 * k as f64 * p[2].cos(), p[1] + 0.02 * k as f64 * p[2].sin(), [255, 0, 0]);
        }
        let _ = std::fs::create_dir_all(dir);
        let f = format!("{dir}/blocked_{:06}.ppm", (self.nav.now * 10.0) as u64);
        let mut out = format!("P6\n{} {}\n255\n", g.w, g.h).into_bytes();
        out.extend_from_slice(&img);
        let _ = std::fs::write(f, out);
    }
}

impl Robot {
    /// 뷰어 겹침(map 좌표): 지나온 길, 계획 경로, 목표, 지난 관측의 목표 id, 자세
    pub fn overlay(&self) -> Value {
        let n = &self.nav;
        let rp = |v: &[[f64; 2]]| v.iter().map(|q| [r2(q[0]), r2(q[1])]).collect::<Vec<_>>();
        let goal = match &self.active {
            Some(crate::Active { motion: crate::Motion::Nav(m), .. }) => json!({"label": m.label, "xy": [r2(m.goal[0]), r2(m.goal[1])], "mode": if m.kind == NavKind::GoTo { "go_to" } else { "probe" }}),
            _ => Value::Null,
        };
        json!({"stamp": r2(n.now), "pose": [r2(n.pose[0]), r2(n.pose[1]), r1(n.pose[2].to_degrees())],
            "trail": rp(&n.path_log), "plan": rp(&n.last_path), "goal": goal,
            "targets": n.targets.iter().map(|t| json!({"id": t.id, "xy": [r2(t.goal[0]), r2(t.goal[1])], "path_m": r1(t.path_m)})).collect::<Vec<_>>(),
            "odo_m": r2(n.odo_m), "gt_cov": n.gt_coverage().map(|c| (c * 1000.0).round() / 1000.0), "contacts": n.contacts})
    }
}

fn can_turn_now(dp: &DwaParams, cm: &Costmap, xy: [f64; 2], yaw: f64, target: f64) -> bool {
    dp.fp.turn_clear(cm, xy[0], xy[1], yaw, target) >= TURN_MIN_CLEAR
}

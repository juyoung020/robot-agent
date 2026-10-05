//! 가짜 세계(가짜 실행기): 시뮬레이터 없이 에이전트 반복문 전체를 돌려 보는 곳.
//! 물체 몇 개와 두 손 로봇 하나를 2차원으로 흉내 내고, 지시(구조체)를 받아 스텝마다 진행한다.
//! - 이동하면 base 속도가 나오고(→ 오도메트리·이동 멈춤 경계), 집으면 그리퍼가 닫힌다(→ 그리퍼 경계).
//! - 가까이 가지 않고 집으려 하면 아무 일도 안 일어나 예산이 끝난다(→ 재시도·재계획 경로 시험).
//! - 물체는 로봇 4 m 안에 들어와야 그래프에 나타난다(scenemap 이 점점 지도를 채우는 것 흉내).
//! - "move to <이름>" 은 이름이 맞는 가장 가까운 물체로 간다(VLA 가 문장만 보고 찾아가는 것 흉내).
//! - 그래프 노드는 scenemap 물체 표의 뜻(처음 자리 first_position, handled = 옮겨짐)을 따른다.

use crate::graph::{Node, SharedGraph, StaticGraph};
use crate::instruction::Instruction;
use crate::monitor::MonitorCfg;
use crate::odom::Pose;
use crate::planner::{Agent, Decision};
use crate::session::EnvSession;
use crate::util::{norm_category, Rng};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct WObj {
    pub id: String,
    pub label: String,
    pub pos: [f64; 3],
    /// 처음 자리(scenemap first_position 흉내)
    pub orig: [f64; 3],
    pub on: Option<String>,
    pub inside: Option<String>,
    pub room: String,
    pub toggled: bool,
    pub open: bool,
}

#[derive(Debug, Clone)]
pub enum Goal {
    Toggled(String),
    AllInside(Vec<String>, String),
}

#[derive(Debug, Clone)]
struct Exec {
    ins: Instruction,
    t: u64,
    dur: u64,
    ok: bool,
}

pub struct World {
    pub objs: Vec<WObj>,
    pub robot: Pose,
    pub vel: [f64; 3],
    /// 손에 든 것(최대 2, 앞이 오른손)
    pub held: Vec<String>,
    pub grip: [f64; 2],
    pub step: u64,
    pub rng: Rng,
    pub p_success: f64,
    pub see_radius: f64,
    pub seen: BTreeSet<String>,
    pub goal: Goal,
    pub rooms: Vec<(String, [f64; 2])>,
    cur: Option<Exec>,
    pub log: Vec<String>,
}

/// 이름이 같은 종류인가("radio receiver" ↔ "radio" 처럼 BDDL 이름과 장면 이름이 달라도)
fn label_match(want: &str, label: &str) -> bool {
    norm_category(label) == want || crate::graph::text_score(want, label) >= 0.6
}

fn obj(id: &str, label: &str, x: f64, y: f64, z: f64, on: Option<&str>, room: &str) -> WObj {
    WObj { id: id.into(), label: label.into(), pos: [x, y, z], orig: [x, y, z], on: on.map(|s| s.into()), inside: None, room: room.into(), toggled: false, open: false }
}

impl World {
    pub fn scenario(name: &str, seed: u64, p_success: f64) -> Option<World> {
        let (objs, goal, rooms) = match name {
            "radio" => (
                vec![
                    obj("coffee_table_koagbh_0", "coffee table", 3.5, 2.0, 0.45, None, "living_room"),
                    obj("radio_89", "radio", 3.5, 2.0, 0.55, Some("coffee_table_koagbh_0"), "living_room"),
                    obj("sofa_lugrhk_1", "sofa", 4.8, 3.2, 0.4, None, "living_room"),
                    obj("fridge_petcxr_0", "fridge", -1.5, 0.5, 0.9, None, "kitchen"),
                ],
                Goal::Toggled("radio_89".into()),
                vec![("kitchen".to_string(), [-1.0, 0.0]), ("living_room".to_string(), [3.5, 2.0])],
            ),
            "trash" => (
                vec![
                    obj("floors_ulujpr_0", "floors", 4.5, 0.5, 0.0, None, "living_room"),
                    obj("can_of_soda_113", "can of soda", 5.0, 1.0, 0.05, Some("floors_ulujpr_0"), "living_room"),
                    obj("can_of_soda_114", "can of soda", 5.4, -0.6, 0.05, Some("floors_ulujpr_0"), "living_room"),
                    obj("can_of_soda_115", "can of soda", 6.1, 0.6, 0.05, Some("floors_ulujpr_0"), "living_room"),
                    obj("trash_can_116", "trash can", -1.8, 1.0, 0.3, None, "kitchen"),
                ],
                Goal::AllInside(vec!["can_of_soda_113".into(), "can_of_soda_114".into(), "can_of_soda_115".into()], "trash_can_116".into()),
                vec![("kitchen".to_string(), [-1.5, 0.5]), ("living_room".to_string(), [5.0, 0.5])],
            ),
            _ => return None,
        };
        let mut w = World {
            objs,
            robot: Pose::default(),
            vel: [0.0; 3],
            held: vec![],
            grip: [0.1, 0.1],
            step: 0,
            rng: Rng::new(seed),
            p_success,
            see_radius: 4.0,
            seen: BTreeSet::new(),
            goal,
            rooms,
            cur: None,
            log: vec![],
        };
        w.update_seen();
        Some(w)
    }

    pub fn task_name(scn: &str) -> &'static str {
        match scn {
            "trash" => "picking_up_trash",
            _ => "turning_on_radio",
        }
    }

    fn find(&self, id: &str) -> Option<&WObj> {
        self.objs.iter().find(|o| o.id == id)
    }

    /// id → 그 물체, 이름 → 이름이 맞는 가장 가까운(안 든) 물체, 방 이름 → 방 가운데.
    fn target_xy(&self, id: &str) -> Option<[f64; 2]> {
        if let Some(o) = self.find(id) {
            return Some([o.pos[0], o.pos[1]]);
        }
        let want = norm_category(id);
        let near = self
            .objs
            .iter()
            .filter(|o| label_match(&want, &o.label) && !self.held.contains(&o.id) && o.inside.is_none())
            .min_by(|a, b| {
                let da = (a.pos[0] - self.robot.x).hypot(a.pos[1] - self.robot.y);
                let db = (b.pos[0] - self.robot.x).hypot(b.pos[1] - self.robot.y);
                da.partial_cmp(&db).unwrap()
            });
        if let Some(o) = near {
            return Some([o.pos[0], o.pos[1]]);
        }
        let key = id.replace(' ', "_");
        self.rooms.iter().find(|(r, _)| *r == key || r.replace('_', " ") == id).map(|(_, c)| *c)
    }

    fn resolve_id(&self, id: &str) -> Option<String> {
        if self.find(id).is_some() {
            return Some(id.to_string());
        }
        let want = norm_category(id);
        self.objs
            .iter()
            .filter(|o| label_match(&want, &o.label) && o.inside.is_none())
            .min_by(|a, b| {
                let da = (a.pos[0] - self.robot.x).hypot(a.pos[1] - self.robot.y);
                let db = (b.pos[0] - self.robot.x).hypot(b.pos[1] - self.robot.y);
                da.partial_cmp(&db).unwrap()
            })
            .map(|o| o.id.clone())
    }

    fn dist(&self, id: &str) -> f64 {
        self.target_xy(id).map(|[x, y]| (x - self.robot.x).hypot(y - self.robot.y)).unwrap_or(f64::INFINITY)
    }

    fn update_seen(&mut self) {
        let (rx, ry) = (self.robot.x, self.robot.y);
        let r = self.see_radius;
        for o in &self.objs {
            if (o.pos[0] - rx).hypot(o.pos[1] - ry) <= r {
                self.seen.insert(o.id.clone());
            }
        }
    }

    pub fn graph_nodes(&self) -> Vec<Node> {
        self.objs
            .iter()
            .filter(|o| self.seen.contains(&o.id))
            .map(|o| {
                let moved = (0..3).map(|i| (o.pos[i] - o.orig[i]).powi(2)).sum::<f64>().sqrt() > 0.15;
                Node {
                    id: o.id.clone(),
                    label: o.label.clone(),
                    score: 1.0,
                    center: o.pos,
                    extent: [0.3, 0.3, 0.3],
                    room: Some(o.room.clone()),
                    on: o.on.clone().or(o.inside.clone()),
                    num_observations: 10,
                    original: Some(o.orig),
                    handled: moved,
                    structural: o.label == "floors",
                }
            })
            .collect()
    }

    pub fn start(&mut self, ins: Option<&Instruction>) {
        self.cur = ins.map(|i| {
            let nav = i.skill == "move to";
            let dur = if nav { 0 } else { self.rng.range(120, 320) };
            let ok = self.rng.f64() < self.p_success;
            Exec { ins: i.clone(), t: 0, dur, ok }
        });
    }

    pub fn goal_met(&self) -> bool {
        match &self.goal {
            Goal::Toggled(id) => self.find(id).map(|o| o.toggled).unwrap_or(false),
            Goal::AllInside(ids, c) => ids.iter().all(|i| self.find(i).and_then(|o| o.inside.clone()).as_deref() == Some(c.as_str())),
        }
    }

    fn hand_of(&self, id: &str) -> usize {
        // 앞의 것 = 오른손(1), 두 번째 = 왼손(0)
        match self.held.iter().position(|h| h == id) {
            Some(0) => 1,
            Some(_) => 0,
            None => if self.held.is_empty() { 1 } else { 0 },
        }
    }

    /// 한 스텝 진행. 반환: (base_qvel, 그리퍼 벌어짐)
    pub fn tick(&mut self) -> ([f64; 3], [f64; 2]) {
        let mut v = [0.0; 3];
        if let Some(mut ex) = self.cur.take() {
            ex.t += 1;
            let first = ex.ins.objects.first().cloned().unwrap_or_default();
            let first_id = self.resolve_id(&first).unwrap_or(first.clone());
            let finished = match ex.ins.skill.as_str() {
                "move to" => match self.target_xy(&first) {
                    Some([tx, ty]) => {
                        let (dx, dy) = (tx - self.robot.x, ty - self.robot.y);
                        let d = dx.hypot(dy);
                        if d <= 0.8 {
                            true
                        } else {
                            let err = (dy.atan2(dx) - self.robot.yaw + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
                            v[2] = (2.0 * err).clamp(-1.0, 1.0);
                            v[0] = if err.abs() < 0.4 { 0.6 } else { 0.05 };
                            false
                        }
                    }
                    None => ex.t > 60,
                },
                skill if matches!(skill, "pick up from" | "hold" | "lift") && self.held.contains(&first_id) => true, // 이미 들고 있음
                skill => {
                    let near = self.dist(&first_id) < 1.3 || self.held.contains(&first_id);
                    let placing = matches!(skill, "place on" | "place in" | "place on next to" | "place in next to" | "place under");
                    // 놓기는 받침·통도 가까이 있거나 손에 있어야 한다
                    let dest_ok = !placing
                        || ex.ins.objects.get(1).and_then(|s| self.resolve_id(s)).map(|d| self.dist(&d) < 1.5 || self.held.contains(&d)).unwrap_or(true);
                    if !near || !dest_ok {
                        false // 전제 불충족: 멈춰 있음(예산이 끝날 때까지)
                    } else {
                        let hand = self.hand_of(&first_id);
                        if matches!(skill, "pick up from" | "hold" | "lift") && ex.t > ex.dur / 2 && self.held.len() < 2 && !self.held.contains(&first_id) {
                            // 그리퍼가 다 닫히는 순간 잡기가 끝난다
                            let target = if ex.ok { 0.035 } else { 0.0 };
                            self.grip[hand] += (target - self.grip[hand]) * 0.35;
                            (self.grip[hand] - target).abs() < 0.003
                        } else if placing && ex.t > ex.dur / 2 && self.held.contains(&first_id) {
                            self.grip[hand] += (0.1 - self.grip[hand]) * 0.35;
                            (self.grip[hand] - 0.1).abs() < 0.003
                        } else {
                            ex.t >= ex.dur
                        }
                    }
                }
            };
            if finished {
                self.finish(&ex, &first_id);
            } else {
                self.cur = Some(ex);
            }
        }
        // 운동: 월드 좌표 = 출발점 기준(오도메트리와 같은 전진 오일러)
        let (s, c) = self.robot.yaw.sin_cos();
        self.robot.x += (c * self.vel[0] - s * self.vel[1]) / 30.0;
        self.robot.y += (s * self.vel[0] + c * self.vel[1]) / 30.0;
        self.robot.yaw += self.vel[2] / 30.0;
        self.vel = v;
        for (i, h) in self.held.clone().iter().enumerate() {
            let side = if i == 0 { -0.25 } else { 0.25 };
            let (fx, fy) = (0.4 * self.robot.yaw.cos() - side * self.robot.yaw.sin(), 0.4 * self.robot.yaw.sin() + side * self.robot.yaw.cos());
            let (x, y) = (self.robot.x + fx, self.robot.y + fy);
            if let Some(o) = self.objs.iter_mut().find(|o| o.id == *h) {
                o.pos = [x, y, 0.9];
                o.on = None;
            }
        }
        // 든 통 안의 물건은 통을 따라간다
        let conts: Vec<(String, [f64; 3])> = self.objs.iter().map(|o| (o.id.clone(), o.pos)).collect();
        for o in self.objs.iter_mut() {
            if let Some(c) = &o.inside {
                if let Some((_, p)) = conts.iter().find(|(id, _)| id == c) {
                    o.pos = [p[0], p[1], p[2] + 0.05];
                }
            }
        }
        self.update_seen();
        self.step += 1;
        (self.vel, self.grip)
    }

    fn finish(&mut self, ex: &Exec, first: &str) {
        let first = first.to_string();
        let second = ex.ins.objects.get(1).and_then(|s| self.resolve_id(s));
        let hand = self.hand_of(&first);
        let msg = match ex.ins.skill.as_str() {
            "move to" => format!("arrived near {}", ex.ins.objects.first().cloned().unwrap_or_default()),
            "pick up from" | "hold" | "lift" if self.held.contains(&first) => format!("already holding {first}"),
            "pick up from" | "hold" | "lift" if ex.ok && self.held.len() < 2 && !self.held.contains(&first) => {
                self.held.push(first.clone());
                if let Some(o) = self.objs.iter_mut().find(|o| o.id == first) {
                    o.inside = None;
                }
                format!("holding {first}")
            }
            "pick up from" | "hold" | "lift" => {
                self.grip[hand] = 0.1;
                format!("grasp of {first} missed")
            }
            "place on" | "place in" | "place on next to" | "place in next to" | "place under" | "release" if self.held.contains(&first) => {
                self.held.retain(|h| *h != first);
                self.grip[hand] = 0.1;
                if let Some(sup) = second.clone() {
                    let sp = self.find(&sup).map(|o| o.pos);
                    let inside = ex.ins.skill == "place in";
                    if let Some(o) = self.objs.iter_mut().find(|o| o.id == first) {
                        if let Some(p) = sp {
                            o.pos = [p[0], p[1], if inside { p[2] + 0.05 } else { p[2] + 0.1 }];
                        }
                        if inside {
                            o.inside = Some(sup.clone());
                            o.on = None;
                        } else {
                            o.on = Some(sup.clone());
                        }
                    }
                } else if let Some(o) = self.objs.iter_mut().find(|o| o.id == first) {
                    o.pos[2] = 0.05;
                }
                format!("placed {first}")
            }
            "press" | "turn on switch" if ex.ok => {
                if let Some(o) = self.objs.iter_mut().find(|o| o.id == first) {
                    o.toggled = true;
                }
                format!("{first} toggled on")
            }
            "open door" | "open drawer" | "open lid" if ex.ok => {
                if let Some(o) = self.objs.iter_mut().find(|o| o.id == first) {
                    o.open = true;
                }
                format!("{first} opened")
            }
            s => format!("{s} {first}: {}", if ex.ok { "ok" } else { "no effect" }),
        };
        self.log.push(format!("step {}: {msg}", self.step));
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EpisodeResult {
    pub success: bool,
    pub steps: u64,
    pub decisions: u64,
    pub llm_calls: u64,
    pub fallbacks: u64,
    pub prompts: Vec<String>,
    pub world_log: Vec<String>,
}

/// 에피소드 하나: 중계기와 같은 세션·경계 규칙으로 에이전트를 부른다.
pub fn run_episode(world: &mut World, agent: &mut Agent, graph: &SharedGraph, mcfg: MonitorCfg, max_steps: u64, episode: u32, send_images: bool) -> EpisodeResult {
    let mut sess = EnvSession::new(0, mcfg, 30.0, max_steps, &agent.core.task.prompt);
    sess.reset(episode);
    agent.reset_episode(episode);
    let s0 = agent.core.stats.clone();
    let mut prompts = Vec::new();
    let mut qvel = [0.0; 3];
    let mut grips = world.grip;
    let mut success = false;
    let img = crate::image::Rgb::pattern(64, 64, 7).jpeg(70);
    for _ in 0..max_steps {
        *graph.0.lock().unwrap() = StaticGraph { nodes: world.graph_nodes(), name: "mockworld".into() };
        if let Some(t) = sess.on_obs(qvel, grips) {
            let images = if send_images { vec![("head".to_string(), img.clone())] } else { vec![] };
            let ev = sess.event(t, grips, images);
            let d = agent.decide(&ev);
            agent.maintain();
            sess.apply(&d);
            match &d {
                Decision::Issue { instruction, prompt, .. } => {
                    prompts.push(prompt.clone());
                    world.start(Some(instruction));
                }
                Decision::Finish { .. } => world.start(None),
                Decision::Continue { .. } => {}
            }
        }
        let (v, g) = world.tick();
        qvel = v;
        grips = g;
        sess.mon.advance();
        if world.goal_met() {
            success = true;
            break;
        }
    }
    let st = &agent.core.stats;
    EpisodeResult {
        success,
        steps: world.step,
        decisions: st.decisions - s0.decisions,
        llm_calls: st.llm_calls - s0.llm_calls,
        fallbacks: st.fallbacks - s0.fallbacks,
        prompts,
        world_log: world.log.clone(),
    }
}

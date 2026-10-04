//! [`Robot`] 의 VLA 실행기 쪽(LIMO + OMX-F). 주행 층(자세 적분·지도·몸통 여유)은 move_robot 과 같은 것을 쓴다.
//! 그래서 VLA 행동도 `go_to`/`delta` 와 같은 지도 장애물·깊이 정지 코드([`Robot::free_ahead`])를 지난다(POLICY 7.1).

use crate::limo::{LimoState, ACTION_DIM as A8};
use crate::goal::{self, GoalRef};
use crate::verify::{self, MemObject, ObjState};
use crate::vla::{self, build_slots, make_policy, parse_call, progress_check, FilterInfo, SkillKind, VlaObs, VlaOut, VlaRun};
use crate::{Robot, Tick};
use serde_json::{json, Value};
use std::f64::consts::PI;

impl Robot {
    pub fn vla_busy(&self) -> bool {
        self.vla.run.is_some()
    }

    /// 시뮬 시각(s) — 기억 물체의 last_seen 과 같은 시계
    pub fn sim_now(&self) -> f64 {
        self.nav.now + self.vla.time_offset
    }

    /// 기억 물체 바꾸기(keyframe 마다, scenemap 스냅숏 그대로). `now` = 같은 시계의 지금 시각.
    pub fn vla_set_objects(&mut self, objs: Vec<MemObject>, now: f64) {
        let (nn, pose, odo) = (self.nav.now, self.nav.pose, self.nav.odo_m);
        self.vla.set_objects(objs, now, nn, pose, odo);
    }

    /// 접촉 누적 수(바닥 아닌 것): 몸통(차체·바퀴) / 팔·그리퍼. 몸통 접촉이 늘면 VLA 단계는 failed(unsafe).
    pub fn vla_set_contacts(&mut self, body: u64, arm: u64) {
        self.vla.contacts_body = body;
        self.vla.contacts_arm = arm;
        self.nav.contacts = body + arm;
    }

    fn vla_immediate(&mut self, v: Value) -> bool {
        self.result = Some(v);
        true
    }

    /// VLA 단계 시작. `true` = 바로 끝남(오류·handback, 결과 준비), `false` = 실행 시작.
    pub fn vla_start(&mut self, args: &Value) -> bool {
        if self.vla.run.is_some() {
            return self.vla_immediate(json!({"status": "error", "executor": "vla", "message": "a VLA step is already running"}));
        }
        if self.busy() {
            return self.vla_immediate(json!({"status": "error", "executor": "vla", "message": "move_robot is still executing the previous call"}));
        }
        let call = match parse_call(args) {
            Ok(c) => c,
            Err(v) => return self.vla_immediate(v),
        };
        let policy = match make_policy(&call.policy) {
            Ok(p) => p,
            Err(e) => return self.vla_immediate(json!({"status": "unavailable", "executor": "vla", "message": e})),
        };
        let mut call = call;
        let p = self.vla.params.clone();
        let now = self.sim_now();
        let pose = self.nav.pose;
        // 놓을 지점 검사(바닥 빈칸·면 윗면, 닿음) → 가장 가까운 맞는 자리로 옮김(goal.rs check_point)
        let mut pcheck = None;
        if let Some((xy, z)) = call.point_req {
            let checked = if self.nav.have_map {
                let (m, a) = self.nav.analyze();
                goal::check_point(xy, z, &self.vla.objects, Some(&goal::PlaceMap { grid: &m.grid, a: &a }))
            } else {
                goal::check_point(xy, z, &self.vla.objects, None)
            };
            match checked {
                Ok(c) => {
                    call.goal.place = Some(GoalRef::Point(c.point));
                    pcheck = Some((c, [xy[0], xy[1], z.unwrap_or(f64::NAN)]));
                }
                Err(e) => {
                    let mut v = vla::result("error", "invalid_point", json!({"why": e}), None, &call, 0, None);
                    v["message"] = json!(e);
                    v["hint"] = json!("tap a free floor spot or the top of a low surface (0.05–0.52 m) the robot can reach");
                    return self.vla_immediate(v);
                }
            }
        }
        let hb = |reason: &str, ev: Value| vla::result("handback", reason, ev, None, &call, 0, None);
        // 시작 조건(POLICY 1.2): 기준이 지점이면 1.5 m 안만, 물체면 기억에 있음·1.5 m 안·보이거나 불확실도 작음
        if let Some(pt) = call.anchor_point() {
            let rel = verify::to_robot(pose, pt);
            let dist = rel[0].hypot(rel[1]);
            let mut m = json!({"dist_m": (dist * 100.0).round() / 100.0, "rel_m": [(rel[0] * 100.0).round() / 100.0, (rel[1] * 100.0).round() / 100.0]});
            if let Some((c, asked)) = &pcheck {
                m["point"] = c.to_json(*asked);
            }
            if dist > p.start_max_m {
                return self.vla_immediate(hb("too_far", json!({"why": format!("the point is {dist:.2} m away (VLA starts within {:.1} m)", p.start_max_m), "m": m, "next": "move_robot go_to"})));
            }
            if call.kind == SkillKind::Place {
                if let Some(id) = call.pick_id() {
                    if self.vla.find(id).is_none() {
                        let v = hb("not_in_map", json!({"why": format!("{id} (the held object) is not in the object memory")}));
                        return self.vla_immediate(v);
                    }
                }
            }
        } else {
            let id = call.anchor_id().unwrap_or("").to_string();
            let Some(o) = self.vla.find(&id).cloned() else {
                let known: Vec<String> = self.vla.objects.iter().take(20).map(|o| o.id.clone()).collect();
                let v = hb("not_in_map", json!({"why": format!("{id} is not in the object memory"), "known_ids": known}));
                return self.vla_immediate(v);
            };
            let rel = verify::to_robot(pose, o.pos);
            let dist = rel[0].hypot(rel[1]);
            let unc = self.vla.uncertainty(&o, now, pose, self.nav.odo_m);
            let visible = now - o.last_seen <= p.visible_s && o.state != ObjState::Gone;
            let m = json!({"dist_m": (dist * 100.0).round() / 100.0, "rel_m": [(rel[0] * 100.0).round() / 100.0, (rel[1] * 100.0).round() / 100.0], "visible": visible, "unc_m": (unc * 100.0).round() / 100.0, "state": o.state.name()});
            if o.state == ObjState::Gone {
                return self.vla_immediate(hb("not_visible", json!({"why": format!("{id} is marked gone in memory"), "m": m})));
            }
            if dist > p.start_max_m {
                return self.vla_immediate(hb("too_far", json!({"why": format!("{id} is {dist:.2} m away (VLA starts within {:.1} m)", p.start_max_m), "m": m, "next": "move_robot go_to"})));
            }
            if !visible && unc > p.unc_max_m {
                return self.vla_immediate(hb("not_visible", json!({"why": format!("{id} not seen for {:.1} s and memory uncertainty {unc:.2} m > {:.2} m", now - o.last_seen, p.unc_max_m), "m": m})));
            }
        }
        let tgt = call.objects.first().and_then(|id| self.vla.find(id)).map(|o| o.pos);
        self.vla.filter.init = false; // 다음 관측에서 유지값을 다시 잡는다(튀지 않게)
        let c0 = (self.vla.contacts_body, self.vla.contacts_arm);
        let mut run = VlaRun::new(call, policy, &LimoState::default(), pose, tgt, c0);
        run.point_check = pcheck;
        self.vla.run = Some(run);
        false
    }

    /// 실행 중인 VLA 단계를 멈춘다(사용자 "그만", 상태 기계). 결과 = handback(reason).
    pub fn vla_stop(&mut self, reason: &str) -> bool {
        let Some(run) = self.vla.run.take() else { return false };
        let contacts = (self.vla.contacts_body + self.vla.contacts_arm).saturating_sub(run.contacts0.0 + run.contacts0.1);
        let mc = self.nav.have_map.then_some(run.min_clear);
        self.result = Some(vla::result("handback", reason, json!({"why": "stopped from outside"}), Some(&run), &run.call, contacts, mc));
        self.vla.filter.prev[0] = 0.0;
        self.vla.filter.prev[1] = 0.0;
        self.vla.last_cmd = Some(self.vla.filter.prev);
        true
    }

    /// 거르개만(학습된 엔진이 따로 돌 때 행동 하나 거르기 = POLICY 7.1 `mr_filter`). 자세는 움직이지 않는다.
    pub fn vla_filter(&mut self, proprio: &[f32], input: &[f64; A8]) -> Result<([f64; A8], FilterInfo), String> {
        let st = LimoState::from_proprio(proprio)?;
        Ok(self.filter_one(&st, input))
    }

    fn filter_one(&mut self, st: &LimoState, input: &[f64; A8]) -> ([f64; A8], FilterInfo) {
        let pose = self.nav.pose;
        let anchor = [pose[0], pose[1]];
        let (fwd, back) = if self.nav.have_map {
            (Some(self.free_ahead(pose, pose[2], true, anchor)), Some(self.free_ahead(pose, pose[2] + PI, false, anchor)))
        } else {
            (None, None)
        };
        let free = move |f: bool| if f { fwd } else { back };
        let margin = self.nav.params.stop_margin;
        let p = self.vla.params.clone();
        self.vla.filter.filter(input, st, self.dt, &p, &free, margin)
    }

    /// 한 스텝(각본·재생 정책): LIMO proprio(24) → 거른 행동 8.
    pub fn vla_tick(&mut self, proprio: &[f32], out: &mut [f32]) -> Tick {
        self.vla_tick_inner(proprio, None, out)
    }

    /// 한 스텝(밖의 정책): 그 정책의 행동 8·끝 신호·확신 낮음을 넣는다. 거르기·끝 판정은 같다.
    pub fn vla_tick_ext(&mut self, proprio: &[f32], action: &[f64; A8], end_prob: f64, unsure: f64, out: &mut [f32]) -> Tick {
        self.vla_tick_inner(proprio, Some(VlaOut { action: *action, end_prob, unsure }), out)
    }

    fn write8(out: &mut [f32], a: &[f64; A8]) {
        for (o, x) in out.iter_mut().zip(a.iter()) {
            *o = *x as f32;
        }
    }

    fn vla_tick_inner(&mut self, proprio: &[f32], ext: Option<VlaOut>, out: &mut [f32]) -> Tick {
        self.ticks += 1;
        let st = match LimoState::from_proprio(proprio) {
            Ok(s) => s,
            Err(_) => {
                let mut h = self.vla.filter.prev;
                h[0] = 0.0;
                h[1] = 0.0;
                Self::write8(out, &h);
                return Tick::BadObs;
            }
        };
        let dt = self.dt;
        match self.gt_pose.take() {
            Some(g) => self.nav.integrate_gt(g, dt),
            None => self.nav.integrate(st.base_v, dt),
        }
        let Some(mut run) = self.vla.run.take() else {
            if !self.vla.filter.init {
                self.vla.filter.hold_from(&st);
            }
            let mut h = self.vla.filter.prev;
            h[0] = 0.0;
            h[1] = 0.0;
            Self::write8(out, &h);
            return Tick::Idle;
        };
        let p = self.vla.params.clone();
        let now = self.sim_now();
        let pose = self.nav.pose;
        let odo = self.nav.odo_m;
        if !self.vla.filter.init {
            self.vla.filter.hold_from(&st);
        }
        run.steps += 1;
        run.t = run.steps as f64 * dt;
        if run.steps == 1 {
            let tgt = run.call.objects.first().and_then(|id| self.vla.find(id)).map(|o| o.pos);
            run.prog = (0.0, st.arm, st.grip, pose, tgt);
        }
        if let Some(e) = ext {
            run.ext = Some(e);
        }
        // 목표 칸(매 스텝, 지금 지도에서 id 풀기 — 사라지면 마지막으로 안 자리 + 잃음)
        run.goal_now = goal::build(&run.call.goal, &self.vla.objects, pose, st.eef, &mut run.goal_mem);
        // 정책(10 Hz): 실행기 스텝 3 개마다 한 번
        let every = ((1.0 / (dt * p.policy_hz)).round() as u64).max(1);
        if (run.steps - 1) % every == 0 {
            if run.policy.is_external() {
                if let Some(e) = run.ext {
                    run.out = e;
                } else {
                    run.out = VlaOut { action: self.vla.filter.prev, end_prob: 0.0, unsure: 0.0 };
                }
            } else {
                let ctx = &self.vla;
                let slots = build_slots(&ctx.objects, pose, &run.call.objects, now, &|o| ctx.uncertainty(o, now, pose, odo), p.visible_s);
                let last = ctx.filter.prev;
                let obs = VlaObs { skill: &run.call.skill, kind: run.call.kind, body: &st, slots: &slots, t: run.t - dt, last_cmd: &last, goal: &run.goal_now };
                run.out = run.policy.act(&obs);
            }
            run.policy_steps += 1;
        }
        let input = run.out.action;
        let (a, info) = self.filter_one(&st, &input);
        run.n_clipped += info.clipped as u64;
        run.n_base_stops += info.base_stop.map_or(0, |(_, r)| (r <= 0.01) as u64);
        run.n_arm_blocked += info.arm_blocked as u64;
        if self.nav.have_map {
            let c = self.nav.dwa.fp.clear(&self.nav.cm, pose[0], pose[1], pose[2]);
            run.min_clear = run.min_clear.min(c);
        }
        // 자동 확인(verify.rs) — 대상·받침은 set_plan id 로
        let kind = run.call.kind;
        let point = run.call.anchor_point();
        let target = run.call.objects.first().and_then(|id| self.vla.find(id)).cloned();
        let support = if kind == SkillKind::Place && point.is_none() { run.call.objects.get(1).and_then(|i| self.vla.find(i)).cloned() } else { None };
        let anchor = run.call.anchor_id().and_then(|id| self.vla.find(id)).cloned();
        let ev = verify::Evidence {
            kind,
            target: target.as_ref(),
            support: support.as_ref(),
            point,
            pose,
            grip: st.grip,
            grip_cmd: a[7],
            eef: st.eef,
            base_speed: st.base_v[0].hypot(st.base_v[1]),
            base_turn: st.base_v[2],
        };
        let verdict = verify::evidence(&ev);
        let (goal_visible, goal_unc) = match &anchor {
            Some(o) => (now - o.last_seen <= p.visible_s && o.state != ObjState::Gone, self.vla.uncertainty(o, now, pose, odo)),
            None => (false, f64::INFINITY),
        };
        let body_contacts = self.vla.contacts_body.saturating_sub(run.contacts0.0);
        let stalled = progress_check(&mut run, &p, &st, pose, target.as_ref().map(|o| o.pos));
        let decision = run.decide(&p, &info, body_contacts, anchor.as_ref(), point.is_some(), goal_visible, goal_unc, verdict, stalled);
        match decision {
            None => {
                Self::write8(out, &a);
                self.vla.run = Some(run);
                Tick::Moving
            }
            Some((status, reason, evidence)) => {
                // 끝: 베이스 0, 팔·그리퍼는 마지막 지령 유지(든 물체를 놓지 않게)
                let mut h = a;
                h[0] = 0.0;
                h[1] = 0.0;
                self.vla.filter.prev = h;
                self.vla.last_cmd = Some(h);
                Self::write8(out, &h);
                let contacts = (self.vla.contacts_body + self.vla.contacts_arm).saturating_sub(run.contacts0.0 + run.contacts0.1);
                let mc = self.nav.have_map.then_some(run.min_clear);
                self.result = Some(vla::result(status, &reason, evidence, Some(&run), &run.call, contacts, mc));
                Tick::Done
            }
        }
    }

    /// 이번 스텝의 목표 칸(원값 32, 정규화 32 — goal.rs). 실행 중이 아니면 None
    pub fn vla_goal_entries(&self) -> Option<&crate::goal::GoalEntries> {
        self.vla.run.as_ref().map(|r| &r.goal_now)
    }

    /// 지금 물체 칸(시험·기록용): set_plan id 로 목표인지가 켜진 칸들
    pub fn vla_slots(&self, goals: &[String]) -> Vec<vla::Slot> {
        let now = self.sim_now();
        let (pose, odo) = (self.nav.pose, self.nav.odo_m);
        let ctx = &self.vla;
        let g: Vec<String> = goals.iter().map(|s| verify::norm_id(s)).collect();
        build_slots(&ctx.objects, pose, &g, now, &|o| ctx.uncertainty(o, now, pose, odo), ctx.params.visible_s)
    }
}

//! 에이전트 = 상태([`Core`]) + 결정기([`Decider`]).
//!
//! **결정 인터페이스**(plan.md 0절: 나중에 강화학습 플래너로 바꾸거나 병행): 결정기는 경계마다
//! `(Core 의 상태, BoundaryEvent) → Decision` 만 책임진다. 상태 관리(단계 기록·계획 체크리스트·물체 기억·
//! 참조 풀기·예산·문장 만들기)는 [`Core`] 가 하므로 결정기를 바꿔도 그대로다.
//! - [`LlmDecider`]: OpenAI 호환 도구 호출 반복문(수업 week02/03 구조를 이 문제에 맞게 키움)
//! - [`PriorDecider`]: LLM 없이 참고 순서를 따르는 결정적 기준선(강화학습 자리의 본보기)
//! - 강화학습 결정기는 [`Core::decision_input`](구조화된 상태, 기록에도 남음)을 입력으로,
//!   [`Core::make_instruction`] 으로 행동(단계 이름 + 물체)을 문장으로 바꿔 `Decision` 을 내면 된다.
//!
//! LLM 결정기의 한 경계:
//! 1. 맥락 조립(system 하나 + 최근 turn + 이번 사건 user 메시지(머리 카메라 영상))
//! 2. 반복(최대 `max_iterations`, 벽시계 `decision_budget_s`): LLM → 도구 실행 → 도구 결과를 관찰값으로 되돌림.
//!    끝내는 도구(issue_command / continue_current / finish) 가 검증을 통과하면 결정.
//!    도구 오류·잘못된 인자는 크래시가 아니라 관찰값으로 돌려준다. 도구 없이 글만 오면 재촉.
//! 3. 실패·시간 초과면 결정적 대체([`Core::fallback`]).
//!
//! 요약(기억 압축)은 [`Agent::maintain`] 에서, 결정을 돌려준 **뒤** 에 한다.

use crate::catalog::{Catalog, TaskCard};
use crate::context;
use crate::graph::{ObjectMemory, SceneGraph};
use crate::instruction::{Format, Instruction};
use crate::llm::{ChatRequest, Llm, Msg, Part};
use crate::memory::{Memory, Outcome};
use crate::monitor::Trigger;
use crate::odom::Pose;
use crate::plan::{Plan, Status};
use crate::trace::Tracer;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerCfg {
    pub format: Format,
    pub max_prompt_tokens: usize,
    pub max_iterations: usize,
    /// 한 경계에서 LLM 에 쓰는 벽시계 한도(초). 평가기 웹소켓 ping_timeout=300 s 보다 훨씬 짧게.
    pub decision_budget_s: f64,
    /// 표본추출 값(결정론: temperature 0, top_k 1, seed 0 …). 요청마다 명시하고 기록에 남긴다.
    pub sampling: crate::llm::Sampling,
    pub max_out_tokens: u32,
    /// 맥락(system + turn) 추정 토큰 한도. 넘으면 참고 순서 2·3위 → 물체 표 → 오래된 turn 순으로 줄인다.
    pub ctx_budget_tokens: usize,
    pub thinking: bool,
    pub send_images: bool,
    pub image_side: usize,
    pub jpeg_quality: u8,
    pub max_retries: u32,
    pub max_continues: u32,
    /// 지시가 바뀌면 VLA 의 남은 행동 묶음을 버리고 바로 새로 추론
    pub flush_on_change: bool,
    pub keep_turns: usize,
    pub summarize_batch: usize,
}

impl Default for PlannerCfg {
    fn default() -> Self {
        PlannerCfg {
            format: Format::Subtask,
            max_prompt_tokens: crate::instruction::DEFAULT_MAX_TOKENS,
            max_iterations: 8,
            decision_budget_s: 120.0,
            sampling: crate::llm::Sampling::default(),
            max_out_tokens: 768,
            ctx_budget_tokens: 9000,
            thinking: false,
            send_images: true,
            image_side: 448,
            jpeg_quality: 80,
            max_retries: 2,
            max_continues: 3,
            flush_on_change: true,
            keep_turns: 3,
            summarize_batch: 3,
        }
    }
}

/// 중계기(또는 가짜 세계)가 경계에서 넘기는 것.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryEvent {
    pub env: usize,
    pub episode: u32,
    pub step: u64,
    pub trigger: Trigger,
    pub max_steps: u64,
    pub pose: Pose,
    pub stage_steps: u64,
    pub stage_budget: u64,
    pub moved_in_stage: f64,
    pub base_speed: f64,
    /// base 가 연달아 멈춰 있던 스텝 수(이동 끝·막힘 증거)
    #[serde(default)]
    pub still_steps: u32,
    pub grippers: [f64; 2],
    /// (카메라 이름, JPEG). 기록에는 파일 이름만.
    #[serde(skip)]
    pub images: Vec<(String, Vec<u8>)>,
    #[serde(default)]
    pub image_files: Vec<String>,
}

impl BoundaryEvent {
    pub fn steps_left(&self) -> u64 {
        self.max_steps.saturating_sub(self.step)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Decision {
    Issue { instruction: Instruction, prompt: String, budget: u64, check_every: u64, previous: String, flush: bool, source: String },
    Continue { extra: u64, check_every: u64, reason: String, source: String },
    Finish { prompt: String, reason: String, source: String },
}

impl Decision {
    pub fn prompt(&self) -> Option<&str> {
        match self {
            Decision::Issue { prompt, .. } | Decision::Finish { prompt, .. } => Some(prompt),
            Decision::Continue { .. } => None,
        }
    }
    pub fn source(&self) -> &str {
        match self {
            Decision::Issue { source, .. } | Decision::Continue { source, .. } | Decision::Finish { source, .. } => source,
        }
    }
    /// 기록 비교용(원천 표시 제외)
    pub fn key(&self) -> String {
        match self {
            Decision::Issue { prompt, budget, previous, .. } => format!("issue|{prompt}|{budget}|{previous}"),
            Decision::Continue { extra, .. } => format!("continue|{extra}"),
            Decision::Finish { prompt, .. } => format!("finish|{prompt}"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PlannerStats {
    pub decisions: u64,
    pub llm_calls: u64,
    pub llm_ms: u64,
    pub llm_errors: u64,
    pub tool_calls: u64,
    pub tool_errors: u64,
    pub fallbacks: u64,
    pub summaries: u64,
    pub decision_ms_max: u64,
}

/// 결정 인터페이스. LLM·강화학습·규칙 결정기가 같은 자리에 들어간다.
pub trait Decider: Send {
    fn name(&self) -> String;
    fn decide(&mut self, core: &mut Core, ev: &BoundaryEvent) -> Decision;
    /// 결정 뒤 여유 시간에 할 일(기억 요약 등)
    fn maintain(&mut self, core: &mut Core) {
        if let Some(r) = core.mem.pending_summary() {
            core.mem.fallback_summary(r);
        }
    }
    /// 당분간 결정할 일이 없을 때(로컬 LLM 서버 내리기)
    fn release(&mut self) {}
}

/// 결정기에 넘기는 구조화된 상태(강화학습 입력·기록용). 글 맥락과 같은 정보를 숫자·목록으로.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionInput {
    pub task_index: u32,
    pub task: String,
    pub step: u64,
    pub steps_left: u64,
    pub trigger: Trigger,
    pub pose: Pose,
    pub grippers: [f64; 2],
    pub holding: Vec<String>,
    pub current_skill: Option<String>,
    pub current_objects: Vec<String>,
    pub stage_steps: u64,
    pub stage_budget: u64,
    pub stage_attempt: u32,
    pub plan_done: usize,
    pub plan_total: usize,
    pub next_plan_step: Option<(String, Vec<String>)>,
    /// (id, 이름, 로봇 기준 앞, 왼쪽, 처음 자리에서 움직인 거리, 다룬 횟수)
    pub objects: Vec<(String, String, f64, f64, f64, u32)>,
    pub finished: bool,
}

/// 에이전트 상태. 결정기가 바뀌어도 그대로.
pub struct Core {
    pub cfg: PlannerCfg,
    pub task: TaskCard,
    pub budgets: HashMap<String, u64>,
    pub graph: Box<dyn SceneGraph>,
    pub mem: Memory,
    pub plan: Plan,
    pub objects: ObjectMemory,
    pub holding: Vec<String>,
    pub finished: bool,
    pub continues: u32,
    pub tracer: Option<Tracer>,
    pub env: usize,
    pub episode: u32,
    pub stats: PlannerStats,
    /// 결정 중에만 채워짐(도구가 참고)
    pub ev: Option<BoundaryEvent>,
    pub extra_msgs: Vec<Msg>,
    /// 이번 경계에서 자동 증거와 다른 판단에 한 번 되물었나
    pub pushback: bool,
}

pub fn tool_err(msg: impl Into<String>) -> Value {
    json!({"status": "error", "message": msg.into()})
}

impl Core {
    pub fn new(cfg: PlannerCfg, task: TaskCard, catalog: &Catalog, graph: Box<dyn SceneGraph>, env: usize) -> Core {
        let budgets = catalog.skills.iter().map(|s| (s.name.clone(), s.budget_steps)).collect();
        Core {
            mem: Memory { keep_turns: cfg.keep_turns, summarize_batch: cfg.summarize_batch, ..Default::default() },
            plan: Plan::default(),
            cfg,
            task,
            budgets,
            graph,
            objects: ObjectMemory::default(),
            holding: vec![],
            finished: false,
            continues: 0,
            tracer: None,
            env,
            episode: 0,
            stats: PlannerStats::default(),
            ev: None,
            extra_msgs: vec![],
            pushback: false,
        }
    }

    /// 새 판. 기억·계획·물체 기억을 비우고, 참고 순서 1위로 체크리스트를 채운다.
    pub fn reset_episode(&mut self, episode: u32) {
        self.episode = episode;
        self.mem = Memory { keep_turns: self.cfg.keep_turns, summarize_batch: self.cfg.summarize_batch, ..Default::default() };
        self.plan = Plan::from_prior(self.task.priors.first());
        self.objects = ObjectMemory::default();
        self.holding.clear();
        self.finished = false;
        self.continues = 0;
        let labels = self.task_labels();
        if let Err(e) = self.graph.reset(&labels) {
            self.trace("graph_error", json!({"op": "reset", "error": e}));
        }
        self.trace("episode_start", json!({"episode": episode, "task": self.task.name, "prompt": self.task.prompt, "plan_source": self.plan.source}));
    }

    /// 이 과제에 나오는 물체 이름들(그래프 CLIP 이름표 후보, 맥락 관련도)
    pub fn task_labels(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        for (o, _) in &self.task.problem.objects {
            let n = crate::util::display_name(o);
            if !n.is_empty() && n != "agent" && !v.contains(&n) {
                v.push(n);
            }
        }
        for p in self.task.priors.iter().take(3) {
            for s in &p.steps {
                for o in &s.objects {
                    for part in o.split(", ") {
                        let n = part.to_string();
                        if !n.is_empty() && !v.contains(&n) {
                            v.push(n);
                        }
                    }
                }
            }
        }
        v
    }

    pub fn trace(&self, kind: &str, data: Value) {
        if let Some(t) = &self.tracer {
            t.event(self.env, kind, data);
        }
    }

    pub fn budget_for(&self, skill: &str) -> u64 {
        self.budgets.get(skill).copied().unwrap_or_else(|| crate::vocab::lookup(skill).map(|k| k.default_budget()).unwrap_or(600))
    }

    pub fn ctx<'a>(&'a self, ev: &'a BoundaryEvent) -> context::Ctx<'a> {
        context::Ctx {
            cfg: &self.cfg,
            task: &self.task,
            mem: &self.mem,
            plan: &self.plan,
            objects: &self.objects,
            holding: &self.holding,
            ev,
            finished: self.finished,
            shrink: 0,
            evidence: self.evidence(ev),
        }
    }

    /// 경계 시작: 그래프 새로 읽기, 물체 기억 갱신, 기록.
    pub fn begin(&mut self, ev: &BoundaryEvent) {
        self.stats.decisions += 1;
        let nodes = match self.graph.all() {
            Ok(n) => n,
            Err(e) => {
                self.trace("graph_error", json!({"op": "all", "error": e}));
                vec![]
            }
        };
        self.objects.observe(&nodes, ev.step);
        if ev.grippers.iter().all(|&g| g > 0.09) {
            self.holding.clear();
        }
        self.ev = Some(ev.clone());
        self.pushback = false;
        let input = self.decision_input(ev);
        let (verdict, why) = self.evidence(ev);
        self.trace("boundary", json!({"event": ev, "nodes": nodes, "holding": self.holding, "input": input, "evidence": {"verdict": verdict, "why": why}}));
    }

    /// 지금 단계가 끝났는지에 대한 자동 증거(감시 신호 + 그래프). LLM 에 힌트로 보여 주고, 대체 결정에도 쓴다.
    /// 반환: (판정, 근거 글). 판정은 "done" / "failed" / "running" / "unknown".
    pub fn evidence(&self, ev: &BoundaryEvent) -> (&'static str, String) {
        let Some(cur) = self.mem.current() else { return ("unknown", "nothing is running".into()) };
        let ins = &cur.instruction;
        let skill = ins.skill.as_str();
        // 지시할 때 그래프에 없던 물체(이름으로 보냄)는 지금 그래프에서 이름으로 다시 찾는다
        let target = ins.objects.first().and_then(|o| {
            self.objects.known.get(o).or_else(|| {
                let name = ins.names.first().cloned().unwrap_or_else(|| o.clone());
                self.objects.resolve(&name, &ev.pose).and_then(|r| self.objects.known.get(&r.id))
            })
        });
        let closed_on = ev.grippers.iter().any(|&w| w > 0.005 && w < 0.07);
        let closed_empty = ev.grippers.iter().any(|&w| w <= 0.005);
        let all_open = ev.grippers.iter().all(|&w| w > 0.09);
        match skill {
            "move to" => {
                if let Some(k) = target {
                    let d = (k.last_center[0] - ev.pose.x).hypot(k.last_center[1] - ev.pose.y);
                    if d <= 1.2 && ev.base_speed < 0.05 {
                        return ("done", format!("robot stopped {d:.1} m from {}", k.id));
                    }
                    if ev.base_speed >= 0.05 {
                        return ("running", format!("base still moving ({:.2} m/s), {d:.1} m to {}", ev.base_speed, k.id));
                    }
                    if ev.still_steps >= 60 && ev.moved_in_stage < 0.1 {
                        return ("failed", format!("base has not moved for {} steps and is {d:.1} m from {}", ev.still_steps, k.id));
                    }
                    return ("unknown", format!("{d:.1} m from {}", k.id));
                }
                if ev.moved_in_stage > 0.3 && ev.still_steps >= 20 {
                    return ("unknown", "stopped after travelling; target not in the scene graph yet".into());
                }
                ("unknown", "target not in the scene graph yet".into())
            }
            "pick up from" | "hold" | "lift" => {
                let lifted = target.map(|k| k.last_center[2] - k.first_center[2]).unwrap_or(0.0);
                let moved = target.map(|k| k.moved()).unwrap_or(0.0);
                if closed_on && (lifted > 0.1 || moved > 0.15 || target.is_none()) {
                    return ("done", format!("a gripper is closed on an object; target lifted {lifted:.2} m, moved {moved:.2} m"));
                }
                if closed_on {
                    return ("unknown", "a gripper is closed on something but the target has not moved".into());
                }
                if closed_empty {
                    return ("failed", "a gripper closed on nothing (grasp missed)".into());
                }
                ("running", "grippers still open".into())
            }
            "place on" | "place in" | "place on next to" | "place in next to" | "place under" | "release" | "hang" | "insert" | "attach" => {
                let sup = ins.objects.get(1).and_then(|o| self.objects.known.get(o));
                let near_sup = match (target, sup) {
                    (Some(t), Some(s)) => ((t.last_center[0] - s.last_center[0]).hypot(t.last_center[1] - s.last_center[1])) < 0.6,
                    _ => true,
                };
                let opened = all_open || ev.trigger == Trigger::GripperChange;
                if opened && near_sup {
                    return ("done", "a gripper opened and the object is at the destination".into());
                }
                if all_open {
                    return ("failed", "grippers opened but the object is not at the destination".into());
                }
                ("running", "still holding".into())
            }
            _ => ("unknown", "not measurable from proprioception; judge from the image".into()),
        }
    }

    pub fn decision_input(&self, ev: &BoundaryEvent) -> DecisionInput {
        let cur = self.mem.current();
        let (done, total) = self.plan.progress();
        let mut objects: Vec<(String, String, f64, f64, f64, u32)> = self
            .objects
            .known
            .values()
            .map(|k| {
                let (f, l) = ev.pose.to_robot(k.last_center[0], k.last_center[1]);
                (k.id.clone(), self.objects.name_of(&k.id).unwrap_or_default(), f, l, k.moved(), k.handled)
            })
            .collect();
        objects.sort_by(|a, b| a.2.hypot(a.3).partial_cmp(&b.2.hypot(b.3)).unwrap());
        DecisionInput {
            task_index: self.task.index,
            task: self.task.name.clone(),
            step: ev.step,
            steps_left: ev.steps_left(),
            trigger: ev.trigger,
            pose: ev.pose,
            grippers: ev.grippers,
            holding: self.holding.clone(),
            current_skill: cur.map(|c| c.instruction.skill.clone()),
            current_objects: cur.map(|c| c.instruction.objects.clone()).unwrap_or_default(),
            stage_steps: ev.stage_steps,
            stage_budget: ev.stage_budget,
            stage_attempt: cur.map(|c| c.attempt).unwrap_or(0),
            plan_done: done,
            plan_total: total,
            next_plan_step: self.plan.next_todo().map(|s| (s.skill.clone(), s.objects.clone())),
            objects,
            finished: self.finished,
        }
    }

    /// 결정적 대체. LLM 이 없어도 로봇이 멈추지 않게. 감시 신호를 증거로 쓴다:
    /// - 끝남 증거(이동 단계에서 이동 뒤 멈춤, 집기에서 그리퍼가 무언가에 닫힘, 놓기에서 열림,
    ///   예산 소진이어도 이동 단계면 0.3 m 넘게 움직임) → 다음 단계
    /// - 증거 없음 + 확인 경계 → 계속(한도까지)
    /// - 증거 없음 + 예산 소진 → 같은 단계 재시도(한도까지), 그다음은 실패로 두고 다음 단계
    /// - 남은 단계가 없으면 과제 문장
    pub fn fallback(&mut self, ev: &BoundaryEvent, why: &str) -> Decision {
        let src = format!("fallback: {why}");
        let mut previous = "none".to_string();
        if let Some(cur) = self.mem.current().cloned() {
            let skill = cur.instruction.skill.as_str();
            let nav = skill == "move to";
            let grasp = matches!(skill, "pick up from" | "hold" | "lift");
            let place = matches!(skill, "place on" | "place in" | "place on next to" | "place in next to" | "place under" | "release" | "hang" | "insert" | "attach");
            let holds_something = ev.grippers.iter().any(|&w| w > 0.005 && w < 0.07);
            let verdict = self.evidence(ev).0;
            let done = verdict == "done"
                || match ev.trigger {
                    Trigger::Settled => nav,
                    Trigger::GripperChange => (grasp && holds_something) || place || (!grasp && !nav),
                    Trigger::BudgetExhausted => nav && ev.moved_in_stage > 0.3,
                    _ => false,
                };
            if !done {
                if matches!(ev.trigger, Trigger::CheckDue | Trigger::Settled | Trigger::GripperChange) && self.continues < self.cfg.max_continues {
                    return Decision::Continue { extra: 0, check_every: 0, reason: "keep executing the current step".into(), source: src };
                }
                let fails = self.mem.consecutive_failures(&cur.instruction);
                if ev.trigger == Trigger::BudgetExhausted && fails < self.cfg.max_retries && cur.attempt <= self.cfg.max_retries {
                    let ins = cur.instruction.clone();
                    let prompt = ins.render(self.cfg.format, &self.task.prompt, self.cfg.max_prompt_tokens);
                    let budget = ins.budget_steps.min(ev.steps_left().max(1));
                    return Decision::Issue { instruction: ins, prompt, budget, check_every: 0, previous: "failed".into(), flush: self.cfg.flush_on_change, source: src };
                }
            }
            previous = if done { "done" } else { "failed" }.to_string();
            if done {
                self.belief_after_done(&cur.instruction);
            }
        }
        let doing = self.plan.doing().map(|s| s.id);
        let next = self.plan.steps.iter().find(|s| matches!(s.status, Status::Todo | Status::Failed) && Some(s.id) != doing).cloned();
        if let Some(step) = next {
            if let Ok(ins) = self.make_instruction(&step.skill, &step.objects, step.memory.as_deref(), &step.spatial, "", "", None) {
                let prompt = ins.render(self.cfg.format, &self.task.prompt, self.cfg.max_prompt_tokens);
                let budget = ins.budget_steps;
                return Decision::Issue { instruction: ins, prompt, budget, check_every: 0, previous, flush: self.cfg.flush_on_change, source: src };
            }
        }
        Decision::Finish { prompt: self.task.prompt.clone(), reason: "no remaining plan step; fall back to the whole-task instruction".into(), source: src }
    }

    /// 끝난 단계가 손에 든 것을 어떻게 바꾸나(여러 번 불러도 같은 결과).
    pub fn belief_after_done(&mut self, ins: &Instruction) {
        let Some(o) = ins.objects.first().cloned() else { return };
        match ins.skill.as_str() {
            "pick up from" | "hold" | "lift" => {
                if !self.holding.contains(&o) {
                    self.holding.push(o);
                }
            }
            "place on" | "place in" | "place on next to" | "place in next to" | "place under" | "release" | "hang" | "insert" | "attach" => self.holding.retain(|h| *h != o),
            _ => {}
        }
    }

    /// 결정을 상태에 반영.
    pub fn commit(&mut self, ev: &BoundaryEvent, d: &Decision) {
        match d {
            Decision::Issue { instruction, prompt, previous, .. } => {
                self.continues = 0;
                self.finished = false;
                let outcome = match previous.as_str() {
                    "done" => Outcome::Done,
                    "failed" => Outcome::Failed,
                    "partial" => Outcome::Partial,
                    _ => Outcome::Replaced,
                };
                if let Some(cur) = self.mem.current().cloned() {
                    self.mem.close_current(outcome, ev.step, "");
                    if let Some(id) = self.plan.doing().map(|s| s.id) {
                        self.plan.set_status(
                            id,
                            match outcome {
                                Outcome::Done => Status::Done,
                                Outcome::Failed | Outcome::Partial => Status::Failed,
                                _ => Status::Todo,
                            },
                        );
                    }
                    if outcome == Outcome::Done {
                        let ins = &cur.instruction;
                        if let Some(o) = ins.objects.first() {
                            // "다룸" 은 조작 단계만(이동은 아님): the other 가 방금 다가간 물체를 빼지 않게.
                            // 그래프에도 알린다(scenemap mark_handled).
                            if ins.skill != "move to" {
                                self.objects.mark_handled(o, &ins.skill);
                                if self.objects.known.contains_key(o) {
                                    if let Err(e) = self.graph.mark_handled(o) {
                                        self.trace("graph_error", json!({"op": "mark_handled", "id": o, "error": e}));
                                    }
                                }
                            }
                        }
                        self.belief_after_done(&cur.instruction);
                    }
                    self.trace("stage_close", json!({"stage": cur.idx, "outcome": outcome, "step": ev.step}));
                }
                self.mem.open(instruction.clone(), prompt.clone(), ev.step);
                let first_name = instruction.names.first().cloned().unwrap_or_default();
                self.plan.start_matching(&instruction.skill, &first_name);
            }
            Decision::Continue { .. } => {
                self.continues += 1;
            }
            Decision::Finish { .. } => {
                self.continues = 0;
                self.finished = true;
                if self.mem.current().is_some() {
                    self.mem.close_current(Outcome::Done, ev.step, "finish");
                    if let Some(id) = self.plan.doing().map(|s| s.id) {
                        self.plan.set_status(id, Status::Done);
                    }
                }
            }
        }
    }

    /// 단계 구조체 만들기(참조 풀기·예산·숫자 명령). issue_command·대체·다른 결정기가 같이 쓴다.
    #[allow(clippy::too_many_arguments)]
    pub fn make_instruction(
        &mut self,
        skill: &str,
        objects: &[String],
        memory: Option<&str>,
        spatial: &[String],
        purpose: &str,
        expected: &str,
        budget: Option<u64>,
    ) -> Result<Instruction, String> {
        let k = crate::vocab::lookup(skill).ok_or_else(|| format!("unknown skill '{skill}'. Use one of: {}", crate::vocab::names().join(", ")))?;
        if objects.is_empty() {
            return Err(format!("skill '{}' needs objects in this order: {}", k.name, k.slots.join(", ")));
        }
        let pose = self.ev.as_ref().map(|e| e.pose).unwrap_or_default();
        let step = self.ev.as_ref().map(|e| e.step).unwrap_or(0);
        let memory = memory.map(|m| m.trim().to_lowercase()).filter(|m| !m.is_empty() && m != "none");
        let mut ids: Vec<String> = Vec::new();
        let mut names = Vec::new();
        let mut centers = Vec::new();
        let current_obj = self.mem.current().and_then(|c| c.instruction.objects.first().cloned());
        for (i, raw) in objects.iter().enumerate() {
            let raw = raw.trim();
            let raw_clean = raw.trim_start_matches("the other ").trim_start_matches("the ").to_string();
            let r = if i == 0 && memory.as_deref() == Some("the other") {
                // 이미 구체 id 를 줬고 아직 안 다룬 것이면 그대로. 아니면 같은 종류 중 안 다룬 것(지금 다가간 것 우선).
                match self.objects.known.get(&raw_clean).filter(|k| !k.is_handled()) {
                    Some(k) => Some(crate::graph::Resolved { id: k.id.clone(), name: self.objects.name_of(&k.id).unwrap_or_default(), center: Some(k.last_center), how: "given id".into() }),
                    None => {
                        let excl: Vec<String> = self.objects.known.values().filter(|k| k.is_handled()).map(|k| k.id.clone()).collect();
                        let cat = self.objects.name_of(&raw_clean).unwrap_or_else(|| raw_clean.clone());
                        self.objects.resolve_other(&cat, &excl, &pose, current_obj.as_deref()).or_else(|| self.objects.resolve(&raw_clean, &pose))
                    }
                }
            } else if i == 1 && memory.as_deref() == Some("back") && (raw.is_empty() || raw == "back" || raw.contains("original")) {
                let first = ids.first().cloned().unwrap_or_default();
                let bddl_sup = self.bddl_support_of(&first);
                self.objects.resolve_back(&first, bddl_sup.as_deref(), &pose)
            } else {
                let placing = i == 0
                    && matches!(k.name, "place on" | "place in" | "place on next to" | "place in next to" | "place under" | "release" | "hang" | "insert" | "attach" | "pour" | "hand over");
                // 받침·통 칸(i ≥ 1)은 "안 다룬 것 먼저" 를 쓰지 않는다(들고 다닌 통도 받침이 될 수 있다)
                let r0 = if i == 0 { self.objects.resolve_ctx(&raw_clean, &pose, &self.holding, placing) } else { self.objects.resolve(&raw_clean, &pose) };
                r0.or_else(|| {
                    let hit = self.graph.query(&raw_clean, 1).ok().and_then(|v| v.into_iter().next()).filter(|n| n.score >= 0.5);
                    hit.map(|n| {
                        self.objects.observe(std::slice::from_ref(&n), step);
                        crate::graph::Resolved { id: n.id.clone(), name: crate::util::display_name(&n.label), center: Some(n.center), how: "graph query".into() }
                    })
                })
            };
            match r {
                Some(r) => {
                    names.push(if r.name.is_empty() { crate::util::display_name(&raw_clean) } else { r.name });
                    ids.push(r.id);
                    centers.push(r.center);
                }
                None => {
                    // 아직 못 본 물체·방 이름: 이름 그대로(탐색용 move to 등)
                    ids.push(raw_clean.clone());
                    names.push(crate::util::display_name(&raw_clean));
                    centers.push(None);
                }
            }
        }
        let default_budget = self.budget_for(k.name);
        let steps_left = self.ev.as_ref().map(|e| e.steps_left()).unwrap_or(u64::MAX).max(1);
        let budget = budget.map(|b| b.clamp(60, default_budget * 3)).unwrap_or(default_budget).min(steps_left);
        let metric = if k.kind == crate::vocab::Kind::Navigation { centers.first().copied().flatten().map(|c| nav_metric(&pose, c)) } else { None };
        Ok(Instruction {
            skill: k.name.to_string(),
            objects: ids,
            names,
            spatial: spatial.to_vec(),
            memory,
            purpose: purpose.to_string(),
            expected: expected.to_string(),
            metric,
            budget_steps: budget,
        })
    }

    fn bddl_support_of(&self, obj_id: &str) -> Option<String> {
        let name = self.objects.name_of(obj_id).unwrap_or_else(|| crate::util::display_name(obj_id));
        self.task
            .problem
            .placements
            .iter()
            .find(|(o, _, _)| {
                let n = crate::util::display_name(o);
                crate::graph::text_score(&name, &n) >= 0.5 || crate::graph::text_score(&n, &name) >= 0.5
            })
            .map(|(_, _, s)| crate::util::display_name(s))
    }
}

/// 이동 단계 숫자 명령: 목표 앞 0.7 m 에 서도록, 로봇 기준 이동량과 회전(도).
pub fn nav_metric(pose: &Pose, c: [f64; 3]) -> crate::instruction::Metric {
    let (f, l) = pose.to_robot(c[0], c[1]);
    let d = f.hypot(l);
    let bearing = l.atan2(f);
    let go = (d - 0.7).max(0.0);
    crate::instruction::Metric { forward_m: go * bearing.cos(), left_m: go * bearing.sin(), turn_deg: bearing.to_degrees() }
}

/// 에이전트 = 상태 + 결정기.
pub struct Agent {
    pub core: Core,
    pub decider: Box<dyn Decider>,
}

impl Agent {
    pub fn new(core: Core, decider: Box<dyn Decider>) -> Agent {
        let mut a = Agent { core, decider };
        a.core.trace("agent", json!({"decider": a.decider.name(), "cfg": a.core.cfg}));
        a.core.reset_episode(0);
        a
    }

    pub fn set_tracer(&mut self, t: Option<Tracer>) {
        self.core.tracer = t;
        self.core.trace("agent", json!({"decider": self.decider.name(), "cfg": self.core.cfg, "task": self.core.task.name}));
    }

    pub fn reset_episode(&mut self, episode: u32) {
        self.core.reset_episode(episode);
    }

    pub fn decide(&mut self, ev: &BoundaryEvent) -> Decision {
        let t0 = Instant::now();
        self.core.begin(ev);
        let d = self.decider.decide(&mut self.core, ev);
        self.core.commit(ev, &d);
        self.core.ev = None;
        let ms = t0.elapsed().as_millis() as u64;
        self.core.stats.decision_ms_max = self.core.stats.decision_ms_max.max(ms);
        self.core.trace("decision", json!({"decision": d, "decision_ms": ms, "step": ev.step}));
        d
    }

    pub fn maintain(&mut self) {
        self.decider.maintain(&mut self.core);
    }

    pub fn release(&mut self) {
        self.decider.release();
    }
}

// ---------------- LLM 결정기 ----------------

pub struct LlmDecider {
    pub llm: Box<dyn Llm>,
}

impl LlmDecider {
    fn event_message(core: &Core, ev: &BoundaryEvent) -> Msg {
        let text = context::event_text(&core.ctx(ev));
        if core.cfg.send_images && ev.images.iter().any(|(c, _)| c == "head") {
            let mut parts = vec![Part::Text { text }];
            for (cam, jpg) in ev.images.iter().filter(|(c, _)| c == "head") {
                parts.push(Part::Text { text: format!("[{cam} camera, now]") });
                parts.push(Part::ImageUrl { image_url: crate::llm::ImageUrl { url: crate::image::data_url_jpeg(jpg) } });
            }
            Msg::user_parts(parts)
        } else {
            Msg::user(text)
        }
    }
}

impl Decider for LlmDecider {
    fn name(&self) -> String {
        format!("llm({})", self.llm.describe())
    }

    fn decide(&mut self, core: &mut Core, ev: &BoundaryEvent) -> Decision {
        let t0 = Instant::now();
        let tools = crate::tools::definitions(core.cfg.send_images && ev.images.len() > 1);
        let mut turn: Vec<Msg> = vec![Self::event_message(core, ev)];
        let mut decision: Option<Decision> = None;
        let mut nudges = 0;
        let mut errors = 0;
        let mut fail_reason = String::new();
        for _ in 0..core.cfg.max_iterations {
            if t0.elapsed().as_secs_f64() > core.cfg.decision_budget_s {
                fail_reason = "decision time budget exceeded".into();
                break;
            }
            // 맥락 예산을 넘으면 단계적으로 줄인다
            let mut req = None;
            for shrink in 0..=3u8 {
                let mut ctx = core.ctx(ev);
                ctx.shrink = shrink;
                let mut messages = vec![Msg::system(context::system(&ctx))];
                let recent = core.mem.recent_turns();
                if shrink >= 3 {
                    // 오래된 turn 은 빼고 마지막 turn 하나만
                    let start = recent.iter().rposition(|m| m.role == "user" && m.tool_call_id.is_none()).unwrap_or(0);
                    messages.extend(recent[start..].iter().cloned());
                } else {
                    messages.extend(recent);
                }
                messages.extend(turn.iter().cloned());
                let r = ChatRequest {
                    messages,
                    tools: Some(tools.clone()),
                    sampling: core.cfg.sampling.clone(),
                    max_tokens: Some(core.cfg.max_out_tokens),
                    thinking: core.cfg.thinking,
                    purpose: "plan".into(),
                };
                let fits = r.approx_tokens() <= core.cfg.ctx_budget_tokens;
                req = Some(r);
                if fits {
                    break;
                }
            }
            let req = req.unwrap();
            let fp = req.fingerprint();
            core.stats.llm_calls += 1;
            let res = self.llm.chat(&req);
            let r = match res {
                Err(e) => {
                    core.stats.llm_errors += 1;
                    errors += 1;
                    core.trace("llm_error", json!({"purpose": "plan", "error": e, "fingerprint": fp}));
                    fail_reason = format!("LLM error: {e}");
                    if errors >= 2 {
                        break;
                    }
                    continue;
                }
                Ok(r) => r,
            };
            core.stats.llm_ms += r.latency_ms;
            let mut req_json = serde_json::to_value(&req.messages).unwrap_or(Value::Null);
            crate::trace::strip_data_urls(&mut req_json);
            core.trace("llm", json!({"purpose": "plan", "fingerprint": fp, "request": req_json, "result": r}));
            let msg = r.msg;
            turn.push(msg.clone());
            if msg.calls().is_empty() {
                nudges += 1;
                if nudges > 2 {
                    fail_reason = "model answered without tool calls".into();
                    break;
                }
                turn.push(Msg::user("Answer with tool calls only. End with exactly one of issue_command, continue_current or finish."));
                continue;
            }
            for call in msg.calls() {
                if decision.is_some() {
                    turn.push(Msg::tool(&call.id, &json!({"status": "skipped", "message": "a decision was already made in this turn"})));
                    continue;
                }
                core.stats.tool_calls += 1;
                let args: Result<Value, String> = if call.function.arguments.trim().is_empty() {
                    Ok(json!({}))
                } else {
                    serde_json::from_str(&call.function.arguments).map_err(|e| format!("arguments are not valid JSON: {e}"))
                };
                let out = match args {
                    Ok(a) if a.is_object() => core.run_tool(&call.function.name, &a),
                    Ok(_) => crate::tools::ToolOut::obs(tool_err("arguments must be a JSON object")),
                    Err(e) => crate::tools::ToolOut::obs(tool_err(e)),
                };
                if out.result.get("status").and_then(|s| s.as_str()) == Some("error") {
                    core.stats.tool_errors += 1;
                }
                core.trace("tool", json!({"name": call.function.name, "args": call.function.arguments, "result": out.result}));
                turn.push(Msg::tool(&call.id, &out.result));
                if out.decision.is_some() {
                    decision = out.decision;
                }
            }
            turn.append(&mut core.extra_msgs);
            if decision.is_some() {
                break;
            }
        }
        core.mem.push_turn(turn);
        match decision {
            Some(d) => d,
            None => {
                core.stats.fallbacks += 1;
                let why = if fail_reason.is_empty() { "no decision within the iteration limit".to_string() } else { fail_reason };
                core.fallback(ev, &why)
            }
        }
    }

    fn maintain(&mut self, core: &mut Core) {
        let Some(range) = core.mem.pending_summary() else { return };
        let t0 = Instant::now();
        let req = ChatRequest {
            messages: core.mem.summary_request(&core.task.prompt, range.clone()),
            tools: None,
            sampling: core.cfg.sampling.clone(),
            max_tokens: Some(400),
            thinking: false,
            purpose: "summary".into(),
        };
        let fp = req.fingerprint();
        core.stats.llm_calls += 1;
        match self.llm.chat(&req) {
            Ok(r) if !r.msg.text().trim().is_empty() => {
                core.stats.summaries += 1;
                let text = r.msg.text();
                core.trace("llm", json!({"purpose": "summary", "fingerprint": fp, "result": r}));
                core.mem.apply_summary(&text, range);
            }
            other => {
                if let Err(e) = &other {
                    core.trace("llm_error", json!({"purpose": "summary", "error": e, "fingerprint": fp}));
                }
                core.mem.fallback_summary(range);
            }
        }
        core.trace("summary", json!({"summary": core.mem.summary, "latency_ms": t0.elapsed().as_millis() as u64}));
    }

    fn release(&mut self) {
        self.llm.release();
    }
}

/// LLM 없이 참고 순서를 그대로 따르는 결정기(기준선, 강화학습 결정기의 본보기).
/// 확인 경계에서는 계속, 이동이 멈추거나 그리퍼가 바뀌거나 예산이 끝나면 다음 단계로.
pub struct PriorDecider;

impl Decider for PriorDecider {
    fn name(&self) -> String {
        "prior".into()
    }
    fn decide(&mut self, core: &mut Core, ev: &BoundaryEvent) -> Decision {
        match ev.trigger {
            Trigger::CheckDue => Decision::Continue { extra: 0, check_every: 0, reason: "periodic check".into(), source: "prior".into() },
            _ => {
                let mut d = {
                    // 현재 단계를 "끝남" 으로 보고 다음 단계
                    let doing = core.plan.doing().map(|s| s.id);
                    let next = core.plan.steps.iter().find(|s| matches!(s.status, Status::Todo | Status::Failed) && Some(s.id) != doing).cloned();
                    match next {
                        Some(step) => match core.make_instruction(&step.skill, &step.objects, step.memory.as_deref(), &step.spatial, "", "", None) {
                            Ok(ins) => {
                                let prompt = ins.render(core.cfg.format, &core.task.prompt, core.cfg.max_prompt_tokens);
                                Decision::Issue { budget: ins.budget_steps, instruction: ins, prompt, check_every: 0, previous: "done".into(), flush: core.cfg.flush_on_change, source: "prior".into() }
                            }
                            Err(e) => core.fallback(ev, &e),
                        },
                        None => Decision::Finish { prompt: core.task.prompt.clone(), reason: "reference order finished".into(), source: "prior".into() },
                    }
                };
                if let Decision::Issue { previous, .. } = &mut d {
                    if core.mem.current().is_none() {
                        *previous = "none".into();
                    }
                }
                d
            }
        }
    }
}

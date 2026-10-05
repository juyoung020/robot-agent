//! 맥락 조립. llama.cpp 의 Qwen 템플릿도 맨 앞 system 메시지 하나만 제대로 다루므로(수업 week03 build_context 와 같은 이유)
//! 역할 설명·과제 카드·계획·기억·물체 표를 **system 하나로 합친다**. 그 뒤에 최근 turn, 그리고 이번 사건 user 메시지.

use crate::catalog::TaskCard;
use crate::graph::ObjectMemory;
use crate::memory::Memory;
use crate::plan::Plan;
use crate::planner::{BoundaryEvent, PlannerCfg};
use crate::util::display_name;

pub const SYSTEM_PROMPT: &str = include_str!("../prompts/system.md");

pub struct Ctx<'a> {
    pub cfg: &'a PlannerCfg,
    pub task: &'a TaskCard,
    pub mem: &'a Memory,
    pub plan: &'a Plan,
    pub objects: &'a ObjectMemory,
    pub holding: &'a [String],
    pub ev: &'a BoundaryEvent,
    pub finished: bool,
    /// 맥락 줄이기 단계: 1 참고 순서 1위만, 2 물체 표 15줄
    pub shrink: u8,
    /// 지금 단계에 대한 자동 증거(판정, 근거)
    pub evidence: (&'static str, String),
}

fn relevance(c: &Ctx, label: &str) -> f64 {
    let mut best: f64 = 0.0;
    for (o, _) in &c.task.problem.objects {
        best = best.max(crate::graph::text_score(&display_name(o), label));
    }
    if let Some(p) = c.task.priors.first() {
        for s in &p.steps {
            for o in &s.objects {
                best = best.max(crate::graph::text_score(o, label));
            }
        }
    }
    best
}

pub fn objects_table(c: &Ctx, max: usize) -> String {
    let pose = &c.ev.pose;
    let mut rows: Vec<(f64, f64, String)> = c
        .objects
        .known
        .values()
        .map(|k| {
            let (f, l) = pose.to_robot(k.last_center[0], k.last_center[1]);
            let d = f.hypot(l);
            let name = c.objects.name_of(&k.id).unwrap_or_default();
            let moved = k.moved();
            let mut extra = Vec::new();
            if moved > 0.2 {
                extra.push(format!("moved {moved:.1} m from first seen"));
            }
            if k.handled > 0 || k.graph_handled {
                extra.push(format!("handled {}x ({})", k.handled, k.last_skill.clone().unwrap_or_default()));
            }
            if c.holding.contains(&k.id) {
                extra.push("in hand".into());
            }
            if let Some(r) = &k.room {
                extra.push(format!("room {r}"));
            }
            let row = format!(
                "{} | {} | {:+.1} fwd, {:+.1} left, dist {:.1} m, height {:.1} | {}",
                k.id,
                name,
                f,
                l,
                d,
                k.last_center[2],
                if extra.is_empty() { "-".into() } else { extra.join("; ") }
            );
            (-relevance(c, &name), d, row)
        })
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
    if rows.is_empty() {
        return "(nothing seen yet — explore)".into();
    }
    let n = rows.len();
    let mut out: Vec<String> = rows.into_iter().take(max).map(|r| r.2).collect();
    if n > max {
        out.push(format!("(+{} more; use graph_query)", n - max));
    }
    out.join("\n")
}

fn priors_text(t: &TaskCard, only_first: bool) -> String {
    if t.priors.is_empty() {
        return "(no demonstrations for this task)".into();
    }
    let mut out = Vec::new();
    for (i, p) in t.priors.iter().enumerate().take(if only_first { 1 } else { usize::MAX }) {
        let label = if p.kind == "medoid" { "most typical demonstration".to_string() } else { format!("order #{}", i + 1) };
        let limit = if i == 0 { 60 } else { 30 };
        if i > 0 && p.steps.len() > limit {
            out.push(format!("{label} ({} of {} demos, {} steps — omitted, too long)", p.count, t.n_episodes, p.steps.len()));
            continue;
        }
        out.push(format!("{label} ({} of {} demos = {:.0}%):", p.count, t.n_episodes, p.freq * 100.0));
        for (j, s) in p.steps.iter().take(limit).enumerate() {
            out.push(format!("  {}. {}", j + 1, s.text()));
        }
        if p.steps.len() > limit {
            out.push(format!("  … ({} more)", p.steps.len() - limit));
        }
    }
    out.join("\n")
}

pub fn system(c: &Ctx) -> String {
    let t = c.task;
    let mut s = SYSTEM_PROMPT.replace("{SKILLS}", &crate::vocab::names().join(", "));
    s.push_str(&format!("\n# Task\n{}\nTime limit: {} simulator steps (30 per second). Demonstrations use about {} steps per episode.\n", t.prompt, t.max_steps, t.human_steps as u64));
    s.push_str("\n# Goal conditions (BDDL; the episode succeeds when all hold)\n");
    for g in &t.problem.goals {
        s.push_str(&format!("- {g}\n"));
    }
    if !t.problem.placements.is_empty() || !t.problem.rooms.is_empty() {
        s.push_str("\n# Initial placements (BDDL :init)\n");
        for (o, r, sup) in t.problem.placements.iter().filter(|(o, _, _)| !o.starts_with("agent")).take(30) {
            s.push_str(&format!("- {o} {r} {sup}\n"));
        }
        for (o, room) in t.problem.rooms.iter().take(20) {
            s.push_str(&format!("- {o} in room {room}\n"));
        }
    }
    s.push_str(&format!("\n# Reference step orders from human demonstrations\n{}\n", priors_text(t, c.shrink >= 1)));
    s.push_str(&format!("\n# Your plan checklist (status | skill | objects | memory)\n{}\n", c.plan.render(25)));
    if !c.mem.summary.is_empty() {
        s.push_str(&format!("\n# Memory summary of earlier boundaries\n{}\n", c.mem.summary));
    }
    if !c.mem.facts.is_empty() {
        s.push_str("\n# Facts you stored\n");
        for f in &c.mem.facts {
            s.push_str(&format!("- {f}\n"));
        }
    }
    s.push_str(&format!("\n# Stage log (instructions sent to the low-level policy)\n{}\n", c.mem.stage_log()));
    s.push_str(&format!("\n# Known objects (scene graph; id | name | position relative to the robot | notes)\n{}\n", objects_table(c, if c.shrink >= 2 { 15 } else { 25 })));
    s.push_str(&format!(
        "\n# Instruction format\nThe low-level policy currently receives instructions in the '{}' format and at most {} tokens. \
You only choose skill, objects, memory, spatial, purpose and expected; the text is generated for you.\n",
        c.cfg.format.as_str(),
        c.cfg.max_prompt_tokens
    ));
    s
}

pub fn event_text(c: &Ctx) -> String {
    let ev = c.ev;
    let mut s = format!(
        "Boundary at step {} of {} ({} steps left). Trigger: {}.\n",
        ev.step,
        ev.max_steps,
        ev.steps_left(),
        ev.trigger.describe()
    );
    match c.mem.current() {
        Some(cur) => s.push_str(&format!(
            "Current step #{} (attempt {}): \"{}\" — skill '{}' on [{}], {} of {} budget steps used.\n",
            cur.idx,
            cur.attempt,
            cur.prompt,
            cur.instruction.skill,
            cur.instruction.objects.join(", "),
            ev.stage_steps,
            ev.stage_budget
        )),
        None if c.finished => s.push_str("You called finish earlier, but the episode has not ended: some goal condition is still unmet.\n"),
        None => s.push_str("No step has been issued yet in this episode.\n"),
    }
    s.push_str(&format!(
        "Robot (odometry from the start): x {:.2} m, y {:.2} m, heading {:.0} deg; moved {:.2} m during this step; base speed {:.2} m/s{}.\n",
        ev.pose.x,
        ev.pose.y,
        ev.pose.yaw.to_degrees(),
        ev.moved_in_stage,
        ev.base_speed,
        if ev.still_steps >= 30 { format!(", base not moving for the last {} steps", ev.still_steps) } else { String::new() }
    ));
    // 그리퍼는 지금 보이는 증거로(믿음과 따로): 닫혀 있고 벌어짐이 남아 있으면 무언가를 쥔 것
    let g = |w: f64| {
        if w < 0.005 {
            "closed on nothing (grasp missed or released)"
        } else if w < 0.07 {
            "closed on an object (likely holding it)"
        } else if w > 0.09 {
            "open"
        } else {
            "partly closed"
        }
    };
    s.push_str(&format!(
        "Grippers now: left {:.3} m = {}; right {:.3} m = {}.\n",
        ev.grippers[0],
        g(ev.grippers[0]),
        ev.grippers[1],
        g(ev.grippers[1]),
    ));
    s.push_str(&format!(
        "Objects in hand according to steps you already confirmed: {}.\n",
        if c.holding.is_empty() { "none".to_string() } else { c.holding.join(", ") }
    ));
    if let Some(cur) = c.mem.current() {
        for o in &cur.instruction.objects {
            if let Some(k) = c.objects.known.get(o) {
                let (f, l) = ev.pose.to_robot(k.last_center[0], k.last_center[1]);
                s.push_str(&format!(
                    "Target {} ({}) is {:+.2} m fwd, {:+.2} m left, height {:.2} m; moved {:.2} m since first seen.\n",
                    o,
                    c.objects.name_of(o).unwrap_or_default(),
                    f,
                    l,
                    k.last_center[2],
                    k.moved()
                ));
            }
        }
    }
    if c.mem.current().is_some() {
        let (v, why) = &c.evidence;
        let label = match *v {
            "done" => "LOOKS DONE",
            "failed" => "LOOKS FAILED",
            "running" => "STILL IN PROGRESS",
            _ => "UNCLEAR",
        };
        s.push_str(&format!("Automatic check of the current step (from proprioception and the scene graph): {label} — {why}.\n"));
    }
    if c.cfg.send_images && ev.images.iter().any(|(c, _)| c == "head") {
        s.push_str("The head camera image is attached. ");
    }
    s.push_str("Judge the current step, then end with issue_command, continue_current or finish.");
    s
}

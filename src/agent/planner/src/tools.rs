//! LLM 이 부르는 도구: 정의(OpenAI `tools` JSON 스키마)와 실행.
//!
//! 끝내는 도구(결정): `issue_command`, `continue_current`, `finish`.
//! 보조 도구: `graph_query`, `resolve_reference`, `look`, `robot_state`, `goal_status`, `set_plan`, `remember`.
//! 오류는 `{"status": "error", "message": …}` 관찰값으로 돌려준다(수업 week02: 오류는 크래시가 아니라 모델이 보고 고칠 관찰값).

use crate::llm::{tool_def, Msg, Part};
use crate::planner::{tool_err, Core, Decision};
use serde_json::{json, Value};

pub struct ToolOut {
    pub result: Value,
    pub decision: Option<Decision>,
}

impl ToolOut {
    pub fn obs(v: Value) -> ToolOut {
        ToolOut { result: v, decision: None }
    }
}

/// 도구 정의. `images` 가 거짓이면 `look` 을 뺀다(없는 영상을 찾느라 호출을 낭비하지 않게).
pub fn definitions(images: bool) -> Vec<Value> {
    let mut v = all_definitions();
    if !images {
        v.retain(|t| t["function"]["name"] != "look");
    }
    v
}

fn all_definitions() -> Vec<Value> {
    let skills: Vec<&str> = crate::vocab::names();
    vec![
        tool_def(
            "issue_command",
            "End this boundary by sending ONE step to the low-level policy (next step, or a retry). Also report how the previous step went.",
            json!({"type": "object", "properties": {
                "previous": {"type": "string", "enum": ["done", "failed", "partial", "none"], "description": "Outcome of the step that was running (none if nothing was running)."},
                "note": {"type": "string", "description": "What you saw that supports the judgement (short)."},
                "skill": {"type": "string", "enum": skills, "description": "Step name from the demonstration vocabulary."},
                "objects": {"type": "array", "items": {"type": "string"}, "description": "Objects in the skill's slot order, scene-graph ids when known."},
                "memory": {"type": "string", "enum": ["none", "back", "the other", "the same"], "description": "back = original place of objects[0]; the other = a not-yet-handled object of the same kind."},
                "spatial": {"type": "array", "items": {"type": "string"}, "description": "Optional spatial modifier per object slot, e.g. [\"\", \"to_the_edge_of\"], [\"right_door\"], [\"left\"]."},
                "purpose": {"type": "string", "description": "Why this step (one short English clause, no numbers)."},
                "expected": {"type": "string", "description": "What the robot should physically do (one short English clause, no numeric distances or angles)."},
                "budget_steps": {"type": "integer", "description": "Optional step budget (30 steps = 1 s). Default comes from demonstration statistics."}
            }, "required": ["previous", "skill", "objects", "purpose", "expected"]}),
        ),
        tool_def(
            "continue_current",
            "End this boundary and keep the current step running (it is still making progress).",
            json!({"type": "object", "properties": {
                "reason": {"type": "string"},
                "extra_steps": {"type": "integer", "description": "Extra steps to add to the budget (0 = keep budget)."}
            }, "required": ["reason"]}),
        ),
        tool_def(
            "finish",
            "End this boundary: all goal conditions look satisfied. The policy then gets the whole-task instruction.",
            json!({"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}),
        ),
        tool_def(
            "graph_query",
            "Find objects in the scene graph by name. Returns id, name, position relative to the robot (m), distance, room, how far it moved since first seen, times handled.",
            json!({"type": "object", "properties": {"text": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["text"]}),
        ),
        tool_def(
            "resolve_reference",
            "Turn 'back' (original support/place of an object) or 'the other' (same kind, not handled yet) into a concrete object.",
            json!({"type": "object", "properties": {
                "object": {"type": "string", "description": "Object id or name. For 'the other', the kind (e.g. 'can of soda')."},
                "modifier": {"type": "string", "enum": ["back", "the other"]}
            }, "required": ["object", "modifier"]}),
        ),
        tool_def(
            "look",
            "Attach the latest image of a wrist camera (the head image is already attached to the event).",
            json!({"type": "object", "properties": {"camera": {"type": "string", "enum": ["left_wrist", "right_wrist"]}}, "required": ["camera"]}),
        ),
        tool_def("robot_state", "Pose, grippers, what is believed to be in hand, steps used and left.", json!({"type": "object", "properties": {}})),
        tool_def("goal_status", "BDDL goal conditions and plan progress.", json!({"type": "object", "properties": {}})),
        tool_def(
            "set_plan",
            "Replace the remaining steps of your checklist (replanning). Finished steps are kept.",
            json!({"type": "object", "properties": {
                "steps": {"type": "array", "items": {"type": "object", "properties": {
                    "skill": {"type": "string", "enum": crate::vocab::names()},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "memory": {"type": "string"}
                }, "required": ["skill", "objects"]}},
                "reason": {"type": "string"}
            }, "required": ["steps", "reason"]}),
        ),
        tool_def(
            "remember",
            "Store a long-lived fact (kept even when old history is summarized), e.g. where an object came from.",
            json!({"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]}),
        ),
    ]
}

fn s(a: &Value, k: &str) -> String {
    a.get(k).and_then(|v| v.as_str()).unwrap_or("").trim().to_string()
}

fn str_list(a: &Value, k: &str) -> Vec<String> {
    match a.get(k) {
        Some(Value::Array(v)) => v.iter().map(|x| x.as_str().map(|s| s.to_string()).unwrap_or_else(|| x.to_string())).collect(),
        Some(Value::String(x)) if !x.is_empty() => x.split(',').map(|p| p.trim().to_string()).collect(),
        _ => vec![],
    }
}

impl Core {
    fn node_view(&self, id: &str) -> Value {
        let pose = self.ev.as_ref().map(|e| e.pose).unwrap_or_default();
        match self.objects.known.get(id) {
            Some(k) => {
                let (f, l) = pose.to_robot(k.last_center[0], k.last_center[1]);
                json!({"id": k.id, "name": self.objects.name_of(&k.id), "forward_m": round2(f), "left_m": round2(l),
                       "distance_m": round2(f.hypot(l)), "height_m": round2(k.last_center[2]), "room": k.room,
                       "moved_since_first_seen_m": round2(k.moved()), "handled": k.handled, "in_hand": self.holding.contains(&k.id)})
            }
            None => json!({"id": id}),
        }
    }

    pub fn run_tool(&mut self, name: &str, a: &Value) -> ToolOut {
        match name {
            "graph_query" => {
                let text = s(a, "text");
                if text.is_empty() {
                    return ToolOut::obs(tool_err("text is required"));
                }
                let k = a.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5).clamp(1, 10) as usize;
                match self.graph.query(&text, k) {
                    Ok(nodes) => {
                        let step = self.ev.as_ref().map(|e| e.step).unwrap_or(0);
                        self.objects.observe(&nodes, step);
                        let v: Vec<Value> = nodes
                            .iter()
                            .map(|n| {
                                let mut x = self.node_view(&n.id);
                                x["score"] = json!(round2(n.score));
                                x
                            })
                            .collect();
                        if v.is_empty() {
                            ToolOut::obs(json!({"status": "ok", "matches": [], "hint": "not seen yet — explore rooms or furniture where it is likely"}))
                        } else {
                            ToolOut::obs(json!({"status": "ok", "matches": v}))
                        }
                    }
                    Err(e) => ToolOut::obs(tool_err(format!("scene graph unavailable: {e}"))),
                }
            }
            "resolve_reference" => {
                let obj = s(a, "object");
                let pose = self.ev.as_ref().map(|e| e.pose).unwrap_or_default();
                let r = match s(a, "modifier").as_str() {
                    "back" => {
                        let id = self.objects.resolve(&obj, &pose).map(|r| r.id).unwrap_or(obj.clone());
                        let name = self.objects.name_of(&id).unwrap_or_else(|| crate::util::display_name(&id));
                        let bddl = self.task.problem.placements.iter().find(|(o, _, _)| crate::graph::text_score(&name, &crate::util::display_name(o)) >= 0.5).map(|(_, _, s)| crate::util::display_name(s));
                        self.objects.resolve_back(&id, bddl.as_deref(), &pose)
                    }
                    "the other" => {
                        let excl: Vec<String> = self.objects.known.values().filter(|k| k.is_handled()).map(|k| k.id.clone()).collect();
                        let cur = self.mem.current().and_then(|c| c.instruction.objects.first().cloned());
                        let cat = self.objects.name_of(&obj).unwrap_or_else(|| obj.clone());
                        self.objects.resolve_other(&cat, &excl, &pose, cur.as_deref())
                    }
                    m => return ToolOut::obs(tool_err(format!("modifier must be 'back' or 'the other', got '{m}'"))),
                };
                match r {
                    Some(r) => ToolOut::obs(json!({"status": "ok", "id": r.id, "name": r.name, "how": r.how, "object": self.node_view(&r.id)})),
                    None => ToolOut::obs(tool_err(format!("could not resolve '{obj}' — use graph_query or explore"))),
                }
            }
            "look" => {
                let cam = s(a, "camera");
                let img = self.ev.as_ref().and_then(|e| e.images.iter().find(|(c, _)| *c == cam).map(|(_, j)| j.clone()));
                match img {
                    Some(j) if self.cfg.send_images => {
                        self.extra_msgs.push(Msg::user_parts(vec![
                            Part::Text { text: format!("[{cam} camera, now]") },
                            Part::ImageUrl { image_url: crate::llm::ImageUrl { url: crate::image::data_url_jpeg(&j) } },
                        ]));
                        ToolOut::obs(json!({"status": "ok", "message": format!("{cam} image attached below")}))
                    }
                    _ => ToolOut::obs(tool_err(format!("no {cam} image at this boundary"))),
                }
            }
            "robot_state" => {
                let Some(ev) = self.ev.clone() else { return ToolOut::obs(tool_err("no boundary")) };
                let cur = self.mem.current();
                ToolOut::obs(json!({"status": "ok", "x_m": round2(ev.pose.x), "y_m": round2(ev.pose.y), "heading_deg": round2(ev.pose.yaw.to_degrees()),
                    "grippers_m": [round3(ev.grippers[0]), round3(ev.grippers[1])], "in_hand": self.holding,
                    "step": ev.step, "steps_left": ev.steps_left(), "current_step": cur.map(|c| c.prompt.clone()),
                    "current_step_steps_used": ev.stage_steps, "current_step_budget": ev.stage_budget}))
            }
            "goal_status" => {
                let (done, total) = self.plan.progress();
                ToolOut::obs(json!({"status": "ok", "goals": self.task.problem.goals, "plan_done": done, "plan_total": total,
                    "note": "goal truth is not observable directly; judge from the scene graph and images"}))
            }
            "set_plan" => {
                let Some(steps) = a.get("steps").and_then(|v| v.as_array()) else { return ToolOut::obs(tool_err("steps must be an array")) };
                let mut new = Vec::new();
                for (i, st) in steps.iter().enumerate() {
                    let skill = s(st, "skill");
                    let Some(k) = crate::vocab::lookup(&skill) else {
                        return ToolOut::obs(tool_err(format!("step {}: unknown skill '{skill}'", i + 1)));
                    };
                    let objs = str_list(st, "objects");
                    if objs.is_empty() {
                        return ToolOut::obs(tool_err(format!("step {}: objects required ({})", i + 1, k.slots.join(", "))));
                    }
                    let mem = Some(s(st, "memory")).filter(|m| !m.is_empty() && m != "none");
                    new.push((k.name.to_string(), objs, mem, vec![]));
                }
                if new.is_empty() {
                    return ToolOut::obs(tool_err("empty plan"));
                }
                self.plan.replace_remaining(new);
                self.trace("plan", json!({"reason": s(a, "reason"), "plan": self.plan.render(40)}));
                ToolOut::obs(json!({"status": "ok", "plan": self.plan.render(40)}))
            }
            "remember" => {
                let f = s(a, "fact");
                if f.is_empty() {
                    return ToolOut::obs(tool_err("fact is empty"));
                }
                if !self.mem.facts.contains(&f) {
                    self.mem.facts.push(f.chars().take(200).collect());
                    if self.mem.facts.len() > 20 {
                        self.mem.facts.remove(0);
                    }
                }
                ToolOut::obs(json!({"status": "ok", "facts": self.mem.facts.len()}))
            }
            "issue_command" => self.tool_issue(a),
            "continue_current" => {
                let Some(ev) = self.ev.clone() else { return ToolOut::obs(tool_err("no boundary")) };
                if self.mem.current().is_none() {
                    return ToolOut::obs(tool_err("nothing is running; use issue_command"));
                }
                if self.continues >= self.cfg.max_continues {
                    return ToolOut::obs(tool_err(format!(
                        "the current step was already continued {} times; decide with issue_command (retry or next) or finish",
                        self.continues
                    )));
                }
                if !self.pushback {
                    let (v, why) = self.evidence(&ev);
                    if v == "done" || v == "failed" {
                        self.pushback = true;
                        return ToolOut::obs(tool_err(format!(
                            "The automatic check says the current step already looks {v} ({why}); continuing it wastes steps. \
Call continue_current again if you disagree, otherwise use issue_command."
                        )));
                    }
                }
                let extra = a.get("extra_steps").and_then(|v| v.as_u64()).unwrap_or(0).min(ev.stage_budget.max(300)).min(ev.steps_left());
                ToolOut { result: json!({"status": "ok", "extra_steps": extra}), decision: Some(Decision::Continue { extra, check_every: 0, reason: s(a, "reason"), source: "llm".into() }) }
            }
            "finish" => ToolOut {
                result: json!({"status": "ok"}),
                decision: Some(Decision::Finish { prompt: self.task.prompt.clone(), reason: s(a, "reason"), source: "llm".into() }),
            },
            other => ToolOut::obs(tool_err(format!("unknown tool '{other}'"))),
        }
    }

    fn tool_issue(&mut self, a: &Value) -> ToolOut {
        let Some(ev) = self.ev.clone() else { return ToolOut::obs(tool_err("no boundary")) };
        let previous = match s(a, "previous").as_str() {
            "" => "none".to_string(),
            p @ ("done" | "failed" | "partial" | "none") => p.to_string(),
            p => return ToolOut::obs(tool_err(format!("previous must be done, failed, partial or none (got '{p}')"))),
        };
        let objects = str_list(a, "objects");
        let memory = Some(s(a, "memory")).filter(|m| !m.is_empty());
        let spatial = str_list(a, "spatial");
        let budget = a.get("budget_steps").and_then(|v| v.as_u64());
        // 이전 단계가 끝났다고 판단했으면, 다음 단계의 참조를 풀기 전에 손에 든 것부터 갱신한다
        // (예: 방금 집은 캔 → "place can in trash can" 의 캔은 손에 든 그것).
        if previous == "done" {
            if let Some(cur) = self.mem.current().map(|c| c.instruction.clone()) {
                self.belief_after_done(&cur);
            }
        }
        let ins = match self.make_instruction(&s(a, "skill"), &objects, memory.as_deref(), &spatial, &s(a, "purpose"), &s(a, "expected"), budget) {
            Ok(i) => i,
            Err(e) => return ToolOut::obs(tool_err(e)),
        };
        // 자동 증거는 끝났다는데 같은 단계를 다시 보내려 하면 한 번 되묻는다(작은 모델의 과한 재시도 방지)
        let same = self.mem.current().map(|c| c.instruction.same_step(&ins)).unwrap_or(false);
        if same && previous != "done" && !self.pushback {
            let (v, why) = self.evidence(&ev);
            if v == "done" {
                self.pushback = true;
                return ToolOut::obs(tool_err(format!(
                    "The automatic check says the current step already looks done ({why}). Re-issuing the same step usually wastes time. \
If you still think it failed, call issue_command again with the same arguments; otherwise set previous='done' and send the next step."
                )));
            }
        }
        // 재시도 한도: 같은 단계가 연달아 실패했으면 다른 방법을 요구
        if let Some(cur) = self.mem.current() {
            if cur.instruction.same_step(&ins) && previous != "done" {
                let fails = self.mem.consecutive_failures(&ins) + 1;
                if fails > self.cfg.max_retries {
                    return ToolOut::obs(tool_err(format!(
                        "'{}' already failed {fails} times in a row. Change the approach: re-approach with 'move to', choose another object, or call set_plan.",
                        cur.prompt
                    )));
                }
            }
        }
        let prompt = ins.render(self.cfg.format, &self.task.prompt, self.cfg.max_prompt_tokens);
        let budget = ins.budget_steps;
        let note = s(a, "note");
        if !note.is_empty() {
            self.mem.close_note(&note);
        }
        let flush = self.cfg.flush_on_change && self.mem.current().map(|c| c.prompt != prompt).unwrap_or(true);
        let view = json!({"status": "ok", "sent": prompt, "budget_steps": budget, "objects": ins.objects, "names": ins.names});
        let _ = ev;
        ToolOut { result: view, decision: Some(Decision::Issue { instruction: ins, prompt, budget, check_every: 0, previous, flush, source: "llm".into() }) }
    }
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

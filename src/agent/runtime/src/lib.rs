//! 에이전트 런타임 — 스킬과 무관한 원형 도구 호출 루프(프레임워크 없음, 수업 규칙):
//! messages + tools → tool_calls 면 실행해 role:"tool" 로 붙이고 다시 → 글이면 끝.
//!
//! - 스킬 = 지시문(데이터)뿐: `src/agent/skills/<skill>/` 의 `system.md`·`task.md`·도구 설명(`tool.md`)·`skill.json`.
//!   런타임은 공통 `src/agent/prompts/common.md` 와 그것들을 실행할 때 읽고([`Prompts`], 판·FNV-64 지문을 기록마다 남김),
//!   도구는 이름으로 고른다([`tools::by_name`]). 끝 조건·되묻기·지표 칸은 `skill.json`([`SkillSpec`]).
//! - 정책 둘: `Llm`(Qwen3.5-9B, KAU) / `Baseline`(LLM 없음: `skill.json` 의 `baseline` 호출을 되풀이 — explore 는 move_robot
//!   모드 `explore` 의 프런티어 탐사). 같은 도구·같은 관측.
//! - 기록: decisions.jsonl(실제로 한 도구 실행 하나에 한 줄, `src/agent/decision_log.md`), timeline.jsonl(실행마다 `_m` 측정값),
//!   trace.jsonl(LLM 요청·응답 요약), summary.json.
//! - 맥락 16k: 결과는 LLM 에게 보일 것만 남겨 짧게(도구의 `view`), 최근 `keep_pairs` 쌍만 그대로 두고 오래된 것은 한 줄 요약으로 접는다.

pub mod prompts;
pub mod skill;
pub mod tools;

pub use prompts::Prompts;
pub use skill::SkillSpec;

use bagent::llm::{sanitize, ChatRequest, Llm, Msg, Sampling};
use move_robot::link::Backend;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use tools::Tool;

fn metric(r: &Value, k: &str) -> Option<f64> {
    r.get("_m").and_then(|m| m.get(k)).and_then(|v| v.as_f64())
}

// ---------------------------------------------------------------- 결정 기록

/// decisions.jsonl 한 줄(필드 뜻은 src/agent/decision_log.md)
pub struct DecisionLog {
    f: std::fs::File,
}

impl DecisionLog {
    pub fn create(p: &std::path::Path) -> std::io::Result<DecisionLog> {
        Ok(DecisionLog { f: std::fs::OpenOptions::new().create(true).append(true).open(p)? })
    }
    pub fn write(&mut self, v: &Value) {
        let _ = writeln!(self.f, "{}", v);
    }
}

// ---------------------------------------------------------------- 실행

#[derive(Clone, Debug, PartialEq)]
pub enum Policy {
    Llm { thinking: bool },
    /// skill.json `baseline` 호출(LLM 없음)
    Baseline,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub policy: Policy,
    pub limits: skill::Limits,
    pub out_dir: PathBuf,
    pub task: String,
    pub run_id: String,
    /// LLM 이 끝 조건 전에 끝내면 되묻기(skill.json `end.nudge`)
    pub nudge: bool,
}

impl Config {
    pub fn new(spec: &SkillSpec) -> Config {
        Config { policy: Policy::Baseline, limits: spec.limits.clone(), out_dir: PathBuf::from("out"), task: spec.name.clone(), run_id: "run".into(), nudge: true }
    }
}

struct Writer {
    timeline: std::fs::File,
    trace: std::fs::File,
    dec: DecisionLog,
}

fn now_iso() -> String {
    std::process::Command::new("date").arg("+%Y-%m-%dT%H:%M:%S").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

fn one_decimal(t0: &Instant) -> f64 {
    (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0
}

/// 스킬 한 판. 끝나면 요약(summary.json 과 같은 것).
pub fn run(spec: &SkillSpec, cfg: &Config, backend: &mut dyn Backend, mut llm: Option<&mut dyn Llm>, prompts: &Prompts) -> Value {
    std::fs::create_dir_all(&cfg.out_dir).ok();
    let open = |n: &str| std::fs::OpenOptions::new().create(true).append(true).open(cfg.out_dir.join(n)).unwrap();
    let mut w = Writer { timeline: open("timeline.jsonl"), trace: open("trace.jsonl"), dec: DecisionLog::create(&cfg.out_dir.join("decisions.jsonl")).unwrap() };
    let t0 = Instant::now();
    let lim = &cfg.limits;
    let pol: String = match cfg.policy {
        Policy::Llm { .. } => "llm".into(),
        Policy::Baseline => spec.baseline.as_ref().map_or("baseline".into(), |b| b.name.clone()),
    };
    // 도구: 이름으로 고르고, 스킬 폴더의 설명·mode 목록으로 정의를 만든다
    let mut tools: Vec<Box<dyn Tool>> = vec![];
    let mut defs: Vec<Value> = vec![];
    for (i, t) in spec.tools.iter().enumerate() {
        let Some(tool) = tools::by_name(&t.name) else {
            return json!({"end": format!("unknown tool in skill.json: {} (known: {:?})", t.name, tools::KNOWN)});
        };
        defs.push(tool.definition(&prompts.tool_descs[i], t.modes.as_deref()));
        tools.push(tool);
    }
    let names: Vec<String> = tools.iter().map(|t| t.name().to_string()).collect();
    let find = |n: &str| names.iter().position(|x| x == n);
    // 첫 관측
    let mut last = match &spec.observe {
        Some(c) => {
            let i = find(&c.tool).unwrap_or(0);
            tools[i].exec(&c.args, backend, &Value::Null)
        }
        None => Value::Null,
    };
    let tl = |w: &mut Writer, k: usize, r: &Value, llm_calls: usize, tok: (u64, u64), t0: &Instant| {
        let mut row = json!({"k": k, "wall_s": one_decimal(t0), "llm_calls": llm_calls, "prompt_tokens": tok.0, "completion_tokens": tok.1});
        for (out, key) in &spec.timeline {
            row[out] = json!(metric(r, key));
        }
        for (out, key) in &spec.timeline_raw {
            row[out] = r["_m"][key].clone();
        }
        let _ = writeln!(w.timeline, "{row}");
    };
    tl(&mut w, 0, &last, 0, (0, 0), &t0);
    let first_obs = tools[0].view(&last);
    let msgs: Vec<Msg> = vec![Msg::system(prompts.system.clone()), Msg::user(prompts.task.replace("{observation}", &first_obs.to_string()))];
    let base_user = prompts.task.clone();
    let mut older: Vec<String> = vec![];
    let mut pairs: Vec<(Msg, Msg)> = vec![];
    let (mut llm_calls, mut ptok, mut ctok, mut llm_ms) = (0usize, 0u64, 0u64, 0u64);
    let mut calls = 0usize;
    let mut end_reason = String::from("max_calls");
    let mut nudged = 0usize;
    let mut cov_marks: Vec<(f64, Value)> = vec![];
    let mut mode_count: std::collections::BTreeMap<String, usize> = Default::default();
    let mut final_text = String::new();
    'outer: while calls < lim.max_calls {
        if metric(&last, "sim_s").unwrap_or(0.0) > lim.max_sim_s {
            end_reason = "max_sim_s".into();
            break;
        }
        if t0.elapsed().as_secs_f64() > lim.max_wall_s {
            end_reason = "max_wall_s".into();
            break;
        }
        let pre = tools[0].view(&last);
        let end_met = spec.end.as_ref().map_or(false, |e| skill::end_met(e, &pre));
        // ---- 결정: (도구 이름, 인자, LLM 정보)
        let (tname, args, llm_info): (String, Value, Value) = match (&cfg.policy, llm.as_deref_mut()) {
            (Policy::Baseline, _) => {
                let Some(b) = &spec.baseline else {
                    end_reason = "no baseline in skill.json".into();
                    break;
                };
                if end_met {
                    end_reason = spec.end.as_ref().unwrap().name.clone();
                    break;
                }
                (b.call.tool.clone(), b.call.args.clone(), Value::Null)
            }
            (Policy::Llm { thinking }, Some(l)) => {
                // 맥락 조립: system, user(과제 + 지난 요약), 최근 쌍
                let mut m = msgs.clone();
                if !older.is_empty() {
                    let hist = older.iter().rev().take(12).rev().cloned().collect::<Vec<_>>().join("\n");
                    let txt = format!("{}\nEarlier moves ({} total, oldest dropped):\n{}", base_user.replace("{observation}", &first_obs.to_string()), older.len(), hist);
                    m[1] = Msg::user(txt);
                }
                for (a, t) in &pairs {
                    m.push(a.clone());
                    m.push(t.clone());
                }
                let req = ChatRequest {
                    messages: m,
                    tools: Some(defs.clone()),
                    sampling: Sampling { temperature: lim.temperature, ..Sampling::default() },
                    max_tokens: Some(lim.max_tokens),
                    thinking: *thinking,
                    purpose: spec.name.clone(),
                };
                let approx = req.approx_tokens();
                // 429(요청 한도)·5xx·연결 오류는 기다렸다 다시(최대 6 번, 10·20·…초)
                let mut tries = 0;
                let res = loop {
                    match l.chat(&req) {
                        Err(e) if tries < 6 && (e.contains("429") || e.contains("HTTP 5") || e.contains("curl") || e.contains("timed out")) => {
                            tries += 1;
                            let _ = writeln!(w.trace, "{}", json!({"k": calls, "llm_retry": tries, "error": e.chars().take(120).collect::<String>()}));
                            std::thread::sleep(std::time::Duration::from_secs(10 * tries));
                        }
                        r => break r,
                    }
                };
                let res = match res {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = writeln!(w.trace, "{}", json!({"k": calls, "llm_error": e}));
                        end_reason = format!("llm_error: {e}");
                        break;
                    }
                };
                llm_calls += 1;
                llm_ms += res.latency_ms;
                let pt = res.usage["prompt_tokens"].as_u64().unwrap_or(approx as u64);
                let ct = res.usage["completion_tokens"].as_u64().unwrap_or(0);
                ptok += pt;
                ctok += ct;
                let msg = sanitize(res.msg);
                let cs = msg.calls().to_vec();
                let _ = writeln!(w.trace, "{}", json!({"k": calls, "latency_ms": res.latency_ms, "prompt_tokens": pt, "completion_tokens": ct,
                    "approx_tokens": approx, "text": msg.text(), "calls": cs.iter().map(|c| json!({"name": c.function.name, "args": c.function.arguments})).collect::<Vec<_>>(),
                    "finish": res.finish_reason, "reasoning_chars": res.reasoning_chars}));
                if cs.is_empty() {
                    final_text = msg.text();
                    if let Some(e) = &spec.end {
                        if !end_met && cfg.nudge && nudged < e.max_nudges {
                            if let Some(txt) = &e.nudge {
                                nudged += 1;
                                let n = skill::path(&pre, &e.when_empty).as_array().map_or(0, |a| a.len());
                                // 되묻기: (assistant 글, user 한 줄) 쌍
                                let a = Msg { role: "assistant".into(), content: Some(bagent::llm::Content::Text(if final_text.is_empty() { "done".into() } else { final_text.clone() })), tool_calls: None, tool_call_id: None };
                                pairs.push((a, Msg::user(txt.replace("{n}", &n.to_string()))));
                                continue 'outer;
                            }
                        }
                        end_reason = if end_met { format!("llm_done_{}", e.name) } else { "llm_done_early".into() };
                    } else {
                        end_reason = "llm_done".into();
                    }
                    break;
                }
                let c = cs[0].clone();
                let a: Value = serde_json::from_str(&c.function.arguments).unwrap_or(Value::String(c.function.arguments.clone()));
                (c.function.name.clone(), a, json!({"latency_ms": res.latency_ms, "prompt_tokens": pt, "completion_tokens": ct, "call_id": c.id, "name": c.function.name, "n_calls": cs.len()}))
            }
            (Policy::Llm { .. }, None) => {
                end_reason = "no llm".into();
                break;
            }
        };
        // ---- 실행
        let ti = find(&tname);
        let tool = &mut tools[ti.unwrap_or(0)];
        let tw = Instant::now();
        let res = match ti {
            Some(_) => tool.exec(&args, backend, &last),
            None => {
                let msg = if names.len() == 1 { format!("unknown tool '{}'; the only tool is {}", tname, names[0]) } else { format!("unknown tool '{}'; tools: {}", tname, names.join(", ")) };
                tool.error(&msg)
            }
        };
        let exec_ms = tw.elapsed().as_millis() as u64;
        // 실제로 한 실행들: 묶음 모드(move_robot explore)는 `_calls`, 아니면 이 호출 하나
        let mut execs: Vec<(Value, Value)> = match res.get("_calls").and_then(|c| c.as_array()) {
            Some(cs) => cs.iter().map(|c| (c["args"].clone(), c["result"].clone())).collect(),
            None => vec![],
        };
        if execs.is_empty() {
            if cfg.policy == Policy::Baseline && res.get("_calls").is_some() {
                // 기준선 호출이 할 일이 없다고 함(예: 닿을 수 있는 프런티어 없음)
                end_reason = res["end"].as_str().unwrap_or("baseline_done").to_string();
                break;
            }
            // 보통 호출, 또는 LLM 이 부른 묶음 모드가 할 일 없이 돌아옴: 이 호출 하나로 기록
            let mut r = res.clone();
            if let Some(o) = r.as_object_mut() {
                o.remove("_calls");
            }
            execs.push((args.clone(), r));
        }
        for (a, r) in &execs {
            if tool.link_down(r) {
                end_reason = format!("sim_link_down: {}", r["message"].as_str().unwrap_or(""));
                break 'outer;
            }
            calls += 1;
            let pre_i = tool.view(&last);
            let mode = tool.mode_key(a);
            *mode_count.entry(mode.clone()).or_default() += 1;
            let mut outcome = tool.outcome(r);
            if let Some(cv) = &spec.coverage {
                outcome.insert(format!("{}_before", cv.key), json!(metric(&last, &cv.key)));
                outcome.insert(format!("{}_after", cv.key), json!(metric(r, &cv.key)));
            }
            for k in &spec.decision {
                outcome.insert(k.clone(), json!(metric(r, k)));
            }
            outcome.insert("exec_wall_ms".into(), json!(exec_ms));
            let row = json!({
                "run": cfg.run_id, "task": cfg.task, "skill": spec.name, "policy": pol, "step": calls, "t": now_iso(),
                "sim_s": metric(r, "sim_s"), "prompt": {"version": prompts.version, "sha": prompts.sha},
                "obs": tool.obs_features(&pre_i),
                "executor": tool.name(), "mode": mode, "args": a,
                "target_space": tool.target_space(a),
                "outcome": Value::Object(outcome),
                "llm": llm_info,
                "label": tool.label(a, &pre_i, r),
            });
            w.dec.write(&row);
            tl(&mut w, calls, r, llm_calls, (ptok, ctok), &t0);
            if let Some(cv) = &spec.coverage {
                if let Some(c) = metric(r, &cv.key) {
                    for &th in &cv.marks {
                        if c >= th && !cov_marks.iter().any(|(t, _)| *t == th) {
                            let mut mk = json!({"cov": th, "calls": calls, "llm_calls": llm_calls, "tokens": ptok + ctok});
                            for f in &cv.fields {
                                mk[f] = json!(metric(r, f));
                            }
                            cov_marks.push((th, mk));
                        }
                    }
                }
            }
            last = tool.next_obs(r, &last);
        }
        // 맥락: 결과 쌍 붙이고 오래된 것 접기(LLM 이 부른 호출 하나 = 쌍 하나)
        let comp = tool.view(&res);
        if let (Policy::Llm { .. }, Some(id)) = (&cfg.policy, llm_info.get("call_id").and_then(|v| v.as_str())) {
            let call = bagent::llm::ToolCall { id: id.to_string(), kind: "function".into(), function: bagent::llm::FnCall { name: tool.name().into(), arguments: args.to_string() } };
            pairs.push((Msg::assistant_calls(vec![call]), Msg::tool(id, &comp)));
            while pairs.len() > lim.keep_pairs {
                pairs.remove(0);
            }
        }
        older.push(tool.fold_line(calls, &args, &res));
    }
    let mut summary = json!({
        "run": cfg.run_id, "task": cfg.task, "policy": pol, "prompt": {"version": prompts.version, "sha": prompts.sha, "files": prompts.files},
        "end": end_reason, "final_text": final_text, "nudges": nudged,
        "calls": calls, "llm_calls": llm_calls, "prompt_tokens": ptok, "completion_tokens": ctok, "llm_ms": llm_ms,
        "wall_s": one_decimal(&t0),
        "modes": mode_count, "coverage_marks": cov_marks.into_iter().map(|c| c.1).collect::<Vec<_>>(),
    });
    for (out, key) in &spec.summary {
        summary[out] = json!(metric(&last, key));
    }
    std::fs::write(cfg.out_dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).ok();
    summary
}

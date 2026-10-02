//! 스킬 `explore`(집 탐사) 실행기. 원형 도구 호출 루프(프레임워크 없음, 수업 규칙):
//! messages + tools → tool_calls 면 실행해 role:"tool" 로 붙이고 다시 → 글이면 끝.
//!
//! - 프롬프트는 실행할 때 파일에서 읽는다: 공통 `src/agent/prompts/common.md` + 스킬 `system.md`·`task.md`·`tool.md`
//!   (`tool.md` = 이 스킬에서 LLM 에게 보이는 move_robot 설명). 판·지문(FNV-64)을 기록마다 남긴다.
//! - 정책 둘: `Llm`(Qwen3.5-9B, KAU) / `Frontier`(LLM 없는 기준선: 경로 가장 짧은 프런티어로 go_to). 같은 도구·같은 관측.
//! - 기록: decisions.jsonl(결정 하나에 한 줄, `decision_log.md`), timeline.jsonl(호출마다 덮은 넓이·경로·시뮬 시각),
//!   trace.jsonl(LLM 요청·응답 요약), summary.json.
//! - 맥락 16k: 결과는 LLM 에게 보일 것만 남겨 짧게([`compact`]), 오래된 호출 쌍은 한 줄 요약으로 접는다.

use bagent::llm::{sanitize, ChatRequest, Llm, Msg, Sampling};
use move_robot::link::{run_tool, Backend};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------- 프롬프트

pub struct Prompts {
    pub system: String,
    pub task: String,
    pub tool_desc: String,
    pub version: String,
    pub sha: String,
    pub files: Vec<String>,
}

fn fnv64(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in b {
        h ^= *x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn strip_header(s: &str) -> (String, Option<String>) {
    let t = s.trim_start();
    if let Some(rest) = t.strip_prefix("<!--") {
        if let Some(end) = rest.find("-->") {
            let head = &rest[..end];
            let ver = head.split("version:").nth(1).map(|v| v.trim().to_string());
            return (rest[end + 3..].trim().to_string(), ver);
        }
    }
    (t.trim().to_string(), None)
}

impl Prompts {
    /// skill_dir = src/agent/skills/explore, common = src/agent/prompts/common.md
    pub fn load(skill_dir: &Path, common: &Path) -> Result<Prompts, String> {
        let rd = |p: &Path| std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()));
        let names = [common.to_path_buf(), skill_dir.join("system.md"), skill_dir.join("task.md"), skill_dir.join("tool.md")];
        let raw: Vec<String> = names.iter().map(|p| rd(p)).collect::<Result<_, _>>()?;
        let mut all = vec![];
        for r in &raw {
            all.extend_from_slice(r.as_bytes());
        }
        let parts: Vec<(String, Option<String>)> = raw.iter().map(|r| strip_header(r)).collect();
        let version = parts.iter().filter_map(|p| p.1.clone()).collect::<Vec<_>>().join("+");
        Ok(Prompts {
            system: format!("{}\n\n{}", parts[0].0, parts[1].0),
            task: parts[2].0.clone(),
            tool_desc: parts[3].0.replace('\n', " "),
            version,
            sha: format!("{:016x}", fnv64(&all)),
            files: names.iter().map(|p| p.display().to_string()).collect(),
        })
    }
}

// ---------------------------------------------------------------- 관측 줄이기

/// 결과 → LLM 에게 보일 것만. 측정값(`_m`)·단위·내부 상태는 뺀다.
pub fn compact(r: &Value) -> Value {
    let mut o = serde_json::Map::new();
    for k in ["status", "mode", "target", "moved_m", "turned_deg", "stopped_by", "clear_m", "hint", "message", "part"] {
        if let Some(v) = r.get(k) {
            if k == "part" && v == "base" {
                continue;
            }
            o.insert(k.into(), v.clone());
        }
    }
    if r.get("mode").is_none() && r.get("part") != Some(&json!("base")) {
        if let Some(v) = r.get("state") {
            o.insert("state".into(), v.clone());
        }
    }
    if let Some(m) = r.get("map") {
        o.insert("map".into(), m.clone());
    }
    Value::Object(o)
}

fn metric(r: &Value, k: &str) -> Option<f64> {
    r.get("_m").and_then(|m| m.get(k)).and_then(|v| v.as_f64())
}

// ---------------------------------------------------------------- 결정 기록

/// decisions.jsonl 한 줄(필드 뜻은 src/agent/decision_log.md)
pub struct DecisionLog {
    f: std::fs::File,
}

impl DecisionLog {
    pub fn create(p: &Path) -> std::io::Result<DecisionLog> {
        Ok(DecisionLog { f: std::fs::OpenOptions::new().create(true).append(true).open(p)? })
    }
    pub fn write(&mut self, v: &Value) {
        let _ = writeln!(self.f, "{}", v);
    }
}

/// 결정 직전 관측에서 뽑는 특징
fn obs_features(obs: &Value) -> Value {
    let map = &obs["map"];
    let fr = map["frontiers"].as_array().cloned().unwrap_or_default();
    let nearest = fr.iter().min_by(|a, b| a["path_m"].as_f64().unwrap_or(1e9).total_cmp(&b["path_m"].as_f64().unwrap_or(1e9)));
    json!({
        "free_m2": map["free_m2"], "n_frontiers": fr.len(),
        "best": fr.first().map(|f| json!({"id": f["id"], "path_m": f["path_m"], "new_area_m2": f["new_area_m2"]})),
        "nearest": nearest.map(|f| json!({"id": f["id"], "path_m": f["path_m"]})),
        "in_room": map.get("in_room").cloned().unwrap_or(Value::Null),
        "front": map["around"]["F"],
        "last_status": obs["status"],
    })
}

/// 사후 표지: 고른 것이 나빴나(계산할 수 있을 때만)
fn label(args: &Value, pre: &Value, res: &Value) -> Value {
    let mode = args["mode"].as_str().unwrap_or("");
    let status = res["status"].as_str().unwrap_or("");
    let gain = res["map"]["new_free_m2"].as_f64().unwrap_or(0.0);
    let mut l = vec![];
    if status == "error" {
        l.push("invalid_call");
        if res["message"].as_str().map_or(false, |m| m.contains("UNKNOWN")) {
            l.push("goto_into_unknown");
        }
    }
    let fr = pre["map"]["frontiers"].as_array().cloned().unwrap_or_default();
    let mut rank = Value::Null;
    if mode == "go_to" {
        if let Some(t) = args["target"].as_str() {
            if let Some(k) = fr.iter().position(|f| f["id"].as_str() == Some(&t.to_uppercase())) {
                rank = json!(k + 1);
            }
        }
    }
    if mode == "probe" {
        let fwd = args["values"].get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let moved = res["moved_m"].as_f64().unwrap_or(0.0);
        if fwd > 0.0 && status == "blocked" && moved < 0.3 {
            l.push("wasted_probe");
        }
        if fwd == 0.0 && gain < 0.5 {
            l.push("no_gain_look");
        }
    }
    if mode == "delta" && args["part"] == "base" {
        l.push("delta_for_explore");
    }
    if status == "reached" && gain < 0.2 && mode == "go_to" {
        l.push("no_gain_move");
    }
    json!({"tags": l, "rank": rank, "vla_better": Value::Null})
}

// ---------------------------------------------------------------- 실행

#[derive(Clone, Debug, PartialEq)]
pub enum Policy {
    Llm { thinking: bool },
    Frontier,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub policy: Policy,
    pub max_calls: usize,
    pub max_sim_s: f64,
    pub max_wall_s: f64,
    pub out_dir: PathBuf,
    /// 맥락에 그대로 남길 최근 호출 쌍 수
    pub keep_pairs: usize,
    pub task: String,
    pub run_id: String,
    /// LLM 이 프런티어가 남았는데 끝내면 한 번 되묻기
    pub nudge: bool,
    pub max_tokens: u32,
    pub temperature: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            policy: Policy::Frontier,
            max_calls: 80,
            max_sim_s: 900.0,
            max_wall_s: 3600.0,
            out_dir: PathBuf::from("out"),
            keep_pairs: 3,
            task: "explore".into(),
            run_id: "run".into(),
            nudge: true,
            max_tokens: 400,
            temperature: 0.0,
        }
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

/// 탐사 한 판. 끝나면 요약(summary.json 과 같은 것).
pub fn run(cfg: &Config, backend: &mut dyn Backend, mut llm: Option<&mut dyn Llm>, prompts: &Prompts) -> Value {
    std::fs::create_dir_all(&cfg.out_dir).ok();
    let open = |n: &str| std::fs::OpenOptions::new().create(true).append(true).open(cfg.out_dir.join(n)).unwrap();
    let mut w = Writer { timeline: open("timeline.jsonl"), trace: open("trace.jsonl"), dec: DecisionLog::create(&cfg.out_dir.join("decisions.jsonl")).unwrap() };
    let t0 = Instant::now();
    let pol = match cfg.policy {
        Policy::Llm { .. } => "llm",
        Policy::Frontier => "frontier",
    };
    let tool = move_robot::definition_with(&prompts.tool_desc);
    // 첫 관측: 베이스 읽기
    let mut last = run_tool(&json!({"part":"base","mode":"delta","values":[0,0,0]}), backend);
    let tl = |w: &mut Writer, k: usize, r: &Value, llm_calls: usize, tok: (u64, u64), t0: &Instant| {
        let row = json!({"k": k, "sim_s": metric(r, "sim_s"), "wall_s": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
            "gt_cov": metric(r, "gt_cov"), "free_m2": metric(r, "free_m2"), "path_m": metric(r, "path_m"),
            "contacts": metric(r, "contacts"), "stalls": metric(r, "stalls"), "blocked": metric(r, "blocked"),
            "min_clear_m": metric(r, "min_clear_m"), "frontiers": metric(r, "frontiers_all"), "llm_calls": llm_calls,
            "prompt_tokens": tok.0, "completion_tokens": tok.1, "pose": r["_m"]["pose"],
            "obs_us": metric(r, "obs_us"), "plan_us": metric(r, "plan_us"), "costmap_us": metric(r, "costmap_us")});
        let _ = writeln!(w.timeline, "{row}");
    };
    tl(&mut w, 0, &last, 0, (0, 0), &t0);
    let first_obs = compact(&last);
    let msgs: Vec<Msg> = vec![
        Msg::system(prompts.system.clone()),
        Msg::user(prompts.task.replace("{observation}", &first_obs.to_string())),
    ];
    let base_user = prompts.task.clone();
    let mut older: Vec<String> = vec![];
    let mut pairs: Vec<(Msg, Msg)> = vec![];
    let (mut llm_calls, mut ptok, mut ctok, mut llm_ms) = (0usize, 0u64, 0u64, 0u64);
    let mut calls = 0usize;
    let mut end_reason = String::from("max_calls");
    let mut nudged = 0usize;
    let mut failed_targets: Vec<(f64, f64)> = vec![];
    let mut cov_marks: Vec<(f64, Value)> = vec![];
    let thresholds = [0.5, 0.8, 0.9, 0.95];
    let mut mode_count: std::collections::BTreeMap<String, usize> = Default::default();
    let mut final_text = String::new();
    'outer: while calls < cfg.max_calls {
        if metric(&last, "sim_s").unwrap_or(0.0) > cfg.max_sim_s {
            end_reason = "max_sim_s".into();
            break;
        }
        if t0.elapsed().as_secs_f64() > cfg.max_wall_s {
            end_reason = "max_wall_s".into();
            break;
        }
        let pre = compact(&last);
        let no_frontier = pre["map"]["frontiers"].as_array().map_or(true, |a| a.is_empty());
        // ---- 결정
        let (args, llm_info): (Value, Value) = match (&cfg.policy, llm.as_deref_mut()) {
            (Policy::Frontier, _) => {
                if no_frontier {
                    end_reason = "no_frontier".into();
                    break;
                }
                // 경로가 가장 짧은 프런티어(실패한 자리 빼고) — Yamauchi 고전 방식
                let fr = pre["map"]["frontiers"].as_array().unwrap();
                let pick = fr.iter().filter(|f| f["path_m"].as_f64().is_some()).min_by(|a, b| a["path_m"].as_f64().unwrap().total_cmp(&b["path_m"].as_f64().unwrap()));
                match pick {
                    Some(f) => (json!({"part":"base","mode":"go_to","target": f["id"]}), Value::Null),
                    None => {
                        end_reason = "no_frontier".into();
                        break;
                    }
                }
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
                let req = ChatRequest { messages: m, tools: Some(vec![tool.clone()]), sampling: Sampling { temperature: cfg.temperature, ..Sampling::default() }, max_tokens: Some(cfg.max_tokens), thinking: *thinking, purpose: "explore".into() };
                let approx = req.approx_tokens();
                let res = match l.chat(&req) {
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
                    if !no_frontier && cfg.nudge && nudged < 1 {
                        nudged += 1;
                        let n = pre["map"]["frontiers"].as_array().map_or(0, |a| a.len());
                        // 되묻기: (assistant 글, user 한 줄) 쌍
                        let a = Msg { role: "assistant".into(), content: Some(bagent::llm::Content::Text(if final_text.is_empty() { "done".into() } else { final_text.clone() })), tool_calls: None, tool_call_id: None };
                        pairs.push((a, Msg::user(format!("The map still shows {n} reachable frontier(s). Continue exploring with go_to, or reply done again if you are sure."))));
                        continue 'outer;
                    }
                    end_reason = if no_frontier { "llm_done_no_frontier".into() } else { "llm_done_early".into() };
                    break;
                }
                let c = cs[0].clone();
                let a: Value = serde_json::from_str(&c.function.arguments).unwrap_or(Value::String(c.function.arguments.clone()));
                (a, json!({"latency_ms": res.latency_ms, "prompt_tokens": pt, "completion_tokens": ct, "call_id": c.id, "name": c.function.name, "n_calls": cs.len()}))
            }
            (Policy::Llm { .. }, None) => {
                end_reason = "no llm".into();
                break;
            }
        };
        // ---- 실행
        let tw = Instant::now();
        let res = if llm_info.get("name").and_then(|n| n.as_str()).map_or(false, |n| n != move_robot::TOOL_NAME) {
            move_robot::error_obs(&format!("unknown tool '{}'; the only tool is move_robot", llm_info["name"].as_str().unwrap_or("")))
        } else {
            run_tool(&args, backend)
        };
        let exec_ms = tw.elapsed().as_millis() as u64;
        if res["hint"].as_str().map_or(false, |h| h.contains("simulator link is down")) {
            end_reason = format!("sim_link_down: {}", res["message"].as_str().unwrap_or(""));
            break;
        }
        calls += 1;
        let mode = format!("{}:{}", args["part"].as_str().unwrap_or("?"), args["mode"].as_str().unwrap_or("?"));
        *mode_count.entry(mode.clone()).or_default() += 1;
        if pol == "frontier" && res["status"] != "reached" {
            if let Some(p) = res["_m"]["pose"].as_array() {
                failed_targets.push((p[0].as_f64().unwrap_or(0.0), p[1].as_f64().unwrap_or(0.0)));
            }
        }
        let cov_before = metric(&last, "gt_cov");
        let cov_after = metric(&res, "gt_cov");
        let row = json!({
            "run": cfg.run_id, "task": cfg.task, "skill": "explore", "policy": pol, "step": calls, "t": now_iso(),
            "sim_s": metric(&res, "sim_s"), "prompt": {"version": prompts.version, "sha": prompts.sha},
            "obs": obs_features(&pre),
            "executor": "move_robot", "mode": mode, "args": args,
            "target_space": match args["mode"].as_str() { Some("go_to") => "known", Some("probe") => "unknown", _ => "n/a" },
            "outcome": {"status": res["status"], "moved_m": res["moved_m"], "time_s": res["time_s"], "stopped_by": res.get("stopped_by").cloned().unwrap_or(Value::Null),
                "replans": res.get("replans").cloned().unwrap_or(json!(0)), "new_free_m2": res["map"]["new_free_m2"],
                "gt_cov_before": cov_before, "gt_cov_after": cov_after,
                "contacts": metric(&res, "contacts"), "stalls": metric(&res, "stalls"), "min_clear_m": metric(&res, "min_clear_m"), "exec_wall_ms": exec_ms,
                "message": res.get("message").cloned().unwrap_or(Value::Null)},
            "llm": llm_info,
            "label": label(&args, &pre, &res),
        });
        w.dec.write(&row);
        tl(&mut w, calls, &res, llm_calls, (ptok, ctok), &t0);
        if let Some(c) = cov_after {
            for th in thresholds {
                if c >= th && !cov_marks.iter().any(|(t, _)| *t == th) {
                    cov_marks.push((th, json!({"cov": th, "calls": calls, "sim_s": metric(&res, "sim_s"), "path_m": metric(&res, "path_m"), "llm_calls": llm_calls, "tokens": ptok + ctok})));
                }
            }
        }
        // 맥락: 결과 쌍 붙이고 오래된 것 접기
        let comp = compact(&res);
        if let (Policy::Llm { .. }, Some(id)) = (&cfg.policy, llm_info.get("call_id").and_then(|v| v.as_str())) {
            let call = bagent::llm::ToolCall {
                id: id.to_string(),
                kind: "function".into(),
                function: bagent::llm::FnCall { name: move_robot::TOOL_NAME.into(), arguments: args.to_string() },
            };
            pairs.push((Msg::assistant_calls(vec![call]), Msg::tool(id, &comp)));
            while pairs.len() > cfg.keep_pairs {
                pairs.remove(0);
            }
        }
        older.push(format!(
            "{}. {} -> {}{}, +{} m2",
            calls,
            short_args(&args),
            res["status"].as_str().unwrap_or("?"),
            res.get("stopped_by").and_then(|v| v.as_str()).map(|s| format!(" ({s})")).unwrap_or_default(),
            res["map"]["new_free_m2"].as_f64().unwrap_or(0.0)
        ));
        last = if res["map"].is_object() { res } else { merge_map(&res, &last) };
    }
    let ms = metric(&last, "sim_s");
    let summary = json!({
        "run": cfg.run_id, "task": cfg.task, "policy": pol, "prompt": {"version": prompts.version, "sha": prompts.sha, "files": prompts.files},
        "end": end_reason, "final_text": final_text, "nudges": nudged,
        "calls": calls, "llm_calls": llm_calls, "prompt_tokens": ptok, "completion_tokens": ctok, "llm_ms": llm_ms,
        "wall_s": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0, "sim_s": ms,
        "gt_cov": metric(&last, "gt_cov"), "free_m2": metric(&last, "free_m2"), "path_m": metric(&last, "path_m"),
        "contacts": metric(&last, "contacts"), "stalls": metric(&last, "stalls"), "blocked": metric(&last, "blocked"), "replans": metric(&last, "replans"),
        "min_clear_m": metric(&last, "min_clear_m"), "frontiers_left": metric(&last, "frontiers_all"),
        "modes": mode_count, "coverage_marks": cov_marks.into_iter().map(|c| c.1).collect::<Vec<_>>(),
    });
    std::fs::write(cfg.out_dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).ok();
    summary
}

fn short_args(a: &Value) -> String {
    match a["mode"].as_str() {
        Some("go_to") => format!("go_to {}", a.get("target").and_then(|t| t.as_str()).map(|s| s.to_string()).unwrap_or_else(|| a["values"].to_string())),
        Some(m) => format!("{} {} {}", a["part"].as_str().unwrap_or("?"), m, a["values"]),
        None => a.to_string(),
    }
}

/// 지도 없는 결과(오류)는 지난 지도·측정값을 이어 쓴다
fn merge_map(r: &Value, last: &Value) -> Value {
    let mut o = r.clone();
    o["map"] = last["map"].clone();
    o["_m"] = last["_m"].clone();
    o
}

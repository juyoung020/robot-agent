//! 시험용 가짜들: 가짜 LLM(규칙으로 도구 호출; 함수형·HTTP 서버형), 가짜 VLA 서버, 가짜 평가기 클라이언트.
//! 가짜 LLM 은 system 메시지의 계획 체크리스트와 사건 글만 읽고 답한다 → 글 계약(맥락 형식)까지 같이 시험된다.

use crate::llm::{FnCall, Msg, ToolCall};
use crate::msgpack;
use crate::ws::{self, Accepted, Buf, Conn, OP_BIN, OP_CLOSE};
use serde_json::{json, Value};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ---------------- 가짜 LLM(규칙) ----------------

struct PlanLine {
    status: String,
    skill: String,
    objects: Vec<String>,
    memory: Option<String>,
}

fn parse_plan(system: &str) -> Vec<PlanLine> {
    let mut out = Vec::new();
    let mut on = false;
    for l in system.lines() {
        if l.starts_with("# Your plan checklist") {
            on = true;
            continue;
        }
        if on {
            if l.starts_with('#') || l.trim().is_empty() {
                break;
            }
            let l = l.trim();
            if !l.starts_with('[') {
                continue;
            }
            let parts: Vec<&str> = l.split(" | ").collect();
            let head = parts[0];
            let status = head[1..head.find(']').unwrap_or(1)].to_string();
            let after = head[head.find(']').map(|i| i + 1).unwrap_or(0)..].trim();
            let skill = after.split_once(' ').map(|(_, s)| s.trim().to_string()).unwrap_or_default();
            let objects = parts.get(1).map(|o| o.split(", ").map(|x| x.trim().to_string()).collect()).unwrap_or_default();
            let memory = parts.get(2).map(|m| m.trim().to_string()).filter(|m| !m.is_empty() && !m.starts_with("(attempts"));
            out.push(PlanLine { status, skill, objects, memory });
        }
    }
    out
}

fn call(name: &str, args: Value, n: usize) -> ToolCall {
    ToolCall { id: format!("call_{n}"), kind: "function".into(), function: FnCall { name: name.into(), arguments: args.to_string() } }
}

/// 규칙 답: 판 시작이면 graph_query 한 번, 정기 확인이면 계속, 그 밖에는 체크리스트 다음 단계.
pub fn oracle_reply(messages: &[Msg], has_tools: bool) -> Msg {
    if !has_tools {
        // 요약 요청
        let n = messages.last().map(|m| m.text().matches("[Boundary").count()).unwrap_or(0);
        return Msg { role: "assistant".into(), content: Some(crate::llm::Content::Text(format!("Summary of {n} boundaries: steps progressed as planned."))), tool_calls: None, tool_call_id: None };
    }
    let system = messages.first().map(|m| m.text()).unwrap_or_default();
    let ev_idx = messages.iter().rposition(|m| m.role == "user" && m.text().starts_with("Boundary at step")).unwrap_or(0);
    let event = messages[ev_idx].text();
    let tools_after = messages[ev_idx..].iter().filter(|m| m.role == "tool").count();
    let plan = parse_plan(&system);
    let next = plan.iter().find(|p| p.status == "todo" || p.status == "failed");
    let has_current = event.contains("Current step #");
    if event.contains("Trigger: episode start") && tools_after == 0 {
        let q = next.and_then(|p| p.objects.first().cloned()).unwrap_or_else(|| "object".into());
        return Msg::assistant_calls(vec![call("graph_query", json!({"text": q}), 0)]);
    }
    if event.contains("Trigger: periodic progress check") && has_current {
        return Msg::assistant_calls(vec![call("continue_current", json!({"reason": "still progressing"}), 0)]);
    }
    let previous = if has_current { "done" } else { "none" };
    match next {
        Some(p) => {
            let mut args = json!({"previous": previous, "note": "judged from event", "skill": p.skill, "objects": p.objects,
                                  "purpose": format!("progress the task: {}", p.skill), "expected": ""});
            if let Some(m) = &p.memory {
                args["memory"] = json!(m);
            }
            Msg::assistant_calls(vec![call("issue_command", args, 0)])
        }
        None => Msg::assistant_calls(vec![call("finish", json!({"reason": "checklist complete"}), 0)]),
    }
}

/// 규칙 답을 주는 함수형 LLM.
pub fn oracle_llm() -> crate::llm::FnLlm<impl FnMut(&crate::llm::ChatRequest) -> Result<Msg, String> + Send> {
    crate::llm::FnLlm { f: |r: &crate::llm::ChatRequest| Ok(oracle_reply(&r.messages, r.tools.is_some())), name: "oracle".into() }
}

/// 가짜 OpenAI 호환 서버(HTTP). `delay_ms` 로 응답 지연 흉내.
pub fn serve_mock_llm(l: TcpListener, delay_ms: u64) {
    for s in l.incoming() {
        let Ok(mut s) = s else { continue };
        std::thread::spawn(move || {
            let Ok(req) = crate::http::read_request(&mut s) else { return };
            let (code, body) = match (req.method.as_str(), req.path.as_str()) {
                ("GET", p) if p.ends_with("/models") => (200, json!({"object": "list", "data": [{"id": "mock-oracle", "object": "model"}]})),
                ("GET", p) if p.ends_with("/health") => (200, json!({"status": "ok"})),
                ("POST", p) if p.ends_with("/chat/completions") => match serde_json::from_slice::<Value>(&req.body) {
                    Ok(v) => {
                        let msgs: Vec<Msg> = serde_json::from_value(v["messages"].clone()).unwrap_or_default();
                        if delay_ms > 0 {
                            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                        }
                        let m = oracle_reply(&msgs, v.get("tools").is_some());
                        let fr = if m.tool_calls.is_some() { "tool_calls" } else { "stop" };
                        (200, json!({"id": "mock", "object": "chat.completion", "model": "mock-oracle",
                                     "choices": [{"index": 0, "message": m, "finish_reason": fr}],
                                     "usage": {"prompt_tokens": v["messages"].to_string().len() / 4, "completion_tokens": 40}}))
                    }
                    Err(e) => (400, json!({"error": e.to_string()})),
                },
                _ => (404, json!({"error": "not found"})),
            };
            let _ = crate::http::write_response(&mut s, code, body.to_string().as_bytes(), "application/json");
        });
    }
}

// ---------------- 가짜 VLA 서버 ----------------

#[derive(Default, Debug)]
pub struct FakePiLog {
    /// 주입 키를 뺀 관측의 지문(평가기가 보낸 원본과 같아야 함)
    pub obs_hashes: Vec<String>,
    pub prompts: Vec<Value>,
    pub flushes: usize,
    pub resp_hashes: Vec<String>,
    pub resets: usize,
    pub steps: usize,
}

/// 평가기 웹소켓 정책 서버와 같은 순서: 접속하면 metadata, 관측마다 {"action": …, "server_timing": …}, reset 은 응답 없음.
pub fn serve_fake_pi(l: TcpListener, log: Option<Arc<Mutex<FakePiLog>>>, once: bool) {
    for s in l.incoming() {
        let Ok(s) = s else { continue };
        let log = log.clone();
        // /healthz 같은 HTTP 요청은 여기서 답하고 다음 접속으로
        let Ok(Accepted::Ws(mut c)) = Conn::accept(s) else { continue };
        let h = std::thread::spawn(move || {
            let mut meta = Vec::new();
            msgpack::w_json(&mut meta, &json!({"fake_pi": true, "action_dim": 23}));
            if c.send(OP_BIN, &[&meta]).is_err() {
                return;
            }
            let mut buf = Buf::default();
            let mut out = Vec::new();
            let mut step: u64 = 0;
            loop {
                let Ok((op, mask)) = c.read_message(&mut buf) else { return };
                if op == OP_CLOSE {
                    let _ = c.send_close(1000);
                    return;
                }
                if let Some(k) = mask {
                    ws::mask_at(buf.data_mut(), k, 0);
                }
                let data = buf.data();
                let top = match msgpack::scan_top(&msgpack::Plain(data)) {
                    Ok(t) => t,
                    Err(_) => return,
                };
                if top.find(|k| k == "reset").is_some() {
                    if let Some(l) = &log {
                        l.lock().unwrap().resets += 1;
                    }
                    continue;
                }
                let batch = top
                    .find(|k| k.ends_with("::proprio"))
                    .and_then(|e| msgpack::parse_nd(&msgpack::Plain(data), e.val_off).ok().flatten())
                    .map(|nd| if nd.shape.len() == 2 { nd.shape[0] } else { 0 })
                    .unwrap_or(0);
                let vals: Vec<u8> = (0..23 * batch.max(1)).flat_map(|i| ((step as f32) * 0.001 + i as f32 * 0.01).to_le_bytes()).collect();
                out.clear();
                msgpack::w_map(&mut out, 2);
                msgpack::w_str(&mut out, "action");
                if batch > 0 {
                    msgpack::w_ndarray(&mut out, "<f4", &[batch, 23], &vals);
                } else {
                    msgpack::w_ndarray(&mut out, "<f4", &[23], &vals);
                }
                msgpack::w_str(&mut out, "server_timing");
                msgpack::w_map(&mut out, 1);
                msgpack::w_str(&mut out, "infer_ms");
                msgpack::w_f64(&mut out, 0.0);
                if let Some(l) = &log {
                    let (stripped, prompt) = crate::wire::strip_injected(data).unwrap_or_default();
                    let flush = top.find(|k| k == crate::wire::FLUSH_KEY).is_some();
                    let mut g = l.lock().unwrap();
                    g.obs_hashes.push(crate::codec::sha1_hex(&stripped));
                    g.prompts.push(prompt.unwrap_or(Value::Null));
                    g.flushes += usize::from(flush);
                    g.resp_hashes.push(crate::codec::sha1_hex(&out));
                    g.steps += 1;
                }
                if c.send(OP_BIN, &[&out]).is_err() {
                    return;
                }
                step += 1;
            }
        });
        if once {
            let _ = h.join();
            return;
        }
    }
}

// ---------------- 가짜 평가기 ----------------

#[derive(Clone, Debug)]
pub struct ObsSpec {
    pub batch: usize,
    pub rgbd: bool,
    pub robot: String,
    pub task_id: i64,
}

impl Default for ObsSpec {
    fn default() -> Self {
        ObsSpec { batch: 1, rgbd: false, robot: "robot_r1".into(), task_id: 0 }
    }
}

/// 평가기와 같은 모양의 관측(msgpack 평문). base 속도·그리퍼는 스텝에 따라 움직인다(경계가 생기도록).
pub fn make_obs(spec: &ObsSpec, step: u64, noise: &[u8]) -> Vec<u8> {
    let b = spec.batch;
    let r = &spec.robot;
    let (head, wrist) = if spec.rgbd { (720usize, 480usize) } else { (224, 224) };
    let n_items = 3 + 3 + if spec.rgbd { 3 } else { 0 };
    let mut o = Vec::with_capacity(8 << 20);
    msgpack::w_map(&mut o, n_items);
    // proprio
    let mut prop = vec![0f32; b * 61];
    for e in 0..b {
        let s = step as f32;
        prop[e * 61] = if (30..120).contains(&step) { 0.5 } else { 0.0 };
        prop[e * 61 + 2] = if (30..60).contains(&step) { 0.3 } else { 0.0 };
        let g = if step > 200 { 0.015 } else { 0.05 };
        prop[e * 61 + 24] = g;
        prop[e * 61 + 25] = g;
        prop[e * 61 + 49] = 0.05;
        prop[e * 61 + 50] = 0.05;
        prop[e * 61 + 10] = s * 1e-4;
    }
    let pb: Vec<u8> = prop.iter().flat_map(|f| f.to_le_bytes()).collect();
    msgpack::w_str(&mut o, &format!("{r}::proprio"));
    msgpack::w_ndarray(&mut o, "<f4", &[b, 61], &pb);
    for (name, side) in [("zed_link", head), ("left_realsense_link", wrist), ("right_realsense_link", wrist)] {
        let n = b * side * side * 4;
        let mut px = Vec::with_capacity(n);
        while px.len() < n {
            let take = (n - px.len()).min(noise.len().max(1));
            px.extend_from_slice(&noise[..take]);
        }
        px[0] = step as u8;
        msgpack::w_str(&mut o, &format!("{r}::{r}:{name}:Camera:0::rgb"));
        msgpack::w_ndarray(&mut o, "|u1", &[b, side, side, 4], &px);
        if spec.rgbd {
            let d: Vec<u8> = (0..b * side * side).flat_map(|i| (1.0f32 + (i % 97) as f32 * 0.01).to_le_bytes()).collect();
            msgpack::w_str(&mut o, &format!("{r}::{r}:{name}:Camera:0::depth_linear"));
            msgpack::w_ndarray(&mut o, "<f4", &[b, side, side], &d);
        }
    }
    let cp: Vec<u8> = (0..b * 21).flat_map(|i| (i as f32 * 0.1).to_le_bytes()).collect();
    msgpack::w_str(&mut o, &format!("{r}::cam_rel_poses"));
    msgpack::w_ndarray(&mut o, "<f4", &[b, 21], &cp);
    msgpack::w_str(&mut o, "task_id");
    msgpack::w_ndarray(&mut o, "<i8", &[1], &spec.task_id.to_le_bytes());
    o
}

pub fn reset_msg() -> Vec<u8> {
    let mut o = Vec::new();
    msgpack::w_map(&mut o, 1);
    msgpack::w_str(&mut o, "reset");
    msgpack::w_bool(&mut o, true);
    o
}

#[derive(Default, Debug)]
pub struct EvalLog {
    pub rtt_us: Vec<u32>,
    pub obs_hashes: Vec<String>,
    pub resp_hashes: Vec<String>,
    pub bytes_per_obs: usize,
}

/// 가짜 평가기: 접속 → metadata → (reset → 관측 n 개) × episodes. 관측 만들기는 시간 재기 밖.
pub fn run_fake_eval(addr: &str, spec: &ObsSpec, steps: u64, episodes: u32, record: bool) -> std::io::Result<EvalLog> {
    let mut c = Conn::connect(addr)?;
    let mut buf = Buf::default();
    let _meta = c.read_message(&mut buf)?;
    let mut rng = crate::util::Rng::new(1);
    let noise: Vec<u8> = (0..65536).map(|_| rng.next_u64() as u8).collect();
    let mut log = EvalLog::default();
    for _ in 0..episodes {
        c.send(OP_BIN, &[&reset_msg()])?;
        for step in 0..steps {
            let obs = make_obs(spec, step, &noise);
            log.bytes_per_obs = obs.len();
            if record {
                log.obs_hashes.push(crate::codec::sha1_hex(&obs));
            }
            let t0 = Instant::now();
            c.send(OP_BIN, &[&obs])?;
            let (op, _) = c.read_message(&mut buf)?;
            log.rtt_us.push(t0.elapsed().as_micros() as u32);
            if op != OP_BIN {
                return Err(std::io::Error::other(format!("응답 op {op}: {}", String::from_utf8_lossy(buf.data()))));
            }
            if record {
                log.resp_hashes.push(crate::codec::sha1_hex(buf.data()));
            }
        }
    }
    let _ = c.send_close(1000);
    Ok(log)
}

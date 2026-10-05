//! OpenAI 호환 Chat Completions 원형 그대로: `messages`(system/user/assistant/tool), `tools`, `tool_calls`, `role: "tool"`.
//! 대상: KAU API(vLLM, `https://agent.kau.ac.kr/v1`, 수업 저장소와 같은 엔드포인트) 또는 로컬 llama.cpp `llama-server`(CUDA).
//! 수업 예제(week02 agent_tool_calling) 처럼 assistant 메시지는 **표준 필드만** 남긴다
//! (reasoning_content 등 비표준 필드는 다음 요청에 되돌려 보내지 않는다).
//!
//! 전송: `http://` 는 손으로 짠 HTTP([`crate::http`]), `https://` 는 시스템 `curl` 을 부른다(TLS 크레이트를 들이지 않으려고;
//! 경계에서만 부르므로 프로세스 띄우는 수 ms 는 LLM 응답 시간에 묻힌다). 키는 **환경변수로만** 읽고 curl 에는 표준입력
//! 설정(`-K -`)으로 넘겨 명령줄·기록에 남지 않게 한다.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

impl Content {
    pub fn text(&self) -> String {
        match self {
            Content::Text(s) => s.clone(),
            Content::Parts(p) => p
                .iter()
                .map(|x| match x {
                    Part::Text { text } => text.clone(),
                    Part::ImageUrl { .. } => "[image]".into(),
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
    pub fn has_image(&self) -> bool {
        matches!(self, Content::Parts(p) if p.iter().any(|x| matches!(x, Part::ImageUrl { .. })))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FnCall {
    pub name: String,
    /// JSON 문자열(OpenAI 원형)
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "fn_type")]
    pub kind: String,
    pub function: FnCall,
}

fn fn_type() -> String {
    "function".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Msg {
    pub role: String,
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Msg {
    pub fn system(t: impl Into<String>) -> Msg {
        Msg { role: "system".into(), content: Some(Content::Text(t.into())), tool_calls: None, tool_call_id: None }
    }
    pub fn user(t: impl Into<String>) -> Msg {
        Msg { role: "user".into(), content: Some(Content::Text(t.into())), tool_calls: None, tool_call_id: None }
    }
    pub fn user_parts(p: Vec<Part>) -> Msg {
        Msg { role: "user".into(), content: Some(Content::Parts(p)), tool_calls: None, tool_call_id: None }
    }
    pub fn tool(id: &str, content: &Value) -> Msg {
        Msg { role: "tool".into(), content: Some(Content::Text(content.to_string())), tool_calls: None, tool_call_id: Some(id.into()) }
    }
    pub fn assistant_calls(calls: Vec<ToolCall>) -> Msg {
        Msg { role: "assistant".into(), content: Some(Content::Text(String::new())), tool_calls: Some(calls), tool_call_id: None }
    }
    pub fn text(&self) -> String {
        self.content.as_ref().map(|c| c.text()).unwrap_or_default()
    }
    /// 영상 부분을 "[image]" 로 바꾼 사본(오래된 turn 을 맥락에 다시 넣을 때 토큰 절약).
    pub fn without_images(&self) -> Msg {
        match &self.content {
            Some(c) if c.has_image() => Msg { content: Some(Content::Text(c.text())), ..self.clone() },
            _ => self.clone(),
        }
    }
    pub fn calls(&self) -> &[ToolCall] {
        self.tool_calls.as_deref().unwrap_or(&[])
    }
}

/// 결정론을 위한 표본추출 값. 서버 기본값(예: llama-server presence 1.5, temp 0.7)에 기대지 않고 요청마다 명시한다.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub repetition_penalty: f64,
    pub seed: Option<u64>,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling { temperature: 0.0, top_p: 1.0, top_k: 1, min_p: 0.0, presence_penalty: 0.0, frequency_penalty: 0.0, repetition_penalty: 1.0, seed: Some(0) }
    }
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<Msg>,
    pub tools: Option<Vec<Value>>,
    pub sampling: Sampling,
    pub max_tokens: Option<u32>,
    /// Qwen3 계열 생각 모드. false 면 `chat_template_kwargs.enable_thinking=false`.
    pub thinking: bool,
    /// 기록·재생용 구분("plan" / "summary"). 서버로는 안 보낸다.
    pub purpose: String,
}

impl ChatRequest {
    pub fn to_json(&self, model: &str) -> Value {
        let s = &self.sampling;
        let mut v = json!({
            "model": model,
            "messages": self.messages,
            "stream": false,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "top_k": s.top_k,
            "min_p": s.min_p,
            "presence_penalty": s.presence_penalty,
            "frequency_penalty": s.frequency_penalty,
            // vLLM 이름 / llama.cpp 이름 (모르는 필드는 두 서버 모두 무시)
            "repetition_penalty": s.repetition_penalty,
            "repeat_penalty": s.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        });
        if let Some(seed) = s.seed {
            v["seed"] = json!(seed);
        }
        if let Some(t) = &self.tools {
            v["tools"] = Value::Array(t.clone());
            v["tool_choice"] = json!("auto");
            v["parallel_tool_calls"] = json!(false);
        }
        if let Some(m) = self.max_tokens {
            v["max_tokens"] = json!(m);
        }
        v
    }
    /// 요청 지문(재생 때 같은 요청인지 비교). 영상은 빼고 글만.
    pub fn fingerprint(&self) -> String {
        let text: String = self.messages.iter().map(|m| format!("{}:{}|{:?}", m.role, m.text(), m.calls())).collect();
        crate::codec::sha1_hex(text.as_bytes())[..16].to_string()
    }
    /// 대략 토큰 수(영어 글 기준 4글자당 1, 영상 1장 256)
    pub fn approx_tokens(&self) -> usize {
        let chars: usize = self.messages.iter().map(|m| m.text().len() + m.calls().iter().map(|c| c.function.arguments.len() + 20).sum::<usize>()).sum();
        let tools: usize = self.tools.as_ref().map(|t| t.iter().map(|x| x.to_string().len()).sum()).unwrap_or(0);
        let images = self.messages.iter().filter(|m| m.content.as_ref().map(|c| c.has_image()).unwrap_or(false)).count();
        (chars + tools) / 4 + images * 256
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResult {
    pub msg: Msg,
    pub finish_reason: String,
    pub latency_ms: u64,
    #[serde(default)]
    pub usage: Value,
    #[serde(default)]
    pub timings: Value,
    /// 서버가 준 생각 글(reasoning_content) 길이 — 기록만, 다음 요청에는 안 넣는다.
    #[serde(default)]
    pub reasoning_chars: usize,
}

pub trait Llm: Send {
    fn chat(&mut self, req: &ChatRequest) -> Result<ChatResult, String>;
    fn describe(&self) -> String;
    /// 결정·요약이 끝나 당분간 안 부를 때(로컬 서버 내리기 훅)
    fn release(&mut self) {}
}

/// `<think>…</think>` 제거, `<tool_call>{…}</tool_call>`(Qwen 원형 글) 을 도구 호출로 바꾼다
/// — 서버가 도구 호출을 못 풀어 글로 돌려준 경우의 대비.
pub fn sanitize(mut msg: Msg) -> Msg {
    let mut text = msg.text();
    while let (Some(a), Some(b)) = (text.find("<think>"), text.find("</think>")) {
        if b < a {
            break;
        }
        text.replace_range(a..b + "</think>".len(), "");
    }
    if let Some(b) = text.find("</think>") {
        text.replace_range(..b + "</think>".len(), "");
    }
    let mut calls = msg.tool_calls.take().unwrap_or_default();
    if calls.is_empty() {
        let mut rest = text.clone();
        let mut n = 0;
        while let (Some(a), Some(b)) = (rest.find("<tool_call>"), rest.find("</tool_call>")) {
            if b < a {
                break;
            }
            let body = rest[a + "<tool_call>".len()..b].trim().to_string();
            if let Ok(v) = serde_json::from_str::<Value>(&body) {
                if let Some(name) = v.get("name").and_then(|x| x.as_str()) {
                    let args = v.get("arguments").cloned().unwrap_or(json!({}));
                    let args = if let Some(s) = args.as_str() { s.to_string() } else { args.to_string() };
                    calls.push(ToolCall { id: format!("call_text_{n}"), kind: "function".into(), function: FnCall { name: name.into(), arguments: args } });
                    n += 1;
                }
            }
            rest.replace_range(a..b + "</tool_call>".len(), "");
        }
        if !calls.is_empty() {
            text = rest;
        }
    }
    for (i, c) in calls.iter_mut().enumerate() {
        if c.id.is_empty() {
            c.id = format!("call_{i}");
        }
        c.kind = "function".into();
    }
    Msg {
        role: "assistant".into(),
        content: Some(Content::Text(text.trim().to_string())),
        tool_calls: if calls.is_empty() { None } else { Some(calls) },
        tool_call_id: None,
    }
}

pub fn parse_response(v: &Value, latency_ms: u64) -> Result<ChatResult, String> {
    let ch = v.get("choices").and_then(|c| c.get(0)).ok_or_else(|| format!("choices 없음: {}", v.to_string().chars().take(300).collect::<String>()))?;
    let m = ch.get("message").ok_or("message 없음")?;
    let reasoning_chars = ["reasoning_content", "reasoning"].iter().filter_map(|k| m.get(*k).and_then(|x| x.as_str())).map(|s| s.len()).sum();
    let raw = Msg {
        role: "assistant".into(),
        content: m.get("content").and_then(|c| c.as_str()).map(|s| Content::Text(s.to_string())),
        tool_calls: m.get("tool_calls").and_then(|t| serde_json::from_value::<Vec<ToolCall>>(normalize_calls(t)).ok()),
        tool_call_id: None,
    };
    Ok(ChatResult {
        msg: sanitize(raw),
        finish_reason: ch.get("finish_reason").and_then(|x| x.as_str()).unwrap_or("").to_string(),
        latency_ms,
        usage: v.get("usage").cloned().unwrap_or(Value::Null),
        timings: v.get("timings").cloned().unwrap_or(Value::Null),
        reasoning_chars,
    })
}

/// arguments 가 객체로 오는 서버도 있어 문자열로 맞춘다.
fn normalize_calls(t: &Value) -> Value {
    let mut t = t.clone();
    if let Some(a) = t.as_array_mut() {
        for c in a {
            if let Some(args) = c.pointer("/function/arguments").cloned() {
                if !args.is_string() {
                    c["function"]["arguments"] = Value::String(args.to_string());
                }
            }
            if c.get("id").map(|x| x.is_null()).unwrap_or(true) {
                c["id"] = json!("");
            }
        }
    }
    t
}

/// OpenAI 호환 HTTP 서버.
pub struct HttpLlm {
    pub base: String,
    pub model: String,
    /// 키가 든 환경변수 이름(값은 기록하지 않는다)
    pub key_env: Option<String>,
    pub timeout: Duration,
}

impl HttpLlm {
    pub fn new(base: &str, model: &str, key_env: Option<&str>, timeout_s: u64) -> HttpLlm {
        HttpLlm { base: base.trim_end_matches('/').to_string(), model: model.to_string(), key_env: key_env.map(|s| s.to_string()), timeout: Duration::from_secs(timeout_s) }
    }

    /// KAU API: KAU_BASE_URL / KAU_MODEL / KAU_API_KEY 환경변수(`~/.config/behavior-2026/kau.env`).
    pub fn kau(timeout_s: u64) -> Result<HttpLlm, String> {
        let base = std::env::var("KAU_BASE_URL").unwrap_or_else(|_| "https://agent.kau.ac.kr/v1".into());
        let model = std::env::var("KAU_MODEL").unwrap_or_else(|_| "qwen3.5-9b".into());
        if std::env::var("KAU_API_KEY").map(|k| k.trim().is_empty()).unwrap_or(true) {
            return Err("KAU_API_KEY 환경변수가 비었다 (set -a; . ~/.config/behavior-2026/kau.env; set +a)".into());
        }
        Ok(HttpLlm::new(&base, &model, Some("KAU_API_KEY"), timeout_s))
    }

    fn key(&self) -> Option<String> {
        self.key_env.as_ref().and_then(|n| std::env::var(n).ok()).filter(|k| !k.trim().is_empty())
    }

    pub fn post(&self, path: &str, body: &Value) -> Result<(u16, Vec<u8>), String> {
        let url = format!("{}{}", self.base, path);
        if url.starts_with("https://") {
            curl_post(&url, body, self.key().as_deref(), self.timeout)
        } else {
            let auth = self.key().map(|k| format!("Bearer {k}"));
            let headers: Vec<(&str, &str)> = auth.as_deref().map(|a| vec![("Authorization", a)]).unwrap_or_default();
            let r = crate::http::post_json(&url, body, &headers, self.timeout)?;
            Ok((r.status, r.body))
        }
    }
}

/// curl 로 HTTPS POST. 키는 표준입력 설정으로(명령줄에 안 보이게), 본문은 임시 파일로.
pub fn curl_post(url: &str, body: &Value, key: Option<&str>, timeout: Duration) -> Result<(u16, Vec<u8>), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let dir = std::env::temp_dir();
    let tag = format!("bagent_{}_{}", std::process::id(), crate::util::Rng::from_time().next_u64());
    let inp = dir.join(format!("{tag}_in.json"));
    let out = dir.join(format!("{tag}_out.json"));
    std::fs::write(&inp, serde_json::to_vec(body).map_err(|e| e.to_string())?).map_err(|e| format!("임시 파일: {e}"))?;
    let mut child = Command::new("curl")
        .args(["-sS", "-K", "-", "--max-time", &format!("{}", timeout.as_secs().max(1)), "-H", "Content-Type: application/json", "-o"])
        .arg(&out)
        .args(["-w", "%{http_code}", "--data-binary"])
        .arg(format!("@{}", inp.display()))
        .arg(url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("curl 실행 실패: {e}"))?;
    {
        let mut stdin = child.stdin.take().ok_or("curl stdin")?;
        if let Some(k) = key {
            let _ = writeln!(stdin, "header = \"Authorization: Bearer {}\"", k.trim().replace('"', ""));
        }
    }
    let o = child.wait_with_output().map_err(|e| format!("curl 대기 실패: {e}"))?;
    let _ = std::fs::remove_file(&inp);
    let resp = std::fs::read(&out).unwrap_or_default();
    let _ = std::fs::remove_file(&out);
    let code: u16 = String::from_utf8_lossy(&o.stdout).trim().parse().unwrap_or(0);
    if code == 0 {
        return Err(format!("curl 실패: {}", String::from_utf8_lossy(&o.stderr).trim()));
    }
    Ok((code, resp))
}

impl Llm for HttpLlm {
    fn chat(&mut self, req: &ChatRequest) -> Result<ChatResult, String> {
        let t0 = Instant::now();
        let (code, body) = self.post("/chat/completions", &req.to_json(&self.model))?;
        if code != 200 {
            return Err(format!("HTTP {code}: {}", String::from_utf8_lossy(&body).chars().take(400).collect::<String>()));
        }
        let v: Value = serde_json::from_slice(&body).map_err(|e| format!("응답 JSON: {e}"))?;
        parse_response(&v, t0.elapsed().as_millis() as u64)
    }
    fn describe(&self) -> String {
        format!("http {} model={}", self.base, self.model)
    }
}

/// 로컬 서버를 계획할 때만 올리고 내리는 감싸개(API 모드에서는 쓰지 않는다).
pub struct ManagedLlm {
    pub inner: Box<dyn Llm>,
    pub up_cmd: Option<String>,
    pub down_cmd: Option<String>,
    pub health_url: Option<String>,
    pub ready_timeout: Duration,
    pub up: bool,
}

impl ManagedLlm {
    fn ensure_up(&mut self) -> Result<(), String> {
        if self.up {
            return Ok(());
        }
        if let Some(c) = &self.up_cmd {
            let st = std::process::Command::new("bash").arg("-c").arg(c).status().map_err(|e| format!("up 명령: {e}"))?;
            if !st.success() {
                return Err(format!("up 명령 실패: {st}"));
            }
        }
        if let Some(h) = &self.health_url {
            let t0 = Instant::now();
            loop {
                if crate::http::get(h, Duration::from_secs(2)).map(|r| r.status == 200).unwrap_or(false) {
                    break;
                }
                if t0.elapsed() > self.ready_timeout {
                    return Err(format!("{h} 가 {:?} 안에 준비되지 않음", self.ready_timeout));
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        }
        self.up = true;
        Ok(())
    }
}

impl Llm for ManagedLlm {
    fn chat(&mut self, req: &ChatRequest) -> Result<ChatResult, String> {
        self.ensure_up()?;
        self.inner.chat(req)
    }
    fn describe(&self) -> String {
        format!("managed({})", self.inner.describe())
    }
    fn release(&mut self) {
        if !self.up {
            return;
        }
        if let Some(c) = &self.down_cmd {
            let _ = std::process::Command::new("bash").arg("-c").arg(c).status();
        }
        self.up = false;
    }
}

/// 함수로 답하는 가짜 LLM(단위 시험·가짜 세계).
pub struct FnLlm<F: FnMut(&ChatRequest) -> Result<Msg, String> + Send> {
    pub f: F,
    pub name: String,
}

impl<F: FnMut(&ChatRequest) -> Result<Msg, String> + Send> Llm for FnLlm<F> {
    fn chat(&mut self, req: &ChatRequest) -> Result<ChatResult, String> {
        let t0 = Instant::now();
        let msg = (self.f)(req)?;
        Ok(ChatResult { msg: sanitize(msg), finish_reason: "stop".into(), latency_ms: t0.elapsed().as_millis() as u64, usage: Value::Null, timings: Value::Null, reasoning_chars: 0 })
    }
    fn describe(&self) -> String {
        self.name.clone()
    }
}

/// 기록된 응답을 순서대로 돌려주는 LLM(재생). 목적별로 줄을 따로 둔다.
pub struct ReplayLlm {
    /// 목적별: (요청 지문, 응답, 기록된 요청 메시지)
    pub queues: std::collections::HashMap<String, std::collections::VecDeque<(String, ChatResult, Value)>>,
    pub mismatches: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl Llm for ReplayLlm {
    fn chat(&mut self, req: &ChatRequest) -> Result<ChatResult, String> {
        let q = self.queues.get_mut(&req.purpose).ok_or_else(|| format!("기록에 '{}' 응답이 없음", req.purpose))?;
        let (fp, r, recorded) = q.pop_front().ok_or_else(|| format!("기록된 '{}' 응답이 바닥남", req.purpose))?;
        let now = req.fingerprint();
        if !fp.is_empty() && fp != now {
            // 처음 달라지는 메시지·글자 위치를 같이 남긴다
            let mut detail = String::new();
            if let Some(rec) = recorded.as_array() {
                for (i, m) in req.messages.iter().enumerate() {
                    let a = rec.get(i).and_then(|x| serde_json::from_value::<Msg>(x.clone()).ok()).map(|x| x.text()).unwrap_or_default();
                    let b = m.text();
                    if a != b {
                        let p = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
                        let ctx = |s: &str| s.chars().skip(p.saturating_sub(40)).take(120).collect::<String>();
                        detail = format!("메시지 {i}({}) 글자 {p}: 기록 «{}» / 지금 «{}»", m.role, ctx(&a), ctx(&b));
                        break;
                    }
                }
            }
            self.mismatches.lock().unwrap().push(format!("{}: 요청 지문 다름 기록 {fp} / 지금 {now} {detail}", req.purpose));
        }
        Ok(r)
    }
    fn describe(&self) -> String {
        "replay".into()
    }
}

pub fn tool_def(name: &str, desc: &str, params: Value) -> Value {
    json!({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_fields_only() {
        let v = json!({"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": null,
            "reasoning_content": "long thoughts", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "finish", "arguments": {"reason": "done"}}}]}}]});
        let r = parse_response(&v, 5).unwrap();
        assert_eq!(r.reasoning_chars, 13);
        let s = serde_json::to_string(&r.msg).unwrap();
        assert!(!s.contains("reasoning"));
        assert_eq!(r.msg.calls()[0].function.arguments, r#"{"reason":"done"}"#);
    }

    #[test]
    fn qwen_text_tool_call() {
        let m = Msg { role: "assistant".into(), content: Some(Content::Text("<think>hmm</think>ok <tool_call>\n{\"name\": \"finish\", \"arguments\": {\"reason\": \"x\"}}\n</tool_call>".into())), tool_calls: None, tool_call_id: None };
        let s = sanitize(m);
        assert_eq!(s.calls().len(), 1);
        assert_eq!(s.calls()[0].function.name, "finish");
        assert_eq!(s.text(), "ok");
    }

    #[test]
    fn request_shape() {
        let r = ChatRequest {
            messages: vec![Msg::system("s"), Msg::user("u")],
            tools: Some(vec![tool_def("x", "d", json!({"type": "object", "properties": {}}))]),
            sampling: Sampling::default(),
            max_tokens: Some(100),
            thinking: false,
            purpose: "plan".into(),
        };
        let v = r.to_json("m");
        assert_eq!(v["chat_template_kwargs"]["enable_thinking"], json!(false));
        assert_eq!(v["tools"][0]["function"]["name"], "x");
        assert_eq!(v["presence_penalty"], json!(0.0));
        assert_eq!(v["seed"], json!(0));
        assert!(v.get("purpose").is_none());
    }
}

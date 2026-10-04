//! `search-objects` — 도구를 손으로 / 각본으로 / LLM 으로 불러 보는 명령.
//!
//! ```text
//! search-objects schema                            도구 정의 세 개(OpenAI tools 배열)
//! search-objects live-check SM_LIB MEM             진짜 scenemap(SM_LIB = libsgrt.so 등)에 합성 스트림 → 실시간 기억 도구 끝까지(livecheck.rs)
//! search-objects call MEM TOOL '<args>' […]        기억 폴더 MEM 에 도구 호출(TOOL '<args>' 쌍을 차례로, 한 색인으로)
//! search-objects demo MEM [QUERY]                  "라디오 가져와" 각본(LLM 없이): 찾기 → 되묻는 말 → 확인 → 다시 찾기
//! search-objects llm MEM "<지시>" [--turns 6]      (--features llm) KAU Qwen 원형 도구 호출 루프(도구 두 개)
//! ```
//! MEM 에는 `cache/objsearch/`(벡터·이름 캐시)와 `confirmations.jsonl`(확인 기록)이 생긴다 — 시험은 복사본에서.

use search_objects::sys::Paths;
use search_objects::{ObjectSearch, CONFIRM, SEARCH};
use serde_json::{json, Value};
use std::path::Path;

fn usage() -> ! {
    eprintln!("usage: search-objects schema | live-check SM_LIB MEM | call MEM TOOL '<json>' [TOOL '<json>' …] | demo MEM [QUERY] | llm MEM \"<instruction>\" [--turns N]");
    std::process::exit(2)
}

fn open(mem: &str) -> ObjectSearch {
    ObjectSearch::open(Path::new(mem), &Paths::default()).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else { usage() };
    match cmd.as_str() {
        "schema" => println!("{}", serde_json::to_string_pretty(&Value::Array(search_objects::definitions())).unwrap()),
        "call" => {
            let mut s = open(args.get(1).unwrap_or_else(|| usage()));
            for pair in args[2..].chunks(2) {
                let [tool, a] = pair else { usage() };
                println!("{}", s.run_tool(tool, &Value::String(a.clone())));
            }
        }
        "demo" => demo(args.get(1).unwrap_or_else(|| usage()), args.get(2).map(String::as_str).unwrap_or("라디오")),
        "llm" => llm(&args),
        "live-check" => std::process::exit(search_objects::livecheck::run(
            args.get(1).unwrap_or_else(|| usage()),
            args.get(2).unwrap_or_else(|| usage()),
        )),
        _ => usage(),
    }
}

/// 각본 대화. LLM 자리의 말은 결과 칸으로 만든 틀 문장이다(실제 LLM 은 `llm` 명령).
fn demo(mem: &str, query: &str) {
    let mut s = open(mem);
    let say = |who: &str, t: &str| println!("{who:>6} │ {t}");
    let call = |s: &mut ObjectSearch, tool: &str, a: Value| -> Value {
        println!("  call │ {tool} {a}");
        let r = s.run_tool(tool, &a);
        println!("result │ {r}");
        r
    };
    say("user", &format!("{query} 가져와"));
    let r = call(&mut s, SEARCH, json!({"query": query}));
    let Some(m) = r["matches"].as_array().and_then(|a| a.first()).cloned() else {
        say("agent", &format!("{query}(을)를 기억에서 못 찾았어요. 어디 있는지 알려 주시거나 둘러볼까요?"));
        return;
    };
    let id = m["id"].as_str().unwrap_or("").to_string();
    let place = match (m["room"].as_str(), m["landmark"]["name"].as_str()) {
        (Some(r), Some(l)) => format!("{r}의 {l} 근처에"),
        (Some(r), None) => format!("{r}에"),
        (None, Some(l)) => format!("{l} 근처에"),
        _ => "".into(),
    };
    if m["match_type"] == "appearance" {
        let reg = m["registered"].as_str().or(m["name"].as_str()).unwrap_or("다른 이름");   // 등록 이름(보여 주는 이름과 같으면 registered 칸이 없음)
        say("agent", &format!("{query}(으)로 등록된 건 없는데, {place} '{reg}'(으)로 등록된 {id}가 {query}일 수 있어요(모습 확률 {:.2}). 가져올까요?", m["p_query"].as_f64().unwrap_or(0.0)));
        say("user", "응, 그거 라디오 맞아");
        call(&mut s, CONFIRM, json!({"id": id, "name": query, "source": "user"}));   // 사용자가 확인한 것 = 질의한 물건
        say("agent", &format!("{id} 를 {query}(으)로 기억할게요. 가져올게요."));
        println!("  (VLA) │ {}", json!({"executor": "vla", "skill": "pick up radio", "objects": [id], "max_s": 30}));
        say("user", &format!("{query} 어디 있어?"));
        let r2 = call(&mut s, SEARCH, json!({"query": query}));
        let m2 = &r2["matches"][0];
        say("agent", &format!("{place} 있어요({}, 이름으로 찾음: {}).", m2["id"].as_str().unwrap_or(""), m2["match_type"].as_str().unwrap_or("")));
    } else {
        say("agent", &format!("{place} {id}({}) 가 있어요. 가져올게요.", m["name"].as_str().unwrap_or("")));
    }
}

#[cfg(not(feature = "llm"))]
fn llm(_: &[String]) {
    eprintln!("build with --features llm");
    std::process::exit(2)
}

/// 수업 week02 원형 루프(move_robot 과 같음): tool_calls 면 실행해 role:"tool" 로 붙이고 다시, 글이면 끝.
#[cfg(feature = "llm")]
fn llm(args: &[String]) {
    use bagent::llm::{sanitize, ChatRequest, HttpLlm, Llm, Msg, Sampling};
    let mem = args.get(1).unwrap_or_else(|| usage());
    let instruction = args.get(2).cloned().unwrap_or_else(|| usage());
    let turns: usize = args.iter().position(|a| a == "--turns").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(6);
    let mut s = open(mem);
    let mut llm = HttpLlm::kau(120).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    let mut msgs = vec![
        Msg::system(
            "You are a home robot assistant. The user speaks Korean; answer in short Korean. \
Use search_objects to find objects in the robot's memory (you may rewrite the query, e.g. Korean to an English noun). \
If a result has ask_user, ask the user before acting. Call confirm_object only after the user confirmed an object's identity. \
When you need the user's answer, reply with your question and no tool call.",
        ),
        Msg::user(instruction),
    ];
    let tools = search_objects::definitions();   // 세 개(list_place 포함)
    for turn in 0..turns {
        let req = ChatRequest { messages: msgs.clone(), tools: Some(tools.clone()), sampling: Sampling::default(), max_tokens: Some(512), thinking: false, purpose: "search_objects".into() };
        let res = match llm.chat(&req) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("llm: {e}");
                std::process::exit(1)
            }
        };
        let msg = sanitize(res.msg);
        let calls = msg.calls().to_vec();
        eprintln!("[turn {turn}] {} ms, {} call(s)", res.latency_ms, calls.len());
        if calls.is_empty() {
            println!(" agent │ {}", msg.text());
            return;
        }
        msgs.push(Msg::assistant_calls(calls.clone()));
        for c in calls {
            let out = s.run_tool(&c.function.name, &Value::String(c.function.arguments.clone()));
            println!("  call │ {} {}\nresult │ {}", c.function.name, c.function.arguments, out);
            msgs.push(Msg::tool(&c.id, &out));
        }
    }
    println!("(turn limit)");
}

//! `move-robot` — 도구 하나를 손으로 / LLM 으로 불러 보는 명령.
//!
//! ```text
//! move-robot schema                         도구 정의(OpenAI tools 항목) 출력
//! move-robot call '<args>' [--addr A]       시뮬 접착부에 한 번 호출 (기본 127.0.0.1:8771)
//! move-robot mock '<args>' ['<args>' …]     가짜 로봇에서 차례로 호출
//! move-robot vla-mock '<call>' ['<call>' …] 가짜 LIMO + OMX-F(컵 O1·상자 O2·먼 의자 O3)에서 VLA 호출을 차례로(각본 정책 대역)
//! move-robot llm "<지시>" [--addr A | --mock] [--turns 8]   (--features llm) KAU Qwen 원형 도구 호출 루프
//! ```

use move_robot::link::{run_tool, LimoMockBackend, Mock, TcpSim, DEFAULT_ADDR};
#[cfg(feature = "llm")]
use move_robot::link::Backend;
use serde_json::Value;

fn usage() -> ! {
    eprintln!("usage: move-robot schema | call '<json>' [--addr host:port] | mock '<json>'... | vla-mock '<vla call>'... | llm \"<instruction>\" [--addr A | --mock] [--turns N]\n\n{}", move_robot::link::part_help());
    std::process::exit(2)
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else { usage() };
    match cmd.as_str() {
        "schema" => println!("{}", serde_json::to_string_pretty(&move_robot::definition()).unwrap()),
        "call" => {
            let a: Value = serde_json::from_str(args.get(1).unwrap_or_else(|| usage())).unwrap_or_else(|e| {
                eprintln!("args: {e}");
                std::process::exit(2)
            });
            let mut sim = TcpSim::new(&flag(&args, "--addr").unwrap_or_else(|| DEFAULT_ADDR.into()));
            println!("{}", run_tool(&a, &mut sim));
        }
        "mock" => {
            let mut m = Mock::default();
            for s in args.iter().skip(1) {
                let a: Value = serde_json::from_str(s).unwrap_or(Value::String(s.clone()));
                println!("{}", run_tool(&a, &mut m));
            }
        }
        "vla-mock" => {
            let mut b = LimoMockBackend(move_robot::limo_mock::LimoMock::new(move_robot::limo_mock::demo_scene()));
            for s in args.iter().skip(1) {
                let a: Value = serde_json::from_str(s).unwrap_or(Value::String(s.clone()));
                println!("{}", run_tool(&a, &mut b));
            }
        }
        "llm" => llm(&args),
        _ => usage(),
    }
}

#[cfg(not(feature = "llm"))]
fn llm(_: &[String]) {
    eprintln!("build with --features llm");
    std::process::exit(2)
}

/// 수업 week02 원형 루프: messages + tools → tool_calls 면 실행해 role:"tool" 로 붙이고 다시, 글이면 끝.
#[cfg(feature = "llm")]
fn llm(args: &[String]) {
    use bagent::llm::{sanitize, ChatRequest, HttpLlm, Llm, Msg, Sampling};
    let instruction = args.get(1).cloned().unwrap_or_else(|| usage());
    let turns: usize = flag(args, "--turns").and_then(|s| s.parse().ok()).unwrap_or(8);
    let mut backend: Box<dyn Backend> =
        if args.iter().any(|a| a == "--mock") { Box::new(Mock::default()) } else { Box::new(TcpSim::new(&flag(args, "--addr").unwrap_or_else(|| DEFAULT_ADDR.into()))) };
    let mut llm = HttpLlm::kau(120).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    let mut msgs = vec![
        Msg::system(
            "You control an R1Pro mobile robot (holonomic base, 4-joint torso, two 7-joint arms, two grippers) with one tool. \
Call move_robot one part at a time; read the result before the next call. If unsure of the current joint values, call delta with zeros first. \
When the request is done (or impossible), answer the user in one short sentence without calling tools.",
        ),
        Msg::user(instruction),
    ];
    let tools = vec![move_robot::definition()];
    for turn in 0..turns {
        let req = ChatRequest { messages: msgs.clone(), tools: Some(tools.clone()), sampling: Sampling::default(), max_tokens: Some(512), thinking: false, purpose: "move_robot".into() };
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
            println!("{}", msg.text());
            return;
        }
        msgs.push(Msg::assistant_calls(calls.clone()));
        for c in calls {
            let out = if c.function.name == move_robot::TOOL_NAME {
                run_tool(&Value::String(c.function.arguments.clone()), backend.as_mut())
            } else {
                move_robot::error_obs(&format!("unknown tool '{}'; the only tool is move_robot", c.function.name))
            };
            println!("call {} -> {}", c.function.arguments, out);
            msgs.push(Msg::tool(&c.id, &out));
        }
    }
    println!("(turn limit)");
}

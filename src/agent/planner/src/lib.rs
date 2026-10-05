//! BEHAVIOR Challenge 2026 상위 계획 에이전트.
//!
//! 두 층으로 나뉜다.
//! - **매 스텝 경로(중계기)**: 평가기 ↔ [relay] ↔ VLA 서버. 관측·행동 바이트를 복사·변경 없이 넘기고
//!   ([`ws`], [`msgpack`], [`wire`]), 오도메트리([`odom`])와 단계 감시([`monitor`])만 산술로 돌린다. LLM 없음.
//! - **단계 경계 경로(에이전트)**: [`planner`] 가 OpenAI 호환 Chat Completions(도구 호출)로 Qwen 을 부르고
//!   ([`llm`], [`tools`], [`context`], [`memory`], [`plan`]), 씬그래프([`graph`])로 `back`·`the other` 를 풀어
//!   다음 단계 지시([`instruction`])를 정한다. 기록은 [`trace`](JSONL)로 남기고 재생한다.
//!
//! 설계와 이유: `docs/에이전트_설계.md`.

pub mod bddl;
pub mod catalog;
pub mod codec;
pub mod context;
pub mod fakes;
pub mod fk;
pub mod graph;
pub mod http;
pub mod image;
pub mod instruction;
pub mod link;
pub mod llm;
pub mod memory;
pub mod mockworld;
pub mod monitor;
pub mod msgpack;
pub mod odom;
pub mod plan;
pub mod pose;
pub mod planner;
pub mod relay;
pub mod replay;
pub mod session;
pub mod setup;
pub mod tools;
pub mod trace;
pub mod util;
pub mod vocab;
pub mod wire;
pub mod ws;

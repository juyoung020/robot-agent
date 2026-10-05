//! 베이스 모드 `explore`(에이전트 쪽, 10-06): LLM 없는 프런티어 탐사(Yamauchi) — 지난 지도 요약에서 경로가 가장 짧은
//! 프런티어로 `go_to` 를 되풀이한다. 옛 스킬 explore 의 기준선 정책을 도구로 옮긴 것(같은 고름, 같은 결과).
//!
//! ```json
//! {"part": "base", "mode": "explore", "max_steps": 1}     (max_steps 1..=50, 없으면 1)
//! ```
//! - 한 번 부르면 `max_steps` 번까지 go_to 한다. 닿을 수 있는 프런티어(`path_m` 이 숫자)가 없으면 거기서 멈춘다.
//! - 결과 = 마지막 go_to 결과 + `explore{steps, targets, end: "no_frontier"|"max_steps"|"link_down"}` + `_calls[{args, result}]`
//!   (실제로 한 go_to 하나하나 — 런타임이 결정 기록을 go_to 마다 한 줄씩 남긴다).
//!   한 번도 못 가면 `{"status":"done","mode":"explore","end":"no_frontier",…,"_calls":[]}`.
//! - 지난 관측이 필요하므로 시뮬 접착부(`mr_command`)가 아니라 에이전트 쪽 [`run_tool_ctx`] 에서만 돈다.
//!   기본 도구 정의(`definition()`)의 mode enum 에는 없고, 쓰려는 스킬이 [`crate::definition_modes`] 로 보인다.

use crate::link::{run_tool, Backend};
use crate::llm_view::{link_down, next_obs};
use serde_json::{json, Value};

pub const MODE: &str = "explore";
pub const MAX_STEPS: u64 = 50;

fn as_obj(args: &Value) -> Option<Value> {
    match args {
        Value::String(s) => serde_json::from_str(s).ok(),
        v => Some(v.clone()),
    }
}

/// `{"part":"base","mode":"explore",…}` 인가
pub fn is_explore_call(args: &Value) -> bool {
    as_obj(args).map_or(false, |a| a["mode"] == MODE)
}

/// 경로가 가장 짧은 프런티어 id(`path_m` 이 숫자인 것 중). 관측 = 지난 베이스 결과(지도 요약 `map` 이 있는 것)
pub fn pick(obs: &Value) -> Option<Value> {
    let fr = obs["map"]["frontiers"].as_array()?;
    fr.iter()
        .filter(|f| f["path_m"].as_f64().is_some())
        .min_by(|a, b| a["path_m"].as_f64().unwrap().total_cmp(&b["path_m"].as_f64().unwrap()))
        .map(|f| f["id"].clone())
}

/// 모드 `explore` 실행. `last` = 지난 결과(지도 요약이 있는 것).
pub fn run(args: &Value, backend: &mut dyn Backend, last: &Value) -> Value {
    let Some(a) = as_obj(args) else {
        return crate::error_obs("arguments are not JSON");
    };
    if a["part"] != "base" {
        return crate::error_obs("mode explore is for part base only");
    }
    let max_steps = match a.get("max_steps") {
        None | Some(Value::Null) => 1,
        Some(v) => match v.as_u64() {
            Some(n) if (1..=MAX_STEPS).contains(&n) => n,
            _ => return crate::error_obs(&format!("max_steps must be an integer 1..{MAX_STEPS}")),
        },
    };
    let mut last = last.clone();
    let mut calls: Vec<Value> = vec![];
    let mut targets: Vec<Value> = vec![];
    let mut end = "max_steps";
    let mut res = Value::Null;
    while (calls.len() as u64) < max_steps {
        let Some(id) = pick(&last) else {
            end = "no_frontier";
            break;
        };
        let go = json!({"part": "base", "mode": "go_to", "target": id});
        res = run_tool(&go, backend);
        calls.push(json!({"args": go, "result": res}));
        targets.push(id);
        if link_down(&res) {
            end = "link_down";
            break;
        }
        last = next_obs(&res, &last);
    }
    if calls.is_empty() {
        return json!({"status": "done", "mode": MODE, "end": end, "message": "no reachable frontier left",
            "hint": "exploration is complete (or every frontier is unreachable)", "_calls": []});
    }
    let mut out = res;
    out["explore"] = json!({"steps": calls.len(), "targets": targets, "end": end});
    out["_calls"] = Value::Array(calls);
    out
}

/// 에이전트 쪽 도구 호출: 모드 `explore` 면 [`run`], 아니면 [`run_tool`] 그대로.
pub fn run_tool_ctx(args: &Value, backend: &mut dyn Backend, last: &Value) -> Value {
    if is_explore_call(args) {
        run(args, backend, last)
    } else {
        run_tool(args, backend)
    }
}

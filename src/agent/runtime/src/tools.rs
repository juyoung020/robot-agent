//! 도구를 이름으로 고르기. 런타임은 도구 코드를 갖지 않고, 각 도구 크레이트(`src/agent/tools/<tool>`)가 내놓는 함수를 이 이름표로 잇는다.
//! 지금 루프에 붙는 것은 `move_robot` 하나(`search_objects` 는 자기 크레이트의 루프에서 돈다 — 옮기면 여기 한 줄).

use move_robot::link::Backend;
use serde_json::{Map, Value};

/// 런타임이 도구에게 묻는 것. 결과 해석(LLM 에게 보일 것·결정 기록 칸)은 도구 쪽 지식이다.
pub trait Tool {
    fn name(&self) -> &str;
    /// OpenAI `tools` 항목. desc = 스킬 폴더의 설명, modes = 스킬이 보일 mode enum
    fn definition(&self, desc: &str, modes: Option<&[String]>) -> Value;
    /// 호출 실행. last = 지난 관측(도구 결과 그대로) — 지난 지도가 필요한 모드용
    fn exec(&mut self, args: &Value, backend: &mut dyn Backend, last: &Value) -> Value;
    /// 결과 → LLM 에게 보일 것만
    fn view(&self, res: &Value) -> Value;
    fn error(&self, msg: &str) -> Value;
    fn next_obs(&self, res: &Value, last: &Value) -> Value;
    fn link_down(&self, res: &Value) -> bool;
    // 결정 기록 칸
    fn mode_key(&self, args: &Value) -> String;
    fn target_space(&self, args: &Value) -> &'static str;
    fn obs_features(&self, pre_view: &Value) -> Value;
    fn outcome(&self, res: &Value) -> Map<String, Value>;
    fn label(&self, args: &Value, pre_view: &Value, res: &Value) -> Value;
    fn fold_line(&self, n: usize, args: &Value, res: &Value) -> String;
}

pub struct MoveRobot;

impl Tool for MoveRobot {
    fn name(&self) -> &str {
        move_robot::TOOL_NAME
    }
    fn definition(&self, desc: &str, modes: Option<&[String]>) -> Value {
        match modes {
            Some(m) => move_robot::definition_modes(desc, &m.iter().map(|s| s.as_str()).collect::<Vec<_>>()),
            None => move_robot::definition_with(desc),
        }
    }
    fn exec(&mut self, args: &Value, backend: &mut dyn Backend, last: &Value) -> Value {
        move_robot::frontier::run_tool_ctx(args, backend, last)
    }
    fn view(&self, res: &Value) -> Value {
        move_robot::llm_view::compact(res)
    }
    fn error(&self, msg: &str) -> Value {
        move_robot::error_obs(msg)
    }
    fn next_obs(&self, res: &Value, last: &Value) -> Value {
        move_robot::llm_view::next_obs(res, last)
    }
    fn link_down(&self, res: &Value) -> bool {
        move_robot::llm_view::link_down(res)
    }
    fn mode_key(&self, args: &Value) -> String {
        move_robot::llm_view::mode_key(args)
    }
    fn target_space(&self, args: &Value) -> &'static str {
        move_robot::llm_view::target_space(args)
    }
    fn obs_features(&self, pre_view: &Value) -> Value {
        move_robot::llm_view::obs_features(pre_view)
    }
    fn outcome(&self, res: &Value) -> Map<String, Value> {
        move_robot::llm_view::outcome(res)
    }
    fn label(&self, args: &Value, pre_view: &Value, res: &Value) -> Value {
        move_robot::llm_view::label(args, pre_view, res)
    }
    fn fold_line(&self, n: usize, args: &Value, res: &Value) -> String {
        move_robot::llm_view::fold_line(n, args, res)
    }
}

/// 이름 → 도구. 모르는 이름이면 None.
pub fn by_name(name: &str) -> Option<Box<dyn Tool>> {
    match name {
        n if n == move_robot::TOOL_NAME => Some(Box::new(MoveRobot)),
        _ => None,
    }
}

pub const KNOWN: &[&str] = &[move_robot::TOOL_NAME];

//! 한 일 기억. 공간 기억(어디에 무엇이, 처음 자리)은 씬그래프 + [`crate::graph::ObjectMemory`] 가 맡는다.
//!
//! 규칙(수업 week03 build_context 를 이 문제에 맞게 확장):
//! - 단계 기록([`StageRecord`])은 전부 들고 있고, 맥락에는 최근 `stage_lines` 개를 한 줄씩 넣는다.
//! - 경계마다의 대화(turn = 사건 user 메시지 + assistant 도구 호출 + tool 결과)는 최근 `keep_turns` 개만 원문으로 넣는다
//!   (영상은 "[image]" 로 바꿔서). 그보다 오래된 turn 이 `summarize_batch` 개 쌓이면 LLM 에 요약을 시켜 `summary` 에 합친다.
//! - 요약은 단계 결정을 돌려준 **뒤** 에(계획기 스레드에서) 한다 → 매 스텝·경계 지연에 안 들어간다.
//! - LLM 요약이 실패하면 단계 기록으로 결정적 요약을 만든다.
//! - `facts`: LLM 이 `remember` 로 남긴 오래 갈 사실(예: "radio_89 was on coffee_table_1"). 요약과 따로 둔다.

use crate::instruction::Instruction;
use crate::llm::Msg;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Running,
    Done,
    Failed,
    Partial,
    Replaced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageRecord {
    pub idx: u32,
    pub step_start: u64,
    pub step_end: Option<u64>,
    pub instruction: Instruction,
    pub prompt: String,
    pub outcome: Outcome,
    #[serde(default)]
    pub note: String,
    pub attempt: u32,
}

impl StageRecord {
    pub fn line(&self) -> String {
        let used = self.step_end.map(|e| e.saturating_sub(self.step_start)).map(|u| format!("{u} steps")).unwrap_or_else(|| "running".into());
        let note = if self.note.is_empty() { String::new() } else { format!(" — {}", self.note) };
        format!("#{} \"{}\" -> {:?} ({used}, attempt {}){note}", self.idx, self.prompt, self.outcome, self.attempt).replace("Running", "running")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub stages: Vec<StageRecord>,
    pub turns: Vec<Vec<Msg>>,
    pub summary: String,
    /// 요약에 이미 들어간 turn 수(앞에서부터)
    pub summarized: usize,
    pub facts: Vec<String>,
    pub keep_turns: usize,
    pub summarize_batch: usize,
    pub stage_lines: usize,
}

impl Default for Memory {
    fn default() -> Self {
        Memory { stages: vec![], turns: vec![], summary: String::new(), summarized: 0, facts: vec![], keep_turns: 3, summarize_batch: 3, stage_lines: 12 }
    }
}

impl Memory {
    pub fn current(&self) -> Option<&StageRecord> {
        self.stages.last().filter(|s| s.outcome == Outcome::Running)
    }
    pub fn current_mut(&mut self) -> Option<&mut StageRecord> {
        self.stages.last_mut().filter(|s| s.outcome == Outcome::Running)
    }

    pub fn close_current(&mut self, outcome: Outcome, step: u64, note: &str) {
        if let Some(c) = self.current_mut() {
            c.outcome = outcome;
            c.step_end = Some(step);
            if !note.is_empty() {
                c.note = note.chars().take(200).collect();
            }
        }
    }

    /// 지금 단계에 관찰 메모(판단 근거)를 남긴다.
    pub fn close_note(&mut self, note: &str) {
        if let Some(c) = self.current_mut() {
            c.note = note.chars().take(200).collect();
        }
    }

    pub fn open(&mut self, ins: Instruction, prompt: String, step: u64) -> u32 {
        let attempt = self
            .stages
            .iter()
            .rev()
            .take_while(|s| s.instruction.same_step(&ins) && s.outcome != Outcome::Done)
            .count() as u32
            + 1;
        let idx = self.stages.len() as u32 + 1;
        self.stages.push(StageRecord { idx, step_start: step, step_end: None, instruction: ins, prompt, outcome: Outcome::Running, note: String::new(), attempt });
        idx
    }

    /// 지금 단계와 같은 단계가 연달아 실패한 횟수
    pub fn consecutive_failures(&self, ins: &Instruction) -> u32 {
        self.stages
            .iter()
            .rev()
            .filter(|s| s.outcome != Outcome::Running)
            .take_while(|s| s.instruction.same_step(ins) && matches!(s.outcome, Outcome::Failed | Outcome::Partial))
            .count() as u32
    }

    pub fn stage_log(&self) -> String {
        let n = self.stages.len();
        let start = n.saturating_sub(self.stage_lines);
        let mut out = Vec::new();
        if start > 0 {
            let done = self.stages[..start].iter().filter(|s| s.outcome == Outcome::Done).count();
            out.push(format!("({start} earlier stages, {done} done — see summary)"));
        }
        out.extend(self.stages[start..].iter().map(|s| s.line()));
        if out.is_empty() {
            "(nothing yet)".into()
        } else {
            out.join("\n")
        }
    }

    pub fn push_turn(&mut self, turn: Vec<Msg>) {
        self.turns.push(turn.into_iter().map(|m| m.without_images()).collect());
    }

    pub fn recent_turns(&self) -> Vec<Msg> {
        // 창 밖이지만 아직 요약에 안 들어간 turn 은 요약될 때까지 원문으로 둔다(정보가 비는 틈이 없게).
        let n = self.turns.len();
        let start = n.saturating_sub(self.keep_turns).min(self.summarized);
        self.turns[start..].iter().flatten().cloned().collect()
    }

    /// 요약할 turn 범위(맥락에서 빠진 것 중 아직 요약 안 한 것)
    pub fn pending_summary(&self) -> Option<std::ops::Range<usize>> {
        let out_of_window = self.turns.len().saturating_sub(self.keep_turns);
        if out_of_window >= self.summarized + self.summarize_batch {
            Some(self.summarized..out_of_window)
        } else {
            None
        }
    }

    pub fn summary_request(&self, task_prompt: &str, range: std::ops::Range<usize>) -> Vec<Msg> {
        let mut hist = String::new();
        for (i, t) in self.turns[range.clone()].iter().enumerate() {
            hist.push_str(&format!("[Boundary {}]\n{}\n\n", range.start + i + 1, turn_to_text(t)));
        }
        vec![
            Msg::system(
                "You compress the history of a household robot planner. The summary replaces the original messages. \
Keep every fact needed later: which steps succeeded or failed and why, where objects were found (ids and positions), \
original places of moved objects, which objects of a kind were already handled, and open problems. \
Do not invent anything. At most 12 short lines.",
            ),
            Msg::user(format!(
                "Task: {task_prompt}\n\nPrevious summary:\n{}\n\nNew history to add:\n{hist}\nWrite the updated summary.",
                if self.summary.is_empty() { "(none)" } else { &self.summary }
            )),
        ]
    }

    pub fn apply_summary(&mut self, text: &str, range: std::ops::Range<usize>) {
        self.summary = text.trim().chars().take(2000).collect();
        self.summarized = range.end;
    }

    /// LLM 없이: 단계 기록으로 만든 요약(대체용)
    pub fn fallback_summary(&mut self, range: std::ops::Range<usize>) {
        let done: Vec<String> = self.stages.iter().filter(|s| s.outcome == Outcome::Done).map(|s| s.prompt.clone()).collect();
        let failed: Vec<String> = self.stages.iter().filter(|s| matches!(s.outcome, Outcome::Failed | Outcome::Partial)).map(|s| s.prompt.clone()).collect();
        self.summary = format!(
            "Done steps ({}): {}. Failed attempts ({}): {}.",
            done.len(),
            done.join("; "),
            failed.len(),
            failed.join("; ")
        )
        .chars()
        .take(2000)
        .collect();
        self.summarized = range.end;
    }
}

/// turn → 사람이 읽는 대화록(요약 요청용). 도구 메시지를 그대로 보내면 tools 없는 요청에서 형식 오류가 날 수 있다(week03 turn_to_text).
pub fn turn_to_text(turn: &[Msg]) -> String {
    let mut lines = Vec::new();
    for m in turn {
        match m.role.as_str() {
            "user" => lines.push(format!("Event: {}", m.text().chars().take(600).collect::<String>())),
            "assistant" => {
                for c in m.calls() {
                    lines.push(format!("Planner called {}({})", c.function.name, c.function.arguments.chars().take(300).collect::<String>()));
                }
                let t = m.text();
                if !t.is_empty() {
                    lines.push(format!("Planner: {}", t.chars().take(300).collect::<String>()));
                }
            }
            "tool" => lines.push(format!("Result: {}", m.text().chars().take(300).collect::<String>())),
            _ => {}
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_and_summary_range() {
        let mut m = Memory::default();
        for i in 0..7 {
            m.push_turn(vec![Msg::user(format!("e{i}"))]);
        }
        // 최근 3개가 창, 앞 4개가 창 밖 → 3개 이상이면 요약
        let r = m.pending_summary().unwrap();
        assert_eq!(r, 0..4);
        m.apply_summary("s", r);
        assert_eq!(m.pending_summary(), None);
        let recent: Vec<String> = m.recent_turns().iter().map(|x| x.text()).collect();
        assert_eq!(recent, vec!["e4", "e5", "e6"]);
    }

    #[test]
    fn attempts_count() {
        let mut m = Memory::default();
        let ins = Instruction { skill: "pick up from".into(), objects: vec!["r".into()], ..Default::default() };
        m.open(ins.clone(), "p".into(), 0);
        m.close_current(Outcome::Failed, 10, "slipped");
        assert_eq!(m.open(ins.clone(), "p".into(), 10), 2);
        assert_eq!(m.stages[1].attempt, 2);
        m.close_current(Outcome::Failed, 20, "");
        assert_eq!(m.consecutive_failures(&ins), 2);
    }
}

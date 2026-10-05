//! 지시 계약: 에이전트가 정한 단계(구조체) → VLA 가 받는 문장 한두 줄.
//!
//! VLA 문장 입력 한도(옛 π0.5 기준으로 잰 값 — 그 VLA 는 10-06 지움): 토큰 200개이고 그 안에 로봇 상태가 들어간다. 실측(PaliGemma 토크나이저, 상태 25차원):
//! 빈 지시일 때 "Task: , State: …;\nAction: " 가 101토큰 → 지시에 남는 것 99토큰. 여유를 두고 기본 90토큰
//! (옛 VLA 의 tokenizer, max_token_len=200, proprio 25차원).
//!
//! 형식 4가지(plan.md 4.1 비교용, 실행 중 바꿔 끼울 수 있음):
//! ① task    과제 문장 그대로(기본 체크포인트가 학습한 형태)
//! ② subtask 시연 주석 어휘 문장("pick up radio from coffee table")
//! ③ purpose "Purpose: … . Expected action: … ."(사용자 제안)
//! ④ metric  ② + 로봇 기준 이동량("go forward 2.1 m, 0.4 m to the left, turn left 30 degrees", 앞 +x, 왼쪽 +y, 반시계 +yaw)
//!           — **실험 전용**(사용자 결정 09-29: VLA 에 숫자 명령을 쓰지 않는다). 지시 형식 오프라인 실험(Comet pt50)에서
//!           숫자 명령이 이동 방향 일치를 오히려 낮췄다(cos 0.71 → 0.54). `--format metric` 을 명시할 때만 쓰고,
//!           중계기 agent 모드에서는 거부한다. 그래프 좌표·오도메트리는 계획기 안쪽 판단(완료 판정 등)에만 쓴다.
//!
//! ②③ 문장에서는 거리·각도 숫자를 [`strip_numbers`] 로 거른다(LLM 이 purpose/expected 에 숫자를 써도 VLA 로 안 간다).

use crate::util::tokens_with_margin;
#[cfg(test)]
use crate::util::estimate_tokens;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_TOKENS: usize = 90;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Task,
    Subtask,
    Purpose,
    Metric,
}

impl Format {
    pub fn parse(s: &str) -> Option<Format> {
        match s.trim().to_lowercase().as_str() {
            "task" | "1" => Some(Format::Task),
            "subtask" | "2" => Some(Format::Subtask),
            "purpose" | "3" => Some(Format::Purpose),
            "metric" | "4" => Some(Format::Metric),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Format::Task => "task",
            Format::Subtask => "subtask",
            Format::Purpose => "purpose",
            Format::Metric => "metric",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Metric {
    pub forward_m: f64,
    pub left_m: f64,
    pub turn_deg: f64,
}

impl Metric {
    pub fn text(&self) -> String {
        let mut parts = Vec::new();
        if self.forward_m.abs() >= 0.05 {
            parts.push(format!("go {} {:.1} m", if self.forward_m > 0.0 { "forward" } else { "backward" }, self.forward_m.abs()));
        }
        if self.left_m.abs() >= 0.05 {
            parts.push(format!("{:.1} m to the {}", self.left_m.abs(), if self.left_m > 0.0 { "left" } else { "right" }));
        }
        if self.turn_deg.abs() >= 5.0 {
            parts.push(format!("turn {} {:.0} degrees", if self.turn_deg > 0.0 { "left" } else { "right" }, self.turn_deg.abs()));
        }
        parts.join(", ")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Instruction {
    /// 어휘 35종 중 하나
    pub skill: String,
    /// 물체 참조(그래프 id 또는 이름), 주석 칸 순서
    pub objects: Vec<String>,
    /// 표시 이름(참조를 풀어 둔 것)
    pub names: Vec<String>,
    #[serde(default)]
    pub spatial: Vec<String>,
    #[serde(default)]
    pub memory: Option<String>,
    #[serde(default)]
    pub purpose: String,
    #[serde(default)]
    pub expected: String,
    #[serde(default)]
    pub metric: Option<Metric>,
    pub budget_steps: u64,
}

fn clean(s: &str) -> String {
    let s: String = s.chars().map(|c| if c.is_ascii() && !c.is_ascii_control() { c } else { ' ' }).collect();
    s.split_whitespace().collect::<Vec<_>>().join(" ").trim_end_matches(['.', ' ']).to_string()
}

const UNITS: &[&str] = &[
    "m", "cm", "mm", "km", "meter", "meters", "metre", "metres", "centimeter", "centimeters", "deg", "degree", "degrees", "rad", "radian",
    "radians", "step", "steps", "s", "sec", "second", "seconds", "ft", "feet", "inch", "inches", "percent", "%",
];

const PUNCT: [char; 5] = [',', '.', ';', ':', ')'];

fn tail_punct(t: &str) -> String {
    let n = t.chars().rev().take_while(|c| PUNCT.contains(c)).count();
    t.chars().skip(t.chars().count() - n).collect()
}

/// VLA 문장에서 거리·각도·수치를 뺀다: 숫자가 든 낱말("2.1", "0.4m", "45°")과 바로 뒤 단위 낱말("m", "degrees")을 지우고
/// 남은 문장부호·매달린 전치사를 정리한다. 수량 낱말("two")은 남긴다.
pub fn strip_numbers(s: &str) -> String {
    let toks: Vec<&str> = s.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        if !(t.chars().any(|c| c.is_ascii_digit()) || t.contains('°')) {
            out.push(t.to_string());
            i += 1;
            continue;
        }
        let mut last = t;
        i += 1;
        if i < toks.len() && UNITS.contains(&toks[i].trim_end_matches(PUNCT).to_lowercase().as_str()) {
            last = toks[i];
            i += 1;
        }
        // 지운 낱말 끝의 쉼표·마침표는 앞 낱말에 붙여 문장 모양을 살린다(소수점 "2." 은 제외)
        let p = tail_punct(last);
        if !p.is_empty() && !last.trim_end_matches(PUNCT).is_empty() {
            if let Some(prev) = out.last_mut() {
                if !prev.ends_with(PUNCT) {
                    prev.push_str(&p);
                }
            }
        }
    }
    let mut s = out.join(" ");
    for (a, b) in [(" ,", ","), (",,", ","), ("()", ""), (" :", ":"), (":,", ":"), (",.", "."), (" .", ".")] {
        while s.contains(a) {
            s = s.replace(a, b);
        }
    }
    // 숫자 앞에 있던 말이 끝에 매달리면 뗀다("move forward by" → "move forward")
    let mut s = s.trim().trim_matches([',', ':', ';']).trim().to_string();
    loop {
        let before = s.len();
        let body = s.trim_end_matches(PUNCT).to_string();
        let end: String = s[body.len()..].to_string();
        let mut b = body.clone();
        for w in [" by", " of", " about", " around", " approximately", " roughly", " exactly", " for", " and", " to"] {
            if let Some(x) = b.strip_suffix(w) {
                b = x.trim_end().to_string();
            }
        }
        let end: String = end.chars().filter(|c| !matches!(c, ',' | ';' | ':')).collect();
        s = format!("{}{}", b.trim_end_matches([',', ';', ':']), end);
        if s.len() == before {
            break;
        }
    }
    s
}

impl Instruction {
    pub fn subtask_text(&self) -> String {
        match crate::vocab::lookup(&self.skill) {
            Some(k) => k.render(&self.names, &self.spatial, self.memory.as_deref()),
            None => clean(&format!("{} {}", self.skill, self.names.join(" "))),
        }
    }

    /// 형식에 맞춰 문장을 만들고 토큰 예산 안으로 줄인다.
    pub fn render(&self, fmt: Format, task_prompt: &str, max_tokens: usize) -> String {
        // ②③ 에는 숫자 거리·각도가 들어가지 않는다(과제 문장 ① 은 공식 문장 그대로, ④ 는 실험 전용)
        let sub = strip_numbers(&self.subtask_text());
        let candidates: Vec<String> = match fmt {
            Format::Task => vec![clean(task_prompt) + "."],
            Format::Subtask => vec![sub.clone()],
            Format::Purpose => {
                let p = strip_numbers(&clean(&self.purpose));
                let e0 = strip_numbers(&clean(&self.expected));
                let e = if e0.trim().is_empty() { sub.clone() } else { e0 };
                let mut v = Vec::new();
                if !p.is_empty() {
                    v.push(format!("Purpose: {p}. Expected action: {e}."));
                }
                v.push(format!("Purpose: {}. Expected action: {sub}.", if p.is_empty() { clean(task_prompt) } else { p }));
                v.push(format!("Expected action: {sub}."));
                v
            }
            Format::Metric => {
                let m = self.metric.map(|m| m.text()).unwrap_or_default();
                if m.is_empty() {
                    vec![sub.clone()]
                } else {
                    vec![format!("{sub}: {m}"), sub.clone()]
                }
            }
        };
        for c in &candidates {
            if tokens_with_margin(c) <= max_tokens {
                return c.clone();
            }
        }
        // 마지막: 단어 단위로 자른다
        let mut out = String::new();
        for w in candidates.last().unwrap().split_whitespace() {
            let t = if out.is_empty() { w.to_string() } else { format!("{out} {w}") };
            if tokens_with_margin(&t) > max_tokens {
                break;
            }
            out = t;
        }
        out
    }

    pub fn same_step(&self, other: &Instruction) -> bool {
        self.skill == other.skill && self.objects == other.objects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn radio_pick() -> Instruction {
        Instruction {
            skill: "pick up from".into(),
            objects: vec!["radio_89".into(), "coffee_table_koagbh_0".into()],
            names: vec!["radio".into(), "coffee table".into()],
            purpose: "hold the radio so its button can be pressed".into(),
            expected: "grasp the radio on the coffee table with one hand and lift it".into(),
            metric: Some(Metric { forward_m: 0.6, left_m: -0.2, turn_deg: 12.0 }),
            budget_steps: 800,
            ..Default::default()
        }
    }

    #[test]
    fn four_formats() {
        let i = radio_pick();
        let task = "Turn on the radio receiver that's on the table in the living room.";
        assert_eq!(i.render(Format::Task, task, 90), task);
        assert_eq!(i.render(Format::Subtask, task, 90), "pick up radio from coffee table");
        assert_eq!(
            i.render(Format::Purpose, task, 90),
            "Purpose: hold the radio so its button can be pressed. Expected action: grasp the radio on the coffee table with one hand and lift it."
        );
        assert_eq!(i.render(Format::Metric, task, 90), "pick up radio from coffee table: go forward 0.6 m, 0.2 m to the right, turn left 12 degrees");
    }

    #[test]
    fn numbers_never_reach_vla_in_subtask_and_purpose() {
        assert_eq!(strip_numbers("go forward 2.1 m, 0.4 m to the left, turn left 30 degrees"), "go forward, to the left, turn left");
        assert_eq!(strip_numbers("move forward by 1.5 meters"), "move forward");
        assert_eq!(strip_numbers("turn 45° to face the radio."), "turn to face the radio.");
        assert_eq!(strip_numbers("walk 0.4m left then grasp the radio"), "walk left then grasp the radio");
        assert_eq!(strip_numbers("pick up the two cans"), "pick up the two cans");
        let mut i = radio_pick();
        i.purpose = "get within 0.7 m of the radio".into();
        i.expected = "walk forward 2 meters and turn right 30 degrees".into();
        let p = i.render(Format::Purpose, "t", 90);
        assert!(!p.chars().any(|c| c.is_ascii_digit()), "{p}");
        assert_eq!(p, "Purpose: get within of the radio. Expected action: walk forward and turn right.");
        // 표시 이름에 숫자가 섞여도(이름표 없는 그래프 노드) 빠진다
        i.names = vec!["object 17".into(), "coffee table".into()];
        assert_eq!(i.render(Format::Subtask, "t", 90), "pick up object from coffee table");
        // 숫자 명령(④)은 실험 전용: 명시했을 때만 숫자가 붙는다
        assert!(i.render(Format::Metric, "t", 90).contains("0.6 m"));
    }

    #[test]
    fn budget_shrinks() {
        let mut i = radio_pick();
        i.purpose = "word ".repeat(200);
        let s = i.render(Format::Purpose, "t", 30);
        assert!(estimate_tokens(&s) <= 30, "{s}");
        assert!(s.contains("Expected action"));
    }

    /// 실제 PaliGemma 토크나이저 토큰 수(sentencepiece 로 잰 값)와 비교:
    /// 여유를 더한 추정은 실제 이상, 그리고 너무 크지 않아야 한다.
    #[test]
    fn estimate_is_conservative() {
        for (real, s) in crate::instruction::CALIBRATION {
            let e = tokens_with_margin(s);
            assert!(e >= *real, "여유 추정 {e} < 실제 {real}: {s}");
            assert!(e <= real + real / 3 + 4, "여유 추정 {e} 이 너무 큼(실제 {real}): {s}");
            assert!(estimate_tokens(s) + 3 >= *real);
        }
    }
}

/// (실제 토큰 수, 문장). 잰 방법: docs/에이전트_설계.md 의 토큰 예산 절(556문장 중 추정이 모자랐던 것 포함).
pub const CALIBRATION: &[(usize, &str)] = &[
    (16, "Turn on the radio receiver that's on the table in the living room."),
    (6, "pick up radio from coffee table"),
    (23, "Purpose: turn on the radio so it plays. Expected action: walk to the coffee table and face the radio."),
    (27, "move to radio: go forward 2.1 m, 0.4 m to the left, turn left 30 degrees"),
    (53, "Take the four mousetraps from the cabinet in the bathroom and place them on the bathroom floor. Make sure all four end up on the same floor surface, and ensure that at least two of them are either under or directly next to the same bathroom sink."),
    (5, "move to mousetrap"),
    (21, "Purpose: progress the task 'setting mousetraps'. Expected action: move to mousetrap."),
    (29, "move to mousetrap: go forward 1.5 m, 0.3 m to the right, turn left 20 degrees"),
    (8, "pick up mousetrap from bottom cabinet"),
    (32, "pick up mousetrap from bottom cabinet: go forward 1.5 m, 0.3 m to the right, turn left 20 degrees"),
];

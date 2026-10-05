//! 계획 = 체크리스트. 처음엔 시연 대표 순서(과제 카드 priors)로 채우고, LLM 이 `set_plan` 으로 고친다(재계획).
//! 표기: [done] 끝남, [doing] 지금, [todo] 아직, [failed] 실패, [skip] 건너뜀 — 2위 Comet 계획기의 [o]/[-]/[x] 를
//! 작은 모델이 헷갈리지 않게 낱말로 바꿨다(refs/openpi-comet/src/openpi/shared/client.py:293-314).

use crate::catalog::{PriorStep, SeqPrior};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Todo,
    Doing,
    Done,
    Failed,
    Skip,
}

impl Status {
    pub fn tag(self) -> &'static str {
        match self {
            Status::Todo => "[todo]",
            Status::Doing => "[doing]",
            Status::Done => "[done]",
            Status::Failed => "[failed]",
            Status::Skip => "[skip]",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanStep {
    pub id: u32,
    pub skill: String,
    pub objects: Vec<String>,
    #[serde(default)]
    pub memory: Option<String>,
    #[serde(default)]
    pub spatial: Vec<String>,
    pub status: Status,
    #[serde(default)]
    pub attempts: u32,
}

impl PlanStep {
    pub fn text(&self) -> String {
        PriorStep { skill: self.skill.clone(), objects: self.objects.iter().map(|o| crate::util::display_name(o)).collect(), memory: self.memory.clone(), spatial: self.spatial.clone() }.text()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Plan {
    pub steps: Vec<PlanStep>,
    pub revision: u32,
    pub source: String,
    next_id: u32,
}

impl Plan {
    pub fn from_prior(p: Option<&SeqPrior>) -> Plan {
        let mut plan = Plan { source: "demo prior".into(), ..Default::default() };
        if let Some(p) = p {
            for s in &p.steps {
                plan.push(&s.skill, s.objects.clone(), s.memory.clone(), s.spatial.clone());
            }
        }
        plan
    }

    fn push(&mut self, skill: &str, objects: Vec<String>, memory: Option<String>, spatial: Vec<String>) {
        self.next_id += 1;
        self.steps.push(PlanStep { id: self.next_id, skill: skill.to_string(), objects, memory, spatial, status: Status::Todo, attempts: 0 });
    }

    /// 재계획: 끝난 단계는 남기고 나머지를 새 목록으로 바꾼다.
    pub fn replace_remaining(&mut self, new: Vec<(String, Vec<String>, Option<String>, Vec<String>)>) {
        self.steps.retain(|s| matches!(s.status, Status::Done | Status::Doing));
        for s in self.steps.iter_mut() {
            if s.status == Status::Doing {
                s.status = Status::Skip;
            }
        }
        for (k, o, m, sp) in new {
            self.push(&k, o, m, sp);
        }
        self.revision += 1;
        self.source = "llm".into();
    }

    pub fn next_todo(&self) -> Option<&PlanStep> {
        self.steps.iter().find(|s| matches!(s.status, Status::Todo | Status::Failed))
    }

    pub fn doing(&self) -> Option<&PlanStep> {
        self.steps.iter().find(|s| s.status == Status::Doing)
    }

    pub fn set_status(&mut self, id: u32, st: Status) {
        if let Some(s) = self.steps.iter_mut().find(|s| s.id == id) {
            s.status = st;
        }
    }

    /// 지금 내린 지시에 맞는 단계를 [doing] 으로. 같은 단계 이름 + 첫 물체 종류가 맞는 첫 미완료 단계.
    pub fn start_matching(&mut self, skill: &str, first_obj_name: &str) -> Option<u32> {
        for s in self.steps.iter_mut() {
            if s.status == Status::Doing {
                s.status = Status::Todo;
            }
        }
        let target = crate::util::norm_category(first_obj_name);
        let pick = self
            .steps
            .iter()
            .position(|s| {
                matches!(s.status, Status::Todo | Status::Failed)
                    && s.skill == skill
                    && s.objects.first().map(|o| {
                        let n = crate::util::norm_category(o);
                        n == target || crate::graph::text_score(&n, &target) >= 0.5
                    }).unwrap_or(true)
            })
            .or_else(|| self.steps.iter().position(|s| matches!(s.status, Status::Todo | Status::Failed) && s.skill == skill));
        pick.map(|i| {
            self.steps[i].status = Status::Doing;
            self.steps[i].attempts += 1;
            self.steps[i].id
        })
    }

    pub fn render(&self, max_lines: usize) -> String {
        if self.steps.is_empty() {
            return "(empty — call set_plan)".into();
        }
        // 끝난 것은 뒤쪽 몇 개만, 남은 것은 앞쪽부터
        let first_open = self.steps.iter().position(|s| !matches!(s.status, Status::Done | Status::Skip)).unwrap_or(self.steps.len());
        let start = first_open.saturating_sub(3);
        let mut out = Vec::new();
        if start > 0 {
            out.push(format!("({} earlier steps done)", start));
        }
        for s in self.steps.iter().skip(start).take(max_lines) {
            let extra = if s.attempts > 1 { format!(" (attempts {})", s.attempts) } else { String::new() };
            out.push(format!("{} #{} {} | {} | {}{}", s.status.tag(), s.id, s.skill, s.objects.join(", "), s.memory.clone().unwrap_or_default(), extra).trim_end_matches(" | ").to_string());
        }
        let shown = start + max_lines;
        if self.steps.len() > shown {
            out.push(format!("(+{} more steps)", self.steps.len() - shown));
        }
        out.join("\n")
    }

    pub fn progress(&self) -> (usize, usize) {
        (self.steps.iter().filter(|s| s.status == Status::Done).count(), self.steps.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_then_match() {
        let prior = SeqPrior {
            kind: "top".into(),
            count: 10,
            freq: 0.5,
            steps: vec![
                PriorStep { skill: "move to".into(), objects: vec!["radio".into()], memory: None, spatial: vec![] },
                PriorStep { skill: "pick up from".into(), objects: vec!["radio".into(), "coffee table".into()], memory: None, spatial: vec![] },
            ],
        };
        let mut p = Plan::from_prior(Some(&prior));
        assert_eq!(p.start_matching("move to", "radio"), Some(1));
        p.set_status(1, Status::Done);
        assert_eq!(p.next_todo().unwrap().id, 2);
        assert!(p.render(10).contains("[done] #1 move to | radio"));
        p.replace_remaining(vec![("press".into(), vec!["radio".into()], None, vec![])]);
        assert_eq!(p.steps.len(), 2);
        assert_eq!(p.next_todo().unwrap().skill, "press");
    }
}

//! 단계 어휘 35종 — 시연 주석 `skill_annotation[].skill_description` 그대로
//! (`data/2026-challenge-demos/annotations/skill_summary.csv`).
//!
//! 문장 틀은 주석의 물체 칸 순서(`object_id` 목록)를 따른다. 예: pick up from = [집을 것, 받침],
//! place on next to = [놓을 것, 받침, 기준 물체], chop = [칼, 자를 것]. 공간 수식어(`spatial_prefix`)는 칸별로 붙는다.
//! 틀 문법: `{oN}` N번 물체, `{sN}` N번 칸 공간 수식어(접두형: "left " 등, 없으면 빈칸),
//! `{sN:기본}` 관계형 수식어(없으면 기본 말), `{m}` 기억 수식어("the other "), `{b}` back("back ").

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Navigation,
    Uncoordinated,
    Coordinated,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Navigation => "navigation",
            Kind::Uncoordinated => "uncoordinated",
            Kind::Coordinated => "coordinated",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Skill {
    pub name: &'static str,
    pub id: u32,
    pub kind: Kind,
    /// (최소 물체 수, 틀) — 물체 수에 맞는 것 중 가장 긴 것을 쓴다.
    pub templates: &'static [(usize, &'static str)],
    /// 시연 평균 길이(초), skill_summary.csv
    pub mean_s: f64,
    pub slots: &'static [&'static str],
}

use Kind::*;

pub const SKILLS: &[Skill] = &[
    Skill { name: "move to", id: 1, kind: Navigation, templates: &[(1, "move to {m}{o0}")], mean_s: 18.64, slots: &["target"] },
    Skill { name: "pick up from", id: 2, kind: Uncoordinated, templates: &[(1, "pick up {m}{o0}"), (2, "pick up {m}{o0} from {o1}")], mean_s: 17.47, slots: &["object", "support"] },
    Skill { name: "place in", id: 4, kind: Uncoordinated, templates: &[(2, "place {m}{o0} {b}in the {s1}{o1}")], mean_s: 13.31, slots: &["object", "container"] },
    Skill { name: "place on", id: 3, kind: Uncoordinated, templates: &[(2, "place {m}{o0} {b}on {s1}{o1}")], mean_s: 13.01, slots: &["object", "support"] },
    Skill { name: "push to", id: 90, kind: Uncoordinated, templates: &[(2, "push {m}{o0} {s1:to} {o1}")], mean_s: 25.48, slots: &["object", "destination"] },
    Skill { name: "open door", id: 10, kind: Coordinated, templates: &[(1, "open the {s0}door of {o0}")], mean_s: 29.48, slots: &["furniture"] },
    Skill { name: "place on next to", id: 91, kind: Uncoordinated, templates: &[(2, "place {m}{o0} on {o1}"), (3, "place {m}{o0} on {o1} {s2:next to} {o2}")], mean_s: 12.91, slots: &["object", "support", "reference"] },
    Skill { name: "close door", id: 12, kind: Coordinated, templates: &[(1, "close the {s0}door of {o0}")], mean_s: 19.99, slots: &["furniture"] },
    Skill { name: "chop", id: 34, kind: Coordinated, templates: &[(1, "chop {m}{o0}"), (2, "chop {m}{o1} with {o0}")], mean_s: 4.82, slots: &["tool", "object"] },
    Skill { name: "sweep surface", id: 50, kind: Coordinated, templates: &[(1, "sweep {o0}"), (2, "sweep {m}{o1} with {o0}")], mean_s: 8.21, slots: &["tool", "surface"] },
    Skill { name: "pour", id: 28, kind: Coordinated, templates: &[(1, "pour {o0}"), (2, "pour {o0} from {o1}"), (3, "pour {o0} from {o1} onto {o2}")], mean_s: 11.12, slots: &["contents", "container", "destination"] },
    Skill { name: "turn on switch", id: 69, kind: Coordinated, templates: &[(1, "turn on {m}{o0}")], mean_s: 11.97, slots: &["object"] },
    Skill { name: "hand over", id: 5, kind: Coordinated, templates: &[(1, "hand over {o0}"), (3, "hand over {o0} from the {o1} hand to the {o2} hand")], mean_s: 12.63, slots: &["object", "from_hand", "to_hand"] },
    Skill { name: "close lid", id: 14, kind: Coordinated, templates: &[(1, "close the lid of {m}{o0}")], mean_s: 19.19, slots: &["object"] },
    Skill { name: "turn off switch", id: 70, kind: Coordinated, templates: &[(1, "turn off {m}{o0}")], mean_s: 9.54, slots: &["object"] },
    Skill { name: "open lid", id: 13, kind: Coordinated, templates: &[(1, "open the lid of {m}{o0}")], mean_s: 33.72, slots: &["object"] },
    Skill { name: "wipe hard", id: 46, kind: Coordinated, templates: &[(1, "wipe {o0}"), (2, "wipe {m}{o1} with {o0}")], mean_s: 22.11, slots: &["tool", "surface"] },
    Skill { name: "turn to", id: 93, kind: Uncoordinated, templates: &[(1, "turn {o0}"), (2, "turn {o0} {s1:to face} {o1}")], mean_s: 17.63, slots: &["object", "reference"] },
    Skill { name: "insert", id: 6, kind: Coordinated, templates: &[(2, "insert {m}{o0} into {o1}")], mean_s: 14.67, slots: &["object", "container"] },
    Skill { name: "close drawer", id: 11, kind: Uncoordinated, templates: &[(1, "close the {s0}drawer of {o0}")], mean_s: 11.62, slots: &["furniture"] },
    Skill { name: "open drawer", id: 9, kind: Uncoordinated, templates: &[(1, "open the {s0}drawer of {o0}")], mean_s: 18.02, slots: &["furniture"] },
    Skill { name: "spray", id: 95, kind: Coordinated, templates: &[(1, "spray {o0}"), (2, "spray {m}{o1} with {o0}")], mean_s: 52.22, slots: &["tool", "target"] },
    Skill { name: "tip over", id: 99, kind: Uncoordinated, templates: &[(1, "tip over {m}{o0}")], mean_s: 9.46, slots: &["object"] },
    Skill { name: "hold", id: 94, kind: Uncoordinated, templates: &[(1, "hold {o0}")], mean_s: 9.42, slots: &["object"] },
    Skill { name: "release", id: 8, kind: Uncoordinated, templates: &[(1, "release {o0}")], mean_s: 5.44, slots: &["object"] },
    Skill { name: "attach", id: 19, kind: Coordinated, templates: &[(2, "attach {o0} to {o1}")], mean_s: 13.19, slots: &["object", "target"] },
    Skill { name: "place in next to", id: 92, kind: Uncoordinated, templates: &[(2, "place {o0} in the {s1}{o1}"), (3, "place {o0} in the {s1}{o1} {s2:next to} {o2}")], mean_s: 12.54, slots: &["object", "container", "reference"] },
    Skill { name: "sweep off", id: 102, kind: Coordinated, templates: &[(2, "sweep {m}{o0} off {o1}")], mean_s: 5.63, slots: &["objects", "surface"] },
    Skill { name: "place under", id: 98, kind: Uncoordinated, templates: &[(2, "place {m}{o0} under {o1}")], mean_s: 17.72, slots: &["object", "reference"] },
    Skill { name: "pull tray", id: 101, kind: Coordinated, templates: &[(1, "pull out the {s0}tray of {o0}")], mean_s: 27.2, slots: &["appliance"] },
    Skill { name: "press", id: 67, kind: Coordinated, templates: &[(1, "press {o0}")], mean_s: 9.69, slots: &["object"] },
    Skill { name: "ignite", id: 88, kind: Coordinated, templates: &[(1, "ignite {o0}"), (2, "ignite {m}{o1} with {o0}")], mean_s: 5.33, slots: &["tool", "object"] },
    Skill { name: "hang", id: 61, kind: Coordinated, templates: &[(2, "hang {o0} on {o1}")], mean_s: 24.53, slots: &["object", "support"] },
    Skill { name: "push tray", id: 100, kind: Coordinated, templates: &[(1, "push in the {s0}tray of {o0}")], mean_s: 7.0, slots: &["appliance"] },
    Skill { name: "lift", id: 103, kind: Uncoordinated, templates: &[(1, "lift {o0}")], mean_s: 9.77, slots: &["object"] },
];

pub fn names() -> Vec<&'static str> {
    SKILLS.iter().map(|s| s.name).collect()
}

/// LLM 이 낸 단계 이름을 어휘로 맞춘다(대소문자·밑줄·흔한 동의어).
pub fn lookup(raw: &str) -> Option<&'static Skill> {
    let s = raw.trim().to_lowercase().replace(['_', '-'], " ");
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(k) = SKILLS.iter().find(|k| k.name == s) {
        return Some(k);
    }
    let alias = match s.as_str() {
        "go to" | "navigate to" | "walk to" | "move" | "navigate" | "approach" | "goto" => "move to",
        "pick up" | "pick" | "grasp" | "grab" | "pickup" | "pick up object" => "pick up from",
        "place" | "put on" | "put" | "place onto" | "put down" => "place on",
        "put in" | "put into" | "place into" | "drop in" => "place in",
        "place next to" | "put next to" => "place on next to",
        "turn on" | "switch on" | "toggle on" => "turn on switch",
        "turn off" | "switch off" | "toggle off" => "turn off switch",
        "push" => "push to",
        "open" => "open door",
        "close" => "close door",
        "cut" | "slice" => "chop",
        "sweep" => "sweep surface",
        "wipe" | "clean" => "wipe hard",
        "handover" => "hand over",
        _ => return None,
    };
    SKILLS.iter().find(|k| k.name == alias)
}

/// 접두형 공간 수식어 (물체 이름 앞에 붙음)
pub fn spatial_prefix(s: &str) -> String {
    match s {
        "" => String::new(),
        "high_level" => "upper ".into(),
        "low_level" => "lower ".into(),
        "right_door" | "right" => "right ".into(),
        "left_door" | "left" => "left ".into(),
        "top" => "top ".into(),
        "bottom" => "bottom ".into(),
        other => format!("{} ", other.replace('_', " ")),
    }
}

/// 관계형 공간 수식어 (두 물체 사이 말)
pub fn spatial_relation(s: &str, default: &str) -> String {
    match s {
        "" => default.to_string(),
        "to_the_edge_of" => "to the edge of".into(),
        "in_front_of" => "in front of".into(),
        "right" => "to the right of".into(),
        "left" => "to the left of".into(),
        "face_away" => "to face away from".into(),
        "behind" => "behind".into(),
        other => other.replace('_', " "),
    }
}

impl Skill {
    /// 물체 표시 이름·공간 수식어·기억 수식어로 문장을 만든다.
    pub fn render(&self, objs: &[String], spatial: &[String], memory: Option<&str>) -> String {
        let n = objs.len();
        let tmpl = self
            .templates
            .iter()
            .filter(|(min, _)| *min <= n.max(1))
            .max_by_key(|(min, _)| *min)
            .map(|(_, t)| *t)
            .unwrap_or(self.templates[0].1);
        let m = match memory {
            Some("the other") => "the other ",
            Some("the same") => "the same ",
            _ => "",
        };
        let b = if memory == Some("back") { "back " } else { "" };
        let mut out = String::with_capacity(tmpl.len() + 32);
        let bytes = tmpl.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'{' {
                let end = tmpl[i..].find('}').map(|e| i + e).unwrap_or(bytes.len() - 1);
                let tok = &tmpl[i + 1..end];
                if tok == "m" {
                    out.push_str(m);
                } else if tok == "b" {
                    out.push_str(b);
                } else if let Some(k) = tok.strip_prefix('o') {
                    let k: usize = k.parse().unwrap_or(0);
                    out.push_str(objs.get(k).map(|s| s.as_str()).unwrap_or("it"));
                } else if let Some(rest) = tok.strip_prefix('s') {
                    let (k, def) = match rest.split_once(':') {
                        Some((k, d)) => (k, Some(d)),
                        None => (rest, None),
                    };
                    let k: usize = k.parse().unwrap_or(0);
                    let sp = spatial.get(k).map(|s| s.as_str()).unwrap_or("");
                    match def {
                        Some(d) => out.push_str(&spatial_relation(sp, d)),
                        None => out.push_str(&spatial_prefix(sp)),
                    }
                }
                i = end + 1;
            } else {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
        out.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// 시연 평균 길이 기반 기본 스텝 예산(30 Hz × 평균 × 3, 최소 150). 주석에서 뽑은 p90 이 있으면 그것을 쓴다(catalog).
    pub fn default_budget(&self) -> u64 {
        ((self.mean_s * 30.0 * 3.0) as u64).max(150)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn count_is_35() {
        assert_eq!(SKILLS.len(), 35);
        let mut n = names();
        n.sort();
        n.dedup();
        assert_eq!(n.len(), 35);
    }

    #[test]
    fn render_examples() {
        let k = lookup("place on").unwrap();
        assert_eq!(k.render(&s(&["radio", "coffee table"]), &[], Some("back")), "place radio back on coffee table");
        assert_eq!(lookup("pick up from").unwrap().render(&s(&["radio", "coffee table"]), &[], None), "pick up radio from coffee table");
        assert_eq!(
            lookup("push to").unwrap().render(&s(&["board game", "bed"]), &s(&["", "to_the_edge_of"]), Some("the other")),
            "push the other board game to the edge of bed"
        );
        assert_eq!(lookup("open door").unwrap().render(&s(&["fridge"]), &s(&["right_door"]), None), "open the right door of fridge");
        assert_eq!(
            lookup("place on next to").unwrap().render(&s(&["cauldron", "floors", "coffee table"]), &s(&["", "", "in_front_of"]), None),
            "place cauldron on floors in front of coffee table"
        );
        assert_eq!(lookup("chop").unwrap().render(&s(&["parer", "beet"]), &[], Some("the other")), "chop the other beet with parer");
        assert_eq!(lookup("GO_TO").unwrap().name, "move to");
    }
}

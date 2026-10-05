//! BDDL(과제 정의) 읽기. 평가 때도 같은 정의를 쓴다(규칙 원문: 과제 정보는 허용 관측).
//! `BEHAVIOR-1K/bddl3/bddl/activity_definitions/<과제>/problem0.bddl` 의 :objects, :init, :goal 을 읽어
//! LLM 이 읽기 쉬운 줄로 바꾸고, 처음 자리(ontop/inside/inroom)를 뽑는다(`back` 풀 때 보조).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub enum Sx {
    Atom(String),
    List(Vec<Sx>),
}

impl Sx {
    pub fn atom(&self) -> Option<&str> {
        match self {
            Sx::Atom(a) => Some(a),
            _ => None,
        }
    }
    pub fn list(&self) -> Option<&[Sx]> {
        match self {
            Sx::List(l) => Some(l),
            _ => None,
        }
    }
    pub fn head(&self) -> Option<&str> {
        self.list().and_then(|l| l.first()).and_then(|h| h.atom())
    }
}

pub fn parse(text: &str) -> Result<Sx, String> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        match c {
            '(' | ')' => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
                toks.push(c.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
            }
            ';' => {
                // 주석은 없다고 보지만 안전하게 토큰으로 둔다
                cur.push(c)
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    let mut stack: Vec<Vec<Sx>> = vec![Vec::new()];
    for t in toks {
        match t.as_str() {
            "(" => stack.push(Vec::new()),
            ")" => {
                let l = stack.pop().ok_or("괄호 짝이 안 맞음")?;
                stack.last_mut().ok_or("괄호 짝이 안 맞음")?.push(Sx::List(l));
            }
            _ => stack.last_mut().ok_or("괄호 짝이 안 맞음")?.push(Sx::Atom(t)),
        }
    }
    if stack.len() != 1 {
        return Err("괄호가 닫히지 않음".into());
    }
    let mut top = stack.pop().unwrap();
    if top.len() == 1 {
        Ok(top.pop().unwrap())
    } else {
        Ok(Sx::List(top))
    }
}

/// `?radio_receiver.n.01_1` → `radio_receiver_1`, `can__of__soda.n.01` → `can_of_soda`
pub fn short(name: &str) -> String {
    let n = name.trim_start_matches('?');
    if let Some(p) = n.find(".n.") {
        let base = n[..p].replace("__", "_");
        // 뒤의 ".n.01_3" 에서 인스턴스 번호
        let tail = &n[p + 3..];
        match tail.split_once('_') {
            Some((_, inst)) => format!("{base}_{inst}"),
            None => base,
        }
    } else {
        n.to_string()
    }
}

fn render(sx: &Sx) -> String {
    match sx {
        Sx::Atom(a) => short(a),
        Sx::List(l) => {
            let head = l.first().and_then(|h| h.atom()).unwrap_or("");
            match head {
                "and" => l[1..].iter().map(render).collect::<Vec<_>>().join(" AND "),
                "or" => format!("({})", l[1..].iter().map(render).collect::<Vec<_>>().join(" OR ")),
                "not" => format!("NOT {}", l.get(1).map(render).unwrap_or_default()),
                "imply" => format!("IF {} THEN {}", l.get(1).map(render).unwrap_or_default(), l.get(2).map(render).unwrap_or_default()),
                "forall" | "exists" | "forpairs" | "forn" => {
                    let (n_arg, body_idx) = if head == "forn" { (l.get(1).map(render), 3) } else { (None, 2) };
                    let var = l.get(if head == "forn" { 2 } else { 1 }).and_then(|v| v.list()).and_then(|v| v.first()).map(render).unwrap_or_default();
                    let body = l.get(body_idx).map(render).unwrap_or_default();
                    match head {
                        "forall" => format!("for every {var}: {body}"),
                        "exists" => format!("for some {var}: {body}"),
                        "forn" => format!("for {} of {var}: {body}", n_arg.unwrap_or_default()),
                        _ => format!("for pairs {var}: {body}"),
                    }
                }
                pred => format!("{pred}({})", l[1..].iter().map(render).collect::<Vec<_>>().join(", ")),
            }
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Problem {
    /// 인스턴스 → 종류(synset)
    pub objects: Vec<(String, String)>,
    /// 처음 사실(읽기 쉬운 줄)
    pub init: Vec<String>,
    /// 처음 자리: (물체, 관계, 받침) — ontop/inside/under/nextto/overlaid
    pub placements: Vec<(String, String, String)>,
    /// 방: (물체, 방)
    pub rooms: Vec<(String, String)>,
    /// 목표 조건: 맨 위 and 를 풀어 한 줄씩
    pub goals: Vec<String>,
}

pub fn problem(text: &str) -> Result<Problem, String> {
    let sx = parse(text)?;
    let items = sx.list().ok_or("최상위가 목록이 아님")?;
    let mut p = Problem::default();
    for it in items {
        match it.head() {
            Some(":objects") => {
                let l = it.list().unwrap();
                let mut pending: Vec<String> = Vec::new();
                let mut i = 1;
                while i < l.len() {
                    let a = l[i].atom().unwrap_or("");
                    if a == "-" {
                        let ty = l.get(i + 1).and_then(|x| x.atom()).unwrap_or("").to_string();
                        for o in pending.drain(..) {
                            p.objects.push((o, ty.clone()));
                        }
                        i += 2;
                    } else {
                        pending.push(a.to_string());
                        i += 1;
                    }
                }
            }
            Some(":init") => {
                for f in &it.list().unwrap()[1..] {
                    p.init.push(render(f));
                    if let Some(l) = f.list() {
                        let head = l.first().and_then(|h| h.atom()).unwrap_or("");
                        let args: Vec<String> = l[1..].iter().filter_map(|x| x.atom()).map(short).collect();
                        match (head, args.len()) {
                            ("inroom", 2) => p.rooms.push((args[0].clone(), args[1].clone())),
                            ("ontop" | "inside" | "under" | "nextto" | "overlaid" | "attached" | "draped", 2) => {
                                p.placements.push((args[0].clone(), head.to_string(), args[1].clone()))
                            }
                            _ => {}
                        }
                    }
                }
            }
            Some(":goal") => {
                if let Some(g) = it.list().and_then(|l| l.get(1)) {
                    if g.head() == Some("and") {
                        for c in &g.list().unwrap()[1..] {
                            p.goals.push(render(c));
                        }
                    } else {
                        p.goals.push(render(g));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(p)
}

impl Problem {
    /// 물체(짧은 이름)의 처음 받침
    pub fn initial_support(&self, obj: &str) -> Option<(&str, &str)> {
        self.placements.iter().find(|(o, _, _)| o == obj).map(|(_, r, s)| (r.as_str(), s.as_str()))
    }
    /// 물체가 처음 있던 방(받침을 따라 올라가며)
    pub fn room_of(&self, obj: &str) -> Option<&str> {
        let mut cur = obj.to_string();
        for _ in 0..8 {
            if let Some((_, r)) = self.rooms.iter().find(|(o, _)| *o == cur) {
                return Some(r.as_str());
            }
            match self.initial_support(&cur) {
                Some((_, s)) => cur = s.to_string(),
                None => return None,
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RADIO: &str = "(define (problem turning_on_radio-0)
    (:domain behavior-1k)
    (:objects
     	radio_receiver.n.01_1 - radio_receiver.n.01
    	table.n.02_1 - table.n.02
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    (:init (not (toggled_on radio_receiver.n.01_1)) (ontop radio_receiver.n.01_1 table.n.02_1)
        (inroom table.n.02_1 living_room) (inroom floor.n.01_1 living_room) (ontop agent.n.01_1 floor.n.01_1))
    (:goal (and (toggled_on ?radio_receiver.n.01_1)))
)";

    #[test]
    fn radio() {
        let p = problem(RADIO).unwrap();
        assert_eq!(p.goals, vec!["toggled_on(radio_receiver_1)"]);
        assert_eq!(p.initial_support("radio_receiver_1"), Some(("ontop", "table_1")));
        assert_eq!(p.room_of("radio_receiver_1"), Some("living_room"));
        assert_eq!(p.objects.len(), 4);
    }

    #[test]
    fn forall() {
        let g = "(define (problem x) (:goal (and (forall (?can__of__soda.n.01 - can__of__soda.n.01) (inside ?can__of__soda.n.01 ?ashcan.n.01_1)))))";
        let p = problem(g).unwrap();
        assert_eq!(p.goals, vec!["for every can_of_soda: inside(can_of_soda, ashcan_1)"]);
    }
}

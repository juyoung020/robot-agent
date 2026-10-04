//! BDDL problem parser and goal grounding.
//!
//! Two views of the same goal:
//! * `compile_goal` — the device literal table: every `or` branch is kept (tagged with its group /
//!   branch), `exists`/`forn`/`forpairs` become variables with candidate lists (forn/forpairs
//!   variables carry a "distinct" group: members must bind to different objects).
//! * the doc count (CURRICULUM_BEHAVIOR2026 1.3): literals whose `or` tags are all branch 0
//!   ("or = first option"), each `exists` = one literal, `forn n` = n, `forpairs` = min(|A|,|B|).
//!   Those literals carry `DOC` = true. This is exactly the one-off script behind the doc table.

#[derive(Clone, Debug)]
pub enum Sx {
    A(String),
    L(Vec<Sx>),
}

impl Sx {
    pub fn atom(&self) -> Option<&str> {
        if let Sx::A(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn list(&self) -> Option<&[Sx]> {
        if let Sx::L(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn head(&self) -> Option<&str> {
        self.list().and_then(|v| v.first()).and_then(|h| h.atom())
    }
}

pub fn tokenize(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in src.lines() {
        let line = match line.find(';') {
            Some(i) => &line[..i],
            None => line,
        };
        let mut cur = String::new();
        for ch in line.chars() {
            if ch == '(' || ch == ')' || ch.is_whitespace() {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                if ch == '(' || ch == ')' {
                    out.push(ch.to_string());
                }
            } else {
                cur.push(ch);
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

fn parse_at(t: &[String], i: &mut usize) -> Result<Sx, String> {
    let tok = t.get(*i).ok_or("unexpected end of BDDL")?;
    if tok == "(" {
        *i += 1;
        let mut v = Vec::new();
        loop {
            match t.get(*i).map(|s| s.as_str()) {
                None => return Err("unbalanced '(' in BDDL".into()),
                Some(")") => {
                    *i += 1;
                    return Ok(Sx::L(v));
                }
                _ => v.push(parse_at(t, i)?),
            }
        }
    } else if tok == ")" {
        Err("unexpected ')' in BDDL".into())
    } else {
        *i += 1;
        Ok(Sx::A(tok.clone()))
    }
}

pub struct Problem {
    pub objects: Vec<(String, String)>,       // (instance, synset) in file order
    pub objs: Vec<(String, Vec<String>)>,     // synset -> instances (first-seen synset order)
    pub init: Vec<Sx>,
    pub goal: Sx,
}

impl Problem {
    pub fn insts_of(&self, syn: &str) -> &[String] {
        self.objs.iter().find(|(s, _)| s == syn).map(|(_, v)| v.as_slice()).unwrap_or(&[])
    }
}

pub fn parse_problem(src: &str) -> Result<Problem, String> {
    let t = tokenize(src);
    let mut i = 0;
    let tree = parse_at(&t, &mut i)?;
    let top = tree.list().ok_or("BDDL root is not a list")?;
    let sec = |name: &str| -> Option<&[Sx]> {
        top.iter().find(|x| x.head() == Some(name)).map(|x| &x.list().unwrap()[1..])
    };
    let osec = sec(":objects").ok_or("no :objects")?;
    let mut objects = Vec::new();
    let mut objs: Vec<(String, Vec<String>)> = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    let mut k = 0;
    while k < osec.len() {
        let a = osec[k].atom().ok_or(":objects has a list")?;
        if a == "-" {
            let syn = osec.get(k + 1).and_then(|x| x.atom()).ok_or("':objects' ends with '-'")?.to_string();
            for b in buf.drain(..) {
                objects.push((b.clone(), syn.clone()));
                match objs.iter_mut().find(|(s, _)| *s == syn) {
                    Some((_, v)) => v.push(b),
                    None => objs.push((syn.clone(), vec![b])),
                }
            }
            k += 2;
        } else {
            buf.push(a.to_string());
            k += 1;
        }
    }
    if !buf.is_empty() {
        return Err(format!(":objects has untyped names {buf:?}"));
    }
    let init = sec(":init").ok_or("no :init")?.to_vec();
    let goal = sec(":goal").and_then(|g| g.first()).ok_or("no :goal")?.clone();
    Ok(Problem { objects, objs, init, goal })
}

const CONN: [&str; 9] = ["and", "or", "not", "forall", "exists", "forn", "forpairs", "fornpairs", "imply"];

/// Every atom anywhere in `e` (also under `not`/`or`), as flat strings. Lists inside an atom
/// become "<list>". Same as the doc script's `atoms`.
pub fn atoms(e: &Sx, acc: &mut Vec<Vec<String>>) {
    if let Sx::L(v) = e {
        if let Some(h) = v.first().and_then(|x| x.atom()) {
            if !CONN.contains(&h) && !h.starts_with('?') {
                acc.push(v.iter().map(|x| x.atom().unwrap_or("<list>").to_string()).collect());
            }
        }
        for x in v.iter().skip(1) {
            atoms(x, acc);
        }
    }
}

/// `cabinet.n.01_1` / `?cabinet.n.01` -> `cabinet.n.01`
pub fn syn_of(name: &str) -> String {
    let n = name.trim_start_matches('?');
    if let Some(p) = n.rfind('_') {
        let tail = &n[p + 1..];
        let head = &n[..p];
        if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) && is_synset(head) {
            return head.to_string();
        }
    }
    n.to_string()
}

fn is_synset(s: &str) -> bool {
    // ".+\.n\.\d+$"
    if let Some(p) = s.rfind(".n.") {
        p > 0 && s.len() > p + 3 && s[p + 3..].bytes().all(|b| b.is_ascii_digit())
    } else {
        false
    }
}

// ---------------------------------------------------------------- doc predicate count (analyze)
/// Predicate multiset of the goal with `or` = the option with the fewest literals (first on ties).
/// Only its keys are used (skills column), exactly like the doc script's `count`.
pub fn pred_count(e: &Sx, p: &Problem) -> (i64, Vec<(String, i64)>) {
    fn add(acc: &mut Vec<(String, i64)>, k: &str, v: i64) {
        match acc.iter_mut().find(|(a, _)| a == k) {
            Some((_, x)) => *x += v,
            None => acc.push((k.to_string(), v)),
        }
    }
    let v = match e.list() {
        Some(v) if !v.is_empty() => v,
        _ => return (0, vec![]),
    };
    let h = v[0].atom().unwrap_or("");
    let scale = |(a, b): (i64, Vec<(String, i64)>), k: i64| (a * k, b.into_iter().map(|(x, y)| (x, y * k)).collect());
    let nsyn = |x: &Sx| -> i64 {
        let s = x.list().and_then(|l| l.get(2)).and_then(|a| a.atom()).unwrap_or("");
        let n = p.insts_of(s).len() as i64;
        if n == 0 {
            1
        } else {
            n
        }
    };
    match h {
        "and" => {
            let mut n = 0;
            let mut c = Vec::new();
            for x in &v[1..] {
                let (a, b) = pred_count(x, p);
                n += a;
                for (k, y) in b {
                    add(&mut c, &k, y);
                }
            }
            (n, c)
        }
        "or" => {
            let mut best: Option<(i64, Vec<(String, i64)>)> = None;
            for x in &v[1..] {
                let r = pred_count(x, p);
                if best.as_ref().map_or(true, |b| r.0 < b.0) {
                    best = Some(r);
                }
            }
            best.unwrap_or((0, vec![]))
        }
        "not" => {
            let (a, b) = pred_count(&v[1], p);
            (a, b.into_iter().map(|(k, y)| (format!("not {k}"), y)).collect())
        }
        "forall" => {
            let k = nsyn(&v[1]);
            scale(pred_count(&v[2], p), k)
        }
        "exists" => pred_count(&v[2], p),
        "forn" => {
            let n: i64 = v[1].list().and_then(|l| l[0].atom()).and_then(|s| s.parse().ok()).unwrap_or(1);
            scale(pred_count(&v[3], p), n)
        }
        "forpairs" => {
            let k = nsyn(&v[1]).min(nsyn(&v[2]));
            scale(pred_count(&v[3], p), k)
        }
        "imply" => pred_count(&v[2], p),
        _ => (1, vec![(h.to_string(), 1)]),
    }
}

// ---------------------------------------------------------------- device compile
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arg {
    Obj(u16),  // index into the task's object list (BDDL :objects order)
    Var(u16),  // index into the task's variable table
    Room(u16), // room category id (1-based line of room_categories.txt), inroom only
}

#[derive(Clone, Debug)]
pub struct Lit {
    pub pred: String,
    pub neg: bool,
    pub args: Vec<Arg>,
    pub or_tags: Vec<(u16, u16)>, // (group id, branch) from outermost to innermost
    pub doc: bool,                // counted by the doc (all branches = 0)
}

#[derive(Clone, Debug)]
pub struct Var {
    pub kind: u8, // VAR_EXISTS / VAR_FORN / VAR_PAIRS
    pub group: u16, // distinct group (forn/forpairs), 0xFFFF for exists
    pub cands: Vec<u16>,
}

pub const VAR_EXISTS: u8 = 1;
pub const VAR_FORN: u8 = 2;
pub const VAR_PAIRS: u8 = 3;

pub struct Compiled {
    pub lits: Vec<Lit>,
    pub vars: Vec<Var>,
    pub n_or_groups: u16,
}

#[derive(Clone)]
enum Bind {
    Obj(u16),
    Var(u16),
}

struct Cx<'a> {
    p: &'a Problem,
    lits: Vec<Lit>,
    vars: Vec<Var>,
    groups: u16,
    or_groups: u16,
}

impl<'a> Cx<'a> {
    fn obj(&self, name: &str) -> Result<u16, String> {
        self.p
            .objects
            .iter()
            .position(|(n, _)| n == name)
            .map(|i| i as u16)
            .ok_or(format!("goal references '{name}', not in :objects"))
    }
    fn cands(&self, syn: &str, fallback: bool) -> Result<Vec<u16>, String> {
        let v = self.p.insts_of(syn);
        if v.is_empty() && fallback {
            return Ok(vec![self.obj(&format!("{syn}_1"))?]);
        }
        v.iter().map(|n| self.obj(n)).collect()
    }
    fn new_var(&mut self, kind: u8, group: u16, cands: Vec<u16>) -> u16 {
        self.vars.push(Var { kind, group, cands });
        (self.vars.len() - 1) as u16
    }
    fn rec(&mut self, e: &Sx, env: &[(String, Bind)], neg: bool, tags: &[(u16, u16)]) -> Result<(), String> {
        let v = match e.list() {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(()),
        };
        let h = v[0].atom().ok_or("goal list head is a list")?;
        let var_decl = |x: &Sx| -> Result<(String, String), String> {
            let l = x.list().ok_or("bad variable declaration")?;
            Ok((
                l.first().and_then(|a| a.atom()).ok_or("bad var")?.to_string(),
                l.get(2).and_then(|a| a.atom()).ok_or("bad var type")?.to_string(),
            ))
        };
        let with = |env: &[(String, Bind)], k: &str, b: Bind| {
            let mut e2 = env.to_vec();
            e2.push((k.to_string(), b));
            e2
        };
        match h {
            "and" => {
                for x in &v[1..] {
                    self.rec(x, env, neg, tags)?;
                }
            }
            "or" => {
                if neg {
                    return Err("'or' under 'not' is not supported".into());
                }
                let g = self.or_groups;
                self.or_groups += 1;
                for (b, x) in v[1..].iter().enumerate() {
                    let mut t2 = tags.to_vec();
                    t2.push((g, b as u16));
                    self.rec(x, env, neg, &t2)?;
                }
            }
            "not" => self.rec(&v[1], env, !neg, tags)?,
            "forall" => {
                let (var, syn) = var_decl(&v[1])?;
                for o in self.cands(&syn, true)? {
                    self.rec(&v[2], &with(env, &var, Bind::Obj(o)), neg, tags)?;
                }
            }
            "exists" => {
                if neg {
                    return Err("'exists' under 'not' is not supported".into());
                }
                let (var, syn) = var_decl(&v[1])?;
                let c = self.cands(&syn, true)?;
                let id = self.new_var(VAR_EXISTS, 0xFFFF, c);
                self.rec(&v[2], &with(env, &var, Bind::Var(id)), neg, tags)?;
            }
            "forn" => {
                let n: usize = v[1].list().and_then(|l| l.first()).and_then(|a| a.atom()).and_then(|s| s.parse().ok()).ok_or("bad forn count")?;
                let (var, syn) = var_decl(&v[2])?;
                let c = self.cands(&syn, true)?;
                let g = self.groups;
                self.groups += 1;
                for _ in 0..n {
                    let id = self.new_var(VAR_FORN, g, c.clone());
                    self.rec(&v[3], &with(env, &var, Bind::Var(id)), neg, tags)?;
                }
            }
            "forpairs" => {
                let (va, sa) = var_decl(&v[1])?;
                let (vb, sb) = var_decl(&v[2])?;
                let (ca, cb) = (self.cands(&sa, false)?, self.cands(&sb, false)?);
                let (ga, gb) = (self.groups, self.groups + 1);
                self.groups += 2;
                for _ in 0..ca.len().min(cb.len()) {
                    let ia = self.new_var(VAR_PAIRS, ga, ca.clone());
                    let ib = self.new_var(VAR_PAIRS, gb, cb.clone());
                    let e2 = with(&with(env, &va, Bind::Var(ia)), &vb, Bind::Var(ib));
                    self.rec(&v[3], &e2, neg, tags)?;
                }
            }
            "fornpairs" | "imply" => return Err(format!("'{h}' is not used by any 2026 task; not supported")),
            _ => {
                let mut args = Vec::new();
                for x in &v[1..] {
                    let a = x.atom().ok_or("atom argument is a list")?;
                    let b = env.iter().rev().find(|(k, _)| k == a).map(|(_, b)| b.clone());
                    args.push(match b {
                        Some(Bind::Obj(o)) => Arg::Obj(o),
                        Some(Bind::Var(id)) => Arg::Var(id),
                        None => Arg::Obj(self.obj(a.trim_start_matches('?'))?),
                    });
                }
                let doc = tags.iter().all(|t| t.1 == 0);
                self.lits.push(Lit { pred: h.to_string(), neg, args, or_tags: tags.to_vec(), doc });
            }
        }
        Ok(())
    }
}

pub fn compile_goal(p: &Problem) -> Result<Compiled, String> {
    let mut cx = Cx { p, lits: vec![], vars: vec![], groups: 0, or_groups: 0 };
    cx.rec(&p.goal, &[], false, &[])?;
    Ok(Compiled { lits: cx.lits, vars: cx.vars, n_or_groups: cx.or_groups })
}

/// `:init` -> ground literals. `inroom x <roomtype>` takes a room category as second argument.
pub fn compile_init(p: &Problem, room_cat: &dyn Fn(&str) -> Option<u16>) -> Result<Vec<Lit>, String> {
    let mut out = Vec::new();
    for e in &p.init {
        let (neg, a) = if e.head() == Some("not") { (true, &e.list().unwrap()[1]) } else { (false, e) };
        let v = a.list().ok_or("init item is not a list")?;
        let h = v.first().and_then(|x| x.atom()).ok_or("init head")?;
        if CONN.contains(&h) {
            return Err(format!("init uses '{h}'"));
        }
        let mut args = Vec::new();
        for (k, x) in v[1..].iter().enumerate() {
            let s = x.atom().ok_or("init arg is a list")?;
            if h == "inroom" && k == 1 {
                args.push(Arg::Room(room_cat(s).ok_or(format!("unknown room type '{s}'"))?));
            } else {
                let i = p.objects.iter().position(|(n, _)| n == s).ok_or(format!("init references '{s}', not in :objects"))?;
                args.push(Arg::Obj(i as u16));
            }
        }
        out.push(Lit { pred: h.to_string(), neg, args, or_tags: vec![], doc: false });
    }
    Ok(out)
}

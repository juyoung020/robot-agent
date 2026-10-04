//! LIMO + OMX-F feasibility (CURRICULUM_BEHAVIOR2026 1.3 / 1.5): per goal literal "can this robot
//! make it true, in principle". Port of the doc's one-off script (rules, keyword lists and quirks
//! kept on purpose — the pass criterion is to re-derive the doc's numbers). Geometry mode `Doc`
//! uses the doc's box heights (root z +- bbox_z/2); `World` uses the rotated, offset world box
//! (sensitivity check only).

use crate::bddl::syn_of;
use crate::task::{FeasTask, Info};


#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Th {
    pub name: &'static str,
    pub pick_z: f64,
    pub place_top: f64,
    pub max_mass: f64,
    pub max_w: f64,
    pub tog_z: f64,
}
pub const STRICT: Th = Th { name: "strict", pick_z: 0.45, place_top: 0.50, max_mass: 0.25, max_w: 0.08, tog_z: 0.55 };
pub const LOOSE: Th = Th { name: "loose", pick_z: 0.60, place_top: 0.62, max_mass: 0.5, max_w: 0.10, tog_z: 0.70 };
/// E0 measured (5.3): toggle not measured (kept)
pub const STRICT_E0: Th = Th { name: "strict-E0", pick_z: 0.45, place_top: 0.48, max_mass: 0.25, max_w: 0.04, tog_z: 0.55 };
pub const LOOSE_E0: Th = Th { name: "loose-E0", pick_z: 0.50, place_top: 0.52, max_mass: 0.40, max_w: 0.06, tog_z: 0.70 };

#[derive(Clone, Copy, PartialEq)]
pub enum Geo {
    Doc,
    World,
}

const CLOSED: [&str; 13] = ["cabinet", "refrigerator", "fridge", "microwave", "oven", "washer", "dishwasher", "car", "toolbox", "recycling_bin", "drawer", "hinged_jar", "trunk"];
const SHELF: [&str; 4] = ["bookcase", "shelf", "hall_tree", "rack"];
const REL6: [&str; 6] = ["ontop", "inside", "nextto", "under", "touching", "attached"];

fn any_in(s: &str, l: &[&str]) -> bool {
    l.iter().any(|c| s.contains(c))
}

pub struct TaskFeas {
    pub n: usize,
    pub ok: usize,
    pub why: Vec<(String, usize)>,   // insertion order
    pub kinds: Vec<(String, usize)>, // insertion order
}

fn bump(v: &mut Vec<(String, usize)>, k: &str) {
    match v.iter_mut().find(|(a, _)| a == k) {
        Some((_, n)) => *n += 1,
        None => v.push((k.to_string(), 1)),
    }
}

fn zr(o: &Info, geo: Geo) -> (f64, f64) {
    let bb = o.bbox.unwrap();
    let p = o.pos.unwrap();
    match geo {
        Geo::Doc => (p[2] - bb[2] / 2.0, p[2] + bb[2] / 2.0),
        Geo::World => {
            let h = [bb[0] / 2.0, bb[1] / 2.0, bb[2] / 2.0];
            let (c, wh, _) = crate::util::world_box(p, o.quat, o.off, h);
            (c[2] - wh[2], c[2] + wh[2])
        }
    }
}

pub fn run(t: &FeasTask, th: &Th, geo: Geo) -> TaskFeas {
    let o_of = |inst: &str| -> Option<&Info> { t.get(inst).filter(|v| !v.system && v.bbox.is_some() && v.pos.is_some()) };
    let pick = |inst: &str| -> Vec<&'static str> {
        let Some(o) = o_of(inst) else { return vec!["?"] };
        let bb = o.bbox.unwrap();
        let (z0, _) = zr(o, geo);
        let mut rs = vec![];
        if z0 > th.pick_z {
            rs.push("높이");
        }
        if o.mass.unwrap_or(0.0) > th.max_mass {
            rs.push("무게");
        }
        if bb[0].min(bb[1]) > th.max_w {
            rs.push("폭");
        }
        for a in &t.init_atoms {
            if a[0] == "inside" && a.len() > 2 && a[1] == inst && any_in(&a[2], &CLOSED) {
                rs.push("닫힌곳안");
                break;
            }
        }
        rs
    };
    let dest = |rel: &str, inst: &str| -> Vec<&'static str> {
        let o = o_of(inst);
        let cat = o.map_or(inst, |o| o.cat.as_str());
        if rel == "nextto" || rel == "under" {
            return vec![];
        }
        if rel == "touching" {
            if inst.contains("floor") || cat == "floors" {
                return vec![];
            }
            let shelf = any_in(cat, &SHELF);
            if let Some(o) = o {
                if zr(o, geo).1 > 1.0 && !shelf {
                    return vec!["놓을높이"];
                }
            }
            return if shelf { vec!["선반"] } else { vec![] };
        }
        if rel == "attached" {
            return vec!["붙이기"];
        }
        if inst.contains("floor") || ["floors", "lawn", "driveway"].contains(&cat) {
            return vec![];
        }
        let Some(o) = o else { return vec!["?"] };
        let top = zr(o, geo).1;
        if rel == "inside" {
            if any_in(cat, &CLOSED) || any_in(inst, &CLOSED) {
                return vec!["열기"];
            }
            if any_in(cat, &SHELF) {
                return vec!["선반"];
            }
            return if top <= th.place_top + 0.05 { vec![] } else { vec!["놓을높이"] };
        }
        if rel == "ontop" {
            return if top <= th.place_top { vec![] } else { vec!["놓을높이"] };
        }
        vec![]
    };
    let mut ok = 0;
    let mut why = vec![];
    let mut kinds = vec![];
    for (p, args, neg) in &t.lits {
        bump(&mut kinds, &format!("{}{p}", if *neg { "not " } else { "" }));
        if REL6.contains(&p.as_str()) && args.len() == 2 && !neg {
            let mut best: Option<Vec<&str>> = None;
            for m in &args[0] {
                let pr = pick(m);
                for d in &args[1] {
                    let mut cand = pr.clone();
                    cand.extend(dest(p, d));
                    if best.as_ref().map_or(true, |b| cand.len() < b.len()) {
                        best = Some(cand);
                    }
                }
            }
            let mut best = best.unwrap_or_default();
            if best.is_empty() {
                ok += 1;
            }
            let mut seen = vec![];
            best.retain(|x| if seen.contains(x) { false } else { seen.push(*x); true });
            for x in best {
                bump(&mut why, x);
            }
        } else if p == "toggled_on" {
            let mut best: Option<Vec<&str>> = None;
            for m in &args[0] {
                let z = o_of(m).map_or(9.0, |o| zr(o, geo).1);
                let rs = if z <= th.tog_z { vec![] } else { vec!["토글높이"] };
                if best.as_ref().map_or(true, |b| rs.len() < b.len()) {
                    best = Some(rs);
                }
            }
            let best = best.unwrap_or_default();
            if best.is_empty() {
                ok += 1;
            }
            for x in best {
                bump(&mut why, x);
            }
        } else if p == "open" && *neg {
            bump(&mut why, "닫힘(처음부터 참)");
        } else if *neg && ["inside", "ontop", "touching", "nextto"].contains(&p.as_str()) {
            let pr = args[0].iter().map(|m| pick(m)).min_by_key(|v| v.len()).unwrap_or_default();
            if pr.is_empty() {
                ok += 1;
            }
            for x in pr {
                bump(&mut why, x);
            }
        } else {
            bump(&mut why, p);
        }
    }
    TaskFeas { n: t.lits.len(), ok, why, kinds }
}

impl TaskFeas {
    pub fn q(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.ok as f64 / self.n as f64
        }
    }
    /// most_common(3), stable for ties (insertion order)
    pub fn top3(&self) -> Vec<(String, usize)> {
        let mut v = self.why.clone();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.truncate(3);
        v
    }
}

// ---------------------------------------------------------------- table columns (doc 1.3)
#[allow(dead_code)]
pub struct Cols {
    pub movers: Vec<String>,  // mover categories (sorted, unique)
    pub dests: Vec<String>,   // destination categories not among the shown movers (sorted, unique, first 3)
    pub all_movers: Vec<String>,
    pub skills: u16,          // RASC_SK bits
    pub n_rooms: usize,
    pub maxd: Option<f64>,
}

pub fn cols(t: &FeasTask) -> Cols {
    let objs_of = |s: &str| -> Vec<&Info> { t.info.iter().filter(|(k, v)| syn_of(k) == s && !v.system).map(|(_, v)| v).collect() };
    let mut movers: Vec<String> = vec![];
    let mut dests: Vec<(String, String)> = vec![];
    for a in &t.goal_atoms {
        if REL6.contains(&a[0].as_str()) && a.len() == 3 {
            let m = syn_of(&a[1]);
            if !movers.contains(&m) {
                movers.push(m);
            }
            dests.push((a[0].clone(), syn_of(&a[2])));
        }
    }
    let mut mv: Vec<(String, f64)> = vec![]; // (cat, mass)
    for s in &movers {
        for o in objs_of(s) {
            if o.fixed || o.bbox.is_none() || o.pos.is_none() {
                continue;
            }
            mv.push((o.cat.clone(), o.mass.unwrap_or(0.0)));
        }
    }
    let mut ds: Vec<String> = vec![];
    for (_, s) in &dests {
        for o in objs_of(s) {
            if o.bbox.is_some() && o.pos.is_some() {
                ds.push(o.cat.clone());
            }
        }
    }
    let mut rooms: Vec<&String> = t.info.iter().flat_map(|(_, v)| v.rooms.iter()).collect();
    rooms.sort();
    rooms.dedup();
    let rs = t.robot_start;
    let mut maxd: Option<f64> = None;
    for s in &movers {
        for o in objs_of(s) {
            if let Some(p) = o.pos {
                let d = (p[0] - rs[0]).hypot(p[1] - rs[1]);
                maxd = Some(maxd.map_or(d, |m: f64| m.max(d)));
            }
        }
    }
    // skills (doc skills.py)
    let preds: Vec<String> = t.pred_keys.iter().map(|p| p.replace("not ", "")).collect();
    let has = |k: &str| preds.iter().any(|p| p == k);
    const CL: [&str; 11] = ["cabinet", "refrigerator", "microwave", "oven", "washer", "dishwasher", "car.", "toolbox", "recycling_bin", "drawer", "hinged_jar"];
    let ga = &t.goal_atoms;
    let ia = &t.init_atoms;
    let mut sk = 0u16;
    let bit = |n: &str| 1u16 << crate::format::SKILLS.iter().position(|s| s.0 == n).unwrap();
    if rooms.len() >= 2 || maxd.map_or(false, |d| r1(d) > 5.0) {
        sk |= bit("MULTI_ROOM");
    }
    let rel5 = ["ontop", "inside", "nextto", "under", "touching"];
    if ga.iter().any(|a| rel5.contains(&a[0].as_str()) && a.len() == 3) || ga.iter().any(|a| a[0] == "inside") {
        sk |= bit("PICK");
    }
    let closed_in = |l: &Vec<Vec<String>>| l.iter().any(|a| a[0] == "inside" && a.len() > 2 && any_in(&a[2], &CL));
    if ga.iter().any(|a| a[0] == "open") || closed_in(ia) || closed_in(ga) {
        sk |= bit("OPEN");
    }
    if has("toggled_on") {
        sk |= bit("TOGGLE");
    }
    if ["cooked", "frozen", "hot", "on_fire"].iter().any(|k| has(k)) {
        sk |= bit("HEAT");
    }
    if has("real") || t.future.iter().any(|x| x.contains("half__") || x.contains("diced__")) {
        sk |= bit("SLICE");
    }
    if ga.iter().any(|a| a[0] == "covered") && t.pred_keys.iter().any(|k| k == "not covered") {
        sk |= bit("CLEAN");
    }
    if t.pred_keys.iter().any(|k| k == "covered") || has("filled") || has("contains") {
        sk |= bit("FILL");
    }
    if has("attached") {
        sk |= bit("ATTACH");
    }
    if mv.iter().any(|m| m.1 > 2.0) {
        sk |= bit("HEAVY");
    }
    let mut mc: Vec<String> = mv.iter().map(|m| m.0.clone()).collect();
    mc.sort();
    mc.dedup();
    // doc table display: first 4 movers; destinations not among those 4, first 3
    let mut dc: Vec<String> = ds.into_iter().filter(|d| !mc.iter().take(4).any(|m| m == d)).collect();
    dc.sort();
    dc.dedup();
    let all_movers = mc.clone();
    mc.truncate(4);
    dc.truncate(3);
    Cols { movers: mc, dests: dc, all_movers, skills: sk, n_rooms: rooms.len(), maxd }
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

pub fn skill_letters(sk: u16) -> String {
    crate::format::SKILLS.iter().enumerate().filter(|(i, _)| sk & (1 << i) != 0).map(|(_, s)| s.1).collect()
}

pub fn fmt_q(q: f64) -> String {
    format!("{:.2}", crate::util::py_round2(q))
}

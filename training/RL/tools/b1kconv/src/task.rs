//! Tasks: problem0.bddl + public task template (+ full template for removed objects) + instances.

use crate::bddl::{self, Arg, Problem};
use crate::format::*;
use crate::scene::{cat_flags, Common, Paths, Scene, SCENES};
use crate::util::*;
use serde_json::Value;
use std::collections::HashMap;

pub struct TaskMeta {
    pub index: usize,
    pub name: String,
    pub length: f64, // frames at 30 Hz
    pub dist: f64,
}

pub fn task_list(p: &Paths) -> Result<Vec<TaskMeta>, String> {
    let path = format!("{}/metadata/task.jsonl", p.ti());
    let s = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let mut v = Vec::new();
    for l in s.lines().filter(|l| !l.trim().is_empty()) {
        let j: Value = serde_json::from_str(l).map_err(|e| format!("{path}: {e}"))?;
        v.push(TaskMeta {
            index: j["task_index"].as_u64().ok_or("task_index")? as usize,
            name: j["task_name"].as_str().ok_or("task_name")?.to_string(),
            length: j["length"].as_f64().ok_or("length")?,
            dist: j["distance_traveled"].as_f64().unwrap_or(0.0),
        });
    }
    Ok(v)
}

/// B100_task_misc.csv: task name -> rooms to include
pub fn rooms_csv(p: &Paths) -> Result<HashMap<String, Vec<String>>, String> {
    let path = format!("{}/metadata/B100_task_misc.csv", p.ti());
    let s = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    // minimal CSV: quoted fields may hold newlines
    let mut rows: Vec<Vec<String>> = vec![];
    let (mut row, mut field, mut q) = (vec![], String::new(), false);
    let mut it = s.chars().peekable();
    while let Some(ch) = it.next() {
        match (ch, q) {
            ('"', false) => q = true,
            ('"', true) => {
                if it.peek() == Some(&'"') {
                    field.push('"');
                    it.next();
                } else {
                    q = false
                }
            }
            (',', false) => row.push(std::mem::take(&mut field)),
            ('\n', false) => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            ('\r', false) => {}
            (c, _) => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    let mut m = HashMap::new();
    for r in rows.iter().skip(1) {
        if r.len() >= 3 {
            m.insert(r[1].clone(), r[2].lines().map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect());
        }
    }
    Ok(m)
}

/// The scene of a task = the only scene with a public template (doc rule; '-partial_rooms' preferred).
pub fn find_scene(p: &Paths, task: &str) -> Result<(String, String, String), String> {
    let mut hits = vec![];
    for sc in SCENES {
        let d = format!("{}/scene_test/public/{sc}/json", p.ti());
        let full = format!("{d}/{sc}_task_{task}_0_0_template.json");
        let part = format!("{d}/{sc}_task_{task}_0_0_template-partial_rooms.json");
        let doc = if std::path::Path::new(&part).exists() { part } else { full.clone() };
        if std::path::Path::new(&doc).exists() {
            hits.push((sc.to_string(), doc, full));
        }
    }
    match hits.len() {
        1 => Ok(hits.pop().unwrap()),
        0 => Err(format!("task {task}: no public template in any of the 7 scenes")),
        _ => Err(format!("task {task}: templates in several scenes {:?}", hits.iter().map(|h| &h.0).collect::<Vec<_>>())),
    }
}

/// Dataset defects that are tolerated (everything else that is missing is an error).
/// (task, BDDL instance, split or "" for both, reason)
pub const KNOWN_DEFECTS: &[(&str, &str, &str, &str)] = &[
    ("cook_a_frozen_pie", "tray.n.01_1", "public", "public instance files lack the tray (train files have it)"),
];

// ---------------------------------------------------------------- doc-compatible info
#[derive(Clone, Debug)]
pub struct Info {
    pub system: bool,
    pub cat: String,
    pub bbox: Option<[f64; 3]>,
    pub off: [f64; 3],
    pub pos: Option<[f64; 3]>,
    pub quat: [f64; 4],
    pub fixed: bool,
    pub mass: Option<f64>,
    pub rooms: Vec<String>,
}

#[allow(dead_code)]
pub struct FeasTask {
    pub idx: usize,
    pub task: String,
    pub scene: String,
    pub len_s: f64,
    pub info: Vec<(String, Info)>, // inst_to_name order, agent skipped
    pub init_atoms: Vec<Vec<String>>,
    pub goal_atoms: Vec<Vec<String>>,
    pub future: Vec<String>,
    pub robot_start: [f64; 3],
    pub pred_keys: Vec<String>,
    /// doc literals: (pred, candidate names per arg, neg)
    pub lits: Vec<(String, Vec<Vec<String>>, bool)>,
    pub objs: Vec<(String, Vec<String>)>,
}

impl FeasTask {
    pub fn get(&self, inst: &str) -> Option<&Info> {
        self.info.iter().find(|(k, _)| k == inst).map(|(_, v)| v)
    }
}

#[allow(dead_code)]
pub struct BuiltTask {
    pub feas: FeasTask,
    pub n_inst: usize,
    pub pose_src: [usize; 4],
    pub extra_keys: usize,
    pub not_open_inst: Vec<(usize, usize)>, // per doc 'not open' literal: (instances closed at start, instances)
}

fn template_objects(t: &Value) -> Result<(&serde_json::Map<String, Value>, &serde_json::Map<String, Value>, &serde_json::Map<String, Value>), String> {
    let i2n = get(t, &["metadata", "task", "inst_to_name"]).and_then(|v| v.as_object()).ok_or("template: no inst_to_name")?;
    let oi = get(t, &["objects_info", "init_info"]).and_then(|v| v.as_object()).ok_or("template: no init_info")?;
    let reg = get(t, &["state", "registry", "object_registry"]).and_then(|v| v.as_object()).ok_or("template: no object_registry")?;
    Ok((i2n, oi, reg))
}

/// fault injection for the negative control
#[derive(Clone, Copy, PartialEq)]
pub enum Fault {
    None,
    DropInstanceObject,
    BadGoalObject,
}

pub struct Out<'a> {
    pub tasks: &'a mut Vec<TaskRec>,
    pub tobjs: &'a mut Vec<TaskObjRec>,
    pub lits: &'a mut Vec<LitRec>,
    pub vars: &'a mut Vec<VarRec>,
    pub cands: &'a mut Vec<u16>,
    pub removed: &'a mut Vec<u16>,
    pub insts: &'a mut Vec<InstRec>,
    pub poses: &'a mut Vec<PoseRec>,
}

fn lit_rec(l: &bddl::Lit, flags: u8) -> Result<LitRec, String> {
    let pred = pred_id(&l.pred).ok_or(format!("unknown predicate '{}'", l.pred))?;
    if l.args.is_empty() || l.args.len() > 2 {
        return Err(format!("predicate '{}' with {} args", l.pred, l.args.len()));
    }
    if l.or_tags.len() > 1 {
        return Err("nested 'or' is not supported".into());
    }
    let mut arg = [NONE16; 2];
    for (k, a) in l.args.iter().enumerate() {
        arg[k] = match *a {
            Arg::Obj(o) => o,
            Arg::Var(v) => ARG_VAR | v,
            Arg::Room(r) => ARG_ROOM | r,
        };
    }
    let (g, b) = l.or_tags.first().copied().unwrap_or((NONE16, 0));
    let mut f = flags;
    if l.doc {
        f |= 1;
    }
    if !l.or_tags.is_empty() {
        f |= 4;
    }
    Ok(LitRec { pred, neg: l.neg as u8, nargs: l.args.len() as u8, flags: f, or_group: g, or_branch: b, arg, reserved: 0 })
}

#[allow(clippy::too_many_arguments)]
pub fn build_task(
    p: &Paths,
    c: &mut Common,
    sc: &mut Scene,
    meta: &TaskMeta,
    rooms_inc: &[String],
    doc_template: &str,
    full_template: &str,
    out: &mut Out,
    fault: Fault,
) -> Result<BuiltTask, String> {
    let task = meta.name.as_str();
    let src = std::fs::read_to_string(p.bddl(task)).map_err(|e| format!("{task}: {e}"))?;
    let mut prob: Problem = bddl::parse_problem(&src).map_err(|e| format!("{task}: {e}"))?;
    if fault == Fault::BadGoalObject {
        // rename one :objects entry -> the goal must fail to resolve it
        if let Some(o) = prob.objects.iter_mut().find(|(_, s)| s != "agent.n.01") {
            o.0.push_str("_missing");
        }
    }
    let tmpl = read_json(doc_template)?;
    let (i2n, oi, reg) = template_objects(&tmpl)?;

    // ---- doc-compatible info (CURRICULUM_BEHAVIOR2026 1.3 script)
    let mut info = Vec::new();
    for (inst, on) in i2n {
        if inst.starts_with("agent") {
            continue;
        }
        let on = on.as_str().unwrap_or("");
        let Some(o) = oi.get(on) else {
            info.push((inst.clone(), Info { system: true, cat: String::new(), bbox: None, off: [0.0; 3], pos: None, quat: [0., 0., 0., 1.], fixed: false, mass: None, rooms: vec![] }));
            continue;
        };
        let a = &o["args"];
        let cat = a.get("category").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let model = a.get("model").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let scale = arr3(a.get("scale")).unwrap_or([1.0; 3]);
        let m = c.meta(&cat, &model).map(|m| (m.bbox, m.off));
        let st = reg.get(on);
        info.push((
            inst.clone(),
            Info {
                system: false,
                bbox: m.map(|(b, _)| [b[0] * scale[0], b[1] * scale[1], b[2] * scale[2]]),
                off: m.map_or([0.0; 3], |(_, o)| [o[0] * scale[0], o[1] * scale[1], o[2] * scale[2]]),
                pos: st.and_then(|s| arr3(get(s, &["root_link", "pos"]))),
                quat: st.and_then(|s| arr4(get(s, &["root_link", "ori"]))).unwrap_or([0., 0., 0., 1.]),
                fixed: a.get("fixed_base").and_then(|x| x.as_bool()).unwrap_or(false),
                mass: c.avg_spec(&cat).0,
                rooms: a.get("in_rooms").and_then(|x| x.as_array()).map(|v| v.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default(),
                cat,
            },
        ));
    }
    let rp = get(&tmpl, &["metadata", "task", "robot_poses"]).and_then(|v| v.as_object()).and_then(|m| m.values().next()).and_then(|v| v.get(0));
    let robot_start = rp.and_then(|r| arr3(r.get("position"))).ok_or(format!("{task}: template has no robot pose"))?;
    let mut init_atoms = vec![];
    bddl::atoms(&bddl::Sx::L(std::iter::once(bddl::Sx::A("and".into())).chain(prob.init.iter().cloned()).collect()), &mut init_atoms);
    let mut goal_atoms = vec![];
    bddl::atoms(&prob.goal, &mut goal_atoms);
    let future: Vec<String> = init_atoms.iter().filter(|a| a[0] == "future" && a.len() > 1).map(|a| a[1].clone()).collect();
    let pred_keys: Vec<String> = bddl::pred_count(&prob.goal, &prob).1.into_iter().map(|(k, _)| k).collect();

    // ---- device compile
    let comp = bddl::compile_goal(&prob).map_err(|e| format!("{task}: goal: {e}"))?;
    let rc = |s: &str| c.room_cat(s);
    let init = bddl::compile_init(&prob, &rc).map_err(|e| format!("{task}: init: {e}"))?;
    let name_of = |a: &Arg| -> Vec<String> {
        match *a {
            Arg::Obj(o) => vec![prob.objects[o as usize].0.clone()],
            Arg::Var(v) => comp.vars[v as usize].cands.iter().map(|&o| prob.objects[o as usize].0.clone()).collect(),
            Arg::Room(r) => vec![c.room_cats[r as usize - 1].clone()],
        }
    };
    let lits: Vec<(String, Vec<Vec<String>>, bool)> =
        comp.lits.iter().filter(|l| l.doc).map(|l| (l.pred.clone(), l.args.iter().map(name_of).collect(), l.neg)).collect();

    // BDDL init (closed world) truth of goal literals
    let init_pos: Vec<(String, Vec<u16>)> = init
        .iter()
        .filter(|l| !l.neg)
        .map(|l| (l.pred.clone(), l.args.iter().map(|a| if let Arg::Obj(o) = a { *o } else { NONE16 }).collect()))
        .collect();
    let cand_of = |a: &Arg| -> Vec<u16> {
        match *a {
            Arg::Obj(o) => vec![o],
            Arg::Var(v) => comp.vars[v as usize].cands.clone(),
            Arg::Room(_) => vec![],
        }
    };
    let init_true = |l: &bddl::Lit| -> bool {
        let c0 = cand_of(&l.args[0]);
        let c1 = if l.args.len() > 1 { cand_of(&l.args[1]) } else { vec![NONE16] };
        c0.iter().any(|&a| {
            c1.iter().any(|&b| {
                let want: Vec<u16> = if l.args.len() > 1 { vec![a, b] } else { vec![a] };
                let present = init_pos.iter().any(|(p, args)| *p == l.pred && *args == want);
                present != l.neg
            })
        })
    };

    // ---- task objects
    let full = if full_template == doc_template { None } else { Some(read_json(full_template)?) };
    let (full_i2n, full_oi, full_reg) = match &full {
        Some(f) => template_objects(f)?,
        None => (i2n, oi, reg),
    };
    let obj_off = out.tobjs.len() as u32;
    let mut future_set = std::collections::HashSet::new();
    for l in &init {
        if l.pred == "future" && !l.neg {
            if let Arg::Obj(o) = l.args[0] {
                future_set.insert(o);
            }
        }
    }
    for (k, (inst, syn)) in prob.objects.iter().enumerate() {
        let mut r = TaskObjRec { inst: sc.strings.add(inst), syn: sc.strings.add(syn), cat: NONE16, scene_obj: -1, quat: [0., 0., 0., 1.], mass: f32::NAN, ..Default::default() };
        if future_set.contains(&(k as u16)) {
            r.flags |= flag("FUTURE");
        }
        if inst.ends_with("_*") {
            r.flags |= flag("WILDCARD");
        }
        if syn == "agent.n.01" {
            r.flags |= flag("AGENT");
        }
        // the partial_rooms template first (doc source); an instance it lacks comes from the full template
        // (bringing_in_wood: partial template still says firewood, BDDL + full template + instances say plywood)
        let (i2n, oi, reg) = if i2n.contains_key(inst) { (i2n, oi, reg) } else { (full_i2n, full_oi, full_reg) };
        match i2n.get(inst).and_then(|v| v.as_str()) {
            None => r.flags |= flag("UNMAPPED"),
            Some(on) => {
                r.og_name = sc.strings.add(on);
                if let Some(&si) = sc.obj_index.get(on) {
                    r.scene_obj = si as i32;
                    r.flags |= flag("SCENE_OBJ");
                }
                match oi.get(on) {
                    None => {
                        if r.flags & flag("AGENT") == 0 {
                            r.flags |= flag("SYSTEM");
                        }
                    }
                    Some(o) => {
                        let a = &o["args"];
                        let cat = a.get("category").and_then(|x| x.as_str()).unwrap_or("");
                        let model = a.get("model").and_then(|x| x.as_str()).unwrap_or("");
                        let scale = arr3(a.get("scale")).unwrap_or([1.0; 3]);
                        r.scale = f3(scale);
                        r.flags |= cat_flags(cat);
                        if let Some(m) = c.meta(cat, model) {
                            r.flags |= m.flags;
                            r.half = f3([scale[0] * m.bbox[0] / 2.0, scale[1] * m.bbox[1] / 2.0, scale[2] * m.bbox[2] / 2.0]);
                            r.offset = f3([scale[0] * m.off[0], scale[1] * m.off[1], scale[2] * m.off[2]]);
                        }
                        if a.get("fixed_base").and_then(|x| x.as_bool()) == Some(true) {
                            r.flags |= flag("FIXED_BASE");
                        }
                        if !cat.is_empty() {
                            r.cat = sc.cat(c, cat);
                            r.mass = sc.cats[r.cat as usize].mass;
                        }
                        if let Some(st) = reg.get(on) {
                            if let Some(pv) = arr3(get(st, &["root_link", "pos"])) {
                                r.pos = f3(pv);
                            }
                            if let Some(q) = arr4(get(st, &["root_link", "ori"])) {
                                r.quat = f4(q);
                            }
                            let nj = f64s(st.get("joint_pos")).map_or(0, |v| v.len());
                            r.n_joints = nj as u16;
                            if nj > 0 {
                                r.flags |= flag("ARTICULATED");
                            }
                        }
                    }
                }
            }
        }
        out.tobjs.push(r);
    }

    // ---- literals
    let init_off = out.lits.len() as u32;
    for l in &init {
        out.lits.push(lit_rec(l, 0).map_err(|e| format!("{task}: init: {e}"))?);
    }
    let goal_off = out.lits.len() as u32;
    let mut n_init_true = 0u16;
    let mut n_not_open = 0u16;
    for l in &comp.lits {
        let it = init_true(l);
        if l.doc && it {
            n_init_true += 1;
        }
        if l.doc && l.pred == "open" && l.neg {
            n_not_open += 1;
        }
        out.lits.push(lit_rec(l, if it { 2 } else { 0 }).map_err(|e| format!("{task}: goal: {e}"))?);
    }
    let var_off = out.vars.len() as u32;
    let cand_off = out.cands.len() as u32;
    for v in &comp.vars {
        out.vars.push(VarRec { kind: v.kind, reserved: 0, group: v.group, cand_off: (out.cands.len() as u32 - cand_off) as u16, cand_n: v.cands.len() as u16 });
        out.cands.extend_from_slice(&v.cands);
    }
    let removed_off = out.removed.len() as u32;
    for (i, nm) in sc.obj_names.iter().enumerate() {
        if !full_oi.contains_key(nm) {
            out.removed.push(i as u16);
        }
    }
    let mut rooms_mask = 0u64;
    for rn in rooms_inc {
        let ri = sc.room(rn).ok_or(format!("{task}: 'Rooms to include' {rn} is not a room of {}", sc.name))?;
        rooms_mask |= 1u64 << ri;
    }

    // ---- instances (streamed: one small file at a time)
    let inst_off = out.insts.len() as u32;
    let task_slot = out.tasks.len() as u16;
    let mut counts = [0usize; 2];
    let mut pose_src = [0usize; 4];
    let mut extra_keys = 0usize;
    let not_open_lits: Vec<Vec<u16>> =
        comp.lits.iter().filter(|l| l.doc && l.pred == "open" && l.neg).map(|l| cand_of(&l.args[0])).collect();
    let mut not_open_inst = vec![(0usize, 0usize); not_open_lits.len()];
    for (split, dir) in [(0u16, format!("{}/scenes/{}/json", p.ti(), sc.name)), (1u16, format!("{}/scene_test/public/{}/json", p.ti(), sc.name))] {
        let idir = format!("{dir}/{}_task_{task}_instances", sc.name);
        let prefix = format!("{}_task_{task}_0_", sc.name);
        let mut files: Vec<(u32, String)> = vec![];
        for e in std::fs::read_dir(&idir).map_err(|e| format!("{idir}: {e}"))? {
            let f = e.map_err(|e| e.to_string())?.file_name().to_string_lossy().into_owned();
            if let Some(rest) = f.strip_prefix(&prefix).and_then(|r| r.strip_suffix("_template-tro_state.json")) {
                files.push((rest.parse().map_err(|_| format!("{idir}/{f}: bad id"))?, f));
            }
        }
        files.sort();
        let mut first = true;
        for (id, f) in files {
            let path = format!("{idir}/{f}");
            let mut j = read_json(&path)?;
            if fault == Fault::DropInstanceObject && first {
                if let Some(m) = j.as_object_mut() {
                    let k = m.keys().find(|k| *k != "robot_poses").cloned();
                    if let Some(k) = k {
                        m.remove(&k);
                    }
                }
            }
            first = false;
            let m = j.as_object().ok_or(format!("{path}: not an object"))?;
            let (rp, rflag) = robot_pose(&j).ok_or(format!("{path}: no robot_poses.robot[0] or R1Pro[0]"))?;
            let rpos = arr3(rp.get("position")).ok_or(format!("{path}: robot position"))?;
            let rq = arr4(rp.get("orientation")).ok_or(format!("{path}: robot orientation"))?;
            let pose_off = out.poses.len() as u32;
            for (k, (inst, _)) in prob.objects.iter().enumerate() {
                let tf = out.tobjs[obj_off as usize + k].flags;
                if tf & (flag("AGENT") | flag("SYSTEM") | flag("UNMAPPED") | flag("WILDCARD")) != 0 {
                    out.poses.push(PoseRec { quat: [0., 0., 0., 1.], ..Default::default() });
                    pose_src[0] += 1;
                    continue;
                }
                let pr = match m.get(inst) {
                    Some(st) => {
                        let pos = arr3(get(st, &["root_link", "pos"])).ok_or(format!("{path}: {inst} root_link.pos"))?;
                        let q = arr4(get(st, &["root_link", "ori"])).ok_or(format!("{path}: {inst} root_link.ori"))?;
                        let jp = f64s(st.get("joint_pos")).unwrap_or_default();
                        let jo = sc.joints.len() as u32;
                        sc.joints.extend(jp.iter().map(|&x| x as f32));
                        PoseRec { pos: f3(pos), quat: f4(q), joint_off: jo, n_joints: jp.len() as u16, src: 1 }
                    }
                    None => {
                        let so = out.tobjs[obj_off as usize + k].scene_obj;
                        let known = KNOWN_DEFECTS.iter().any(|d| d.0 == task && d.1 == inst && (d.2.is_empty() || (d.2 == "public") == (split == 1)));
                        if so >= 0 {
                            let o = &sc.objs[so as usize];
                            PoseRec { pos: o.pos, quat: o.quat, joint_off: o.joint_off, n_joints: o.n_joints, src: 2 }
                        } else if known {
                            let t = &out.tobjs[obj_off as usize + k];
                            PoseRec { pos: t.pos, quat: t.quat, joint_off: 0, n_joints: 0, src: 3 }
                        } else {
                            return Err(format!("{path}: task object '{inst}' missing (not a scene object, not a known defect)"));
                        }
                    }
                };
                pose_src[pr.src as usize] += 1;
                out.poses.push(pr);
            }
            extra_keys += m.keys().filter(|k| *k != "robot_poses" && !prob.objects.iter().any(|(n, _)| n == *k)).count();
            // 'not open' at the start of this instance: closed = every joint |q| < 0.05
            for (li, cands) in not_open_lits.iter().enumerate() {
                let closed = cands.iter().all(|&o| {
                    let pr = &out.poses[pose_off as usize + o as usize];
                    (0..pr.n_joints as usize).all(|j| sc.joints[pr.joint_off as usize + j].abs() < 0.05)
                });
                not_open_inst[li].1 += 1;
                if closed {
                    not_open_inst[li].0 += 1;
                }
            }
            out.insts.push(InstRec {
                task: task_slot,
                split,
                inst_id: id,
                pose_off,
                n_pose: prob.objects.len() as u16,
                flags: rflag,
                robot_pos: f3(rpos),
                robot_quat: f4(rq),
                robot_yaw: yaw(rq) as f32,
            });
            counts[split as usize] += 1;
        }
    }

    let human_s = meta.length / 30.0;
    out.tasks.push(TaskRec {
        name: sc.strings.add(task),
        task_index: meta.index as u16,
        n_obj: prob.objects.len() as u16,
        n_init: init.len() as u16,
        n_goal: comp.lits.len() as u16,
        n_var: comp.vars.len() as u16,
        n_cand: (out.cands.len() as u32 - cand_off) as u16,
        n_removed: (out.removed.len() as u32 - removed_off) as u16,
        n_or_groups: comp.n_or_groups,
        n_doc_lits: lits.len() as u16,
        n_not_open,
        n_init_true,
        skills: 0,
        obj_off,
        init_off,
        goal_off,
        var_off,
        cand_off,
        removed_off,
        inst_off,
        n_inst: (counts[0] + counts[1]) as u32,
        n_train: counts[0] as u32,
        n_public: counts[1] as u32,
        human_s: human_s as f32,
        limit_s: (human_s * 1.5) as f32,
        dist_m: meta.dist as f32,
        q_strict: 0.0,
        q_loose: 0.0,
        rooms_mask,
    });

    let feas = FeasTask {
        idx: meta.index,
        task: task.to_string(),
        scene: sc.name.clone(),
        len_s: human_s,
        info,
        init_atoms,
        goal_atoms,
        future,
        robot_start,
        pred_keys,
        lits,
        objs: prob.objs.clone(),
    };
    Ok(BuiltTask { feas, n_inst: counts[0] + counts[1], pose_src, extra_keys, not_open_inst })
}

/// Evaluator rule (eval/evaluator.py): generic "robot" pose if present, else the robot model's.
/// 2025 tasks' train files only have per-model poses (R1Pro, Fetch, R1, Stretch, Tiago): we take
/// R1Pro's (the default robot) and set InstRec.flags bit 0.
pub fn robot_pose(j: &Value) -> Option<(&Value, u16)> {
    let rp = j.get("robot_poses")?.as_object()?;
    if let Some(v) = rp.get("robot") {
        return Some((v.get(0)?, 0));
    }
    let v = rp.iter().find(|(k, _)| k.to_lowercase() == "r1pro")?.1;
    Some((v.get(0)?, 1))
}

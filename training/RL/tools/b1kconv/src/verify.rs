//! Re-read a RASC file and compare its values with the source JSON / BDDL (E1 pass criterion 2).

use crate::bddl;
use crate::format::*;
use crate::scene::{Paths, SCENES};
use crate::util::*;
use serde_json::Value;

pub struct File {
    pub buf: Vec<u8>,
    pub hdr: FileHeader,
}

impl File {
    pub fn open(path: &str) -> Result<File, String> {
        let buf = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let hsz = std::mem::size_of::<FileHeader>();
        if buf.len() < hsz {
            return Err("file shorter than header".into());
        }
        let hdr = read_recs::<FileHeader>(&buf[..hsz])[0];
        if hdr.magic != MAGIC || hdr.version != VERSION || hdr.header_bytes as usize != hsz || hdr.layout_hash != layout_hash() {
            return Err(format!("{path}: magic/version/layout mismatch"));
        }
        for (i, s) in hdr.sections.iter().enumerate() {
            let end = s.offset + s.bytes;
            if s.bytes > 0 && (s.offset % 64 != 0 || end as usize > buf.len()) {
                return Err(format!("{path}: section {} out of range", SEC_NAMES[i]));
            }
            let want = if s.elem == 0 { (s.count as u64).div_ceil(8) } else { s.count as u64 * s.elem as u64 };
            if s.bytes != want {
                return Err(format!("{path}: section {} size {} != {}", SEC_NAMES[i], s.bytes, want));
            }
        }
        Ok(File { buf, hdr })
    }
    pub fn sec(&self, id: usize) -> &[u8] {
        let s = &self.hdr.sections[id];
        &self.buf[s.offset as usize..(s.offset + s.bytes) as usize]
    }
    pub fn recs<T: Rec>(&self, id: usize) -> Vec<T> {
        read_recs(self.sec(id))
    }
    pub fn str(&self, off: u32) -> String {
        cstr(self.sec(S_STRINGS), off)
    }
    pub fn u32s(&self, id: usize) -> Vec<u32> {
        self.sec(id).chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
    pub fn f32s(&self, id: usize) -> Vec<f32> {
        self.sec(id).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
}

#[derive(Default)]
pub struct Cmp {
    pub n: u64,
    pub diffs: Vec<String>,
}
impl Cmp {
    fn eq<T: PartialEq + std::fmt::Debug>(&mut self, what: &dyn Fn() -> String, a: T, b: T) {
        self.n += 1;
        if a != b && self.diffs.len() < 50 {
            self.diffs.push(format!("{}: file {a:?} != source {b:?}", what()));
        } else if a != b {
            self.diffs.push(String::new());
        }
    }
    fn fs(&mut self, what: &dyn Fn() -> String, a: &[f32], b: Option<Vec<f64>>) {
        let b: Vec<u32> = b.unwrap_or_default().iter().map(|&x| (x as f32).to_bits()).collect();
        let a: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
        self.eq(what, a, b);
    }
}

fn sx_str(e: &bddl::Sx) -> String {
    match e {
        bddl::Sx::A(s) => s.clone(),
        bddl::Sx::L(v) => format!("({})", v.iter().map(sx_str).collect::<Vec<_>>().join(" ")),
    }
}

pub fn verify(p: &Paths, path: &str) -> Result<Cmp, String> {
    let f = File::open(path)?;
    let mut c = Cmp::default();
    let scene = SCENES[f.hdr.scene_index as usize];
    let objs: Vec<ObjRec> = f.recs(S_OBJS);
    let cats: Vec<CatRec> = f.recs(S_CATS);
    let rooms: Vec<RoomRec> = f.recs(S_ROOMS);
    let joints = f.f32s(S_JOINTS);
    let best = read_json(&format!("{}/scenes/{scene}/json/{scene}_best.json", p.assets()))?;
    let oi = get(&best, &["objects_info", "init_info"]).and_then(|v| v.as_object()).ok_or("best")?;
    let reg = get(&best, &["state", "registry", "object_registry"]).and_then(|v| v.as_object()).ok_or("best")?;
    c.eq(&|| "object count".into(), objs.len(), oi.len());
    for (i, (name, v)) in oi.iter().enumerate() {
        let Some(o) = objs.get(i) else { break };
        let w = || format!("obj {name}");
        c.eq(&w, f.str(o.name), name.clone());
        let a = &v["args"];
        c.eq(&w, f.str(o.model), a["model"].as_str().unwrap_or("").to_string());
        c.eq(&w, f.str(cats[o.cat as usize].name), a["category"].as_str().unwrap_or("").to_string());
        c.fs(&w, &o.scale, f64s(a.get("scale")).or(Some(vec![1.0; 3])));
        c.eq(&w, o.flags & flag("FIXED_BASE") != 0, a.get("fixed_base").and_then(|x| x.as_bool()) == Some(true));
        let st = &reg[name];
        c.fs(&w, &o.pos, f64s(get(st, &["root_link", "pos"])));
        c.fs(&w, &o.quat, f64s(get(st, &["root_link", "ori"])));
        c.fs(&w, &joints[o.joint_off as usize..(o.joint_off + o.n_joints as u32) as usize], f64s(st.get("joint_pos")).or(Some(vec![])));
        let pool = f.u32s(S_IN_ROOMS);
        let fr: Vec<String> = (0..o.n_in_rooms as usize).map(|k| f.str(pool[o.in_rooms_off as usize + k])).collect();
        for (k, rr) in [o.room_a, o.room_b].iter().enumerate() {
            if *rr != NONE16 {
                c.eq(&w, f.str(rooms[*rr as usize].name), fr.get(k).cloned().unwrap_or_default());
            }
        }
        let sr: Vec<String> = a.get("in_rooms").and_then(|x| x.as_array()).map(|v| v.iter().map(|s| s.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
        c.eq(&w, fr, sr);
    }

    // tasks
    let tasks: Vec<TaskRec> = f.recs(S_TASKS);
    let tobjs: Vec<TaskObjRec> = f.recs(S_TASK_OBJS);
    let lits: Vec<LitRec> = f.recs(S_LITS);
    let insts: Vec<InstRec> = f.recs(S_INSTS);
    let poses: Vec<PoseRec> = f.recs(S_POSES);
    let rc: Vec<String> = std::fs::read_to_string(format!("{}/metadata/room_categories.txt", p.assets())).map_err(|e| e.to_string())?.lines().map(String::from).collect();
    for t in &tasks {
        let tn = f.str(t.name);
        let prob = bddl::parse_problem(&std::fs::read_to_string(p.bddl(&tn)).map_err(|e| e.to_string())?)?;
        let tmpl = read_json(&crate::task::find_scene(p, &tn)?.1)?; // same template as the converter (partial_rooms first)
        let i2n = get(&tmpl, &["metadata", "task", "inst_to_name"]).and_then(|v| v.as_object()).ok_or("i2n")?;
        let ftmpl = read_json(&crate::task::find_scene(p, &tn)?.2)?;
        let fi2n = get(&ftmpl, &["metadata", "task", "inst_to_name"]).and_then(|v| v.as_object()).ok_or("i2n")?;
        c.eq(&|| format!("{tn} n_obj"), t.n_obj as usize, prob.objects.len());
        for (k, (inst, syn)) in prob.objects.iter().enumerate() {
            let o = &tobjs[t.obj_off as usize + k];
            let w = || format!("{tn} obj {inst}");
            c.eq(&w, f.str(o.inst), inst.clone());
            c.eq(&w, f.str(o.syn), syn.clone());
            c.eq(&w, f.str(o.og_name), i2n.get(inst).or(fi2n.get(inst)).and_then(|v| v.as_str()).unwrap_or("").to_string());
        }
        // init literals decoded back to text
        c.eq(&|| format!("{tn} n_init"), t.n_init as usize, prob.init.len());
        for (k, e) in prob.init.iter().enumerate() {
            let l = &lits[t.init_off as usize + k];
            let mut parts = vec![PREDS[l.pred as usize].to_string()];
            for a in &l.arg[..l.nargs as usize] {
                parts.push(if a & ARG_ROOM != 0 { rc[(a & !ARG_ROOM) as usize - 1].clone() } else { f.str(tobjs[t.obj_off as usize + *a as usize].inst) });
            }
            let mut s = format!("({})", parts.join(" "));
            if l.neg != 0 {
                s = format!("(not {s})");
            }
            c.eq(&|| format!("{tn} init {k}"), s, sx_str(e));
        }
        // goal references
        for k in 0..t.n_goal as usize {
            let l = &lits[t.goal_off as usize + k];
            for a in &l.arg[..l.nargs as usize] {
                let ok = if a & ARG_VAR != 0 { ((a & !ARG_VAR) as u16) < t.n_var } else { *a < t.n_obj };
                c.eq(&|| format!("{tn} goal {k} arg valid"), ok, true);
            }
        }
        // instances
        let mut seen = [0u32; 2];
        for ii in 0..t.n_inst as usize {
            let r = &insts[t.inst_off as usize + ii];
            seen[r.split as usize] += 1;
            let dir = if r.split == 0 { format!("{}/scenes/{scene}/json", p.ti()) } else { format!("{}/scene_test/public/{scene}/json", p.ti()) };
            let path = format!("{dir}/{scene}_task_{tn}_instances/{scene}_task_{tn}_0_{}_template-tro_state.json", r.inst_id);
            let j: Value = read_json(&path)?;
            let w = || format!("{tn} inst {}", r.inst_id);
            let rp = crate::task::robot_pose(&j).map(|x| x.0);
            c.fs(&w, &r.robot_pos, rp.and_then(|r| f64s(r.get("position"))));
            c.fs(&w, &r.robot_quat, rp.and_then(|r| f64s(r.get("orientation"))));
            for (k, (inst, _)) in prob.objects.iter().enumerate() {
                let pr = &poses[r.pose_off as usize + k];
                let w2 = || format!("{tn} inst {} {inst}", r.inst_id);
                match pr.src {
                    1 => {
                        let st = &j[inst.as_str()];
                        c.fs(&w2, &pr.pos, f64s(get(st, &["root_link", "pos"])));
                        c.fs(&w2, &pr.quat, f64s(get(st, &["root_link", "ori"])));
                        c.fs(&w2, &joints[pr.joint_off as usize..pr.joint_off as usize + pr.n_joints as usize], f64s(st.get("joint_pos")).or(Some(vec![])));
                    }
                    2 => {
                        c.eq(&w2, j.get(inst.as_str()).is_none(), true);
                        let o = &objs[tobjs[t.obj_off as usize + k].scene_obj as usize];
                        c.eq(&w2, (pr.pos, pr.quat), (o.pos, o.quat));
                    }
                    _ => {}
                }
            }
        }
        c.eq(&|| format!("{tn} splits"), (seen[0], seen[1]), (t.n_train, t.n_public));
    }
    Ok(c)
}

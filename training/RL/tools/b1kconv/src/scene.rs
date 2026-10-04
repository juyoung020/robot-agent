//! Scene static part: `<scene>_best.json` objects + layout PNGs -> categories, rooms, objects,
//! boxes, doors, grids.

use crate::format::*;
use crate::png;
use crate::util::*;
use serde_json::Value;
use std::collections::HashMap;

pub struct Paths {
    pub b1k: String, // .../BEHAVIOR-1K
}
impl Paths {
    pub fn assets(&self) -> String {
        format!("{}/datasets/behavior-1k-assets", self.b1k)
    }
    pub fn ti(&self) -> String {
        format!("{}/datasets/2026-challenge-task-instances", self.b1k)
    }
    pub fn bddl(&self, task: &str) -> String {
        format!("{}/bddl3/bddl/activity_definitions/{task}/problem0.bddl", self.b1k)
    }
}

/// doc 1.2 order; index = FileHeader.scene_index
pub const SCENES: [&str; 7] = [
    "house_single_floor",
    "house_double_floor_lower",
    "house_double_floor_upper",
    "restaurant_diner",
    "Rs_int",
    "hotel_suite_large",
    "office_cubicles_right",
];

pub struct Meta {
    pub bbox: [f64; 3],
    pub off: [f64; 3],
    pub flags: u32,
}

#[derive(Default)]
pub struct Common {
    pub meta: HashMap<(String, String), Option<Meta>>,
    pub avg: Value,
    pub room_cats: Vec<String>,
    pub assets: String,
}

impl Common {
    pub fn load(p: &Paths) -> Result<Common, String> {
        let a = p.assets();
        let avg = read_json(&format!("{a}/metadata/avg_category_specs.json"))?;
        let rc = std::fs::read_to_string(format!("{a}/metadata/room_categories.txt")).map_err(|e| e.to_string())?;
        Ok(Common { meta: HashMap::new(), avg, room_cats: rc.lines().map(|l| l.trim_end().to_string()).collect(), assets: a })
    }
    pub fn room_cat(&self, name: &str) -> Option<u16> {
        self.room_cats.iter().position(|x| x == name).map(|i| (i + 1) as u16)
    }
    pub fn meta(&mut self, cat: &str, model: &str) -> Option<&Meta> {
        let k = (cat.to_string(), model.to_string());
        if !self.meta.contains_key(&k) {
            let path = format!("{}/objects/{cat}/{model}/misc/metadata.json", self.assets);
            let m = read_json(&path).ok().and_then(|j| {
                let bbox = arr3(j.get("bbox_size"))?;
                let off = arr3(j.get("base_link_offset")).unwrap_or([0.0; 3]);
                let mut flags = flag("HAS_BBOX");
                if j.get("openable_joint_ids").and_then(|x| x.as_array()).map_or(false, |a| !a.is_empty()) {
                    flags |= flag("OPENABLE");
                }
                if let Some(tags) = j.get("link_tags").and_then(|x| x.as_object()) {
                    if tags.values().any(|t| t.as_array().map_or(false, |a| a.iter().any(|s| s == "openable"))) {
                        flags |= flag("OPENABLE");
                    }
                }
                if let Some(ml) = j.get("meta_links").and_then(|x| x.as_object()) {
                    for l in ml.values() {
                        if let Some(o) = l.as_object() {
                            for (t, f) in [("togglebutton", "TOGGLEABLE"), ("heatsource", "HEATSOURCE"), ("fluidsource", "FLUID_SOURCE"),
                                ("fluidsink", "FLUID_SINK"), ("attachment", "ATTACHMENT"), ("lights", "LIGHT")] {
                                if o.contains_key(t) {
                                    flags |= flag(f);
                                }
                            }
                        }
                    }
                }
                Some(Meta { bbox, off, flags })
            });
            self.meta.insert(k.clone(), m);
        }
        self.meta[&k].as_ref()
    }
    pub fn avg_spec(&self, cat: &str) -> (Option<f64>, Option<f64>, Option<f64>) {
        let c = self.avg.get(cat);
        let g = |k: &str| c.and_then(|c| c.get(k)).and_then(|v| v.as_f64());
        (g("mass"), g("volume"), g("density"))
    }
}

pub fn cat_flags(cat: &str) -> u32 {
    let mut f = 0;
    if cat == "walls" {
        f |= flag("WALL");
    }
    if cat == "floors" {
        f |= flag("FLOOR");
    }
    if cat == "ceilings" {
        f |= flag("CEILING");
    }
    if cat.contains("door") {
        f |= flag("DOOR");
    }
    if cat.contains("window") {
        f |= flag("WINDOW");
    }
    if cat.contains("stair") {
        f |= flag("STAIRS");
    }
    if cat == "carpet" || cat == "rug" {
        f |= flag("CARPET");
    }
    f
}

#[allow(dead_code)]
pub struct Scene {
    pub name: String,
    pub index: u32,
    pub hdr: FileHeader,
    pub strings: StrPool,
    pub cats: Vec<CatRec>,
    pub cat_names: Vec<String>,
    pub rooms: Vec<RoomRec>,
    pub room_names: Vec<String>,
    pub objs: Vec<ObjRec>,
    pub obj_names: Vec<String>,
    pub obj_index: HashMap<String, usize>,
    pub boxes: Vec<BoxRec>,
    pub doors: Vec<DoorRec>,
    pub joints: Vec<f32>,
    pub in_rooms: Vec<u32>,
    pub room_grid: Vec<u8>,
    pub trav: [Vec<u8>; 4],
    pub trav_free_px: [u64; 4],
    pub warnings: Vec<String>,
}

impl Scene {
    pub fn cat(&mut self, c: &mut Common, name: &str) -> u16 {
        if let Some(i) = self.cat_names.iter().position(|x| x == name) {
            return i as u16;
        }
        let (m, v, d) = c.avg_spec(name);
        let nan = f32::NAN;
        self.cats.push(CatRec {
            name: self.strings.add(name),
            flags: m.is_some() as u32,
            mass: m.map_or(nan, |x| x as f32),
            volume: v.map_or(nan, |x| x as f32),
            density: d.map_or(nan, |x| x as f32),
            ..Default::default()
        });
        self.cat_names.push(name.to_string());
        (self.cats.len() - 1) as u16
    }
    pub fn room(&self, name: &str) -> Option<u16> {
        self.room_names.iter().position(|x| x == name).map(|i| i as u16)
    }
    /// world xy -> (row, col) or None
    pub fn cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let c = ((x - self.hdr.origin[0] as f64) / self.hdr.cell as f64).floor();
        let r = ((y - self.hdr.origin[1] as f64) / self.hdr.cell as f64).floor();
        if c < 0.0 || r < 0.0 || c >= self.hdr.grid_w as f64 || r >= self.hdr.grid_h as f64 {
            None
        } else {
            Some((r as usize, c as usize))
        }
    }
}

fn layout(c: &Common, scene: &str, f: &str) -> Result<png::Gray, String> {
    png::read_gray(&format!("{}/scenes/{scene}/layout/{f}", c.assets))
}

pub fn build(c: &mut Common, scene: &str) -> Result<Scene, String> {
    let index = SCENES.iter().position(|s| *s == scene).ok_or(format!("unknown scene {scene}"))? as u32;
    let mut hdr = FileHeader { magic: MAGIC, version: VERSION, header_bytes: std::mem::size_of::<FileHeader>() as u32, ..Default::default() };
    hdr.layout_hash = layout_hash();
    let nb = scene.as_bytes();
    if nb.len() >= 32 {
        return Err("scene name too long".into());
    }
    hdr.scene_name[..nb.len()].copy_from_slice(nb);
    hdr.scene_index = index;
    hdr.n_levels = 1;

    // ---------------- rooms
    let ins = layout(c, scene, "floor_insseg_0.png")?;
    let sem = layout(c, scene, "floor_semseg_0.png")?;
    if ins.w != ins.h || sem.w != ins.w || sem.h != ins.h {
        return Err(format!("{scene}: insseg/semseg sizes differ or not square"));
    }
    let n = ins.w;
    let res = 0.01f64;
    let cell = 0.10f64;
    let g = n.div_ceil(10);
    let ox = -(n as f64) / 2.0 * res - res / 2.0;
    hdr.origin = [ox as f32, ox as f32];
    hdr.cell = cell as f32;
    hdr.grid_w = g as u32;
    hdr.grid_h = g as u32;
    hdr.src_px = n as u32;
    hdr.src_res = res as f32;

    // OmniGibson SegmentationMap naming: nearest resize to int(n*0.01/0.1), unique ids ascending,
    // sem id of the first pixel (row-major), names <sem>_<k> in id order within each sem.
    let m = ((n as f64) * 0.01 / 0.1) as usize;
    let ifx = 1.0 / (m as f64 / n as f64);
    let mut first_sem_og: [Option<u8>; 256] = [None; 256];
    for r in 0..m {
        let sr = ((r as f64 * ifx).floor() as usize).min(n - 1);
        for cc in 0..m {
            let sc = ((cc as f64 * ifx).floor() as usize).min(n - 1);
            let v = ins.px[sr * n + sc];
            if v != 0 && first_sem_og[v as usize].is_none() {
                first_sem_og[v as usize] = Some(sem.px[sr * n + sc]);
            }
        }
    }
    // full resolution stats
    let mut cnt = [0u64; 256];
    let mut sx = [0f64; 256];
    let mut sy = [0f64; 256];
    let mut bmin = [[f64::MAX; 2]; 256];
    let mut bmax = [[f64::MIN; 2]; 256];
    let mut first_sem_full: [Option<u8>; 256] = [None; 256];
    for r in 0..n {
        let y = (r as f64 - n as f64 / 2.0) * res;
        for cc in 0..n {
            let v = ins.px[r * n + cc] as usize;
            if v == 0 {
                continue;
            }
            let x = (cc as f64 - n as f64 / 2.0) * res;
            cnt[v] += 1;
            sx[v] += x;
            sy[v] += y;
            bmin[v] = [bmin[v][0].min(x), bmin[v][1].min(y)];
            bmax[v] = [bmax[v][0].max(x), bmax[v][1].max(y)];
            if first_sem_full[v].is_none() {
                first_sem_full[v] = Some(sem.px[r * n + cc]);
            }
        }
    }
    hdr.n_rooms_png = cnt.iter().filter(|&&k| k > 0).count() as u32;
    let mut sem_k: HashMap<u8, u32> = HashMap::new();
    let mut room_of_ins = [0u8; 256]; // RoomRec index + 1
    let mut strings = StrPool::new();
    let mut rooms = Vec::new();
    let mut room_names = Vec::new();
    for v in 1..256usize {
        if cnt[v] == 0 && first_sem_og[v].is_none() {
            continue;
        }
        let (s, og) = match first_sem_og[v] {
            Some(s) => (s, true),
            None => (first_sem_full[v].unwrap(), false),
        };
        if s == 0 || s as usize > c.room_cats.len() {
            return Err(format!("{scene}: room id {v} has semantic id {s} outside room_categories.txt"));
        }
        let base = &c.room_cats[s as usize - 1];
        let name = if og {
            let k = sem_k.entry(s).or_insert(0);
            *k += 1;
            format!("{base}_{}", *k - 1)
        } else {
            format!("{base}_px{v}")
        };
        let k = cnt[v].max(1) as f64;
        rooms.push(RoomRec {
            name: strings.add(&name),
            sem: s as u16,
            ins: v as u16,
            og_named: og as u16,
            area_m2: (cnt[v] as f64 * res * res) as f32,
            centroid: [(sx[v] / k) as f32, (sy[v] / k) as f32],
            bmin: [bmin[v][0] as f32, bmin[v][1] as f32],
            bmax: [bmax[v][0] as f32, bmax[v][1] as f32],
            ..Default::default()
        });
        room_names.push(name);
        room_of_ins[v] = rooms.len() as u8;
    }
    if rooms.len() > 64 {
        return Err(format!("{scene}: {} rooms > 64 (rooms_mask is u64)", rooms.len()));
    }
    // room grid: majority of the (up to) 10x10 pixels, ties -> lower id
    let mut room_grid = vec![0u8; g * g];
    for gr in 0..g {
        for gc in 0..g {
            let mut h = [0u16; 256];
            for r in gr * 10..((gr + 1) * 10).min(n) {
                for cc in gc * 10..((gc + 1) * 10).min(n) {
                    h[ins.px[r * n + cc] as usize] += 1;
                }
            }
            let mut best = 0usize;
            for v in 1..256 {
                if h[v] > h[best] {
                    best = v;
                }
            }
            let rid = room_of_ins[best];
            room_grid[gr * g + gc] = rid;
            if rid > 0 {
                rooms[rid as usize - 1].n_cells += 1;
            }
        }
    }
    drop(ins);
    drop(sem);

    // ---------------- traversability
    let mut trav: [Vec<u8>; 4] = Default::default();
    let mut trav_free_px = [0u64; 4];
    let mut warnings = Vec::new();
    for (k, f) in ["floor_trav_0.png", "floor_trav_no_obj_0.png", "floor_trav_no_door_0.png", "floor_trav_open_door_0.png"].iter().enumerate() {
        let t = layout(c, scene, f)?;
        if t.w != n || t.h != n {
            return Err(format!("{scene}: {f} size {}x{} != insseg {n}", t.w, t.h));
        }
        let other = t.px.iter().filter(|&&p| p != 0 && p != 255).count();
        if other > 0 {
            warnings.push(format!("{f}: {other} pixels neither 0 nor 255 (counted as blocked)"));
        }
        trav_free_px[k] = t.px.iter().filter(|&&p| p == 255).count() as u64;
        let mut bits = vec![0u8; (g * g).div_ceil(8)];
        for gr in 0..g {
            for gc in 0..g {
                let mut all = true;
                'b: for r in gr * 10..((gr + 1) * 10).min(n) {
                    for cc in gc * 10..((gc + 1) * 10).min(n) {
                        if t.px[r * n + cc] != 255 {
                            all = false;
                            break 'b;
                        }
                    }
                }
                if all {
                    let i = gr * g + gc;
                    bits[i >> 3] |= 1 << (i & 7);
                }
            }
        }
        trav[k] = bits;
    }
    hdr.trav_area_m2 = (trav_free_px[0] as f64 * res * res) as f32;

    let mut sc = Scene {
        name: scene.to_string(),
        index,
        hdr,
        strings,
        cats: vec![],
        cat_names: vec![],
        rooms,
        room_names,
        objs: vec![],
        obj_names: vec![],
        obj_index: HashMap::new(),
        boxes: vec![],
        doors: vec![],
        joints: vec![],
        in_rooms: vec![],
        room_grid,
        trav,
        trav_free_px,
        warnings,
    };

    // ---------------- objects
    let best = read_json(&format!("{}/scenes/{scene}/json/{scene}_best.json", c.assets))?;
    let oi = get(&best, &["objects_info", "init_info"]).and_then(|v| v.as_object()).ok_or("best.json: no objects_info.init_info")?;
    let reg = get(&best, &["state", "registry", "object_registry"]).and_then(|v| v.as_object()).ok_or("best.json: no object_registry")?;
    let mut errs = Vec::new();
    for (name, v) in oi {
        let a = v.get("args").ok_or(format!("{name}: no args"))?;
        let cat = a.get("category").and_then(|x| x.as_str()).ok_or(format!("{name}: no category"))?;
        let model = a.get("model").and_then(|x| x.as_str()).ok_or(format!("{name}: no model"))?;
        let scale = arr3(a.get("scale")).unwrap_or([1.0; 3]);
        let st = reg.get(name).ok_or(format!("{name}: not in object_registry"))?;
        let pos = arr3(get(st, &["root_link", "pos"])).ok_or(format!("{name}: no root_link.pos"))?;
        let quat = arr4(get(st, &["root_link", "ori"])).ok_or(format!("{name}: no root_link.ori"))?;
        let jp = f64s(st.get("joint_pos")).unwrap_or_default();
        let meta = c.meta(cat, model).map(|m| (m.bbox, m.off, m.flags));
        let Some((bbox, off, mflags)) = meta else {
            errs.push(format!("{name}: objects/{cat}/{model}/misc/metadata.json missing"));
            continue;
        };
        let mut flags = mflags | cat_flags(cat);
        if a.get("fixed_base").and_then(|x| x.as_bool()) == Some(true) {
            flags |= flag("FIXED_BASE");
        }
        if a.get("visual_only").and_then(|x| x.as_bool()) == Some(true) {
            flags |= flag("VISUAL_ONLY");
        }
        if !jp.is_empty() {
            flags |= flag("ARTICULATED");
        }
        let mut rr = [NONE16, NONE16];
        let in_rooms_off = sc.in_rooms.len() as u32;
        if let Some(list) = a.get("in_rooms").and_then(|x| x.as_array()) {
            if list.len() > 2 {
                sc.warnings.push(format!("{name}: {} in_rooms: room_a/room_b hold the first 2, IN_ROOMS has all", list.len()));
            }
            for r in list {
                let s = sc.strings.add(r.as_str().unwrap_or(""));
                sc.in_rooms.push(s);
            }
            for (k, r) in list.iter().take(2).enumerate() {
                let rn = r.as_str().unwrap_or("");
                match sc.room(rn) {
                    Some(i) => rr[k] = i,
                    None => sc.warnings.push(format!("{name}: in_rooms '{rn}' is not a room of floor_insseg_0.png (left as 0xFFFF)")),
                }
            }
        }
        let half = [scale[0] * bbox[0] / 2.0, scale[1] * bbox[1] / 2.0, scale[2] * bbox[2] / 2.0];
        let soff = [scale[0] * off[0], scale[1] * off[1], scale[2] * off[2]];
        let (cen, wh, tilt) = world_box(pos, quat, soff, half);
        if tilt > 5f64.to_radians() {
            flags |= flag("TILTED");
        }
        let ci = sc.cat(c, cat);
        sc.cats[ci as usize].n_scene += 1;
        let joint_off = sc.joints.len() as u32;
        sc.joints.extend(jp.iter().map(|&x| x as f32));
        let o = ObjRec {
            name: sc.strings.add(name),
            model: sc.strings.add(model),
            cat: ci,
            room_a: rr[0],
            room_b: rr[1],
            level: 0,
            flags,
            n_joints: jp.len() as u16,
            joint_off,
            pos: f3(pos),
            quat: f4(quat),
            scale: f3(scale),
            center: f3(cen),
            half: f3(half),
            aabb_min: f3([cen[0] - wh[0], cen[1] - wh[1], cen[2] - wh[2]]),
            aabb_max: f3([cen[0] + wh[0], cen[1] + wh[1], cen[2] + wh[2]]),
            yaw: yaw(quat) as f32,
            mass: c.avg_spec(cat).0.map_or(f32::NAN, |x| x as f32),
            n_in_rooms: (sc.in_rooms.len() as u32 - in_rooms_off) as u16,
            in_rooms_off,
        };
        let oidx = sc.objs.len();
        if flags & (flag("FLOOR") | flag("CEILING")) == 0 {
            sc.boxes.push(BoxRec {
                center: o.center,
                half: o.half,
                yaw: o.yaw,
                cat: ci,
                room: rr[0],
                flags,
                obj: oidx as u32,
                zmin: o.aabb_min[2],
                zmax: o.aabb_max[2],
            });
        }
        if cat.contains("door") {
            let kind = match cat {
                "door" => 0,
                "sliding_door" => 1,
                "garage_door" => 2,
                "elevator_door" => 3,
                _ => 4,
            };
            sc.doors.push(DoorRec {
                obj: oidx as u32,
                room_a: rr[0],
                room_b: rr[1],
                kind,
                n_joints: o.n_joints,
                joint_off,
                center: o.center,
                half: o.half,
                yaw: o.yaw,
                reserved: 0,
            });
        }
        sc.obj_index.insert(name.clone(), oidx);
        sc.obj_names.push(name.clone());
        sc.objs.push(o);
    }
    if !errs.is_empty() {
        return Err(format!("{scene}: {} errors (rooms {:?}):\n  {}", errs.len(), sc.rooms.iter().zip(&sc.room_names).map(|(r, n)| format!("{n} ins{} sem{} og{} px{}", r.ins, r.sem, r.og_named, (r.area_m2 * 1e4).round())).collect::<Vec<_>>(), errs.join("\n  ")));
    }
    Ok(sc)
}

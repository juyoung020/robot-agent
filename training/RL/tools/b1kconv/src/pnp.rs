//! Pick-and-place candidate table (curriculum B3–B5): graspable objects, supports (ontop surfaces,
//! open containers, room floors), and (object, source support, target support) pairs, with
//! approach / reachability from the 0.10 m traversability grid (floor_trav_0).
//!
//! Limits are parameters (`Limits`): `outer` decides inclusion, `inner` only sets flag bits.
//! Defaults = the doc's loose (outer) and strict (inner) criteria; E0 measurements replace them.

use crate::format::*;
use crate::scene::Scene;
use crate::util::{cstr, world_box};
use crate::write::Tables;

pub fn limits_default() -> [LimitsRec; 2] {
    let reach = 0.40; // OMX-F reach (ROBOTIS spec); measured from the base cell center (estimate)
    [
        LimitsRec { pick_z: 0.60, place_top: 0.62, inside_margin: 0.05, max_mass: 0.5, max_w: 0.10, min_side: 0.15, min_top: 0.05, reach },
        LimitsRec { pick_z: 0.45, place_top: 0.50, inside_margin: 0.05, max_mass: 0.25, max_w: 0.08, min_side: 0.15, min_top: 0.05, reach },
    ]
}

/// "pick_z=0.5,max_mass=0.3" -> overrides
pub fn parse_limits(l: &mut LimitsRec, s: &str) -> Result<(), String> {
    for kv in s.split(',').filter(|x| !x.is_empty()) {
        let (k, v) = kv.split_once('=').ok_or(format!("bad limit '{kv}'"))?;
        let v: f32 = v.parse().map_err(|_| format!("bad value in '{kv}'"))?;
        match k {
            "pick_z" => l.pick_z = v,
            "place_top" => l.place_top = v,
            "inside_margin" => l.inside_margin = v,
            "max_mass" => l.max_mass = v,
            "max_w" => l.max_w = v,
            "min_side" => l.min_side = v,
            "min_top" => l.min_top = v,
            "reach" => l.reach = v,
            _ => return Err(format!("unknown limit '{k}'")),
        }
    }
    Ok(())
}

// open-top containers / closed articulated / shelves (doc script keyword lists)
const OPEN_TOP: [&str; 21] = ["basket", "box", "bin", "bucket", "bowl", "bag", "tote", "vase", "trash", "ashcan", "cup", "case", "crate", "container", "carton", "toy_box", "hamper", "plate", "tray", "sheet", "pan"];
const CLOSED: [&str; 13] = ["cabinet", "refrigerator", "fridge", "microwave", "oven", "washer", "dishwasher", "car", "toolbox", "recycling_bin", "drawer", "hinged_jar", "trunk"];
const SHELF: [&str; 4] = ["bookcase", "shelf", "hall_tree", "rack"];
fn any_in(s: &str, l: &[&str]) -> bool {
    l.iter().any(|c| s.contains(c))
}

/// one object in world coordinates
#[derive(Clone)]
struct Body {
    obj: u32,
    inst: u32,
    cat: String,
    flags: u32,
    center: [f32; 3],
    half: [f32; 3],
    yaw: f32,
    lo: [f32; 3],
    hi: [f32; 3],
    mass: f32,
}

const EXCLUDE: &[&str] = &["WALL", "FLOOR", "CEILING", "DOOR", "WINDOW", "STAIRS", "CARPET", "AGENT", "SYSTEM", "UNMAPPED", "WILDCARD"];

pub struct Pnp {
    pub limits: [LimitsRec; 2],
    pub picks: Vec<PickRec>,
    pub places: Vec<PlaceRec>,
    pub pairs: Vec<PairRec>,
    pub ranges: Vec<PnpRange>,
    pub n_comp: usize,
}

struct Grid<'a> {
    sc: &'a Scene,
    comp: Vec<u16>,
}

impl<'a> Grid<'a> {
    fn new(sc: &'a Scene) -> (Self, usize) {
        let (w, h) = (sc.hdr.grid_w as usize, sc.hdr.grid_h as usize);
        let free = |i: usize| sc.trav[0][i >> 3] >> (i & 7) & 1 == 1;
        let mut lab = vec![u32::MAX; w * h];
        let mut sizes: Vec<usize> = vec![];
        let mut q = vec![];
        for s in 0..w * h {
            if !free(s) || lab[s] != u32::MAX {
                continue;
            }
            let id = sizes.len() as u32;
            lab[s] = id;
            q.clear();
            q.push(s);
            let mut n = 0;
            while let Some(i) = q.pop() {
                n += 1;
                let (r, c) = (i / w, i % w);
                let mut nb = |j: usize| {
                    if free(j) && lab[j] == u32::MAX {
                        lab[j] = id;
                        q.push(j);
                    }
                };
                if r > 0 { nb(i - w); }
                if r + 1 < h { nb(i + w); }
                if c > 0 { nb(i - 1); }
                if c + 1 < w { nb(i + 1); }
            }
            sizes.push(n);
        }
        // relabel: largest component = 0; ids >= 0xfffe collapse to 0xfffe ("tiny")
        let mut order: Vec<usize> = (0..sizes.len()).collect();
        order.sort_by(|a, b| sizes[*b].cmp(&sizes[*a]).then(a.cmp(b)));
        let mut rank = vec![0u16; sizes.len()];
        for (k, &o) in order.iter().enumerate() {
            rank[o] = k.min(0xfffe) as u16;
        }
        let comp = lab.iter().map(|&l| if l == u32::MAX { NONE16 } else { rank[l as usize] }).collect();
        (Grid { sc, comp }, sizes.len())
    }
    /// free cells within `reach` of the xy box [lo, hi]; returns (count, dominant component)
    fn approach(&self, lo: [f32; 3], hi: [f32; 3], reach: f32) -> (u32, u16) {
        let hd = &self.sc.hdr;
        let (w, h) = (hd.grid_w as i64, hd.grid_h as i64);
        let cell = hd.cell;
        let c0 = (((lo[0] - reach - hd.origin[0]) / cell).floor() as i64).max(0);
        let c1 = (((hi[0] + reach - hd.origin[0]) / cell).floor() as i64).min(w - 1);
        let r0 = (((lo[1] - reach - hd.origin[1]) / cell).floor() as i64).max(0);
        let r1 = (((hi[1] + reach - hd.origin[1]) / cell).floor() as i64).min(h - 1);
        let mut n = 0u32;
        let mut votes: Vec<(u16, u32)> = vec![];
        for r in r0..=r1 {
            for c in c0..=c1 {
                let i = (r * w + c) as usize;
                let k = self.comp[i];
                if k == NONE16 {
                    continue;
                }
                let x = hd.origin[0] + (c as f32 + 0.5) * cell;
                let y = hd.origin[1] + (r as f32 + 0.5) * cell;
                let dx = (lo[0] - x).max(x - hi[0]).max(0.0);
                let dy = (lo[1] - y).max(y - hi[1]).max(0.0);
                if dx * dx + dy * dy <= reach * reach {
                    n += 1;
                    match votes.iter_mut().find(|v| v.0 == k) {
                        Some(v) => v.1 += 1,
                        None => votes.push((k, 1)),
                    }
                }
            }
        }
        let comp = votes.iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map_or(NONE16, |v| v.0);
        (n, comp)
    }
    fn room(&self, x: f32, y: f32) -> u16 {
        match self.sc.cell(x as f64, y as f64) {
            Some((r, c)) => {
                let v = self.sc.room_grid[r * self.sc.hdr.grid_w as usize + c];
                if v == 0 { NONE16 } else { v as u16 - 1 }
            }
            None => NONE16,
        }
    }
    fn comp_at(&self, x: f32, y: f32) -> u16 {
        match self.sc.cell(x as f64, y as f64) {
            Some((r, c)) => self.comp[r * self.sc.hdr.grid_w as usize + c],
            None => NONE16,
        }
    }
}

fn excluded(flags: u32) -> bool {
    EXCLUDE.iter().any(|f| flags & flag(f) != 0)
}

fn min_w(b: &Body) -> f32 {
    2.0 * b.half[0].min(b.half[1])
}

fn grasp(b: &Body, l: &LimitsRec) -> bool {
    !excluded(b.flags) && b.flags & flag("FIXED_BASE") == 0 && b.flags & flag("HAS_BBOX") != 0 && b.lo[2] <= l.pick_z && (b.mass.is_nan() || b.mass <= l.max_mass) && min_w(b) <= l.max_w
}

/// 0 = not a support, 1 ontop, 2 inside
fn support(b: &Body, l: &LimitsRec) -> u16 {
    if excluded(b.flags) || b.flags & flag("HAS_BBOX") == 0 {
        return 0;
    }
    if any_in(&b.cat, &OPEN_TOP) && !any_in(&b.cat, &CLOSED) && !any_in(&b.cat, &SHELF) {
        return if b.hi[2] <= l.place_top + l.inside_margin { 2 } else { 0 };
    }
    if any_in(&b.cat, &CLOSED) || any_in(&b.cat, &SHELF) {
        return 0; // closed articulated / shelf compartments: out of scope (B6 cut)
    }
    if b.hi[2] <= l.place_top && b.hi[2] >= l.min_top && min_w(b) >= l.min_side {
        1
    } else {
        0
    }
}

fn place_rec(b: &Body, kind: u16, g: &Grid, l: &[LimitsRec; 2], room_hint: u16) -> PlaceRec {
    let (n, comp) = g.approach(b.lo, b.hi, l[0].reach);
    let room = match g.room(b.center[0], b.center[1]) {
        NONE16 => room_hint,
        r => r,
    };
    PlaceRec {
        obj: b.obj,
        inst: b.inst,
        kind,
        room,
        center: b.center,
        half: [b.half[0], b.half[1]],
        yaw: b.yaw,
        top: b.hi[2],
        n_approach: n,
        comp,
        flags: (support(b, &l[1]) == kind) as u16,
    }
}

fn pick_rec(b: &Body, g: &Grid, l: &[LimitsRec; 2], extra: u32) -> PickRec {
    let (n, comp) = g.approach(b.lo, b.hi, l[0].reach);
    let mut f = extra;
    if grasp(b, &l[1]) {
        f |= 1;
    }
    if b.mass.is_nan() {
        f |= 2;
    }
    PickRec {
        obj: b.obj,
        inst: b.inst,
        center: b.center,
        z0: b.lo[2],
        top: b.hi[2],
        min_w: min_w(b),
        mass: b.mass,
        yaw: b.yaw,
        room: g.room(b.center[0], b.center[1]),
        comp,
        n_approach: n,
        src_place: u32::MAX,
        flags: f,
        reserved: [0; 2],
    }
}

/// support under / around a pick: containing open container, else highest surface whose
/// footprint holds the center and whose top is within [-0.15, +0.05] m of the object bottom,
/// else the floor of the room when the object is near z = 0.
fn find_src(p: &PickRec, places: &[PlaceRec], cand: &[u32]) -> u32 {
    let inside_xy = |pl: &PlaceRec, m: f32| {
        let (dx, dy) = (p.center[0] - pl.center[0], p.center[1] - pl.center[1]);
        let (cs, sn) = (pl.yaw.cos(), pl.yaw.sin());
        let (lx, ly) = (cs * dx + sn * dy, -sn * dx + cs * dy);
        lx.abs() <= pl.half[0] + m && ly.abs() <= pl.half[1] + m
    };
    let mut best = u32::MAX;
    let mut best_top = f32::MIN;
    for &k in cand {
        let pl = &places[k as usize];
        if pl.obj == p.obj && pl.inst == p.inst {
            continue;
        }
        match pl.kind {
            // inside an open container: center over it, bottom below its rim (container wins: +10)
            2 if inside_xy(pl, 0.0) && p.z0 < pl.top && p.z0 >= pl.top - 0.6 && pl.top + 10.0 > best_top => {
                best = k;
                best_top = pl.top + 10.0;
            }
            1 if inside_xy(pl, 0.02) && p.z0 >= pl.top - 0.15 && p.z0 <= pl.top + 0.05 && pl.top > best_top => {
                best = k;
                best_top = pl.top;
            }
            _ => {}
        }
    }
    if best == u32::MAX && p.z0 < 0.10 {
        for &k in cand {
            let pl = &places[k as usize];
            if pl.kind == 3 && pl.room == p.room {
                return k;
            }
        }
    }
    best
}

pub fn build(sc: &Scene, t: &Tables, limits: [LimitsRec; 2]) -> Pnp {
    let (g, n_comp) = Grid::new(sc);
    let l = &limits;
    let mut out = Pnp { limits, picks: vec![], places: vec![], pairs: vec![], ranges: vec![], n_comp };

    // ---- scene level (static)
    let scene_bodies: Vec<Body> = sc
        .objs
        .iter()
        .enumerate()
        .map(|(i, o)| Body {
            obj: i as u32,
            inst: u32::MAX,
            cat: sc.cat_names[o.cat as usize].clone(),
            flags: o.flags,
            center: o.center,
            half: o.half,
            yaw: o.yaw,
            lo: o.aabb_min,
            hi: o.aabb_max,
            mass: o.mass,
        })
        .collect();
    let mut s_places = vec![];
    for (ri, r) in sc.rooms.iter().enumerate() {
        let (n, comp) = (r.n_cells, {
            // dominant component of the room's free cells
            let mut votes: std::collections::HashMap<u16, u32> = Default::default();
            for (i, &v) in sc.room_grid.iter().enumerate() {
                if v as usize == ri + 1 && g.comp[i] != NONE16 {
                    *votes.entry(g.comp[i]).or_default() += 1;
                }
            }
            votes.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map_or(NONE16, |v| v.0)
        });
        let free_in_room = sc.room_grid.iter().enumerate().filter(|(i, &v)| v as usize == ri + 1 && g.comp[*i] != NONE16).count() as u32;
        let _ = n;
        out.places.push(PlaceRec {
            obj: u32::MAX,
            inst: u32::MAX,
            kind: 3,
            room: ri as u16,
            center: [r.centroid[0], r.centroid[1], 0.0],
            half: [(r.bmax[0] - r.bmin[0]) / 2.0, (r.bmax[1] - r.bmin[1]) / 2.0],
            yaw: 0.0,
            top: 0.0,
            n_approach: free_in_room,
            comp,
            flags: 1,
        });
        s_places.push(out.places.len() as u32 - 1);
    }
    for b in &scene_bodies {
        let k = support(b, &l[0]);
        if k != 0 {
            out.places.push(place_rec(b, k, &g, l, sc.objs[b.obj as usize].room_a));
            s_places.push(out.places.len() as u32 - 1);
        }
    }
    let mut s_picks = vec![];
    for b in &scene_bodies {
        if grasp(b, &l[0]) {
            let mut p = pick_rec(b, &g, l, 0);
            p.src_place = find_src(&p, &out.places, &s_places);
            out.picks.push(p);
            s_picks.push(out.picks.len() as u32 - 1);
        }
    }
    let scene_pair_off = out.pairs.len() as u32;
    // scene pairs: every scene pick x every scene support in the same TRAV component or same room
    let mk_pair = |p: &PickRec, pi: u32, d: &PlaceRec, di: u32, robot_comp: u16| -> PairRec {
        let mut reach = 0u8;
        if p.comp != NONE16 && p.comp == d.comp {
            reach |= 1;
        }
        if p.flags & 1 != 0 && d.flags & 1 != 0 {
            reach |= 2;
        }
        if robot_comp != NONE16 && robot_comp == p.comp && reach & 1 != 0 {
            reach |= 4;
        }
        PairRec {
            pick: pi,
            src: p.src_place,
            dst: di,
            rel: if d.kind == 2 { pred_id("inside").unwrap() } else { pred_id("ontop").unwrap() },
            reachable: reach,
            room_pick: p.room,
            room_dst: d.room,
            reserved: 0,
            dist: (p.center[0] - d.center[0]).hypot(p.center[1] - d.center[1]),
        }
    };
    let pair_ok = |p: &PickRec, d: &PlaceRec, di: u32| -> bool {
        di != p.src_place && !(d.obj == p.obj && d.inst == p.inst) && ((d.comp != NONE16 && d.comp == p.comp) || (d.room != NONE16 && d.room == p.room))
    };
    // scene level is appended after the instances (PNP_RANGES last entry) — collect now
    let mut scene_pairs = vec![];
    for &pi in &s_picks {
        let p = out.picks[pi as usize];
        for &di in &s_places {
            let d = &out.places[di as usize];
            if pair_ok(&p, d, di) {
                scene_pairs.push(mk_pair(&p, pi, d, di, NONE16));
            }
        }
    }
    let _ = scene_pair_off;
    let scene_range = PnpRange {
        pick_off: 0,
        place_off: 0,
        pair_off: 0,
        n_pick: s_picks.len() as u16,
        n_place: out.places.len() as u16,
        n_pair: scene_pairs.len() as u32,
        robot_comp: NONE16,
        reserved: 0,
    };
    out.pairs = scene_pairs;

    // ---- instance level: task objects at their instance poses
    for (ti, tr) in t.tasks.iter().enumerate() {
        let tobjs = &t.tobjs[tr.obj_off as usize..(tr.obj_off + tr.n_obj as u32) as usize];
        let name = |k: usize| cstr(&sc.strings.bytes, tobjs[k].inst);
        // BDDL init: inside a closed container
        let mut in_closed = vec![false; tobjs.len()];
        for li in &t.lits[tr.init_off as usize..(tr.init_off + tr.n_init as u32) as usize] {
            if PREDS[li.pred as usize] == "inside" && li.neg == 0 && li.nargs == 2 && li.arg[1] & (ARG_VAR | ARG_ROOM) == 0 {
                if any_in(&name(li.arg[1] as usize), &CLOSED) {
                    in_closed[li.arg[0] as usize] = true;
                }
            }
        }
        let room_ok = |r: u16| r != NONE16 && (r >= 64 || tr.rooms_mask & (1u64 << r) != 0 || tr.rooms_mask == 0);
        for ii in tr.inst_off..tr.inst_off + tr.n_inst {
            let ins = &t.insts[ii as usize];
            let poses = &t.poses[ins.pose_off as usize..(ins.pose_off + ins.n_pose as u32) as usize];
            let mut bodies = vec![];
            for (k, o) in tobjs.iter().enumerate() {
                let p = &poses[k];
                if p.src == 0 || o.cat == NONE16 {
                    continue;
                }
                let q = [p.quat[0] as f64, p.quat[1] as f64, p.quat[2] as f64, p.quat[3] as f64];
                let pos = [p.pos[0] as f64, p.pos[1] as f64, p.pos[2] as f64];
                let half = [o.half[0] as f64, o.half[1] as f64, o.half[2] as f64];
                let (c, wh, _) = world_box(pos, q, [o.offset[0] as f64, o.offset[1] as f64, o.offset[2] as f64], half);
                let lo = [(c[0] - wh[0]) as f32, (c[1] - wh[1]) as f32, (c[2] - wh[2]) as f32];
                let hi = [(c[0] + wh[0]) as f32, (c[1] + wh[1]) as f32, (c[2] + wh[2]) as f32];
                bodies.push((
                    k,
                    Body {
                        obj: PNP_TASKOBJ | (tr.obj_off + k as u32),
                        inst: ii,
                        cat: sc.cat_names[o.cat as usize].clone(),
                        flags: o.flags,
                        center: [c[0] as f32, c[1] as f32, c[2] as f32],
                        half: o.half,
                        yaw: crate::util::yaw(q) as f32,
                        lo,
                        hi,
                        mass: o.mass,
                    },
                ));
            }
            let place_off = out.places.len() as u32;
            let mut cand_places: Vec<u32> = s_places.iter().copied().filter(|&k| {
                let pl = &out.places[k as usize];
                // a scene object in the task scope is replaced by its instance entry
                let moved = pl.obj != u32::MAX && tobjs.iter().any(|o| o.scene_obj >= 0 && o.scene_obj as u32 == pl.obj);
                !moved && room_ok(pl.room)
            }).collect();
            for (_, b) in &bodies {
                let k = support(b, &l[0]);
                if k != 0 {
                    out.places.push(place_rec(b, k, &g, l, NONE16));
                    cand_places.push(out.places.len() as u32 - 1);
                }
            }
            let pick_off = out.picks.len() as u32;
            let pair_off = out.pairs.len() as u32;
            let robot_comp = g.comp_at(ins.robot_pos[0], ins.robot_pos[1]);
            for (k, b) in &bodies {
                if !grasp(b, &l[0]) {
                    continue;
                }
                let mut extra = 0;
                if tobjs[*k].scene_obj >= 0 {
                    extra |= 4;
                }
                if in_closed[*k] {
                    extra |= 8;
                }
                let mut p = pick_rec(b, &g, l, extra);
                p.src_place = find_src(&p, &out.places, &cand_places);
                let pi = out.picks.len() as u32;
                out.picks.push(p);
                for &di in &cand_places {
                    let d = &out.places[di as usize];
                    if pair_ok(&p, d, di) {
                        let pr = mk_pair(&p, pi, d, di, robot_comp);
                        out.pairs.push(pr);
                    }
                }
            }
            out.ranges.push(PnpRange {
                pick_off,
                place_off,
                pair_off,
                n_pick: (out.picks.len() as u32 - pick_off) as u16,
                n_place: (out.places.len() as u32 - place_off) as u16,
                n_pair: out.pairs.len() as u32 - pair_off,
                robot_comp,
                reserved: 0,
            });
            let _ = ti;
        }
    }
    out.ranges.push(scene_range);
    out
}

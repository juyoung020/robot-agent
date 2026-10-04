//! Pick-and-place candidate table (curriculum B3–B5): graspable objects, supports (ontop surfaces,
//! open containers, room floors), and (object, source support, target support) pairs, with
//! approach / reachability from the 0.10 m traversability grid (floor_trav_0) cut at floor steps
//! larger than the base can cross.
//!
//! Limits are parameters (`LimitsRec`): `outer` decides inclusion, `inner` only sets flag bits.
//! Defaults = E0 measured values (CURRICULUM_BEHAVIOR2026 5.3): outer = loose, inner = strict.

use crate::format::*;
use crate::scene::Scene;
use crate::util::{cstr, world_box};
use crate::write::Tables;

pub fn limits_default() -> [LimitsRec; 2] {
    // E0 (5.3): side reach 0.27 / front 0.17 m beyond the body edge; body edge from the joint1 axis
    // side 0.11 / front 0.21 m; top-down grasps up to 0.25 m; above that a side grasp within
    // 0.10 m of the surface edge; floor steps 0.02 (strict) / 0.025 m (crossed in sim).
    let common = LimitsRec {
        inside_margin: 0.05,
        min_side: 0.15,
        min_top: 0.05,
        reach_side: 0.27,
        reach_front: 0.17,
        edge_side: 0.11,
        edge_front: 0.21,
        topdown_z: 0.25,
        edge_dist: 0.10,
        ..Default::default()
    };
    [
        LimitsRec { pick_z: 0.50, place_top: 0.52, max_mass: 0.40, max_w: 0.06, threshold: 0.025, ..common },
        LimitsRec { pick_z: 0.45, place_top: 0.48, max_mass: 0.25, max_w: 0.04, threshold: 0.02, ..common },
    ]
}

/// "pick_z=0.5,max_mass=0.3" -> overrides
pub fn parse_limits(l: &mut LimitsRec, s: &str) -> Result<(), String> {
    for kv in s.split(',').filter(|x| !x.is_empty()) {
        let (k, v) = kv.split_once('=').ok_or(format!("bad limit '{kv}'"))?;
        let v: f32 = v.parse().map_err(|_| format!("bad value in '{kv}'"))?;
        let f = match k {
            "pick_z" => &mut l.pick_z,
            "place_top" => &mut l.place_top,
            "inside_margin" => &mut l.inside_margin,
            "max_mass" => &mut l.max_mass,
            "max_w" => &mut l.max_w,
            "min_side" => &mut l.min_side,
            "min_top" => &mut l.min_top,
            "reach_side" => &mut l.reach_side,
            "reach_front" => &mut l.reach_front,
            "edge_side" => &mut l.edge_side,
            "edge_front" => &mut l.edge_front,
            "topdown_z" => &mut l.topdown_z,
            "edge_dist" => &mut l.edge_dist,
            "threshold" => &mut l.threshold,
            _ => return Err(format!("unknown limit '{k}'")),
        };
        *f = v;
    }
    Ok(())
}

/// base-center (joint1 axis) to footprint distance for a top-down grasp / place (best heading)
fn reach_low(l: &LimitsRec) -> f32 {
    (l.edge_side + l.reach_side).max(l.edge_front + l.reach_front)
}
/// side grasp / place above topdown_z: body side edge at the surface edge, object within edge_dist
fn reach_high(l: &LimitsRec) -> f32 {
    l.edge_side + l.edge_dist
}

// open-top containers / closed articulated / shelves (doc script keyword lists)
const OPEN_TOP: [&str; 21] = ["basket", "box", "bin", "bucket", "bowl", "bag", "tote", "vase", "trash", "ashcan", "cup", "case", "crate", "container", "carton", "toy_box", "hamper", "plate", "tray", "sheet", "pan"];
const CLOSED: [&str; 13] = ["cabinet", "refrigerator", "fridge", "microwave", "oven", "washer", "dishwasher", "car", "toolbox", "recycling_bin", "drawer", "hinged_jar", "trunk"];
const SHELF: [&str; 4] = ["bookcase", "shelf", "hall_tree", "rack"];
fn any_in(s: &str, l: &[&str]) -> bool {
    l.iter().any(|c| s.contains(c))
}

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
pub const FLOOR_NONE: i16 = i16::MIN;

pub struct Pnp {
    pub limits: [LimitsRec; 2],
    pub picks: Vec<PickRec>,
    pub places: Vec<PlaceRec>,
    pub pairs: Vec<PairRec>,
    pub ranges: Vec<PnpRange>,
    pub floor_z: Vec<i16>,
    pub n_comp: [usize; 2],
    pub step_cuts: [usize; 2], // free-cell neighbour links cut by the floor-step threshold
}

/// floor top height (mm) per cell: floor objects whose in_rooms holds the cell's room (the one
/// whose box covers the cell, else any of them); cells in no room: smallest floor box covering it
pub fn floor_z(sc: &Scene) -> Vec<i16> {
    let (w, h) = (sc.hdr.grid_w as usize, sc.hdr.grid_h as usize);
    let pool: Vec<u32> = sc.in_rooms.clone();
    let floors: Vec<(usize, Vec<u16>)> = sc
        .objs
        .iter()
        .enumerate()
        .filter(|(_, o)| o.flags & flag("FLOOR") != 0)
        .map(|(i, o)| {
            let rooms = (0..o.n_in_rooms as usize).filter_map(|k| sc.room(&cstr(&sc.strings.bytes, pool[o.in_rooms_off as usize + k]))).collect();
            (i, rooms)
        })
        .collect();
    let mm = |z: f32| (z * 1000.0).round().clamp(-32767.0, 32767.0) as i16;
    let mut out = vec![FLOOR_NONE; w * h];
    for r in 0..h {
        let y = sc.hdr.origin[1] + (r as f32 + 0.5) * sc.hdr.cell;
        for c in 0..w {
            let x = sc.hdr.origin[0] + (c as f32 + 0.5) * sc.hdr.cell;
            let covers = |o: &ObjRec| x >= o.aabb_min[0] && x <= o.aabb_max[0] && y >= o.aabb_min[1] && y <= o.aabb_max[1];
            let room = sc.room_grid[r * w + c];
            let mut best: Option<f32> = None;
            if room > 0 {
                let mine: Vec<&ObjRec> = floors.iter().filter(|(_, rs)| rs.contains(&(room as u16 - 1))).map(|(i, _)| &sc.objs[*i]).collect();
                let cov: Vec<&&ObjRec> = mine.iter().filter(|o| covers(o)).collect();
                let src: Vec<f32> = if cov.is_empty() { mine.iter().map(|o| o.aabb_max[2]).collect() } else { cov.iter().map(|o| o.aabb_max[2]).collect() };
                best = src.into_iter().reduce(f32::max);
            }
            if best.is_none() {
                let mut area = f32::MAX;
                for (i, _) in &floors {
                    let o = &sc.objs[*i];
                    let a = (o.aabb_max[0] - o.aabb_min[0]) * (o.aabb_max[1] - o.aabb_min[1]);
                    if covers(o) && a < area {
                        area = a;
                        best = Some(o.aabb_max[2]);
                    }
                }
            }
            if let Some(z) = best {
                out[r * w + c] = mm(z);
            }
        }
    }
    out
}

struct Grid<'a> {
    sc: &'a Scene,
    comp: [Vec<u16>; 2],
}

/// connected components of free cells; neighbours connect unless their floor heights differ by
/// more than `thr_mm` (unknown height connects). Largest component = 0.
fn components(sc: &Scene, fz: &[i16], thr_mm: i32) -> (Vec<u16>, usize, usize) {
    let (w, h) = (sc.hdr.grid_w as usize, sc.hdr.grid_h as usize);
    let free = |i: usize| sc.trav[0][i >> 3] >> (i & 7) & 1 == 1;
    let link = |a: usize, b: usize| fz[a] == FLOOR_NONE || fz[b] == FLOOR_NONE || (fz[a] as i32 - fz[b] as i32).abs() <= thr_mm;
    let mut cuts = 0;
    for i in 0..w * h {
        if !free(i) {
            continue;
        }
        if i % w + 1 < w && free(i + 1) && !link(i, i + 1) {
            cuts += 1;
        }
        if i + w < w * h && free(i + w) && !link(i, i + w) {
            cuts += 1;
        }
    }
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
                if free(j) && lab[j] == u32::MAX && link(i, j) {
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
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|a, b| sizes[*b].cmp(&sizes[*a]).then(a.cmp(b)));
    let mut rank = vec![0u16; sizes.len()];
    for (k, &o) in order.iter().enumerate() {
        rank[o] = k.min(0xfffe) as u16;
    }
    (lab.iter().map(|&l| if l == u32::MAX { NONE16 } else { rank[l as usize] }).collect(), sizes.len(), cuts)
}

impl<'a> Grid<'a> {
    /// free cells within `reach` of the xy box [lo, hi]: (count, dominant component outer, inner)
    fn approach(&self, lo: [f32; 3], hi: [f32; 3], reach: f32) -> (u32, u16, u16) {
        let hd = &self.sc.hdr;
        let (w, h) = (hd.grid_w as i64, hd.grid_h as i64);
        let cell = hd.cell;
        let c0 = (((lo[0] - reach - hd.origin[0]) / cell).floor() as i64).max(0);
        let c1 = (((hi[0] + reach - hd.origin[0]) / cell).floor() as i64).min(w - 1);
        let r0 = (((lo[1] - reach - hd.origin[1]) / cell).floor() as i64).max(0);
        let r1 = (((hi[1] + reach - hd.origin[1]) / cell).floor() as i64).min(h - 1);
        let mut n = 0u32;
        let mut votes: [Vec<(u16, u32)>; 2] = [vec![], vec![]];
        for r in r0..=r1 {
            for c in c0..=c1 {
                let i = (r * w + c) as usize;
                if self.comp[0][i] == NONE16 {
                    continue;
                }
                let x = hd.origin[0] + (c as f32 + 0.5) * cell;
                let y = hd.origin[1] + (r as f32 + 0.5) * cell;
                let dx = (lo[0] - x).max(x - hi[0]).max(0.0);
                let dy = (lo[1] - y).max(y - hi[1]).max(0.0);
                if dx * dx + dy * dy <= reach * reach {
                    n += 1;
                    for k in 0..2 {
                        let id = self.comp[k][i];
                        match votes[k].iter_mut().find(|v| v.0 == id) {
                            Some(v) => v.1 += 1,
                            None => votes[k].push((id, 1)),
                        }
                    }
                }
            }
        }
        let dom = |v: &Vec<(u16, u32)>| v.iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map_or(NONE16, |v| v.0);
        (n, dom(&votes[0]), dom(&votes[1]))
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
    fn comp_at(&self, k: usize, x: f32, y: f32) -> u16 {
        match self.sc.cell(x as f64, y as f64) {
            Some((r, c)) => self.comp[k][r * self.sc.hdr.grid_w as usize + c],
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

/// size / mass / height limits (the edge condition is applied after the source support is known)
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
    let reach = if b.hi[2] <= l[0].topdown_z { reach_low(&l[0]) } else { reach_high(&l[0]) };
    let (n, comp, comp_inner) = g.approach(b.lo, b.hi, reach);
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
        comp_inner,
        reserved: 0,
    }
}

/// support under / around a pick: containing open container, else highest surface whose
/// footprint holds the center and whose top is within [-0.15, +0.05] m of the object bottom,
/// else the floor of the room when the object is near z = 0.
fn find_src(center: [f32; 3], z0: f32, room: u16, obj: u32, inst: u32, places: &[PlaceRec], cand: &[u32]) -> u32 {
    let inside_xy = |pl: &PlaceRec, m: f32| {
        let (lx, ly) = local(pl, center);
        lx.abs() <= pl.half[0] + m && ly.abs() <= pl.half[1] + m
    };
    let mut best = u32::MAX;
    let mut best_top = f32::MIN;
    for &k in cand {
        let pl = &places[k as usize];
        if pl.obj == obj && pl.inst == inst {
            continue;
        }
        match pl.kind {
            // inside an open container: center over it, bottom below its rim (container wins: +10)
            2 if inside_xy(pl, 0.0) && z0 < pl.top && z0 >= pl.top - 0.6 && pl.top + 10.0 > best_top => {
                best = k;
                best_top = pl.top + 10.0;
            }
            1 if inside_xy(pl, 0.02) && z0 >= pl.top - 0.15 && z0 <= pl.top + 0.05 && pl.top > best_top => {
                best = k;
                best_top = pl.top;
            }
            _ => {}
        }
    }
    if best == u32::MAX && z0 < 0.10 {
        for &k in cand {
            let pl = &places[k as usize];
            if pl.kind == 3 && pl.room == room {
                return k;
            }
        }
    }
    best
}

fn local(pl: &PlaceRec, p: [f32; 3]) -> (f32, f32) {
    let (dx, dy) = (p[0] - pl.center[0], p[1] - pl.center[1]);
    let (cs, sn) = (pl.yaw.cos(), pl.yaw.sin());
    (cs * dx + sn * dy, -sn * dx + cs * dy)
}

/// candidate pick (size/mass/height already passed `outer`) -> record, or None when the side-grasp
/// edge condition fails under `outer`
fn pick_rec(b: &Body, g: &Grid, l: &[LimitsRec; 2], extra: u32, places: &[PlaceRec], cand: &[u32]) -> Option<PickRec> {
    let room = g.room(b.center[0], b.center[1]);
    let src = find_src(b.center, b.lo[2], room, b.obj, b.inst, places, cand);
    let mut f = extra;
    let mut edge_d = f32::NAN;
    if src != u32::MAX && places[src as usize].kind == 1 {
        let pl = &places[src as usize];
        let (lx, ly) = local(pl, b.center);
        edge_d = (pl.half[0] - lx.abs()).min(pl.half[1] - ly.abs()).max(0.0);
    }
    let side = |lim: &LimitsRec| b.lo[2] > lim.topdown_z;
    let edge_ok = |lim: &LimitsRec| !side(lim) || edge_d.is_nan() || edge_d <= lim.edge_dist;
    if !edge_ok(&l[0]) {
        return None;
    }
    if side(&l[0]) {
        f |= 16;
        if edge_d.is_nan() {
            f |= 32;
        }
    }
    if grasp(b, &l[1]) && edge_ok(&l[1]) && !(side(&l[1]) && edge_d.is_nan()) {
        f |= 1;
    }
    if b.mass.is_nan() {
        f |= 2;
    }
    let reach = if side(&l[0]) { reach_high(&l[0]) } else { reach_low(&l[0]) };
    let (n, comp, comp_inner) = g.approach(b.lo, b.hi, reach);
    Some(PickRec {
        obj: b.obj,
        inst: b.inst,
        center: b.center,
        z0: b.lo[2],
        top: b.hi[2],
        min_w: min_w(b),
        mass: b.mass,
        yaw: b.yaw,
        room,
        comp,
        n_approach: n,
        src_place: src,
        flags: f,
        edge_d,
        comp_inner,
        reserved: 0,
    })
}

fn mk_pair(p: &PickRec, pi: u32, d: &PlaceRec, di: u32, robot: [u16; 2]) -> PairRec {
    let mut reach = 0u8;
    if p.comp != NONE16 && p.comp == d.comp {
        reach |= 1;
    }
    if p.flags & 1 != 0 && d.flags & 1 != 0 {
        reach |= 2;
    }
    if robot[0] != NONE16 && robot[0] == p.comp && reach & 1 != 0 {
        reach |= 4;
    }
    if robot[1] != NONE16 && robot[1] == p.comp_inner && p.comp_inner == d.comp_inner {
        reach |= 8;
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
}

fn pair_ok(p: &PickRec, d: &PlaceRec, di: u32) -> bool {
    di != p.src_place && !(d.obj == p.obj && d.inst == p.inst) && ((d.comp != NONE16 && d.comp == p.comp) || (d.room != NONE16 && d.room == p.room))
}

pub fn build(sc: &Scene, t: &Tables, limits: [LimitsRec; 2]) -> Pnp {
    let fz = floor_z(sc);
    let (c0, n0, k0) = components(sc, &fz, (limits[0].threshold * 1000.0).round() as i32);
    let (c1, n1, k1) = components(sc, &fz, (limits[1].threshold * 1000.0).round() as i32);
    let g = Grid { sc, comp: [c0, c1] };
    let l = &limits;
    let mut out = Pnp { limits, picks: vec![], places: vec![], pairs: vec![], ranges: vec![], floor_z: vec![], n_comp: [n0, n1], step_cuts: [k0, k1] };

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
        let dom = |k: usize| {
            let mut votes: std::collections::HashMap<u16, u32> = Default::default();
            for (i, &v) in sc.room_grid.iter().enumerate() {
                if v as usize == ri + 1 && g.comp[k][i] != NONE16 {
                    *votes.entry(g.comp[k][i]).or_default() += 1;
                }
            }
            votes.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map_or(NONE16, |v| v.0)
        };
        let free_in_room = sc.room_grid.iter().enumerate().filter(|(i, &v)| v as usize == ri + 1 && g.comp[0][*i] != NONE16).count() as u32;
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
            comp: dom(0),
            flags: 1,
            comp_inner: dom(1),
            reserved: 0,
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
            if let Some(p) = pick_rec(b, &g, l, 0, &out.places, &s_places) {
                out.picks.push(p);
                s_picks.push(out.picks.len() as u32 - 1);
            }
        }
    }
    let mut scene_pairs = vec![];
    for &pi in &s_picks {
        let p = out.picks[pi as usize];
        for &di in &s_places {
            let d = &out.places[di as usize];
            if pair_ok(&p, d, di) {
                scene_pairs.push(mk_pair(&p, pi, d, di, [NONE16, NONE16]));
            }
        }
    }
    let scene_range = PnpRange {
        pick_off: 0,
        place_off: 0,
        pair_off: 0,
        n_pick: s_picks.len() as u16,
        n_place: out.places.len() as u16,
        n_pair: scene_pairs.len() as u32,
        robot_comp: NONE16,
        robot_comp_inner: NONE16,
    };
    out.pairs = scene_pairs;

    // ---- instance level: task objects at their instance poses
    for tr in t.tasks.iter() {
        let tobjs = &t.tobjs[tr.obj_off as usize..(tr.obj_off + tr.n_obj as u32) as usize];
        let name = |k: usize| cstr(&sc.strings.bytes, tobjs[k].inst);
        let mut in_closed = vec![false; tobjs.len()];
        for li in &t.lits[tr.init_off as usize..(tr.init_off + tr.n_init as u32) as usize] {
            if PREDS[li.pred as usize] == "inside" && li.neg == 0 && li.nargs == 2 && li.arg[1] & (ARG_VAR | ARG_ROOM) == 0 && any_in(&name(li.arg[1] as usize), &CLOSED) {
                in_closed[li.arg[0] as usize] = true;
            }
        }
        let room_ok = |r: u16| r != NONE16 && (r >= 64 || tr.rooms_mask == 0 || tr.rooms_mask & (1u64 << r) != 0);
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
                        lo: [(c[0] - wh[0]) as f32, (c[1] - wh[1]) as f32, (c[2] - wh[2]) as f32],
                        hi: [(c[0] + wh[0]) as f32, (c[1] + wh[1]) as f32, (c[2] + wh[2]) as f32],
                        mass: o.mass,
                    },
                ));
            }
            let place_off = out.places.len() as u32;
            let mut cand_places: Vec<u32> = s_places
                .iter()
                .copied()
                .filter(|&k| {
                    let pl = &out.places[k as usize];
                    // a scene object in the task scope is replaced by its instance entry
                    let moved = pl.obj != u32::MAX && tobjs.iter().any(|o| o.scene_obj >= 0 && o.scene_obj as u32 == pl.obj);
                    !moved && room_ok(pl.room)
                })
                .collect();
            for (_, b) in &bodies {
                let k = support(b, &l[0]);
                if k != 0 {
                    out.places.push(place_rec(b, k, &g, l, NONE16));
                    cand_places.push(out.places.len() as u32 - 1);
                }
            }
            let pick_off = out.picks.len() as u32;
            let pair_off = out.pairs.len() as u32;
            let robot = [g.comp_at(0, ins.robot_pos[0], ins.robot_pos[1]), g.comp_at(1, ins.robot_pos[0], ins.robot_pos[1])];
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
                let Some(p) = pick_rec(b, &g, l, extra, &out.places, &cand_places) else { continue };
                let pi = out.picks.len() as u32;
                out.picks.push(p);
                for &di in &cand_places {
                    let d = &out.places[di as usize];
                    if pair_ok(&p, d, di) {
                        out.pairs.push(mk_pair(&p, pi, d, di, robot));
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
                robot_comp: robot[0],
                robot_comp_inner: robot[1],
            });
        }
    }
    out.ranges.push(scene_range);
    out.floor_z = fz;
    out
}

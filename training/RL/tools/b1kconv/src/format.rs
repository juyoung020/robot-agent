//! RASC v1 — one file per scene. Every record is a #[repr(C)] struct with no implicit padding
//! (checked: sum of field sizes == size_of). The same macro emits the C header (`header_text`),
//! so the C++ loader and this writer cannot disagree; the header carries a layout hash that the
//! file stores too (the loader refuses a file written by a different layout).

pub trait CT {
    const C: &'static str;
    const N: usize;
}
macro_rules! ct {
    ($t:ty, $c:expr) => {
        impl CT for $t {
            const C: &'static str = $c;
            const N: usize = 1;
        }
    };
}
ct!(u8, "uint8_t");
ct!(u16, "uint16_t");
ct!(u32, "uint32_t");
ct!(i32, "int32_t");
ct!(u64, "uint64_t");
ct!(f32, "float");
impl<T: CT, const K: usize> CT for [T; K] {
    const C: &'static str = T::C;
    const N: usize = K * T::N;
}

#[allow(dead_code)]
pub struct FieldInfo {
    pub name: &'static str,
    pub c: &'static str,
    pub n: usize,
    pub off: usize,
    pub size: usize,
    pub doc: &'static str,
}

pub trait Rec: Copy + Default {
    const NAME: &'static str;
    const DOC: &'static str;
    fn fields() -> Vec<FieldInfo>;
}

macro_rules! rec {
    ($name:ident, $doc:expr, { $($f:ident : $t:ty => $fd:expr),* $(,)? }) => {
        #[repr(C)]
        #[derive(Clone, Copy, Default, Debug, PartialEq)]
        pub struct $name { $(pub $f: $t),* }
        impl Rec for $name {
            const NAME: &'static str = stringify!($name);
            const DOC: &'static str = $doc;
            fn fields() -> Vec<FieldInfo> {
                vec![$(FieldInfo {
                    name: stringify!($f), c: <$t as CT>::C, n: <$t as CT>::N,
                    off: std::mem::offset_of!($name, $f), size: std::mem::size_of::<$t>(), doc: $fd
                }),*]
            }
        }
        const _: () = assert!(0 $(+ std::mem::size_of::<$t>())* == std::mem::size_of::<$name>(),
                              concat!(stringify!($name), " has implicit padding"));
    };
}

pub const MAGIC: u32 = 0x4353_4152; // "RASC" little endian
pub const VERSION: u32 = 2;
pub const NSEC: usize = 28;
pub const NONE16: u16 = 0xFFFF;

rec!(Sec, "section table entry", {
    offset: u64 => "byte offset from file start (64-aligned)",
    bytes: u64 => "section size in bytes",
    count: u32 => "number of elements",
    elem: u32 => "element size in bytes (sizeof record, 1 for u8, 0 for bit arrays where count = bits)",
});

impl CT for Sec {
    const C: &'static str = "RascSec";
    const N: usize = 1;
}

rec!(FileHeader, "file header (one per file, at offset 0)", {
    magic: u32 => "'RASC' = 0x43534152",
    version: u32 => "format version (1)",
    header_bytes: u32 => "sizeof(FileHeader)",
    layout_hash: u32 => "FNV-1a of the generated layout text; must equal RASC_LAYOUT_HASH",
    scene_name: [u8; 32] => "scene model name, NUL padded",
    scene_index: u32 => "0..6 (doc 1.2 order: HSF HDL HDU RD RS HSL OCR)",
    n_levels: u32 => "floors (levels) in this scene model; all 2026 scenes have 1 (level id 0)",
    origin: [f32; 2] => "world xy (m) of the lower-left corner of grid cell (row 0, col 0)",
    cell: f32 => "grid cell size (m), 0.10",
    grid_w: u32 => "grid columns (x)",
    grid_h: u32 => "grid rows (y); cell (r,c) center = origin + ((c+.5)*cell, (r+.5)*cell)",
    src_px: u32 => "layout PNG width = height (pixels)",
    src_res: f32 => "layout PNG resolution (m/pixel), 0.01; PNG pixel (r,c) sits at ((c-src_px/2)*res, (r-src_px/2)*res)",
    trav_area_m2: f32 => "free pixels of floor_trav_0.png * res^2 (full resolution)",
    n_rooms_png: u32 => "distinct non-zero room ids in floor_insseg_0.png at full resolution",
    flags: u32 => "reserved (0)",
    sections: [Sec; NSEC] => "section table, indexed by RASC_SEC_*",
    reserved: [u32; 4] => "0",
});

rec!(CatRec, "object category (OmniGibson category name) used by this scene or its tasks", {
    name: u32 => "string offset",
    flags: u32 => "1 = avg_category_specs.json has this category",
    mass: f32 => "category average mass (kg), avg_category_specs.json; NaN if absent",
    volume: f32 => "category average volume (m^3); NaN if absent",
    density: f32 => "category average density (kg/m^3); NaN if absent",
    n_scene: u32 => "scene objects of this category",
    reserved: [u32; 2] => "0",
});

rec!(RoomRec, "room instance (floor_insseg / floor_semseg)", {
    name: u32 => "string offset: OmniGibson room instance name, e.g. 'kitchen_0' (same naming as SegmentationMap)",
    sem: u16 => "room category id: 1-based line of room_categories.txt",
    ins: u16 => "pixel value in floor_insseg_0.png",
    og_named: u16 => "1 if OmniGibson (0.10 m nearest resize) names this room; 0 = only visible at full resolution",
    reserved0: u16 => "0",
    n_cells: u32 => "grid cells whose majority room is this room",
    area_m2: f32 => "pixels * res^2 at full resolution",
    centroid: [f32; 2] => "world xy (m), full resolution",
    bmin: [f32; 2] => "world xy min (m)",
    bmax: [f32; 2] => "world xy max (m)",
    reserved1: u32 => "0",
});

rec!(ObjRec, "scene object (every entry of <scene>_best.json, file order)", {
    name: u32 => "string offset: OmniGibson object name",
    model: u32 => "string offset: model id",
    cat: u16 => "index into CatRec",
    room_a: u16 => "first in_rooms entry (RoomRec index) or 0xFFFF",
    room_b: u16 => "second in_rooms entry (doors) or 0xFFFF",
    level: u16 => "floor/level id (0)",
    flags: u32 => "RASC_F_*",
    n_joints: u16 => "joint_pos length in the scene state",
    n_in_rooms: u16 => "in_rooms entries (all, also unresolved ones) in the IN_ROOMS pool",
    joint_off: u32 => "first joint in the JOINTS pool",
    pos: [f32; 3] => "root link position (m)",
    quat: [f32; 4] => "root link orientation (x, y, z, w)",
    scale: [f32; 3] => "instance scale",
    center: [f32; 3] => "world bbox center = pos + R * (scale * base_link_offset)",
    half: [f32; 3] => "bbox half extents in the object frame = scale * bbox_size / 2",
    aabb_min: [f32; 3] => "world axis-aligned box of the rotated bbox",
    aabb_max: [f32; 3] => "",
    yaw: f32 => "rotation about z (rad) from quat",
    mass: f32 => "category average mass (kg) or NaN",
    in_rooms_off: u32 => "first entry in IN_ROOMS (u32 string offsets of the in_rooms names)",
});

rec!(BoxRec, "static collision box: every scene object except floors and ceilings (walls included, flagged)", {
    center: [f32; 3] => "world center (m)",
    half: [f32; 3] => "half extents in the object frame",
    yaw: f32 => "rotation about z (rad); see RASC_F_TILTED for non-upright objects",
    cat: u16 => "CatRec index",
    room: u16 => "RoomRec index or 0xFFFF",
    flags: u32 => "RASC_F_*",
    obj: u32 => "ObjRec index",
    zmin: f32 => "world AABB z min",
    zmax: f32 => "world AABB z max",
});

rec!(DoorRec, "door-like scene object (category contains 'door')", {
    obj: u32 => "ObjRec index",
    room_a: u16 => "RoomRec index or 0xFFFF",
    room_b: u16 => "RoomRec index or 0xFFFF",
    kind: u16 => "0 door, 1 sliding_door, 2 garage_door, 3 elevator_door, 4 other",
    n_joints: u16 => "",
    joint_off: u32 => "JOINTS pool",
    center: [f32; 3] => "world bbox center",
    half: [f32; 3] => "object-frame half extents",
    yaw: f32 => "",
    reserved: u32 => "0",
});

rec!(TaskRec, "BEHAVIOR task (problem0.bddl) whose scene is this file", {
    name: u32 => "string offset: task name",
    task_index: u16 => "0..99 (metadata/task.jsonl)",
    n_obj: u16 => "task objects (BDDL :objects, incl. agent and systems)",
    n_init: u16 => "init literals",
    n_goal: u16 => "goal literals (all or-branches)",
    n_var: u16 => "goal variables",
    n_cand: u16 => "candidate entries",
    n_removed: u16 => "scene objects the task template does not load",
    n_or_groups: u16 => "or groups",
    n_doc_lits: u16 => "goal literals counted like the doc (first or-branch)",
    n_not_open: u16 => "doc 1.4 'initially true' count: goal 'not open' literals (doc rule)",
    n_init_true: u16 => "doc literals already true under the BDDL :init (closed world)",
    skills: u16 => "RASC_SK_* bits (doc 1.3 skill letters)",
    obj_off: u32 => "TaskObjRec index",
    init_off: u32 => "LitRec index of init literals",
    goal_off: u32 => "LitRec index of goal literals",
    var_off: u32 => "VarRec index",
    cand_off: u32 => "CANDS (u16) index",
    removed_off: u32 => "REMOVED (u16 ObjRec indices) index",
    inst_off: u32 => "InstRec index",
    n_inst: u32 => "instances (train then public)",
    n_train: u32 => "",
    n_public: u32 => "",
    human_s: f32 => "mean human demo length (s) = task.jsonl length / 30",
    limit_s: f32 => "evaluation time limit (s) = human_s * 1.5",
    dist_m: f32 => "human demo base travel (m)",
    q_strict: f32 => "doc 1.5 q upper bound, strict",
    q_loose: f32 => "doc 1.5 q upper bound, loose",
    rooms_mask: u64 => "bit i = RoomRec i is in B100_task_misc.csv 'Rooms to include'",
});

rec!(TaskObjRec, "task object: BDDL instance joined with the task template", {
    inst: u32 => "string offset: BDDL instance, e.g. 'jar__of__honey.n.01_1'",
    syn: u32 => "string offset: synset",
    og_name: u32 => "string offset: OmniGibson name (inst_to_name), '' if unmapped",
    cat: u16 => "CatRec index or 0xFFFF (agent / system / unmapped)",
    n_joints: u16 => "joint_pos length in the template state",
    flags: u32 => "RASC_F_*",
    scene_obj: i32 => "ObjRec index if this is a scene object, else -1",
    scale: [f32; 3] => "",
    half: [f32; 3] => "scale * bbox_size / 2 (object frame); 0 if no bbox",
    offset: [f32; 3] => "scale * base_link_offset (object frame)",
    mass: f32 => "category average mass (kg) or NaN",
    pos: [f32; 3] => "template (instance 0) root position",
    quat: [f32; 4] => "template root orientation (x, y, z, w)",
});

rec!(LitRec, "literal (init or goal)", {
    pred: u8 => "RASC_P_*",
    neg: u8 => "1 = negated",
    nargs: u8 => "1 or 2",
    flags: u8 => "RASC_L_*",
    or_group: u16 => "or group (task-local) or 0xFFFF",
    or_branch: u16 => "branch within the group",
    arg: [u16; 2] => "RASC_ARG_VAR bit -> VarRec index; RASC_ARG_ROOM bit -> room category; else TaskObjRec index (task-local)",
    reserved: u32 => "0",
});

rec!(VarRec, "goal variable: binds to one of its candidates", {
    kind: u8 => "1 exists, 2 forn, 3 forpairs",
    reserved: u8 => "0",
    group: u16 => "distinct group (forn/forpairs: members bind to different objects) or 0xFFFF",
    cand_off: u16 => "task-local CANDS index",
    cand_n: u16 => "",
});

rec!(InstRec, "task instance: robot start + task object states", {
    task: u16 => "TaskRec index in this file",
    split: u16 => "0 train (scenes/, ids 1..300 for tasks 50-99, 0..299 for tasks 0-49), 1 public eval (scene_test/public/, ids 301..320)",
    inst_id: u32 => "instance id from the file name",
    pose_off: u32 => "PoseRec index; n_pose = task n_obj, same order (agent / systems get src 0)",
    n_pose: u16 => "",
    flags: u16 => "bit 0: robot pose is the R1Pro-specific one (2025 train files have no generic pose)",
    robot_pos: [f32; 3] => "robot_poses.robot[0].position",
    robot_quat: [f32; 4] => "robot_poses.robot[0].orientation (x, y, z, w)",
    robot_yaw: f32 => "",
});

rec!(PoseRec, "task object state in one instance", {
    pos: [f32; 3] => "root link position",
    quat: [f32; 4] => "(x, y, z, w)",
    joint_off: u32 => "JOINTS pool",
    n_joints: u16 => "",
    src: u16 => "0 none (agent/system), 1 instance file, 2 scene file (object absent from the instance), 3 template (known dataset defect)",
});

rec!(LimitsRec, "pick-and-place limits used for PICKS/PLACES (2 records: [0] outer = inclusion, [1] inner = flag bit 0)", {
    pick_z: f32 => "max bottom height (m) of a graspable object (top-down grasp from the LIMO base)",
    place_top: f32 => "max top height (m) of an ontop support",
    inside_margin: f32 => "an open container may be this much higher than place_top (m)",
    max_mass: f32 => "max mass (kg; category average)",
    max_w: f32 => "max of the smaller horizontal box side (m) = gripper opening",
    min_side: f32 => "min smaller horizontal side (m) of an ontop support",
    min_top: f32 => "min top height (m) of an ontop support (floor is its own kind)",
    reach: f32 => "max xy distance (m) from a free base cell center to the object's box footprint",
});

rec!(PickRec, "graspable object candidate", {
    obj: u32 => "ObjRec index, or RASC_PNP_TASKOBJ | TaskObjRec index (instance level)",
    inst: u32 => "InstRec index, 0xffffffff = scene level (static)",
    center: [f32; 3] => "world box center",
    z0: f32 => "world AABB bottom",
    top: f32 => "world AABB top",
    min_w: f32 => "smaller horizontal side of the object box (m)",
    mass: f32 => "category average mass (kg) or NaN",
    yaw: f32 => "",
    room: u16 => "RoomRec index at the center or 0xffff",
    comp: u16 => "TRAV connected component of its approach cells (largest), 0xffff = none",
    n_approach: u32 => "TRAV-free cells within reach of the footprint",
    src_place: u32 => "PlaceRec index it rests on / in (same level or scene level), 0xffffffff = unknown",
    flags: u32 => "RASC_PK_*",
    reserved: [u32; 2] => "0",
});

rec!(PlaceRec, "support candidate (ontop surface, open container, or a room's floor)", {
    obj: u32 => "ObjRec index, RASC_PNP_TASKOBJ | TaskObjRec index, or 0xffffffff for a floor",
    inst: u32 => "InstRec index, 0xffffffff = scene level",
    kind: u16 => "1 ontop surface, 2 inside open container, 3 floor of a room",
    room: u16 => "RoomRec index",
    center: [f32; 3] => "world box center (floor: room centroid, z 0)",
    half: [f32; 2] => "object-frame half extents xy (floor: room bbox half)",
    yaw: f32 => "",
    top: f32 => "world AABB top (floor: 0)",
    n_approach: u32 => "TRAV-free cells within reach (floor: free cells of the room)",
    comp: u16 => "TRAV component (largest among approach cells), 0xffff = none",
    flags: u16 => "bit 0: also within the inner (strict) limits",
});

rec!(PairRec, "pick-and-place candidate: object, source support, target support", {
    pick: u32 => "PickRec index",
    src: u32 => "PlaceRec index of the source support (= pick.src_place)",
    dst: u32 => "PlaceRec index of the target support",
    rel: u8 => "RASC_P_ONTOP or RASC_P_INSIDE",
    reachable: u8 => "bit 0: pick and target approach cells share a TRAV component; bit 1: both within inner limits; bit 2: instance robot start in that component",
    room_pick: u16 => "RoomRec index of the object",
    room_dst: u16 => "RoomRec index of the target",
    reserved: u16 => "0",
    dist: f32 => "xy distance object -> target (m)",
});

rec!(PnpRange, "per-instance ranges into PICKS/PLACES/PAIRS (n_inst + 1 entries; the last = scene level)", {
    pick_off: u32 => "",
    place_off: u32 => "",
    pair_off: u32 => "",
    n_pick: u16 => "",
    n_place: u16 => "",
    n_pair: u32 => "",
    robot_comp: u16 => "TRAV component of the robot start cell (0xffff none / scene level)",
    reserved: u16 => "0",
});

pub const PNP_TASKOBJ: u32 = 0x8000_0000;
pub const PK_FLAGS: &[(&str, u32, &str)] = &[
    ("INNER", 1, "also within the inner (strict) limits"),
    ("MASS_UNKNOWN", 2, "category has no average mass (mass limit not applied)"),
    ("SCENE_OBJ", 4, "instance-level entry of a scene object in the task scope (overrides the scene-level one)"),
    ("IN_CLOSED", 8, "BDDL init puts it inside a closed articulated container"),
];

// section ids
pub const SEC_NAMES: [&str; NSEC] = [
    "STRINGS", "CATS", "ROOMS", "OBJS", "BOXES", "DOORS", "JOINTS", "ROOM_GRID", "TRAV", "TRAV_NO_OBJ", "TRAV_NO_DOOR",
    "TRAV_OPEN_DOOR", "TASKS", "TASK_OBJS", "LITS", "VARS", "CANDS", "REMOVED", "INSTS", "POSES", "IN_ROOMS", "LIMITS",
    "PICKS", "PLACES", "PAIRS", "PNP_RANGES", "RES26", "RES27",
];
pub const S_STRINGS: usize = 0;
pub const S_CATS: usize = 1;
pub const S_ROOMS: usize = 2;
pub const S_OBJS: usize = 3;
pub const S_BOXES: usize = 4;
pub const S_DOORS: usize = 5;
pub const S_JOINTS: usize = 6;
pub const S_ROOM_GRID: usize = 7;
pub const S_TRAV: usize = 8;
pub const S_TRAV_NO_OBJ: usize = 9;
pub const S_TRAV_NO_DOOR: usize = 10;
pub const S_TRAV_OPEN_DOOR: usize = 11;
pub const S_TASKS: usize = 12;
pub const S_TASK_OBJS: usize = 13;
pub const S_LITS: usize = 14;
pub const S_VARS: usize = 15;
pub const S_CANDS: usize = 16;
pub const S_REMOVED: usize = 17;
pub const S_INSTS: usize = 18;
pub const S_POSES: usize = 19;
pub const S_IN_ROOMS: usize = 20;
pub const S_LIMITS: usize = 21;
pub const S_PICKS: usize = 22;
pub const S_PLACES: usize = 23;
pub const S_PAIRS: usize = 24;
pub const S_PNP_RANGES: usize = 25;

// flags (objects and task objects)
pub const FLAGS: &[(&str, u32, &str)] = &[
    ("WALL", 1 << 0, "category walls"),
    ("FLOOR", 1 << 1, "category floors"),
    ("CEILING", 1 << 2, "category ceilings"),
    ("DOOR", 1 << 3, "category contains 'door'"),
    ("WINDOW", 1 << 4, "category contains 'window'"),
    ("STAIRS", 1 << 5, "category contains 'stair'"),
    ("FIXED_BASE", 1 << 6, "args.fixed_base"),
    ("ARTICULATED", 1 << 7, "state has joint_pos"),
    ("OPENABLE", 1 << 8, "metadata openable_joint_ids or an 'openable' link tag"),
    ("TOGGLEABLE", 1 << 9, "metadata meta_links has togglebutton"),
    ("VISUAL_ONLY", 1 << 10, "args.visual_only"),
    ("CARPET", 1 << 11, "category carpet/rug (flat)"),
    ("HEATSOURCE", 1 << 12, "meta link heatsource"),
    ("FLUID_SOURCE", 1 << 13, "meta link fluidsource"),
    ("FLUID_SINK", 1 << 14, "meta link fluidsink"),
    ("ATTACHMENT", 1 << 15, "meta link attachment"),
    ("TILTED", 1 << 16, "object z axis more than 5 deg from world z (yaw-only box is approximate)"),
    ("HAS_BBOX", 1 << 17, "metadata.json found (bbox valid)"),
    ("SYSTEM", 1 << 18, "task object is a particle/fluid system (no rigid body)"),
    ("AGENT", 1 << 19, "task object is the robot"),
    ("SCENE_OBJ", 1 << 20, "task object is a scene object (scene_obj >= 0)"),
    ("WILDCARD", 1 << 21, "BDDL wildcard instance (name ends with _*)"),
    ("UNMAPPED", 1 << 22, "not in the template inst_to_name"),
    ("FUTURE", 1 << 23, "BDDL init (future x): created during the task"),
    ("LIGHT", 1 << 24, "meta link lights"),
];
pub fn flag(name: &str) -> u32 {
    FLAGS.iter().find(|f| f.0 == name).map(|f| f.1).unwrap()
}

pub const PREDS: &[&str] = &[
    "", "inroom", "ontop", "inside", "nextto", "under", "touching", "attached", "open", "toggled_on", "cooked",
    "frozen", "hot", "on_fire", "covered", "filled", "contains", "saturated", "insource", "real", "future", "overlaid",
    "draped", "folded", "unfolded", "broken",
];
pub fn pred_id(p: &str) -> Option<u8> {
    PREDS.iter().position(|x| *x == p && !p.is_empty()).map(|i| i as u8)
}

pub const LIT_FLAGS: &[(&str, u8, &str)] = &[
    ("DOC", 1, "counted by the doc rule (all or-tags on branch 0)"),
    ("INIT_TRUE", 2, "goal literal already true under the BDDL :init (closed world; candidates: any)"),
    ("IN_OR", 4, "inside an or-branch"),
];

pub const SKILLS: &[(&str, char)] = &[
    ("MULTI_ROOM", 'M'), ("PICK", 'P'), ("OPEN", 'O'), ("TOGGLE", 'T'), ("HEAT", 'H'), ("SLICE", 'S'),
    ("CLEAN", 'C'), ("FILL", 'F'), ("ATTACH", 'A'), ("HEAVY", 'W'),
];

pub const ARG_VAR: u16 = 0x8000;
pub const ARG_ROOM: u16 = 0x4000;

fn fnv(s: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

fn cname(n: &str) -> String {
    format!("Rasc{n}")
}

fn struct_text<R: Rec>(out: &mut String) {
    let cn = cname(R::NAME);
    out.push_str(&format!("/* {} */\ntypedef struct {cn} {{\n", R::DOC));
    for f in R::fields() {
        let arr = if f.n > 1 { format!("[{}]", f.n) } else { String::new() };
        let doc = if f.doc.is_empty() { String::new() } else { format!(" /* {} */", f.doc) };
        out.push_str(&format!("    {} {}{arr};{doc}\n", f.c, f.name));
    }
    out.push_str(&format!("}} {cn};\n"));
    out.push_str(&format!("RASC_STATIC_ASSERT(sizeof({cn}) == {}, \"{cn} size\");\n", std::mem::size_of::<R>()));
    for f in R::fields() {
        out.push_str(&format!("RASC_STATIC_ASSERT(offsetof({cn}, {}) == {}, \"{cn}.{}\");\n", f.name, f.off, f.name));
    }
    out.push('\n');
}

/// Layout part of the header (hashed). Struct names in C are Rasc<Name>.
fn layout_text() -> String {
    let mut s = String::new();
    struct_text::<Sec>(&mut s);
    struct_text::<FileHeader>(&mut s);
    struct_text::<CatRec>(&mut s);
    struct_text::<RoomRec>(&mut s);
    struct_text::<ObjRec>(&mut s);
    struct_text::<BoxRec>(&mut s);
    struct_text::<DoorRec>(&mut s);
    struct_text::<TaskRec>(&mut s);
    struct_text::<TaskObjRec>(&mut s);
    struct_text::<LitRec>(&mut s);
    struct_text::<VarRec>(&mut s);
    struct_text::<InstRec>(&mut s);
    struct_text::<PoseRec>(&mut s);
    struct_text::<LimitsRec>(&mut s);
    struct_text::<PickRec>(&mut s);
    struct_text::<PlaceRec>(&mut s);
    struct_text::<PairRec>(&mut s);
    struct_text::<PnpRange>(&mut s);
    s.push_str("enum {\n");
    for (i, n) in SEC_NAMES.iter().enumerate() {
        s.push_str(&format!("    RASC_SEC_{n} = {i},\n"));
    }
    s.push_str(&format!("    RASC_NSEC = {NSEC}\n}};\n\n"));
    s.push_str("enum {\n");
    for (n, v, d) in FLAGS {
        s.push_str(&format!("    RASC_F_{n} = 0x{v:x}u, /* {d} */\n"));
    }
    s.push_str("};\n\nenum {\n");
    for (i, p) in PREDS.iter().enumerate().skip(1) {
        s.push_str(&format!("    RASC_P_{} = {i},\n", p.to_uppercase()));
    }
    s.push_str(&format!("    RASC_NPRED = {}\n}};\n\nenum {{\n", PREDS.len()));
    for (n, v, d) in LIT_FLAGS {
        s.push_str(&format!("    RASC_L_{n} = {v}, /* {d} */\n"));
    }
    s.push_str("};\n\nenum {\n");
    for (n, v, d) in PK_FLAGS {
        s.push_str(&format!("    RASC_PK_{n} = {v}, /* {d} */\n"));
    }
    s.push_str("};\n\nenum {\n");
    for (i, (n, c)) in SKILLS.iter().enumerate() {
        s.push_str(&format!("    RASC_SK_{n} = 1 << {i}, /* doc letter {c} */\n"));
    }
    s.push_str(&format!(
        "}};\n\n#define RASC_ARG_VAR 0x{ARG_VAR:x}u\n#define RASC_ARG_ROOM 0x{ARG_ROOM:x}u\n#define RASC_NONE16 0xffffu\n#define RASC_NONE32 0xffffffffu\n#define RASC_PNP_TASKOBJ 0x{PNP_TASKOBJ:x}u\n"
    ));
    s
}

pub fn layout_hash() -> u32 {
    fnv(&layout_text())
}

pub fn header_text() -> String {
    let lay = layout_text();
    let mut s = String::new();
    s.push_str("/* GENERATED by training/RL/tools/b1kconv (`b1kconv header`). Do not edit.\n");
    s.push_str(" * RASC v1: one little-endian file per BEHAVIOR scene. Sections are 64-byte aligned;\n");
    s.push_str(" * a section is an array of the record below (or u8 / u16 / f32 / bit arrays).\n");
    s.push_str(" *   STRINGS        char[]  NUL-terminated names; records hold byte offsets\n");
    s.push_str(" *   ROOM_GRID      u8[grid_h][grid_w]  RoomRec index + 1 (0 = no room), majority of 10x10 pixels\n");
    s.push_str(" *   TRAV*          bits[grid_h*grid_w] (bit i = byte i>>3, bit i&7): 1 = all 100 pixels free\n");
    s.push_str(" *                  TRAV = floor_trav_0, TRAV_NO_OBJ = floor_trav_no_obj_0 (walls only),\n");
    s.push_str(" *                  TRAV_NO_DOOR = floor_trav_no_door_0, TRAV_OPEN_DOOR = floor_trav_open_door_0\n");
    s.push_str(" *   JOINTS         f32 pool (joint positions)\n");
    s.push_str(" *   CANDS, REMOVED u16 pools; IN_ROOMS u32 pool (string offsets)\n */\n");
    s.push_str("#pragma once\n#include <stdint.h>\n#include <stddef.h>\n");
    s.push_str("#ifdef __cplusplus\n#define RASC_STATIC_ASSERT(c, m) static_assert(c, m)\n#else\n#define RASC_STATIC_ASSERT(c, m) _Static_assert(c, m)\n#endif\n\n");
    s.push_str(&format!("#define RASC_MAGIC 0x{MAGIC:08x}u\n#define RASC_VERSION {VERSION}u\n#define RASC_LAYOUT_HASH 0x{:08x}u\n\n", fnv(&lay)));
    s.push_str(&lay);
    s.push_str("\nstatic const char* const rasc_pred_names[] = {");
    for p in PREDS {
        s.push_str(&format!("\"{p}\", "));
    }
    s.push_str("};\nstatic const char* const rasc_sec_names[] = {");
    for p in SEC_NAMES {
        s.push_str(&format!("\"{p}\", "));
    }
    s.push_str("};\n");
    s
}

// ---------------------------------------------------------------- byte helpers
pub fn bytes_of<T: Rec>(v: &[T]) -> &[u8] {
    // SAFETY: Rec types are repr(C) PODs without implicit padding (checked at compile time).
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn read_recs<T: Rec>(b: &[u8]) -> Vec<T> {
    let n = b.len() / std::mem::size_of::<T>();
    let mut v = vec![T::default(); n];
    // SAFETY: same as above; copy into an aligned Vec.
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, n * std::mem::size_of::<T>()) };
    v
}


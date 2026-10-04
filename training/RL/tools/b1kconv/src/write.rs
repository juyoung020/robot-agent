//! Assemble one RASC file: header + 64-byte aligned sections.

use crate::format::*;
use crate::scene::Scene;

pub struct Tables {
    pub tasks: Vec<TaskRec>,
    pub tobjs: Vec<TaskObjRec>,
    pub lits: Vec<LitRec>,
    pub vars: Vec<VarRec>,
    pub cands: Vec<u16>,
    pub removed: Vec<u16>,
    pub insts: Vec<InstRec>,
    pub poses: Vec<PoseRec>,
}

fn raw<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain numbers
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn fnv64(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

pub fn assemble(sc: &mut Scene, t: &Tables) -> Vec<u8> {
    let g = (sc.hdr.grid_w * sc.hdr.grid_h) as u32;
    let secs: Vec<(usize, &[u8], u32, u32)> = vec![
        (S_STRINGS, &sc.strings.bytes, sc.strings.bytes.len() as u32, 1),
        (S_CATS, bytes_of(&sc.cats), sc.cats.len() as u32, std::mem::size_of::<CatRec>() as u32),
        (S_ROOMS, bytes_of(&sc.rooms), sc.rooms.len() as u32, std::mem::size_of::<RoomRec>() as u32),
        (S_OBJS, bytes_of(&sc.objs), sc.objs.len() as u32, std::mem::size_of::<ObjRec>() as u32),
        (S_BOXES, bytes_of(&sc.boxes), sc.boxes.len() as u32, std::mem::size_of::<BoxRec>() as u32),
        (S_DOORS, bytes_of(&sc.doors), sc.doors.len() as u32, std::mem::size_of::<DoorRec>() as u32),
        (S_JOINTS, raw(&sc.joints), sc.joints.len() as u32, 4),
        (S_ROOM_GRID, &sc.room_grid, g, 1),
        (S_TRAV, &sc.trav[0], g, 0),
        (S_TRAV_NO_OBJ, &sc.trav[1], g, 0),
        (S_TRAV_NO_DOOR, &sc.trav[2], g, 0),
        (S_TRAV_OPEN_DOOR, &sc.trav[3], g, 0),
        (S_TASKS, bytes_of(&t.tasks), t.tasks.len() as u32, std::mem::size_of::<TaskRec>() as u32),
        (S_TASK_OBJS, bytes_of(&t.tobjs), t.tobjs.len() as u32, std::mem::size_of::<TaskObjRec>() as u32),
        (S_LITS, bytes_of(&t.lits), t.lits.len() as u32, std::mem::size_of::<LitRec>() as u32),
        (S_VARS, bytes_of(&t.vars), t.vars.len() as u32, std::mem::size_of::<VarRec>() as u32),
        (S_CANDS, raw(&t.cands), t.cands.len() as u32, 2),
        (S_REMOVED, raw(&t.removed), t.removed.len() as u32, 2),
        (S_INSTS, bytes_of(&t.insts), t.insts.len() as u32, std::mem::size_of::<InstRec>() as u32),
        (S_POSES, bytes_of(&t.poses), t.poses.len() as u32, std::mem::size_of::<PoseRec>() as u32),
        (S_IN_ROOMS, raw(&sc.in_rooms), sc.in_rooms.len() as u32, 4),
    ];
    let hsz = std::mem::size_of::<FileHeader>();
    let mut off = hsz.div_ceil(64) * 64;
    for (id, b, count, elem) in &secs {
        sc.hdr.sections[*id] = Sec { offset: off as u64, bytes: b.len() as u64, count: *count, elem: *elem };
        off += b.len().div_ceil(64) * 64;
    }
    let mut out = vec![0u8; off];
    out[..hsz].copy_from_slice(bytes_of(std::slice::from_ref(&sc.hdr)));
    for (id, b, _, _) in &secs {
        let o = sc.hdr.sections[*id].offset as usize;
        out[o..o + b.len()].copy_from_slice(b);
    }
    out
}

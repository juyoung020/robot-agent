//! .trp 쓰개 C ABI. 문자열은 NUL 끝 UTF-8. 실패는 음수/null.
use serde_json::Value;
use std::ffi::{c_char, CStr};
use trainfmt::trp::TrpWriter;

unsafe fn s(p: *const c_char) -> String {
    if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
}
fn csv(t: &str) -> Vec<String> {
    t.split(',').filter(|x| !x.is_empty()).map(|x| x.trim().to_string()).collect()
}

/// cols / slot_cols: 쉼표로 이은 열 이름. head_json: 머리 JSON 객체(meta, dt, objects, joint_map, grid, scene …)
#[no_mangle]
pub unsafe extern "C" fn trp_new(cols: *const c_char, slot_cols: *const c_char, n_slots: i32, head_json: *const c_char) -> *mut TrpWriter {
    let head: Value = match serde_json::from_str(&s(head_json)) {
        Ok(v @ Value::Object(_)) => v,
        _ => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(TrpWriter::new(&csv(&s(cols)), &csv(&s(slot_cols)), n_slots.max(0) as usize, head)))
}
/// row: cols 개, slots: n_slots × slot_cols 개(슬롯 열이 없으면 null)
#[no_mangle]
pub unsafe extern "C" fn trp_frame(w: *mut TrpWriter, row: *const f32, slots: *const f32) {
    let w = &mut *w;
    let r = std::slice::from_raw_parts(row, w.cols.len());
    let n = w.n_slots * w.slot_cols.len();
    let sl: &[f32] = if n == 0 || slots.is_null() { &[] } else { std::slice::from_raw_parts(slots, n) };
    if n > 0 && sl.is_empty() {
        let z = vec![0f32; n];
        w.frame(r, &z);
    } else {
        w.frame(r, sl);
    }
}
#[no_mangle]
pub unsafe extern "C" fn trp_map_rect(w: *mut TrpWriter, frame: u32, gw: i32, gh: i32, res: f64, ox: f64, oy: f64, x0: i32, y0: i32, x1: i32, y1: i32, cells: *const i8) {
    let n = ((x1 - x0 + 1) as usize) * ((y1 - y0 + 1) as usize);
    (*w).map_rect(frame, gw, gh, res, ox, oy, x0, y0, x1, y1, std::slice::from_raw_parts(cells, n));
}
#[no_mangle]
pub unsafe extern "C" fn trp_image(w: *mut TrpWriter, frame: u32, cam: u8, jpeg: *const u8, n: usize) {
    (*w).image(frame, cam, std::slice::from_raw_parts(jpeg, n));
}
#[no_mangle]
pub unsafe extern "C" fn trp_record(w: *mut TrpWriter, name: *const c_char, frame: u32, bytes: *const u8, n: usize) {
    let b: &[u8] = if n == 0 { &[] } else { std::slice::from_raw_parts(bytes, n) };
    (*w).record(&s(name), frame, b);
}
/// 머리 칸 하나를 JSON 값으로 바꿈(예: 판이 끝난 뒤 "meta")
#[no_mangle]
pub unsafe extern "C" fn trp_set_head(w: *mut TrpWriter, key: *const c_char, json: *const c_char) -> i32 {
    match serde_json::from_str::<Value>(&s(json)) {
        Ok(v) => { let w = &mut *w; w.head[s(key)] = v; 0 }
        Err(_) => -1,
    }
}
#[no_mangle]
pub unsafe extern "C" fn trp_n_frames(w: *const TrpWriter) -> i64 { (*w).n_frames() as i64 }
/// .tmp 에 쓰고 이름 바꾸기. 바이트 수(음수 = 실패). 쓰개는 그대로(trp_free 로 놓을 것)
#[no_mangle]
pub unsafe extern "C" fn trp_finish(w: *const TrpWriter, path: *const c_char) -> i64 {
    match (*w).finish(std::path::Path::new(&s(path))) { Ok(n) => n as i64, Err(_) => -1 }
}
#[no_mangle]
pub unsafe extern "C" fn trp_free(w: *mut TrpWriter) {
    if !w.is_null() { drop(Box::from_raw(w)); }
}

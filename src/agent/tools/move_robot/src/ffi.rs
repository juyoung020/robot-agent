//! C ABI (`include/move_robot.h`). 시뮬 쪽 파이썬 접착부가 ctypes 로 부른다 — 평가기 프로세스 안에서 스텝마다
//! `mr_tick` 한 번(할당 없음), 도구 호출마다 `mr_command` 한 번, 끝나면 `mr_take_result`.

use crate::{definition, Robot, Tick, ACTION_DIM};
use serde_json::Value;
use std::ffi::{c_char, c_int, CStr};
use std::sync::OnceLock;

/// 새 실행기. `hz` ≤ 0 이면 30.
#[no_mangle]
pub extern "C" fn mr_new(hz: f64) -> *mut Robot {
    Box::into_raw(Box::new(Robot::new(if hz > 0.0 { hz } else { crate::HZ })))
}

/// # Safety
/// `r` 는 `mr_new` 가 준 것이거나 NULL.
#[no_mangle]
pub unsafe extern "C" fn mr_free(r: *mut Robot) {
    if !r.is_null() {
        drop(Box::from_raw(r));
    }
}

/// 한 스텝. `action` 에 23 개를 쓴다. 반환: 0 대기, 1 움직이는 중, 2 이번에 끝남(결과 준비), -1 proprio 이상(유지값), -2 인자 이상.
///
/// # Safety
/// `proprio` 는 `n` 개, `action` 은 23 개 이상 쓸 수 있는 float 배열.
#[no_mangle]
pub unsafe extern "C" fn mr_tick(r: *mut Robot, proprio: *const f32, n: usize, action: *mut f32) -> c_int {
    if r.is_null() || proprio.is_null() || action.is_null() {
        return -2;
    }
    let p = std::slice::from_raw_parts(proprio, n);
    let a = std::slice::from_raw_parts_mut(action, ACTION_DIM);
    match (*r).tick(p, a) {
        Tick::Idle => 0,
        Tick::Moving => 1,
        Tick::Done => 2,
        Tick::BadObs => -1,
    }
}

/// 이번 스텝의 정답(GT) 자세를 넣는다(map 틀: x, y, yaw[rad]). 다음 `mr_tick` 한 번에서 base_qvel 적분 대신 쓰여 자세·이동 거리·궤적이
/// 정답을 따른다. 시뮬 GT 모드(`SGRT_POSE=gt`)에서 매 스텝 `mr_tick` 앞에 부른다. 반환 0, 인자 이상(NaN·널) -2.
///
/// # Safety
/// `r` 는 `mr_new` 가 준 것.
#[no_mangle]
pub unsafe extern "C" fn mr_set_gt_pose(r: *mut Robot, x: f64, y: f64, yaw: f64) -> c_int {
    if r.is_null() || !x.is_finite() || !y.is_finite() || !yaw.is_finite() {
        return -2;
    }
    (*r).set_gt_pose([x, y, yaw]);
    0
}

/// 도구 호출 시작(`args` = 도구 인자 JSON, UTF-8). 반환: 0 움직이기 시작, 1 바로 끝남(읽기·오류, 결과 준비), -2 인자 이상.
///
/// # Safety
/// `args` 는 NUL 로 끝나는 문자열.
#[no_mangle]
pub unsafe extern "C" fn mr_command(r: *mut Robot, args: *const c_char) -> c_int {
    if r.is_null() || args.is_null() {
        return -2;
    }
    let s = CStr::from_ptr(args).to_string_lossy();
    let v: Value = serde_json::from_str(&s).unwrap_or(Value::String(s.into_owned()));
    if (*r).command(&v) {
        1
    } else {
        0
    }
}

/// 움직이는 중이면 1
///
/// # Safety
/// `r` 는 `mr_new` 가 준 것.
#[no_mangle]
pub unsafe extern "C" fn mr_busy(r: *const Robot) -> c_int {
    if r.is_null() {
        return 0;
    }
    (*r).busy() as c_int
}

/// 새 판: 다음 관측에서 유지값을 다시 잡는다.
///
/// # Safety
/// `r` 는 `mr_new` 가 준 것.
#[no_mangle]
pub unsafe extern "C" fn mr_reset(r: *mut Robot) {
    if !r.is_null() {
        (*r).reset();
    }
}

/// 결과 JSON 을 `buf` 에 NUL 까지 복사하고 지운다. 반환: 쓴 길이(NUL 제외), 결과 없음 0,
/// `cap` 이 모자라면 필요한 길이의 음수(결과는 남는다).
///
/// # Safety
/// `buf` 는 `cap` 바이트.
#[no_mangle]
pub unsafe extern "C" fn mr_take_result(r: *mut Robot, buf: *mut c_char, cap: usize) -> isize {
    if r.is_null() || buf.is_null() {
        return 0;
    }
    let Some(v) = (*r).take_result() else { return 0 };
    let s = v.to_string();
    if s.len() + 1 > cap {
        let need = s.len() as isize + 1;
        // 되돌려 둔다
        (*r).put_result(v);
        return -need;
    }
    std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, s.len());
    *buf.add(s.len()) = 0;
    s.len() as isize
}

/// 도구 정의(OpenAI `tools` 항목) JSON. 정적 문자열 — 해제하지 않는다.
#[no_mangle]
pub extern "C" fn mr_tool_definition() -> *const c_char {
    static DEF: OnceLock<std::ffi::CString> = OnceLock::new();
    DEF.get_or_init(|| std::ffi::CString::new(definition().to_string()).unwrap()).as_ptr()
}

// ---------------------------------------------------------------- 지도(탐사)

/// `sgrt_map_view`(behavior-2026 src/scene_graph/runtime/include/sgrt.h)와 같은 배치. 평가기 접착부는 sgrt_map 이 채운
/// 구조체 포인터를 그대로 [`mr_set_map`] 에 넘긴다(파이썬은 바이트를 만지지 않음).
#[repr(C)]
pub struct SgrtMapView {
    pub stamp: f64,
    pub pose: [f64; 3],
    pub res: f64,
    pub origin: [f64; 2],
    pub w: i32,
    pub h: i32,
    pub cells: *const i8,
    pub room_res: f64,
    pub room_origin: [f64; 2],
    pub room_w: i32,
    pub room_h: i32,
    pub room_ids: *const u32,
    pub n_rooms: i32,
    pub scan_pose: [f64; 3],
    pub scan_origin: [f32; 2],
    pub n_hit: i32,
    pub hit_x: *const f32,
    pub hit_y: *const f32,
    pub n_free: i32,
    pub free_x: *const f32,
    pub free_y: *const f32,
    pub dirty: i32,
    pub dirty_box: [i32; 4],
    pub map_version: u64,
    pub n_movable: i32,
    pub movable_xyr: *const f32,
}

/// 지도 하나 받기. 반환: 0 성공, -2 인자 이상. 걸린 시간(µs)은 결과 `_m.costmap_us`.
///
/// # Safety
/// `v` 는 sgrt_map 이 채운(포인터가 아직 유효한) 구조체.
#[no_mangle]
pub unsafe extern "C" fn mr_set_map(r: *mut Robot, v: *const SgrtMapView) -> c_int {
    if r.is_null() || v.is_null() {
        return -2;
    }
    let v = &*v;
    if v.w <= 0 || v.h <= 0 || v.cells.is_null() {
        return -2;
    }
    let n = (v.w as usize) * (v.h as usize);
    let grid = crate::map::Grid { res: v.res, ox: v.origin[0], oy: v.origin[1], w: v.w as usize, h: v.h as usize, cells: std::slice::from_raw_parts(v.cells, n).to_vec() };
    let rooms = if !v.room_ids.is_null() && v.room_w > 0 && v.room_h > 0 {
        let rn = (v.room_w as usize) * (v.room_h as usize);
        Some(crate::map::RoomGrid { res: v.room_res, ox: v.room_origin[0], oy: v.room_origin[1], w: v.room_w as usize, h: v.room_h as usize, ids: std::slice::from_raw_parts(v.room_ids, rn).to_vec() })
    } else {
        None
    };
    let pts = |n: i32, x: *const f32, y: *const f32| -> Vec<[f64; 2]> {
        if n <= 0 || x.is_null() || y.is_null() {
            return vec![];
        }
        let (xs, ys) = (std::slice::from_raw_parts(x, n as usize), std::slice::from_raw_parts(y, n as usize));
        xs.iter().zip(ys).map(|(a, b)| [*a as f64, *b as f64]).collect()
    };
    let hits = pts(v.n_hit, v.hit_x, v.hit_y);
    let free = pts(v.n_free, v.free_x, v.free_y);
    let scan = if hits.is_empty() && free.is_empty() {
        None
    } else {
        Some(crate::map::Scan::from_base(v.scan_pose, [v.scan_origin[0] as f64, v.scan_origin[1] as f64], &hits, &free, v.stamp))
    };
    let mut movable = vec![];
    if v.n_movable > 0 && !v.movable_xyr.is_null() {
        let m = std::slice::from_raw_parts(v.movable_xyr, 3 * v.n_movable as usize);
        for k in 0..v.n_movable as usize {
            movable.push([m[3 * k] as f64, m[3 * k + 1] as f64, m[3 * k + 2] as f64]);
        }
    }
    let d = crate::nav::MapDelta {
        dirty: if v.dirty != 0 { Some([v.dirty_box[0] as i64, v.dirty_box[1] as i64, v.dirty_box[2] as i64, v.dirty_box[3] as i64]) } else { None },
        unknown_dirty: false,
        map_version: v.map_version,
        movable,
    };
    (*r).set_map(crate::map::MapIn { stamp: v.stamp, pose: v.pose, grid, rooms, n_rooms: v.n_rooms, scan }, &d);
    0
}

/// 정답 기준(측정용, map 좌표): cells[y·w + x] 1 = 닿을 수 있는 바닥.
///
/// # Safety
/// `cells` 는 w·h 바이트.
#[no_mangle]
pub unsafe extern "C" fn mr_set_reference(r: *mut Robot, cells: *const u8, w: i32, h: i32, res: f64, ox: f64, oy: f64) -> c_int {
    if r.is_null() || cells.is_null() || w <= 0 || h <= 0 {
        return -2;
    }
    let n = (w as usize) * (h as usize);
    let g = crate::map::Grid { res, ox, oy, w: w as usize, h: h as usize, cells: std::slice::from_raw_parts(cells, n).iter().map(|v| (*v > 0) as i8).collect() };
    (*r).set_reference(g);
    0
}

/// 시뮬이 센 접촉(로봇 링크 ↔ 바닥 아닌 것) 누적 수 — 결과 `_m.contacts`
///
/// # Safety
/// `r` 는 mr_new 가 준 것.
#[no_mangle]
pub unsafe extern "C" fn mr_set_contacts(r: *mut Robot, n: u64) {
    if !r.is_null() {
        (*r).nav.contacts = n;
    }
}

/// 뷰어용 겹침(JSON, map 좌표): 지나온 길·지금 계획 경로·목표·보인 프런티어·자세. 반환 = 쓴 길이, 모자라면 −필요 길이.
///
/// # Safety
/// `buf` 는 `cap` 바이트.
#[no_mangle]
pub unsafe extern "C" fn mr_overlay_json(r: *mut Robot, buf: *mut c_char, cap: usize) -> isize {
    if r.is_null() || buf.is_null() {
        return 0;
    }
    let s = (*r).overlay().to_string();
    if s.len() + 1 > cap {
        return -(s.len() as isize + 1);
    }
    std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, s.len());
    *buf.add(s.len()) = 0;
    s.len() as isize
}

// ---------------------------------------------------------------- VLA 실행기 (LIMO + OMX-F, POLICY 1.3)

/// scenemap.h `sm_object` 와 같은 배치. 평가기 접착부는 `sm_snap_objects` 가 준 배열 포인터를 그대로 [`mr_vla_set_objects`] 에 넘긴다.
#[repr(C)]
pub struct SmObject {
    pub id: u32,
    pub name: *const c_char,
    pub score: f32,
    pub pos: [f64; 3],
    pub extent: [f64; 3],
    pub first_pos: [f64; 3],
    pub n_obs: u32,
    pub last_seen: f64,
    pub state: i32,
    pub handled: i32,
    pub structural: i32,
}

/// VLA 단계 시작(`{"executor":"vla","skill",...,"objects":[ids],"max_s"}`). 반환: 0 실행 시작, 1 바로 끝남(오류·handback, 결과 준비), -2 인자 이상.
///
/// # Safety
/// `args` 는 NUL 로 끝나는 문자열.
#[no_mangle]
pub unsafe extern "C" fn mr_vla_start(r: *mut Robot, args: *const c_char) -> c_int {
    if r.is_null() || args.is_null() {
        return -2;
    }
    let s = CStr::from_ptr(args).to_string_lossy();
    let v: Value = serde_json::from_str(&s).unwrap_or(Value::String(s.into_owned()));
    (*r).vla_start(&v) as c_int
}

/// VLA 한 스텝(안의 정책: 각본·재생). `proprio` = LIMO 평가기 proprio(24), `out` 에 거른 행동 8
/// [vx m/s, wz rad/s, j1..j5 rad, 그리퍼 0..1]. 반환: 0 대기(유지), 1 실행 중, 2 이번에 끝남(결과 준비), -1 proprio 이상, -2 인자 이상.
///
/// # Safety
/// `proprio` 는 `n` 개, `out` 은 8 개 이상.
#[no_mangle]
pub unsafe extern "C" fn mr_vla_tick(r: *mut Robot, proprio: *const f32, n: usize, out: *mut f32) -> c_int {
    if r.is_null() || proprio.is_null() || out.is_null() {
        return -2;
    }
    let p = std::slice::from_raw_parts(proprio, n);
    let o = std::slice::from_raw_parts_mut(out, crate::limo::ACTION_DIM);
    tick_code((*r).vla_tick(p, o))
}

/// VLA 한 스텝(밖의 정책 = 학습된 엔진 자리): 그 정책의 행동 8(`action`)·끝 신호 확률·확신 낮음(0..1)을 넣으면 거르고 끝을 판정한다.
///
/// # Safety
/// `action`·`out` 은 8 개, `proprio` 는 `n` 개.
#[no_mangle]
pub unsafe extern "C" fn mr_vla_tick_ext(r: *mut Robot, proprio: *const f32, n: usize, action: *const f32, end_prob: f32, unsure: f32, out: *mut f32) -> c_int {
    if r.is_null() || proprio.is_null() || action.is_null() || out.is_null() {
        return -2;
    }
    let p = std::slice::from_raw_parts(proprio, n);
    let a = std::slice::from_raw_parts(action, crate::limo::ACTION_DIM);
    let mut a8 = [0.0; crate::limo::ACTION_DIM];
    for (d, s) in a8.iter_mut().zip(a) {
        *d = *s as f64;
    }
    let o = std::slice::from_raw_parts_mut(out, crate::limo::ACTION_DIM);
    tick_code((*r).vla_tick_ext(p, &a8, end_prob as f64, unsure as f64, o))
}

fn tick_code(t: Tick) -> c_int {
    match t {
        Tick::Idle => 0,
        Tick::Moving => 1,
        Tick::Done => 2,
        Tick::BadObs => -1,
    }
}

/// 거르개만: 행동 8 하나를 안전 한계·장애물 정지로 거른다(POLICY 7.1 `mr_filter`). 반환: 비트 1 잘림, 2 베이스 정지, 4 팔 막힘, 8 NaN; -1 proprio 이상, -2 인자 이상.
///
/// # Safety
/// `action`·`out` 은 8 개, `proprio` 는 `n` 개.
#[no_mangle]
pub unsafe extern "C" fn mr_filter(r: *mut Robot, proprio: *const f32, n: usize, action: *const f32, out: *mut f32) -> c_int {
    if r.is_null() || proprio.is_null() || action.is_null() || out.is_null() {
        return -2;
    }
    let p = std::slice::from_raw_parts(proprio, n);
    let a = std::slice::from_raw_parts(action, crate::limo::ACTION_DIM);
    let mut a8 = [0.0; crate::limo::ACTION_DIM];
    for (d, s) in a8.iter_mut().zip(a) {
        *d = *s as f64;
    }
    match (*r).vla_filter(p, &a8) {
        Err(_) => -1,
        Ok((f, info)) => {
            let o = std::slice::from_raw_parts_mut(out, crate::limo::ACTION_DIM);
            for (d, s) in o.iter_mut().zip(f.iter()) {
                *d = *s as f32;
            }
            (info.clipped as c_int) | ((info.base_stop.map_or(false, |(_, r)| r <= 0.01) as c_int) << 1) | ((info.arm_blocked as c_int) << 2) | ((info.bad_input as c_int) << 3)
        }
    }
}

/// 기억 물체(scenemap 스냅숏 `sm_snap_objects` 배열 그대로)와 그 시계의 지금 시각. id 는 "O<id>". 반환 0, -2 인자 이상.
///
/// # Safety
/// `objs` 는 `n` 개(스냅숏이 살아 있는 동안 — 여기서 복사한다).
#[no_mangle]
pub unsafe extern "C" fn mr_vla_set_objects(r: *mut Robot, objs: *const SmObject, n: i32, now: f64) -> c_int {
    if r.is_null() || (n > 0 && objs.is_null()) || !now.is_finite() {
        return -2;
    }
    let mut v = vec![];
    if n > 0 {
        for o in std::slice::from_raw_parts(objs, n as usize) {
            // 구조물(가구·받침, structural)도 넣는다: 놓기의 받침·서랍·가구 문이 이것들이다
            if o.pos.iter().any(|x| !x.is_finite()) {
                continue;
            }
            let name = if o.name.is_null() { String::new() } else { CStr::from_ptr(o.name).to_string_lossy().into_owned() };
            v.push(crate::verify::MemObject {
                id: format!("O{}", o.id),
                name,
                score: o.score as f64,
                pos: o.pos,
                extent: o.extent,
                first_pos: o.first_pos,
                n_obs: o.n_obs,
                last_seen: o.last_seen,
                state: crate::verify::ObjState::from_i32(o.state),
            });
        }
    }
    (*r).vla_set_objects(v, now);
    0
}

/// 기억 물체 JSON 배열(`[{"id","pos","extent","first_pos","last_seen","state":"seen|gone|moved|held"}]`) — 시험·scenemap 없는 쪽용.
///
/// # Safety
/// `json` 은 NUL 로 끝나는 문자열.
#[no_mangle]
pub unsafe extern "C" fn mr_vla_set_objects_json(r: *mut Robot, json: *const c_char, now: f64) -> c_int {
    if r.is_null() || json.is_null() || !now.is_finite() {
        return -2;
    }
    let s = CStr::from_ptr(json).to_string_lossy();
    let Ok(Value::Array(xs)) = serde_json::from_str::<Value>(&s) else { return -2 };
    let v = xs.iter().filter_map(crate::verify::MemObject::from_json).collect();
    (*r).vla_set_objects(v, now);
    0
}

/// 접촉 누적 수(바닥 아닌 것): 몸통(차체·바퀴) / 팔·그리퍼. VLA 단계 중 몸통 접촉이 늘면 failed(unsafe).
///
/// # Safety
/// `r` 는 mr_new 가 준 것.
#[no_mangle]
pub unsafe extern "C" fn mr_vla_contacts(r: *mut Robot, body: u64, arm: u64) {
    if !r.is_null() {
        (*r).vla_set_contacts(body, arm);
    }
}

/// 실행 중인 VLA 단계를 멈춘다(결과 handback "cancelled" 준비). 반환 1 멈춤, 0 실행 중 아님.
///
/// # Safety
/// `r` 는 mr_new 가 준 것.
#[no_mangle]
pub unsafe extern "C" fn mr_vla_stop(r: *mut Robot) -> c_int {
    if r.is_null() {
        return 0;
    }
    (*r).vla_stop("cancelled") as c_int
}

/// VLA 단계 실행 중이면 1
///
/// # Safety
/// `r` 는 mr_new 가 준 것.
#[no_mangle]
pub unsafe extern "C" fn mr_vla_busy(r: *const Robot) -> c_int {
    if r.is_null() {
        return 0;
    }
    (*r).vla_busy() as c_int
}

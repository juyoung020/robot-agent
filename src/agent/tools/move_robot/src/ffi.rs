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

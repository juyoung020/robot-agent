//! C ABI(`include/search_objects.h`) — Rust 밖(파이썬 ctypes, 다른 에이전트 루프)에서 같은 도구를 부를 때.
//!
//! `so_open` → `so_call`(도구 이름 + 인자 JSON → 결과 JSON) … → `so_close`. 결과 쓰기는 sgsearch 와 같은 규칙:
//! 반환 = 쓴 바이트 수, 버퍼가 모자라면 −필요한 크기(끝 0 포함), 그 밖 < 0 = 인자 오류.

use crate::sys::Paths;
use crate::ObjectSearch;
use serde_json::Value;
use std::ffi::{c_char, CStr};
use std::path::Path;

fn emit(s: &str, out: *mut c_char, cap: i32) -> i32 {
    let need = s.len() + 1;
    if out.is_null() || (cap as usize) < need {
        return -(need as i32);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), out as *mut u8, s.len());
        *out.add(s.len()) = 0;
    }
    s.len() as i32
}

/// 기억 폴더를 열어 도구 실행기를 만든다(라벨 표·글 인코더 경로는 환경 변수·기본값). 실패면 NULL + err
#[no_mangle]
pub extern "C" fn so_open(mem_dir: *const c_char, err: *mut c_char, err_len: i32) -> *mut ObjectSearch {
    if mem_dir.is_null() {
        return std::ptr::null_mut();
    }
    let dir = unsafe { CStr::from_ptr(mem_dir) }.to_string_lossy().into_owned();
    match ObjectSearch::open(Path::new(&dir), &Paths::default()) {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(e) => {
            put_err(&e, err, err_len);
            std::ptr::null_mut()
        }
    }
}

/// 실시간 기억으로 연다: `sm_ctx` = scenemap 문맥(sgrt 면 `sgrt_scenemap(s)`), `sm_lib` = scenemap 이 든 라이브러리 경로
/// (NULL = 프로세스 전체에서 dlsym — 파이썬 ctypes 처럼 RTLD_LOCAL 로 올렸으면 그 경로), `mem_dir` = 지도 저장 폴더(sgrt out_dir:
/// 색인 벡터·확인 기록·pos_sd, view.json 이 있어야 함). 실패면 NULL + err
#[no_mangle]
pub extern "C" fn so_open_live(sm_ctx: *mut std::ffi::c_void, sm_lib: *const c_char, mem_dir: *const c_char, err: *mut c_char, err_len: i32) -> *mut ObjectSearch {
    if mem_dir.is_null() {
        return std::ptr::null_mut();
    }
    let dir = unsafe { CStr::from_ptr(mem_dir) }.to_string_lossy().into_owned();
    let lib = (!sm_lib.is_null()).then(|| unsafe { CStr::from_ptr(sm_lib) }.to_string_lossy().into_owned());
    let r = crate::live::SmCtx::new(sm_ctx, lib.as_deref())
        .and_then(|api| ObjectSearch::open_live(Box::new(api), Path::new(&dir), &Paths::default()));
    match r {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(e) => {
            put_err(&e, err, err_len);
            std::ptr::null_mut()
        }
    }
}

fn put_err(e: &str, err: *mut c_char, err_len: i32) {
    if !err.is_null() && err_len > 0 {
        let b = e.as_bytes();
        let n = b.len().min(err_len as usize - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(b.as_ptr(), err as *mut u8, n);
            *err.add(n) = 0;
        }
    }
}

#[no_mangle]
pub extern "C" fn so_close(h: *mut ObjectSearch) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

/// 도구 호출 하나. tool = "search_objects" | "confirm_object" | "list_place", args = 인자 JSON(UTF-8)
#[no_mangle]
pub extern "C" fn so_call(h: *mut ObjectSearch, tool: *const c_char, args: *const c_char, out: *mut c_char, cap: i32) -> i32 {
    if h.is_null() || tool.is_null() || args.is_null() {
        return i32::MIN;
    }
    let s = unsafe { &mut *h };
    let t = unsafe { CStr::from_ptr(tool) }.to_string_lossy().into_owned();
    let a = unsafe { CStr::from_ptr(args) }.to_string_lossy().into_owned();
    let r = s.run_tool(&t, &Value::String(a));
    emit(&r.to_string(), out, cap)
}

/// 도구 정의 세 개(OpenAI tools 배열 JSON)
#[no_mangle]
pub extern "C" fn so_definitions(out: *mut c_char, cap: i32) -> i32 {
    emit(&Value::Array(crate::definitions()).to_string(), out, cap)
}

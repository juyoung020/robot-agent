//! 실시간 기억 — scenemap C ABI(`sm_snapshot` · `sm_snap_*` · `sm_observe_object_name`,
//! `src/scene_graph/scenemap/include/scenemap.h`) 위의 [`Memory`]. 시뮬(sgrt, `sgrt_scenemap`)과 실제 로봇이 같은 길.
//!
//! - 도구 호출마다 새 스냅숏(참조 카운트, 어느 스레드든 됨)을 잡아 물체·자세·방을 복사하고 바로 놓는다 → 자리·상태·자세는 지금 지도 그대로.
//! - 이름 사후·벡터는 공용 색인(sgsearch)이 지도 저장 폴더(`sm_save_dsg` / sgrt `save_s` 주기 저장, `view.json` + `objects/`)에서 읽는다
//!   → 이름·생김새는 저장 주기(sgrt 1 s)만큼 늦을 수 있다. 저장 전 새 물체는 찾기에는 아직 안 나오고 `list_place` 에는 나온다.
//! - objprob `pos_sd` 는 스냅숏 ABI 에 없어 같은 저장 폴더의 `view.json` 에서(위치 불확실도라 1 s 늦어도 됨).
//! - 함수는 링크하지 않고 `dlsym` 으로 찾는다: 이 프로세스에 scenemap 이 이미 올라와 있어야 한다(sgrt 의 libsgrt.so, 로봇의 scenemap).
//!   파이썬 ctypes 처럼 RTLD_LOCAL 로 올린 라이브러리면 그 경로를 주면 `RTLD_NOLOAD` 로 그 핸들에서 찾는다.
//!
//! 시험은 [`SnapApi`] 를 가짜로 바꿔 끼운다(tests.rs), 진짜 scenemap 으로는 `search-objects live-check`.

use crate::memview::{arr3, MemSource, Memory, ObjInfo};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::PathBuf;
use std::time::SystemTime;

// ---------------------------------------------------------------- C 배치(scenemap.h 와 같음)

#[repr(C)]
#[derive(Clone, Copy)]
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

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SmPose2 {
    pub stamp: f64,
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SmStatus {
    pub last_proprio_stamp: f64,
    pub last_image_stamp: f64,
    pub n_objects: i32,
    pub n_images: i32,
    pub n_proprio: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SmRoom {
    pub id: u32,
    pub name: *const c_char,
    pub typ: *const c_char,
    pub name_conf: f32,
    pub centroid: [f64; 2],
    pub bbox_min: [f64; 2],
    pub bbox_max: [f64; 2],
    pub area_m2: f64,
    pub n_objects: i32,
    pub objects: *const u32,
}

pub const SM_SEEN: i32 = 0;
pub const SM_GONE: i32 = 1;
pub const SM_MOVED: i32 = 2;
pub const SM_HELD: i32 = 3;

pub fn state_name(s: i32) -> &'static str {
    match s {
        SM_GONE => "gone",
        SM_MOVED => "moved",
        SM_HELD => "held",
        _ => "seen",
    }
}

// ---------------------------------------------------------------- 스냅숏 원천

/// 스냅숏 하나에서 복사한 것
#[derive(Clone, Debug, Default)]
pub struct SnapData {
    pub objs: Vec<ObjInfo>,
    pub pose: Option<[f64; 3]>,
    pub now: f64,
    pub rooms: Vec<(i64, String)>,
}

/// 지도 쪽 접점: 진짜는 [`SmCtx`], 시험은 가짜
pub trait SnapApi: Send {
    fn take(&mut self) -> Result<SnapData, String>;
    fn observe_name(&mut self, id: u32, label: &str, log_lr: f32) -> i32;
}

extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, name: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;
const RTLD_NOLOAD: c_int = 4;

/// 이미 올라온 라이브러리 핸들(`lib` = 경로나 soname, None = 프로세스 전체 RTLD_DEFAULT)
pub fn loaded_handle(lib: Option<&str>) -> Result<*mut c_void, String> {
    match lib {
        None => Ok(std::ptr::null_mut()),
        Some(l) => {
            let c = CString::new(l).map_err(|e| e.to_string())?;
            let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_NOLOAD) };
            if h.is_null() {
                Err(format!("{l} is not loaded in this process"))
            } else {
                Ok(h)
            }
        }
    }
}

pub fn sym(h: *mut c_void, name: &str) -> *mut c_void {
    let c = CString::new(name).unwrap();
    unsafe { dlsym(h, c.as_ptr()) }
}

type FSnapshot = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> c_int;
type FRelease = unsafe extern "C" fn(*mut c_void);
type FPose = unsafe extern "C" fn(*const c_void) -> SmPose2;
type FStatus = unsafe extern "C" fn(*const c_void) -> SmStatus;
type FObjects = unsafe extern "C" fn(*const c_void, *mut *const SmObject) -> c_int;
type FMovable = unsafe extern "C" fn(*const c_void, u32) -> c_int;
type FRooms = unsafe extern "C" fn(*const c_void, *mut *const SmRoom) -> c_int;
type FObjRoom = unsafe extern "C" fn(*const c_void, u32) -> u32;
type FObserve = unsafe extern "C" fn(*mut c_void, u32, *const c_char, f32) -> c_int;

/// 진짜 scenemap 문맥(`sm_ctx*`). 함수는 dlsym 으로.
pub struct SmCtx {
    ctx: *mut c_void,
    snapshot: FSnapshot,
    release: FRelease,
    pose: FPose,
    status: FStatus,
    objects: FObjects,
    movable: FMovable,
    rooms: Option<FRooms>,
    obj_room: Option<FObjRoom>,
    observe: Option<FObserve>,
}

// sm_snapshot·sm_snap_*·sm_observe_object_name 은 scenemap 이 잠근다(scenemap.h "스레드") — 다른 스레드에서 불러도 됨
unsafe impl Send for SmCtx {}

impl SmCtx {
    /// `ctx` = `sm_ctx*`(sgrt 면 `sgrt_scenemap`), `lib` = scenemap 이 든 라이브러리(None = 프로세스 전체에서 찾음)
    pub fn new(ctx: *mut c_void, lib: Option<&str>) -> Result<SmCtx, String> {
        if ctx.is_null() {
            return Err("scenemap context is NULL".into());
        }
        let h = loaded_handle(lib)?;
        let need = |n: &str| {
            let p = sym(h, n);
            if p.is_null() {
                Err(format!("scenemap symbol {n} not found (is scenemap/sgrt loaded? pass its library path)"))
            } else {
                Ok(p)
            }
        };
        let opt = |n: &str| Some(sym(h, n)).filter(|p| !p.is_null());
        unsafe {
            Ok(SmCtx {
                ctx,
                snapshot: std::mem::transmute::<*mut c_void, FSnapshot>(need("sm_snapshot")?),
                release: std::mem::transmute::<*mut c_void, FRelease>(need("sm_snapshot_release")?),
                pose: std::mem::transmute::<*mut c_void, FPose>(need("sm_snap_pose")?),
                status: std::mem::transmute::<*mut c_void, FStatus>(need("sm_snap_status")?),
                objects: std::mem::transmute::<*mut c_void, FObjects>(need("sm_snap_objects")?),
                movable: std::mem::transmute::<*mut c_void, FMovable>(need("sm_snap_movable")?),
                rooms: opt("sm_snap_rooms").map(|p| std::mem::transmute::<*mut c_void, FRooms>(p)),
                obj_room: opt("sm_snap_object_room").map(|p| std::mem::transmute::<*mut c_void, FObjRoom>(p)),
                observe: opt("sm_observe_object_name").map(|p| std::mem::transmute::<*mut c_void, FObserve>(p)),
            })
        }
    }
}

fn cs(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

impl SnapApi for SmCtx {
    fn take(&mut self) -> Result<SnapData, String> {
        let mut s: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { (self.snapshot)(self.ctx, &mut s) };
        if rc != 0 || s.is_null() {
            return Err(format!("sm_snapshot failed ({rc})"));
        }
        let mut d = SnapData::default();
        unsafe {
            let p = (self.pose)(s);
            let st = (self.status)(s);
            d.pose = Some([p.x, p.y, p.yaw]);
            d.now = st.last_proprio_stamp.max(st.last_image_stamp).max(p.stamp);
            if let Some(f) = self.rooms {
                let mut r: *const SmRoom = std::ptr::null();
                let n = f(s, &mut r);
                for i in 0..n.max(0) as usize {
                    let x = &*r.add(i);
                    d.rooms.push((x.id as i64, cs(x.name)));
                }
            }
            let mut o: *const SmObject = std::ptr::null();
            let n = (self.objects)(s, &mut o);
            for i in 0..n.max(0) as usize {
                let x = &*o.add(i);
                let room = self.obj_room.map(|f| f(s, x.id)).filter(|r| *r > 0).map(|r| r as i64);
                d.objs.push(ObjInfo {
                    id: x.id,
                    name: cs(x.name),
                    pos: x.pos,
                    extent: x.extent,
                    state: state_name(x.state).into(),
                    last_seen: x.last_seen,
                    room,
                    movable: (self.movable)(s, x.id) != 0,
                    structural: x.structural != 0,
                    n_obs: x.n_obs,
                    pos_sd: None,
                });
            }
            (self.release)(s);
        }
        Ok(d)
    }

    fn observe_name(&mut self, id: u32, label: &str, log_lr: f32) -> i32 {
        let Some(f) = self.observe else { return -9 };
        let c = CString::new(label.replace('\0', " ")).unwrap();
        unsafe { f(self.ctx, id, c.as_ptr(), log_lr) }
    }
}

// ---------------------------------------------------------------- Memory

/// 실시간 기억: 스냅숏 원천 + (선택) 저장 폴더 view.json 의 objprob `pos_sd`
pub struct LiveMem {
    pub api: Box<dyn SnapApi>,
    data: SnapData,
    side: Option<PathBuf>,
    side_mtime: Option<SystemTime>,
    sd: HashMap<u32, [f64; 3]>,
}

impl LiveMem {
    /// `side_view_json` = 지도 저장 폴더의 view.json(pos_sd 용, 없어도 됨)
    pub fn new(api: Box<dyn SnapApi>, side_view_json: Option<PathBuf>) -> Result<LiveMem, String> {
        let mut m = LiveMem { api, data: SnapData::default(), side: side_view_json, side_mtime: None, sd: HashMap::new() };
        m.pull()?;
        Ok(m)
    }

    fn pull(&mut self) -> Result<(), String> {
        self.data = self.api.take()?;
        if let Some(p) = &self.side {
            let mt = std::fs::metadata(p).and_then(|m| m.modified()).ok();
            if mt.is_some() && mt != self.side_mtime {
                self.side_mtime = mt;
                self.sd.clear();
                if let Some(j) = std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()) {
                    for o in j["objects"].as_array().into_iter().flatten() {
                        if o["pos_sd"].as_array().is_some_and(|a| a.len() == 3) {
                            self.sd.insert(o["id"].as_u64().unwrap_or(0) as u32, arr3(&o["pos_sd"]));
                        }
                    }
                }
            }
        }
        for o in &mut self.data.objs {
            o.pos_sd = self.sd.get(&o.id).copied();
        }
        Ok(())
    }
}

impl Memory for LiveMem {
    fn objects(&self) -> &[ObjInfo] {
        &self.data.objs
    }
    fn now(&self) -> f64 {
        self.data.now
    }
    fn pose(&self) -> Option<[f64; 3]> {
        self.data.pose
    }
    fn room_name(&self, room: i64) -> Option<String> {
        self.data.rooms.iter().find(|r| r.0 == room).map(|r| r.1.clone()).filter(|s| !s.is_empty())
    }
    fn find_rooms(&self, s: &str) -> Vec<i64> {
        let t = s.trim().to_lowercase();
        let num = t.trim_start_matches('r').parse::<i64>().ok();
        self.data.rooms.iter().filter(|r| Some(r.0) == num || r.1.to_lowercase() == t).map(|r| r.0).collect()
    }
}

impl MemSource for LiveMem {
    fn mem(&self) -> &dyn Memory {
        self
    }
    fn refresh(&mut self) -> Result<bool, String> {
        self.pull()?;
        Ok(true)
    }
    fn observe_name(&mut self, id: u32, label: &str, log_lr: f32) -> Option<i32> {
        Some(self.api.observe_name(id, label, log_lr))
    }
    fn kind(&self) -> &'static str {
        "live"
    }
}

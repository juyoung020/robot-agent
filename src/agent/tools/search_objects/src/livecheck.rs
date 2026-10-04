//! `search-objects live-check SM_LIB MEM` — 진짜 scenemap(C ABI, 같은 프로세스)에 합성 스트림을 넣으며 실시간 기억 도구를 끝까지 본다.
//!
//! SM_LIB = scenemap 이 든 공유 라이브러리(sgrt 의 `libsgrt.so` — sm_* 를 내보냄). `dlopen` 으로 올리고(RTLD_LOCAL, 파이썬 ctypes 와 같음)
//! [`crate::live::SmCtx`] 가 그 핸들에서 함수를 찾는다. GPU 를 쓰지 않는다(sm_create 는 CPU, 색인은 글·영상 인코더 끔).
//!
//! 장면(LIMO + OMX-F, 로봇은 원점): 낮은 탁자(0.8–1.4 m 앞, 높이 0.25), 탁자 위 컵, 바닥 라디오(왼쪽 앞), 뒤 벽 x = 3.
//! 몸통 깊이 카메라 광선 추적 → 깊이 + 검출 마스크 → `sm_push_proprio`·`sm_push_image`. 확인하는 것:
//! 1. `sm_save_dsg` 한 번 뒤 색인을 열고 찾기·list_place 의 자리가 스냅숏과 같음
//! 2. 저장 없이 컵을 옮기고 로봇을 90° 돌림 → 다음 호출의 `pos`·`rel` 이 바로 바뀜(실시간), 색인 이름은 그대로
//! 3. `confirm_object` → `sm_observe_object_name` 을 부름(A′ 를 안 켰으면 "map":"not_aprime"), confirmations.jsonl 한 줄

use crate::live::{loaded_handle, sym, SmCtx, SmObject};
use crate::sys::Paths;
use crate::{ObjectSearch, CONFIRM, LIST, SEARCH};
use serde_json::{json, Value};
use std::ffi::{c_char, c_int, c_void, CString};
use std::path::Path;

#[repr(C)]
struct SmProprio {
    stamp: f64,
    proprio: *const f32,
    n: c_int,
}
#[repr(C)]
struct SmImage {
    stamp: f64,
    cam: c_int,
    w: c_int,
    h: c_int,
    rgba: *const u8,
    depth: *const f32,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}
#[repr(C)]
struct SmDet {
    stamp: f64,
    cam: c_int,
    img_w: c_int,
    img_h: c_int,
    n: c_int,
    cls: *const i32,
    score: *const f32,
    bx: *const f32,
    mask_w: c_int,
    mask_h: c_int,
    sx: f32,
    sy: f32,
    ox: f32,
    oy: f32,
    bits: *const u32,
}
#[repr(C)]
struct SmFk {
    n_cams: i32,
    n_hands: i32,
    cam_valid: [i32; 3],
    t_cam: [[f64; 12]; 3],
    eef_valid: [i32; 2],
    t_eef: [[f64; 12]; 2],
    grip: [f32; 2],
}

extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
}

struct Api {
    create: unsafe extern "C" fn(*const c_char) -> *mut c_void,
    destroy: unsafe extern "C" fn(*mut c_void),
    labels: unsafe extern "C" fn(*mut c_void, *const *const c_char, c_int) -> c_int,
    proprio: unsafe extern "C" fn(*mut c_void, *const SmProprio) -> c_int,
    image: unsafe extern "C" fn(*mut c_void, *const SmImage, *const SmDet) -> c_int,
    fk: unsafe extern "C" fn(i32, *const f32, i32, *mut SmFk) -> c_int,
    save: unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int,
    snapshot: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> c_int,
    release: unsafe extern "C" fn(*mut c_void),
    objects: unsafe extern "C" fn(*const c_void, *mut *const SmObject) -> c_int,
}

const W: usize = 160;
const H: usize = 120;

struct Rig {
    api: Api,
    /// A′ 켬: 검출마다 넣을 임베딩(라벨 one-hot, 차원 4)
    det_emb: Option<unsafe extern "C" fn(*mut c_void, *const f32, i32, i32) -> c_int>,
    c: *mut c_void,
    t: f64,
    pose: [f64; 3],
    boxes: Vec<([f64; 3], [f64; 3], i32)>,
}

fn slab(o: [f64; 3], r: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> Option<f64> {
    let (mut t0, mut t1) = (0.0f64, 1e9f64);
    for i in 0..3 {
        if r[i].abs() < 1e-12 {
            if o[i] < lo[i] || o[i] > hi[i] {
                return None;
            }
            continue;
        }
        let (mut a, mut b) = ((lo[i] - o[i]) / r[i], (hi[i] - o[i]) / r[i]);
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        t0 = t0.max(a);
        t1 = t1.min(b);
        if t0 > t1 {
            return None;
        }
    }
    (t0 > 0.0).then_some(t0)
}

impl Rig {
    fn q(&self) -> Vec<f32> {
        let mut q = vec![0f32; 12];
        q[0] = self.pose[0] as f32;
        q[1] = self.pose[1] as f32;
        q[2] = self.pose[2] as f32;
        let home = [0.0, 1.3, -1.9, 0.7, 0.0];
        for k in 0..5 {
            q[6 + k] = home[k];
        }
        q[11] = 1.0;
        q
    }

    /// 한 스텝(0.1 s): proprio, image 면 깊이 + 검출도
    fn step(&mut self, image: bool) {
        self.t += 0.1;
        let q = self.q();
        let p = SmProprio { stamp: self.t, proprio: q.as_ptr(), n: 12 };
        unsafe { (self.api.proprio)(self.c, &p) };
        if !image {
            return;
        }
        let mut fk: SmFk = unsafe { std::mem::zeroed() };
        unsafe { (self.api.fk)(1, q.as_ptr(), 12, &mut fk) };
        let tb = fk.t_cam[0];
        // map ← 베이스(자세) ∘ 베이스 ← 카메라
        let (s, c) = self.pose[2].sin_cos();
        let rot = |v: [f64; 3]| [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]];
        let fx = 80.0 / (71.0f64 / 2.0).to_radians().tan();
        let (cx, cy) = (79.5, 59.5);
        let o_b = [tb[3], tb[7], tb[11]];
        let o = {
            let r = rot(o_b);
            [r[0] + self.pose[0], r[1] + self.pose[1], r[2]]
        };
        let mut depth = vec![0f32; W * H];
        let mut hit = vec![-1i32; W * H];
        for v in 0..H {
            for u in 0..W {
                let d = [(u as f64 - cx) / fx, (v as f64 - cy) / fx, 1.0];
                let rb = [0, 1, 2].map(|i| tb[i * 4] * d[0] + tb[i * 4 + 1] * d[1] + tb[i * 4 + 2] * d[2]);
                let r = rot(rb);
                let mut best = 1e9;
                let mut who = -1;
                if r[2] < -1e-9 {
                    best = -o[2] / r[2];
                }
                for (b, (lo, hi, _)) in self.boxes.iter().enumerate() {
                    if let Some(t) = slab(o, r, *lo, *hi) {
                        if t < best {
                            best = t;
                            who = b as i32;
                        }
                    }
                }
                // 벽: x = 3, y = ±3 (상자로)
                for (lo, hi) in [([3.0, -3.0, 0.0], [3.1, 3.0, 2.5]), ([-3.0, 3.0, 0.0], [3.0, 3.1, 2.5]), ([-3.0, -3.1, 0.0], [3.0, -3.0, 2.5])] {
                    if let Some(t) = slab(o, r, lo, hi) {
                        if t < best {
                            best = t;
                            who = -1;
                        }
                    }
                }
                if best < 8.0 {
                    depth[v * W + u] = best as f32;
                }
                hit[v * W + u] = who;
            }
        }
        let words = (W * H).div_ceil(32);
        let (mut cls, mut score, mut bx, mut bits) = (vec![], vec![], vec![], vec![]);
        for (b, (_, _, k)) in self.boxes.iter().enumerate() {
            let (mut x0, mut y0, mut x1, mut y1) = (W, H, 0usize, 0usize);
            let mut any = false;
            let base = bits.len();
            bits.resize(base + words, 0u32);
            for v in 0..H {
                for u in 0..W {
                    if hit[v * W + u] == b as i32 {
                        any = true;
                        x0 = x0.min(u);
                        y0 = y0.min(v);
                        x1 = x1.max(u);
                        y1 = y1.max(v);
                        let i = v * W + u;
                        bits[base + (i >> 5)] |= 1 << (i & 31);
                    }
                }
            }
            if !any {
                bits.truncate(base);
                continue;
            }
            cls.push(*k);
            score.push(0.9f32);
            bx.extend([x0 as f32, y0 as f32, (x1 + 1) as f32, (y1 + 1) as f32]);
        }
        let det = SmDet {
            stamp: self.t,
            cam: 0,
            img_w: W as c_int,
            img_h: H as c_int,
            n: cls.len() as c_int,
            cls: cls.as_ptr(),
            score: score.as_ptr(),
            bx: bx.as_ptr(),
            mask_w: W as c_int,
            mask_h: H as c_int,
            sx: 1.0,
            sy: 1.0,
            ox: 0.0,
            oy: 0.0,
            bits: bits.as_ptr(),
        };
        if let Some(f) = self.det_emb {
            let mut e = vec![0f32; cls.len() * 4];
            for (i, k) in cls.iter().enumerate() {
                e[i * 4 + *k as usize] = 1.0;
            }
            unsafe { f(self.c, e.as_ptr(), cls.len() as i32, 4) };
        }
        let im = SmImage { stamp: self.t, cam: 0, w: W as c_int, h: H as c_int, rgba: std::ptr::null(), depth: depth.as_ptr(), fx, fy: fx, cx, cy };
        unsafe { (self.api.image)(self.c, &im, &det) };
    }

    fn snap_objects(&self) -> Vec<(u32, String, [f64; 3])> {
        let mut s = std::ptr::null_mut();
        let mut out = vec![];
        unsafe {
            (self.api.snapshot)(self.c, &mut s);
            let mut o: *const SmObject = std::ptr::null();
            let n = (self.api.objects)(s, &mut o);
            for i in 0..n.max(0) as usize {
                let x = &*o.add(i);
                out.push((x.id, std::ffi::CStr::from_ptr(x.name).to_string_lossy().into_owned(), x.pos));
            }
            (self.api.release)(s);
        }
        out
    }
}

fn check(ok: bool, what: &str, fails: &mut u32) {
    println!("{} {what}", if ok { "ok  " } else { "FAIL" });
    *fails += !ok as u32;
}

/// 0 = 다 통과
pub fn run(lib: &str, mem: &str) -> i32 {
    let cl = CString::new(lib).unwrap();
    let h = unsafe { dlopen(cl.as_ptr(), 2) };
    if h.is_null() {
        eprintln!("cannot dlopen {lib}");
        return 2;
    }
    macro_rules! f {
        ($n:literal) => {{
            let p = sym(h, $n);
            if p.is_null() {
                eprintln!("{lib}: no symbol {}", $n);
                return 2;
            }
            unsafe { std::mem::transmute::<*mut c_void, _>(p) }
        }};
    }
    let api = Api {
        create: f!("sm_create"),
        destroy: f!("sm_destroy"),
        labels: f!("sm_set_labels"),
        proprio: f!("sm_push_proprio"),
        image: f!("sm_push_image"),
        fk: f!("sm_robot_fk"),
        save: f!("sm_save_dsg"),
        snapshot: f!("sm_snapshot"),
        release: f!("sm_snapshot_release"),
        objects: f!("sm_snap_objects"),
    };
    assert!(loaded_handle(Some(lib)).is_ok());
    let cfg = CString::new(r#"{"robot": "limo_omx"}"#).unwrap();
    let c = unsafe { (api.create)(cfg.as_ptr()) };
    let names: Vec<CString> = ["cup", "table", "radio"].iter().map(|s| CString::new(*s).unwrap()).collect();
    let np: Vec<*const c_char> = names.iter().map(|s| s.as_ptr()).collect();
    unsafe { (api.labels)(c, np.as_ptr(), 3) };
    // SO_LIVE_APRIME=1: A′ 물체 모델을 켬(장난감 글 모델: 라벨마다 차원 4 one-hot) → confirm 이 지도에 "applied" 되는지
    let aprime = std::env::var("SO_LIVE_APRIME").is_ok_and(|v| v == "1");
    let mut det_emb = None;
    if aprime {
        let tm: unsafe extern "C" fn(*mut c_void, *const f32, *const i32, i32, i32, f32, f32) -> c_int = f!("sm_set_text_model");
        let om: unsafe extern "C" fn(*mut c_void, i32) -> c_int = f!("sm_set_object_model");
        let text: Vec<f32> = (0..3).flat_map(|r| (0..4).map(move |d| if d == r { 1.0 } else { 0.0 })).collect();
        let rl = [0i32, 1, 2];
        unsafe {
            tm(c, text.as_ptr(), rl.as_ptr(), 3, 4, 10.0, -5.0);
            om(c, 1);
        }
        det_emb = Some(f!("sm_set_det_embeddings"));
    }
    let mut rig = Rig {
        api,
        det_emb,
        c,
        t: 0.0,
        pose: [0.0; 3],
        boxes: vec![
            ([0.80, -0.30, 0.0], [1.40, 0.30, 0.25], 1), // 낮은 탁자
            ([1.00, -0.04, 0.25], [1.08, 0.04, 0.35], 0), // 탁자 위 컵
            ([0.90, 0.45, 0.0], [1.10, 0.60, 0.12], 2),   // 바닥 라디오
        ],
    };
    let mut fails = 0;
    for _ in 0..12 {
        rig.step(true);
    }
    let objs = rig.snap_objects();
    println!("snapshot objects: {objs:?}");
    let id_of = |n: &str, o: &[(u32, String, [f64; 3])]| o.iter().find(|x| x.1 == n).map(|x| (x.0, x.2));
    let (Some((cup, cup_p)), Some((table, _)), Some((radio, _))) = (id_of("cup", &objs), id_of("table", &objs), id_of("radio", &objs)) else {
        println!("FAIL scenemap did not create cup/table/radio from the synthetic stream");
        return 1;
    };
    let _ = std::fs::remove_dir_all(mem);
    std::fs::create_dir_all(mem).unwrap();
    let cm = CString::new(mem).unwrap();
    unsafe { (rig.api.save)(c, cm.as_ptr()) };
    check(Path::new(mem).join("view.json").exists(), "sm_save_dsg wrote view.json", &mut fails);

    let api = match SmCtx::new(c, Some(lib)) {
        Ok(a) => a,
        Err(e) => {
            println!("FAIL SmCtx: {e}");
            return 1;
        }
    };
    let p = Paths { no_text: true, no_image: true, ..Default::default() };
    let mut s = match ObjectSearch::open_live(Box::new(api), Path::new(mem), &p) {
        Ok(s) => s,
        Err(e) => {
            println!("FAIL open_live: {e}");
            return 1;
        }
    };
    let call = |s: &mut ObjectSearch, t: &str, a: Value| -> Value {
        let r = s.run_tool(t, &a);
        println!("  call {t} {a}\n  -> {r}");
        r
    };
    // 1. 찾기
    let r = call(&mut s, SEARCH, json!({"query": "cup"}));
    let m = &r["matches"][0];
    check(m["id"] == format!("O{cup}"), "search cup -> the live cup id", &mut fails);
    let close = |v: &Value, p: [f64; 3]| (0..3).all(|i| (v[i].as_f64().unwrap_or(9.0) - p[i]).abs() < 0.011);
    check(close(&m["pos"], cup_p), "pos = snapshot pos (cm)", &mut fails);
    check((m["rel"]["x"].as_f64().unwrap_or(0.0) - cup_p[0]).abs() < 0.011 && m["rel"]["bearing_deg"].as_f64().is_some(), "rel at yaw 0 = map pos", &mut fails);
    let r = call(&mut s, LIST, json!({"place": format!("O{table}")}));
    let ids: Vec<&str> = r["objects"].as_array().map(|a| a.iter().filter_map(|o| o["id"].as_str()).collect()).unwrap_or_default();
    check(ids.contains(&format!("O{cup}").as_str()) && ids.contains(&format!("O{radio}").as_str()), "list_place(table) has cup and radio", &mut fails);
    let cup_row = r["objects"].as_array().and_then(|a| a.iter().find(|o| o["id"] == format!("O{cup}")).cloned()).unwrap_or_default();
    check(cup_row["dist_m"].as_f64().is_some_and(|d| d <= 0.3) && cup_row["dz_m"].as_f64().is_some_and(|z| z > 0.0 && z < 0.1), "cup: within 0.3 m of the table box (scenemap sees its front face only), dz > 0 (numbers, no relation word)", &mut fails);

    // 2. 저장 없이: 컵을 탁자 오른쪽 끝으로, 로봇은 제자리에서 왼쪽으로 90° 돌았다가 돌아와 다시 봄
    rig.boxes[1] = ([1.25, -0.24, 0.25], [1.33, -0.16, 0.35], 0);
    // 서 있으면 scenemap 이 같은 자리 keyframe 을 덜 넣으므로, 고개를 한 번 돌렸다 오며 다시 봄
    for k in (1..=10).chain((0..10).rev()) {
        rig.pose[2] = (6.0 * k as f64).to_radians();
        rig.step(true);
    }
    for _ in 0..10 {
        rig.step(true);
    }
    let after = rig.snap_objects();
    println!("snapshot after the move: {after:?}");
    // 새 자리의 컵은 scenemap 이 새 물체로 만든다(옛 것은 아직 안 사라짐). 색인(저장 폴더)에는 아직 없음
    let Some(&(new_cup, _, moved)) = after.iter().find(|x| x.1 == "cup" && x.0 != cup) else {
        println!("FAIL scenemap did not register the cup at its new place");
        return 1;
    };
    for k in 1..=10 {
        rig.pose[2] = (9.0 * k as f64).to_radians();
        rig.step(false);
    }
    let r = call(&mut s, SEARCH, json!({"query": "cup"}));
    let m = r["matches"].as_array().and_then(|a| a.iter().find(|x| x["id"] == format!("O{new_cup}")).cloned()).unwrap_or_default();
    check(close(&m["pos"], moved) && m["unindexed"] == true, "live: the new cup is found before any save (unindexed, live pos)", &mut fails);
    let rel_y = m["rel"]["y"].as_f64().unwrap_or(0.0);
    check(rel_y < -0.9 && m["rel"]["bearing_deg"].as_f64().is_some_and(|b| b < -60.0), "live: robot turned 90° left -> cup is to the right (rel y < 0)", &mut fails);
    unsafe { (rig.api.save)(c, cm.as_ptr()) };
    let r = call(&mut s, SEARCH, json!({"query": "cup"}));
    let m = r["matches"].as_array().and_then(|a| a.iter().find(|x| x["id"] == format!("O{new_cup}")).cloned()).unwrap_or_default();
    check(m.get("unindexed").is_none() && m["name_p"].as_f64().is_some(), "after the next map save the index has it (name probability)", &mut fails);
    // 시간 거르개: 지금(now_s) 이후에 본 것 = 없음, 아주 넉넉한 max_age = 있음
    let now = r["now_s"].as_f64().unwrap_or(0.0);
    let r = call(&mut s, SEARCH, json!({"query": "cup", "seen_after_s": now + 5.0}));
    check(r["matches"].as_array().is_some_and(|a| a.is_empty()), "seen_after_s in the future -> nothing", &mut fails);
    let r = call(&mut s, SEARCH, json!({"query": "cup", "max_age_s": 600}));
    check(!r["matches"].as_array().is_none_or(|a| a.is_empty()), "max_age_s 600 -> found", &mut fails);

    // 3. 확인 → 지도에 이름 관측
    let r = call(&mut s, CONFIRM, json!({"id": format!("O{radio}"), "name": "라디오", "source": "user"}));
    check(r["status"] == "ok" && r["map"].is_string(), "confirm_object called sm_observe_object_name (map status reported)", &mut fails);
    let log = std::fs::read_to_string(Path::new(mem).join("confirmations.jsonl")).unwrap_or_default();
    let line: Value = log.lines().last().and_then(|l| serde_json::from_str(l).ok()).unwrap_or_default();
    check(line["id"] == radio && line["map"] == r["map"], "confirmations.jsonl line with the map status", &mut fails);
    if aprime {
        check(r["map"] == "applied", "A′ on: the map took the name observation", &mut fails);
        unsafe { (rig.api.save)(c, cm.as_ptr()) };
        let v: Value = serde_json::from_str(&std::fs::read_to_string(Path::new(mem).join("view.json")).unwrap_or_default()).unwrap_or_default();
        let np = v["objects"].as_array().and_then(|a| a.iter().find(|o| o["id"] == radio)).map(|o| o["name_post"].clone()).unwrap_or_default();
        println!("  radio name_post after save: {np}");
        check(np["external"] == true && np["top"].is_array(), "saved A′ name_post has external: true (index will not count the confirmation twice)", &mut fails);
        let r = call(&mut s, SEARCH, json!({"query": "radio"}));
        check(r["matches"][0]["id"] == format!("O{radio}") && r["matches"][0]["match_type"] == "name", "radio found by name after reload", &mut fails);
    }
    drop(s);
    unsafe { (rig.api.destroy)(c) };
    println!("{}", if fails == 0 { "OK" } else { "FAILED" });
    fails as i32
}

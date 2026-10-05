//! 공용 물체 색인 C ABI(`src/scene_graph/clip/include/sgclip.h`·`sgsearch.h`)의 날 선언과 얇은 안전 덮개.
//!
//! 이 크레이트는 벡터를 만지지 않는다. 라벨 표·글 인코더(SigLIP 2 글 탑, TensorRT)·영상 인코더(대체 벡터용)·색인을 열고,
//! 찾기·확인은 색인이 돌려주는 JSON 을 그대로 받는다. RecallVLA 실행기는 같은 색인을 `sgs_search_vec` 로(자기 질의 벡터) 쓴다.

use std::ffi::{c_char, c_void, CStr, CString};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SgsConfig {
    pub mem_dir: *const c_char,
    pub cache_dir: *const c_char,
    pub labels: *const c_void,
    pub text: *mut c_void,
    pub encoder: *mut c_void,
    pub threads: i32,
    pub reg_lr: f32,
    pub user_lr: f32,
    pub look_lr: f32,
    pub weak_name: f32,
    pub name_min: f32,
    pub app_min: f32,
    pub app_rank: i32,
    pub exemplar_min: f32,
    pub img_a: f32,
    pub img_c0: f32,
    pub img_min: f32,
    pub attr_min: f32,
}

#[repr(C)]
struct TextConfig {
    engine: *const c_char,
    tokenizer: *const c_char,
    tok_emb: *const c_char,
    device: i32,
    max_batch: i32,
}

#[repr(C)]
struct EncConfig {
    engine: *const c_char,
    device: i32,
    max_batch: i32,
    use_graph: i32,
    margin: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Slot {
    pub id: u32,
    pub score: f32,
    pub cos: f32,
}

extern "C" {
    fn sgc_labels_open_ex(dir: *const c_char, index_dir: *const c_char, img_sample: *const c_char, err: *mut c_char, n: usize) -> *mut c_void;
    fn sgc_labels_close(l: *mut c_void);
    fn sgc_text_default_config(c: *mut TextConfig, dir: *const c_char);
    fn sgc_text_create(c: *const TextConfig, err: *mut c_char, n: usize) -> *mut c_void;
    fn sgc_text_destroy(t: *mut c_void);
    fn sgc_default_config(c: *mut EncConfig);
    fn sgc_create(c: *const EncConfig, err: *mut c_char, n: usize) -> *mut c_void;
    fn sgc_destroy(e: *mut c_void);
    fn sgs_default_config(c: *mut SgsConfig);
    fn sgs_open(c: *const SgsConfig, err: *mut c_char, n: usize) -> *mut c_void;
    fn sgs_close(x: *mut c_void);
    fn sgs_reload(x: *mut c_void, err: *mut c_char, n: usize) -> i32;
    fn sgs_search_json(x: *mut c_void, q: *const c_char, k: i32, force: i32, out: *mut c_char, cap: i32) -> i32;
    fn sgs_search_vec(x: *const c_void, q: *const f32, k: i32, out: *mut Slot) -> i32;
    fn sgs_object_json(x: *const c_void, id: u32, out: *mut c_char, cap: i32) -> i32;
    fn sgs_confirm(x: *mut c_void, id: u32, name: *const c_char, source: *const c_char, query: *const c_char, out: *mut c_char, cap: i32) -> i32;
    fn sgs_stats_json(x: *const c_void, out: *mut c_char, cap: i32) -> i32;
}

fn cstr(s: &str) -> CString {
    CString::new(s.replace('\0', " ")).unwrap()
}

fn err_string(b: &[u8]) -> String {
    CStr::from_bytes_until_nul(b).map(|c| c.to_string_lossy().into_owned()).unwrap_or_default()
}

/// 색인이 JSON 을 쓰는 함수(반환 = 쓴 길이, 모자라면 −필요 길이)를 버퍼를 늘려 가며 부른다.
fn call_json(mut f: impl FnMut(*mut c_char, i32) -> i32) -> Result<String, String> {
    let mut cap = 1 << 16;
    loop {
        let mut buf = vec![0u8; cap];
        let n = f(buf.as_mut_ptr() as *mut c_char, cap as i32);
        if n >= 0 {
            buf.truncate(n as usize);
            return String::from_utf8(buf).map_err(|e| e.to_string());
        }
        if -n as usize > cap {
            cap = -n as usize + 1;
        } else {
            return Err(format!("index call failed ({n})"));
        }
    }
}

/// 여는 데 필요한 경로. 비우면 환경 변수·기본값(README "경로").
#[derive(Clone, Debug, Default)]
pub struct Paths {
    pub labels: Option<String>,
    pub text_dir: Option<String>,
    /// 대체 벡터용 영상 엔진(objprob 벡터가 없는 기억). None + 환경 변수 없음 = 기본 경로가 있으면 씀
    pub image_engine: Option<String>,
    pub no_text: bool,
    pub no_image: bool,
}

fn home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| ".".into())
}

/// 라벨 표 + 글 인코더 + (선택) 영상 인코더 + 색인 하나.
pub struct Index {
    labels: *mut c_void,
    text: *mut c_void,
    enc: *mut c_void,
    idx: *mut c_void,
    _keep: Vec<CString>,
    pub has_text: bool,
}

// 색인은 한 스레드에서 쓴다(sgsearch.h). 소유권을 다른 스레드로 넘기는 것은 괜찮다.
unsafe impl Send for Index {}

impl Index {
    pub fn open(mem_dir: &str, p: &Paths) -> Result<Index, String> {
        let h = home();
        let labels = p.labels.clone().or_else(|| std::env::var("SGRT_LABELS").ok()).unwrap_or_else(|| format!("{h}/embed_work/labels/objects-v1"));
        let sample = std::env::var("SGC_IMG_SAMPLE").unwrap_or_else(|_| format!("{h}/ovdet_models/x86_sm120/siglip2_b32/img_sample_lvis10k.f16"));
        let keep = vec![cstr(&labels), cstr(&format!("{h}/.cache/sgclip")), cstr(&sample), cstr(mem_dir)];
        let mut err = [0u8; 512];
        unsafe {
            let l = sgc_labels_open_ex(keep[0].as_ptr(), keep[1].as_ptr(), keep[2].as_ptr(), err.as_mut_ptr() as *mut c_char, err.len());
            if l.is_null() {
                return Err(format!("label table: {}", err_string(&err)));
            }
            let mut ix = Index { labels: l, text: std::ptr::null_mut(), enc: std::ptr::null_mut(), idx: std::ptr::null_mut(), _keep: keep, has_text: false };
            if !p.no_text {
                let mut tc: TextConfig = std::mem::zeroed();
                let td = p.text_dir.as_deref().map(cstr);
                sgc_text_default_config(&mut tc, td.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()));
                ix.text = sgc_text_create(&tc, err.as_mut_ptr() as *mut c_char, err.len());
                ix.has_text = !ix.text.is_null();
                if !ix.has_text {
                    eprintln!("[search_objects] text encoder off ({}): only label-table names can be searched", err_string(&err));
                }
            }
            if !p.no_image {
                let eng = p.image_engine.clone().or_else(|| std::env::var("SGC_ENGINE").ok())
                    .unwrap_or_else(|| format!("{h}/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan"));
                if std::path::Path::new(&eng).exists() {
                    let ce = cstr(&eng);
                    let mut ec: EncConfig = std::mem::zeroed();
                    sgc_default_config(&mut ec);
                    ec.engine = ce.as_ptr();
                    ec.margin = 0.0; // 기억의 best view 사진은 이미 잘린 것
                    ix.enc = sgc_create(&ec, err.as_mut_ptr() as *mut c_char, err.len());
                    ix._keep.push(ce);
                }
            }
            let mut c: SgsConfig = std::mem::zeroed();
            sgs_default_config(&mut c);
            c.mem_dir = ix._keep[3].as_ptr();
            c.labels = ix.labels;
            c.text = ix.text;
            c.encoder = ix.enc;
            ix.idx = sgs_open(&c, err.as_mut_ptr() as *mut c_char, err.len());
            if ix.idx.is_null() {
                return Err(format!("object index: {}", err_string(&err)));
            }
            Ok(ix)
        }
    }

    pub fn reload(&mut self) -> Result<(), String> {
        let mut err = [0u8; 512];
        let r = unsafe { sgs_reload(self.idx, err.as_mut_ptr() as *mut c_char, err.len()) };
        if r == 0 { Ok(()) } else { Err(err_string(&err)) }
    }

    /// 색인 찾기 JSON(include/sgsearch.h): k = 0 이면 기준을 넘는 것 전부
    pub fn search(&mut self, query: &str, k: i32, force_appearance: bool) -> Result<String, String> {
        let q = cstr(query);
        let idx = self.idx;
        call_json(|out, cap| unsafe { sgs_search_json(idx, q.as_ptr(), k, force_appearance as i32, out, cap) })
    }

    pub fn object(&self, id: u32) -> Result<String, String> {
        let idx = self.idx;
        call_json(|out, cap| unsafe { sgs_object_json(idx, id, out, cap) })
    }

    pub fn confirm(&mut self, id: u32, name: &str, source: &str, query: &str) -> Result<String, String> {
        let (n, s, q) = (cstr(name), cstr(source), cstr(query));
        let idx = self.idx;
        call_json(|out, cap| unsafe { sgs_confirm(idx, id, n.as_ptr(), s.as_ptr(), q.as_ptr(), out, cap) })
    }

    /// 확인 + 기록 한 줄에 덧붙일 칸(`extra` JSON 객체). 새 색인(`sgs_confirm_ex`, objsearch-live)이 없으면
    /// 옛 `sgs_confirm`(덧붙임 없이) — 이때 지도가 받은 확인도 색인이 한 번 더 셀 수 있다(README "실시간 기억").
    pub fn confirm_ex(&mut self, id: u32, name: &str, source: &str, query: &str, extra: &str) -> Result<String, String> {
        type F = unsafe extern "C" fn(*mut c_void, u32, *const c_char, *const c_char, *const c_char, *const c_char, *mut c_char, i32) -> i32;
        let p = clip_sym("sgs_confirm_ex");
        if p.is_null() {
            return self.confirm(id, name, source, query);
        }
        let f: F = unsafe { std::mem::transmute::<*mut c_void, F>(p) };
        let (n, s, q, e) = (cstr(name), cstr(source), cstr(query), cstr(extra));
        let idx = self.idx;
        call_json(|out, cap| unsafe { f(idx, id, n.as_ptr(), s.as_ptr(), q.as_ptr(), e.as_ptr(), out, cap) })
    }

    /// 이름 → 라벨 표 영어 이름(지도 라벨과 맞추기). 새 색인이 없으면 소문자 그대로
    pub fn label_of(&self, name: &str) -> String {
        type F = unsafe extern "C" fn(*const c_void, *const c_char, *mut c_char, i32) -> i32;
        let p = clip_sym("sgs_label_of");
        if p.is_null() {
            return name.trim().to_lowercase();
        }
        let f: F = unsafe { std::mem::transmute::<*mut c_void, F>(p) };
        let n = cstr(name);
        let idx = self.idx;
        call_json(|out, cap| unsafe { f(idx, n.as_ptr(), out, cap) }).unwrap_or_else(|_| name.trim().to_lowercase())
    }

    pub fn stats(&self) -> Result<String, String> {
        let idx = self.idx;
        call_json(|out, cap| unsafe { sgs_stats_json(idx, out, cap) })
    }

    /// RecallVLA 와 같은 길: 질의 벡터(768, L2) → 상위 k
    pub fn search_vec(&self, q: &[f32], k: usize) -> Vec<Slot> {
        if q.len() != 768 || k == 0 {
            return vec![];
        }
        let mut out = vec![Slot::default(); k];
        let n = unsafe { sgs_search_vec(self.idx, q.as_ptr(), k as i32, out.as_mut_ptr()) };
        out.truncate(n.max(0) as usize);
        out
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe {
            if !self.idx.is_null() {
                sgs_close(self.idx);
            }
            if !self.enc.is_null() {
                sgc_destroy(self.enc);
            }
            if !self.text.is_null() {
                sgc_text_destroy(self.text);
            }
            if !self.labels.is_null() {
                sgc_labels_close(self.labels);
            }
        }
    }
}

/// 색인 라이브러리의 (새) 함수: 링크한 libsgclip_c.so 핸들에서 찾음(없으면 NULL)
fn clip_sym(name: &str) -> *mut c_void {
    let h = crate::live::loaded_handle(Some("libsgclip_c.so")).unwrap_or(std::ptr::null_mut());
    crate::live::sym(h, name)
}

/// 확인 우도비(sgs_default_config): (user, close_look)
pub fn confirm_lrs() -> (f32, f32) {
    let mut c: SgsConfig = unsafe { std::mem::zeroed() };
    unsafe { sgs_default_config(&mut c) };
    (c.user_lr, c.look_lr)
}

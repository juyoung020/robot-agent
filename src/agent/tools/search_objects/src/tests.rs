//! 단위 시험 — GPU 없이. 스키마·인자 읽기·결과 만들기(가짜 기억)·끝까지(라벨 표가 있으면: 가짜 기억 폴더 + 공용 색인).

use super::*;
use crate::memview::ObjInfo;
use std::ffi::{c_char, c_void, CString};

fn args(s: &str) -> Value {
    serde_json::from_str(s).unwrap()
}

#[test]
fn schemas_are_small_and_strict() {
    let s = search_definition();
    assert_eq!(s["function"]["name"], SEARCH);
    assert_eq!(s["function"]["parameters"]["required"], json!(["query"]));
    assert_eq!(s["function"]["parameters"]["properties"]["state"]["enum"], json!(STATES));
    let c = confirm_definition();
    assert_eq!(c["function"]["parameters"]["required"], json!(["id", "name", "source"]));
    assert_eq!(c["function"]["parameters"]["properties"]["source"]["enum"], json!(["user", "close_look"]));
    // 9B 맥락 16k: 둘 합쳐 작게
    let n = s.to_string().len() + c.to_string().len();
    assert!(n < 2600, "definitions are {n} bytes");
}

#[test]
fn parse_tolerant_and_errors_fixable() {
    let a = parse_search(&Value::String(r#"{"query":" 라디오 ","k":"3","state":"Seen","near":"o12"}"#.into())).unwrap();
    assert_eq!((a.query.as_str(), a.k, a.state.as_deref(), a.near), ("라디오", 3, Some("seen"), Some(12)));
    assert_eq!(parse_search(&args(r#"{"query":"cup","k":99}"#)).unwrap().k, 10);
    assert!(parse_search(&args(r#"{"k":3}"#)).unwrap_err().contains("query is required"));
    assert!(parse_search(&args(r#"{"query":"cup","state":"lost"}"#)).unwrap_err().contains("seen, moved, held, gone"));
    let c = parse_confirm(&args(r#"{"id":"O27","name":"radio","source":"close look"}"#)).unwrap();
    assert_eq!(c, ConfirmArgs { id: 27, name: "radio".into(), source: "close_look".into() });
    assert!(parse_confirm(&args(r#"{"id":27,"name":"radio","source":"me"}"#)).unwrap_err().contains("user or close_look"));
    assert!(parse_confirm(&args(r#"{"name":"radio","source":"user"}"#)).unwrap_err().contains("id is required"));
}

/// 가짜 기억: 거실(R1) 선반 O33(고정), 그 위 O27, 부엌(R2) O40
struct Fake(Vec<ObjInfo>);
impl Memory for Fake {
    fn objects(&self) -> &[ObjInfo] {
        &self.0
    }
    fn now(&self) -> f64 {
        100.0
    }
    fn pose(&self) -> Option<[f64; 3]> {
        Some([0.0, 0.0, 0.0])
    }
    fn room_name(&self, r: i64) -> Option<String> {
        match r {
            1 => Some("living room".into()),
            2 => Some("kitchen".into()),
            _ => None,
        }
    }
    fn find_rooms(&self, s: &str) -> Vec<i64> {
        match s {
            "living room" | "R1" => vec![1],
            "kitchen" | "R2" => vec![2],
            _ => vec![],
        }
    }
}

fn obj(id: u32, name: &str, pos: [f64; 3], ext: [f64; 3], movable: bool, room: i64) -> ObjInfo {
    ObjInfo { id, name: name.into(), pos, extent: ext, state: "seen".into(), last_seen: 40.0, room: Some(room), movable, structural: !movable, n_obs: 5 }
}

fn fake() -> Fake {
    Fake(vec![
        obj(33, "shelf", [3.0, 4.0, 0.5], [1.2, 0.4, 1.0], false, 1),
        obj(27, "fire extinguisher", [3.1, 4.0, 1.12], [0.25, 0.1, 0.15], true, 1),
        obj(40, "radio", [8.0, 1.0, 0.9], [0.2, 0.1, 0.1], true, 2),
    ])
}

fn hit(id: u32, name: &str, mt: &str, m: f64) -> Value {
    json!({"id": id, "name": name, "name_p": 0.48, "alt": [["radio", 0.39], ["speaker", 0.05], ["lamp", 0.01]], "match_type": mt, "match": m,
           "p_query": 0.39, "p_registered": 0.48, "attrs": ["red", "metal"]})
}

#[test]
fn appearance_hit_asks_user_and_reads_compactly() {
    let core = json!({"step2": true, "hits": [hit(27, "fire extinguisher", "appearance", 0.39)]});
    let a = parse_search(&args(r#"{"query":"radio"}"#)).unwrap();
    let r = format_search(&core, &fake(), &a, &|id| (id == 33).then(|| "bookshelf".to_string()));
    let m = &r["matches"][0];
    assert_eq!(m["id"], "O27");
    assert_eq!(m["match_type"], "appearance");
    assert_eq!((m["p_query"].as_f64(), m["p_registered"].as_f64()), (Some(0.39), Some(0.48)));
    assert!(m.get("registered").is_none(), "registered only when it differs from the shown name");
    assert_eq!(m["alt"].as_array().unwrap().len(), 2, "alternatives below p 0.03 dropped");
    assert_eq!(m["attrs"], json!(["red", "metal", "small"]));
    assert_eq!((m["room"].as_str(), m["state"].as_str(), m["last_seen_ago_s"].as_f64()), (Some("living room"), Some("seen"), Some(60.0)));
    assert_eq!(m["landmark"]["id"], "O33");
    assert_eq!(m["landmark"]["name"], "bookshelf");
    assert_eq!(m["landmark"]["dz_m"].as_f64(), Some(0.12));
    assert!(m.get("relation").is_none() && m.get("on").is_none(), "no inter-object relation words (10-05)");
    assert!(r["ask_user"].as_str().unwrap().contains("O27 is registered as 'fire extinguisher' but looks like 'radio'"));
    assert_eq!(r["searched"], "name+appearance");
    assert!(r.to_string().len() < 700, "one hit is {} bytes", r.to_string().len());
}

#[test]
fn name_hit_no_question_filters_and_ties() {
    let core = json!({"step2": false, "hits": [hit(40, "radio", "name", 0.9), hit(27, "radio", "name", 0.85)]});
    let fk = fake();
    let a = parse_search(&args(r#"{"query":"radio"}"#)).unwrap();
    let r = format_search(&core, &fk, &a, &|_| None);
    assert_eq!(r["matches"].as_array().unwrap().len(), 2);
    assert!(r["ask_user"].as_str().unwrap().contains("score alike"), "two far apart candidates within 0.1 -> ask which: {r}");
    assert_eq!(r["matches"][1]["registered"], "fire extinguisher");
    let a = parse_search(&args(r#"{"query":"radio","room":"kitchen"}"#)).unwrap();
    let r = format_search(&core, &fk, &a, &|_| None);
    assert_eq!(r["matches"].as_array().unwrap().len(), 1);
    assert!(r.get("ask_user").is_none(), "one strong name hit -> no question: {r}");
    let a = parse_search(&args(r#"{"query":"radio","near":"O33"}"#)).unwrap();
    assert_eq!(format_search(&core, &fk, &a, &|_| None)["matches"][0]["id"], "O27");
    let a = parse_search(&args(r#"{"query":"radio","state":"gone"}"#)).unwrap();
    let r = format_search(&core, &fk, &a, &|_| None);
    assert!(r["matches"].as_array().unwrap().is_empty() && r["ask_user"].as_str().unwrap().contains("nothing in memory"));
}

// ---- 끝까지: 라벨 표 + 공용 색인(GPU 없이: 글 인코더·영상 인코더 끔) ----
extern "C" {
    fn sgc_labels_open_ex(dir: *const c_char, index_dir: *const c_char, img_sample: *const c_char, err: *mut c_char, n: usize) -> *mut c_void;
    fn sgc_labels_close(l: *mut c_void);
    fn sgc_labels_find_name(l: *const c_void, t: *const c_char) -> i32;
    fn sgc_labels_text_emb(l: *const c_void, row: i32, out: *mut f32) -> i32;
    fn sgc_f32_to_f16(i: *const f32, o: *mut u16, n: i32);
}

fn labels_dir() -> String {
    std::env::var("SGRT_LABELS").unwrap_or_else(|_| format!("{}/embed_work/labels/objects-v1", std::env::var("HOME").unwrap()))
}

fn write_views(path: &Path, vs: &[Vec<f32>]) {
    let mut bytes = vec![];
    for v in vs {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let u: Vec<f32> = v.iter().map(|x| x / n).collect();
        let mut h = vec![0u16; 768];
        unsafe { sgc_f32_to_f16(u.as_ptr(), h.as_mut_ptr(), 768) };
        bytes.extend(h.iter().flat_map(|x| x.to_le_bytes()));
    }
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn end_to_end_radio_registered_as_fire_extinguisher() {
    let ld = labels_dir();
    if !Path::new(&ld).join("manifest.json").exists() {
        eprintln!("skip: no label table at {ld}");
        return;
    }
    let mem = std::env::temp_dir().join(format!("search_objects_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&mem);
    std::fs::create_dir_all(mem.join("objects")).unwrap();
    let (cd, ci) = (CString::new(ld.clone()).unwrap(), CString::new(mem.join("idx").to_string_lossy().as_ref()).unwrap());
    let mut err = [0i8; 256];
    let l = unsafe { sgc_labels_open_ex(cd.as_ptr(), ci.as_ptr(), std::ptr::null(), err.as_mut_ptr(), 256) };
    assert!(!l.is_null());
    let tv = |n: &str| {
        let c = CString::new(n).unwrap();
        let mut v = vec![0f32; 768];
        unsafe { sgc_labels_text_emb(l, sgc_labels_find_name(l, c.as_ptr()), v.as_mut_ptr()) };
        v
    };
    let (radio, ext, shelf) = (tv("radio"), tv("fire extinguisher"), tv("shelf"));
    let mix: Vec<f32> = radio.iter().zip(&ext).map(|(a, b)| a + 0.35 * b).collect();
    write_views(&mem.join("objects/O27_views.f16"), &[mix]);
    write_views(&mem.join("objects/O33_emb.f16"), &[shelf]);
    unsafe { sgc_labels_close(l) };
    std::fs::write(mem.join("view.json"), json!({"stamp": 100.0, "pose": [0.0, 0.0, 0.0],
        "rooms": [{"id": 1, "name": "living room"}],
        "objects": [{"id": 27, "name": "fire extinguisher", "state": "seen", "pos": [3.1, 4.0, 1.12], "extent": [0.25, 0.1, 0.15], "last_seen": 40.0, "room": 1, "movable": true},
                    {"id": 33, "name": "shelf", "state": "seen", "pos": [3.0, 4.0, 0.5], "extent": [1.2, 0.4, 1.0], "last_seen": 90.0, "room": 1, "movable": false}]}).to_string()).unwrap();
    let p = sys::Paths { labels: Some(ld), no_text: true, no_image: true, ..Default::default() };
    let mut s = ObjectSearch::open(&mem, &p).unwrap();
    // ①② "라디오": 이름 없음 → 생김새로 O27, 물어봐야 함
    let r = s.run_tool(SEARCH, &args(r#"{"query":"라디오"}"#));
    eprintln!("{r}");
    assert_eq!(r["searched"], "name+appearance");
    assert_eq!(r["matches"][0]["id"], "O27");
    assert_eq!(r["matches"][0]["match_type"], "appearance");
    assert_eq!(r["matches"][0]["registered"], "fire extinguisher");
    assert_eq!(r["matches"][0]["landmark"]["id"], "O33");
    assert!(r["ask_user"].is_string());
    // ③ 확인 → 다음엔 이름으로, 묻지 않음
    let c = s.run_tool(CONFIRM, &args(r#"{"id":"O27","name":"radio","source":"user"}"#));
    eprintln!("{c}");
    assert_eq!(c["status"], "ok");
    assert!(c["p_after"].as_f64().unwrap() > 0.9);
    let r = s.run_tool(SEARCH, &args(r#"{"query":"radio"}"#));
    assert_eq!((r["matches"][0]["id"].as_str(), r["matches"][0]["match_type"].as_str(), r["searched"].as_str()), (Some("O27"), Some("name"), Some("name")));
    assert!(r.get("ask_user").is_none(), "{r}");
    let log = std::fs::read_to_string(mem.join("confirmations.jsonl")).unwrap();
    let line: Value = serde_json::from_str(log.lines().next().unwrap()).unwrap();
    assert_eq!((line["id"].as_u64(), line["source"].as_str(), line["query"].as_str()), (Some(27), Some("user"), Some("라디오")), "the query that led to the confirmation is logged");
    // 틀린 인자는 관찰값
    assert_eq!(s.run_tool(CONFIRM, &args(r#"{"id":"O99","name":"radio","source":"user"}"#))["status"], "error");
    assert_eq!(s.run_tool(SEARCH, &args(r#"{"query":"radio","room":"garage"}"#))["status"], "error");
    // 기억이 바뀌면 다시 읽음(새 물체 O50)
    std::thread::sleep(std::time::Duration::from_millis(20));
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(mem.join("view.json")).unwrap()).unwrap();
    v["objects"].as_array_mut().unwrap().push(json!({"id": 50, "name": "radio", "state": "seen", "pos": [1.0, 1.0, 0.5], "extent": [0.2, 0.1, 0.1], "last_seen": 99.0, "room": 1}));
    std::fs::write(mem.join("view.json"), v.to_string()).unwrap();
    let r = s.run_tool(SEARCH, &args(r#"{"query":"radio"}"#));
    assert!(r["matches"].as_array().unwrap().iter().any(|m| m["id"] == "O50"), "{r}");
    let _ = std::fs::remove_dir_all(&mem);
}

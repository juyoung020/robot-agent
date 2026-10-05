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
    let l = list_definition();
    assert_eq!(l["function"]["parameters"]["required"], json!(["place"]));
    assert!(s["function"]["parameters"]["properties"]["max_age_s"].is_object() && s["function"]["parameters"]["properties"]["seen_after_s"].is_object());
    // 9B 맥락 16k: 셋 합쳐 작게
    let n = s.to_string().len() + c.to_string().len() + l.to_string().len();
    assert!(n < 3600, "definitions are {n} bytes");
    assert_eq!(definitions().len(), 3);
}

#[test]
fn parse_tolerant_and_errors_fixable() {
    let a = parse_search(&Value::String(r#"{"query":" 라디오 ","k":"3","state":"Seen"}"#.into())).unwrap();
    assert_eq!((a.query.as_str(), a.k, a.state.as_deref()), ("라디오", 3, Some("seen")));
    assert_eq!(parse_search(&args(r#"{"query":"cup","k":99}"#)).unwrap().k, 10);
    assert!(parse_search(&args(r#"{"k":3}"#)).unwrap_err().contains("query is required"));
    assert!(parse_search(&args(r#"{"query":"cup","state":"lost"}"#)).unwrap_err().contains("seen, moved, held, gone"));
    let c = parse_confirm(&args(r#"{"id":"O27","name":"radio","source":"close look"}"#)).unwrap();
    assert_eq!(c, ConfirmArgs { id: 27, name: "radio".into(), source: "close_look".into() });
    assert!(parse_confirm(&args(r#"{"id":27,"name":"radio","source":"me"}"#)).unwrap_err().contains("user or close_look"));
    assert!(parse_confirm(&args(r#"{"name":"radio","source":"user"}"#)).unwrap_err().contains("id is required"));
    let a = parse_search(&args(r#"{"query":"cup","max_age_s":"30 s","seen_after_s":12.5}"#)).unwrap();
    assert_eq!((a.max_age_s, a.seen_after_s), (Some(30.0), Some(12.5)));
    assert!(parse_search(&args(r#"{"query":"cup","max_age_s":-1}"#)).unwrap_err().contains("max_age_s"));
    assert_eq!(parse_list(&args(r#"{"place":"o12"}"#)).unwrap(), Place::Object(12));
    assert_eq!(parse_list(&args(r#"{"place":"kitchen"}"#)).unwrap(), Place::Room("kitchen".into()));
    assert_eq!(parse_list(&args(r#"{"place":"R2"}"#)).unwrap(), Place::Room("R2".into()));
    assert!(parse_list(&args(r#"{}"#)).unwrap_err().contains("place is required"));
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
    ObjInfo { id, name: name.into(), pos, extent: ext, state: "seen".into(), last_seen: 40.0, room: Some(room), movable, structural: false, n_obs: 5, pos_sd: None }
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
    assert!(r.to_string().len() < 820, "one hit is {} bytes", r.to_string().len());
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
    let a = parse_search(&args(r#"{"query":"radio","state":"gone"}"#)).unwrap();
    let r = format_search(&core, &fk, &a, &|_| None);
    assert!(r["matches"].as_array().unwrap().is_empty() && r["ask_user"].as_str().unwrap().contains("nothing in memory"));
}

/// 로봇이 (1, 1) 에서 왼쪽(+y)을 봄
struct Turned(Fake);
impl Memory for Turned {
    fn objects(&self) -> &[ObjInfo] {
        self.0.objects()
    }
    fn now(&self) -> f64 {
        100.0
    }
    fn pose(&self) -> Option<[f64; 3]> {
        Some([1.0, 1.0, std::f64::consts::FRAC_PI_2])
    }
    fn room_name(&self, r: i64) -> Option<String> {
        self.0.room_name(r)
    }
    fn find_rooms(&self, s: &str) -> Vec<i64> {
        self.0.find_rooms(s)
    }
}

#[test]
fn coordinates_map_and_robot_relative() {
    let mut f = fake();
    f.0[1].pos_sd = Some([0.012, 0.018, 0.004]);
    let mem = Turned(f);
    let core = json!({"step2": false, "hits": [hit(27, "radio", "name", 0.9)]});
    let a = parse_search(&args(r#"{"query":"radio"}"#)).unwrap();
    let r = format_search(&core, &mem, &a, &|_| None);
    let m = &r["matches"][0];
    assert_eq!(m["pos"], json!([3.1, 4.0, 1.12]));
    assert_eq!(m["size"], json!([0.25, 0.1, 0.15]));
    assert_eq!(m["pos_sd"], json!([0.01, 0.02, 0.0]), "cm rounding");
    // 로봇 (1,1) 이 +y 를 봄: 물체 (3.1, 4.0) → 앞 3.0, 왼쪽 −2.1(오른쪽)
    assert_eq!((m["rel"]["x"].as_f64(), m["rel"]["y"].as_f64(), m["rel"]["z"].as_f64()), (Some(3.0), Some(-2.1), Some(1.12)));
    assert_eq!(m["rel"]["dist_m"].as_f64(), Some(3.66));
    assert_eq!(m["rel"]["bearing_deg"].as_f64(), Some(-35.0));
    assert!(m.get("dist_m").is_none(), "robot distance lives in rel now");
    assert_eq!(r["now_s"].as_f64(), Some(100.0));
}

#[test]
fn time_filters() {
    let mut f = fake();
    f.0[2].last_seen = 95.0; // O40 5 s 전, O27 60 s 전
    let core = json!({"step2": false, "hits": [hit(40, "radio", "name", 0.9), hit(27, "radio", "name", 0.85)]});
    let ids = |q: &str| -> Vec<String> {
        let a = parse_search(&args(q)).unwrap();
        format_search(&core, &f, &a, &|_| None)["matches"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap().to_string()).collect()
    };
    assert_eq!(ids(r#"{"query":"radio"}"#), ["O40", "O27"]);
    assert_eq!(ids(r#"{"query":"radio","max_age_s":10}"#), ["O40"]);
    assert_eq!(ids(r#"{"query":"radio","seen_after_s":50}"#), ["O40"]);
    assert_eq!(ids(r#"{"query":"radio","seen_after_s":40}"#), ["O40", "O27"], "inclusive");
    assert!(ids(r#"{"query":"radio","max_age_s":1}"#).is_empty());
}

#[test]
fn list_place_room_and_furniture() {
    let mut f = fake();
    f.0.push(obj(41, "mug", [8.3, 1.0, 0.9], [0.1, 0.1, 0.1], true, 2));
    f.0.push(ObjInfo { structural: true, ..obj(42, "wall", [8.0, 2.0, 1.0], [3.0, 0.1, 2.0], false, 2) });
    f.0.push(obj(43, "counter", [8.0, 1.2, 0.45], [2.0, 0.6, 0.9], false, 2));
    let info = |id: u32| (id == 40).then(|| json!({"name": "radio", "name_p": 0.81, "attrs": ["black"]}));
    let r = format_list(&Place::Room("kitchen".into()), &f, &info);
    let ids: Vec<&str> = r["objects"].as_array().unwrap().iter().map(|o| o["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["O40", "O41", "O43"], "movable first by robot distance, then fixed; structural left out: {r}");
    assert_eq!(r["place"], json!({"room": "kitchen", "id": "R2"}));
    let o40 = &r["objects"][0];
    assert_eq!((o40["name_p"].as_f64(), o40["attrs"].clone()), (Some(0.81), json!(["black", "small"])));
    assert!(o40["pos"].is_array() && o40["rel"]["dist_m"].is_number());
    assert_eq!(r["objects"][2]["fixed"], true);
    // 가구 id: 상자에서 1.5 m 안, dist_m·dz_m 숫자만
    let r = format_list(&Place::Object(33), &f, &|_| None);
    assert_eq!(r["place"]["id"], "O33");
    assert!(r["place"]["pos"].is_array());
    let o = &r["objects"][0];
    assert_eq!((o["id"].as_str(), o["dist_m"].as_f64(), o["dz_m"].as_f64()), (Some("O27"), Some(0.0), Some(0.12)));
    assert_eq!(r["n"], 1);
    for k in ["on", "in", "relation", "next_to", "near"] {
        assert!(!r.to_string().contains(&format!("\"{k}\"")), "no relation words: {k}");
    }
    assert_eq!(format_list(&Place::Room("garage".into()), &f, &|_| None)["status"], "error");
    assert_eq!(format_list(&Place::Object(999), &f, &|_| None)["status"], "error");
    // 15 개까지
    let mut big = fake();
    for i in 0..20 {
        big.0.push(obj(100 + i, "block", [8.0 + 0.01 * i as f64, 1.0, 0.1], [0.05; 3], true, 2));
    }
    let r = format_list(&Place::Room("R2".into()), &big, &|_| None);
    assert_eq!((r["objects"].as_array().unwrap().len(), r["n"].as_u64()), (PLACE_MAX, Some(21)));
    assert!(r["hint"].as_str().unwrap().contains("6 more"));
}

/// 가짜 scenemap: take 마다 다음 상태(자세·자리 바뀜), 이름 관측 기록
struct FakeSnap {
    k: u32,
    observed: std::sync::Arc<std::sync::Mutex<Vec<(u32, String, f32)>>>,
}
impl live::SnapApi for FakeSnap {
    fn take(&mut self) -> Result<live::SnapData, String> {
        self.k += 1;
        let x = 1.0 + 0.5 * self.k as f64;
        Ok(live::SnapData {
            objs: vec![ObjInfo { id: 7, name: "cup".into(), pos: [x, 0.0, 0.4], extent: [0.08; 3], state: "seen".into(), last_seen: self.k as f64, room: Some(3), movable: true, ..Default::default() }],
            pose: Some([0.0, 0.0, 0.0]),
            now: self.k as f64,
            rooms: vec![(3, "kitchen".into())],
        })
    }
    fn observe_name(&mut self, id: u32, label: &str, log_lr: f32) -> i32 {
        self.observed.lock().unwrap().push((id, label.into(), log_lr));
        if label == "cup" { 0 } else { -2 }
    }
}

#[test]
fn live_memory_is_fresh_every_call_and_reads_side_pos_sd() {
    let dir = std::env::temp_dir().join(format!("so_live_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("view.json"), json!({"objects": [{"id": 7, "pos_sd": [0.03, 0.02, 0.01]}]}).to_string()).unwrap();
    let obs = std::sync::Arc::new(std::sync::Mutex::new(vec![]));
    let mut m = live::LiveMem::new(Box::new(FakeSnap { k: 0, observed: obs.clone() }), Some(dir.join("view.json"))).unwrap();
    assert_eq!(m.objects()[0].pos[0], 1.5);
    assert_eq!(m.objects()[0].pos_sd, Some([0.03, 0.02, 0.01]));
    assert!(MemSource::refresh(&mut m).unwrap());
    assert_eq!((m.objects()[0].pos[0], m.now()), (2.0, 2.0), "new snapshot each refresh");
    assert_eq!(m.find_rooms("kitchen"), vec![3]);
    assert_eq!(m.find_rooms("R3"), vec![3]);
    assert_eq!(MemSource::observe_name(&mut m, 7, "cup", 3.9), Some(0));
    assert_eq!(obs.lock().unwrap()[0], (7, "cup".to_string(), 3.9));
    assert_eq!(MemSource::kind(&m), "live");
    let core = json!({"step2": false, "hits": [hit(7, "cup", "name", 0.9)]});
    let a = parse_search(&args(r#"{"query":"cup"}"#)).unwrap();
    let r = format_search(&core, &m, &a, &|_| None);
    assert_eq!(r["matches"][0]["pos"], json!([2.0, 0.0, 0.4]));
    assert_eq!(r["matches"][0]["room"], "kitchen");
    let _ = std::fs::remove_dir_all(&dir);
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
    std::env::var("SGRT_LABELS").unwrap_or_else(|_| format!("{}/../../../../models/labels/objects-v1", env!("CARGO_MANIFEST_DIR")))
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
                    {"id": 33, "name": "shelf", "state": "seen", "pos": [3.0, 4.0, 0.5], "extent": [1.2, 0.4, 1.0], "last_seen": 90.0, "room": 1, "movable": false, "pos_sd": [0.05, 0.05, 0.02]}]}).to_string()).unwrap();
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
    // list_place: 선반 둘레(관계말 없이 거리·높이 숫자)
    let l = s.run_tool(LIST, &args(r#"{"place":"O33"}"#));
    eprintln!("{l}");
    assert_eq!((l["objects"][0]["id"].as_str(), l["objects"][0]["name"].as_str()), (Some("O27"), Some("radio")));
    assert_eq!(l["place"]["pos_sd"], json!([0.05, 0.05, 0.02]));
    assert_eq!(s.run_tool(LIST, &args(r#"{"place":"living room"}"#))["n"], 2);
    // 오프라인 기억엔 지도가 없으므로 확인 결과에 map 칸 없음
    assert!(c.get("map").is_none());
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

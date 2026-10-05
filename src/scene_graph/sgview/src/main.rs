//! sgview — scene-graph viewer server (Rust). Replaces the Python/viser viewer.
//!
//! Reads the folder scenemap writes (`view.json`, `map.pgm` + `map.yaml`, `objects/O<id>_*`) and serves it to a three.js page.
//! No Python, no Spark-DSG, no crates besides std: everything the page needs comes from `view.json`, which already carries
//! objects, rooms, the compact graph, robot pose and grid info. Walls as 2D lines / numbers come from scenemap's C++ (`walls.cpp`).
//!
//!   sgview <memory_dir> [--port 8080] [--bind 0.0.0.0]
//!
//! GET /                      the page
//! GET /api/view?v=<ver>      {"v": ver, "map_v": ver, "view": <view.json>} or {"unchanged": true} (poll; ver = mtime+size)
//! GET /api/map[?since=<ver>] PGM pixels (row 0 = max y); with since only the changed rows (X-Y0/X-Y1/X-Full)
//! GET /api/walls?x=&y=&yaw=  {"segments": [[ax,ay,bx,by]..], "state": [56 floats], "pose": [x,y,yaw]}
//! GET /api/depth?path=<rel>  16-bit grey PNG → u16 little-endian pixels (X-W / X-H)
//! GET /file/<relative path>  any file below the memory dir (PNG crops, PLY points)
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::mpsc::{sync_channel, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

const INDEX_HTML: &str = include_str!("../assets/index.html");
const THREE_JS: &[u8] = include_bytes!("../assets/three.min.js");
const ORBIT_JS: &[u8] = include_bytes!("../assets/OrbitControls.js");
const GLTF_JS: &[u8] = include_bytes!("../assets/GLTFLoader.js");
include!(concat!(env!("OUT_DIR"), "/robot_assets.rs"));   // ROBOT_FILES: the URDF model (GLB meshes + robot.json)

const WALL_STATE_LEN: usize = 56;

/// Non-interlaced 8/16-bit grey or 8-bit RGB(A) PNG → (w, h, channels, bytes per sample, raw samples big-endian as stored).
fn decode_png(b: &[u8]) -> Option<(usize, usize, usize, usize, Vec<u8>)> {
    use std::io::Read;
    if b.len() < 33 || &b[..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let (mut w, mut h, mut depth, mut ctype, mut interlace) = (0usize, 0usize, 0u8, 0u8, 0u8);
    let mut idat = Vec::new();
    let mut i = 8;
    while i + 8 <= b.len() {
        let len = u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
        let ty = &b[i + 4..i + 8];
        let data = b.get(i + 8..i + 8 + len)?;
        match ty {
            b"IHDR" => {
                w = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
                h = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
                depth = data[8];
                ctype = data[9];
                interlace = data[12];
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        i += 12 + len;
    }
    let ch = match ctype {
        0 => 1,
        2 => 3,
        6 => 4,
        _ => return None,
    };
    if interlace != 0 || (depth != 8 && depth != 16) || w == 0 || h == 0 {
        return None;
    }
    let bps = (depth / 8) as usize;
    let bpp = ch * bps;
    let stride = w * bpp;
    let mut raw = Vec::with_capacity((stride + 1) * h);
    flate2::read::ZlibDecoder::new(&idat[..]).read_to_end(&mut raw).ok()?;
    if raw.len() < (stride + 1) * h {
        return None;
    }
    let mut out = vec![0u8; stride * h];
    for y in 0..h {
        let f = raw[y * (stride + 1)];
        let line = &raw[y * (stride + 1) + 1..(y + 1) * (stride + 1)];
        for x in 0..stride {
            let a = if x >= bpp { out[y * stride + x - bpp] as i32 } else { 0 };
            let up = if y > 0 { out[(y - 1) * stride + x] as i32 } else { 0 };
            let c = if x >= bpp && y > 0 { out[(y - 1) * stride + x - bpp] as i32 } else { 0 };
            let p = match f {
                0 => 0,
                1 => a,
                2 => up,
                3 => (a + up) / 2,
                4 => {
                    let pp = a + up - c;
                    let (pa, pb, pc) = ((pp - a).abs(), (pp - up).abs(), (pp - c).abs());
                    if pa <= pb && pa <= pc { a } else if pb <= pc { up } else { c }
                }
                _ => return None,
            };
            out[y * stride + x] = (line[x] as i32 + p) as u8;
        }
    }
    Some((w, h, ch, bps, out))
}

extern "C" {
    fn sgv_wall_segments(cells: *const i8, w: i32, h: i32, res: f64, ox: f64, oy: f64, ignore: *const f64, n_ignore: i32, out: *mut f64, cap: i32, angle_out: *mut f64) -> i32;
    fn sgv_wall_state(cells: *const i8, w: i32, h: i32, res: f64, ox: f64, oy: f64, segs: *const f64, n: i32, pose: *const f64, out: *mut f32);
}

/// Occupancy grid in scenemap layout (row 0 = min y; −1 unknown, 0..100 occupied %), plus its wall segments.
struct WallMap {
    ver: String,
    ig_key: String,
    w: i32,
    h: i32,
    res: f64,
    ox: f64,
    oy: f64,
    cells: Vec<i8>,
    segs: Vec<f64>, // ax ay bx by …
    theta: f64,     // wall direction [rad] (slam maps are in the start-pose frame, so walls can be tilted); ~0 when axis-aligned
}

/// Recent map.pgm versions (newest last): a client that has version V gets only the rows that changed since V.
struct MapHist {
    vers: Vec<(String, Arc<Pgm>)>,
}
const MAP_HIST: usize = 8;

struct State {
    dir: PathBuf,
    walls: Mutex<Option<Arc<WallMap>>>,
    view: Mutex<(String, Arc<String>)>,   // view.json text by version: several clients/polls read the file once
    maps: Mutex<MapHist>,
    live_mode: bool,                      // --ingest: the C++ side streams here, no files
    live: Mutex<Live>,
    clients: Mutex<Vec<SyncSender<Arc<String>>>>,
}

/// Live state fed by the C++ stream (scenemap/stream.hpp): the grid, the latest summary and pose, and the wall segments derived from them.
struct Live {
    w: i32,
    h: i32,
    res: f64,
    ox: f64,
    oy: f64,
    cells: Vec<i8>,          // scenemap layout (row 0 = min y)
    view: Option<Arc<String>>,
    pose: Option<[f64; 4]>,
    joints: Option<String>,  // last joints event (SSE text), resent to new pages
    ig: Vec<f64>,            // furniture footprints from the summary
    segs: Vec<f64>,
    theta: f64,              // wall direction of the grid [rad], with the segments
    seg_ver: u64,            // bumps when the segments actually change
    seg_sent: u64,           // version last broadcast
    segs_dirty: bool,
    last_walls: Instant,
}

fn file_ver(p: &Path) -> Option<String> {
    let m = fs::metadata(p).ok()?;
    let t = m.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_nanos();
    Some(format!("{}-{}", t, m.len()))
}

pub struct Pgm {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

fn read_pgm(p: &Path) -> Option<Pgm> {
    let b = fs::read(p).ok()?;
    let mut i = 0;
    let mut tok = Vec::new();
    while tok.len() < 4 {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < b.len() && b[i] == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while i < b.len() && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        if s == i {
            return None;
        }
        tok.push(String::from_utf8_lossy(&b[s..i]).to_string());
    }
    i += 1; // one whitespace after maxval
    if tok[0] != "P5" || tok[3] != "255" {
        return None;
    }
    let (w, h): (usize, usize) = (tok[1].parse().ok()?, tok[2].parse().ok()?);
    if b.len() < i + w * h {
        return None;
    }
    Some(Pgm { w, h, px: b[i..i + w * h].to_vec() })
}

/// resolution and origin (x, y) of a ROS map.yaml
fn read_yaml(p: &Path) -> (f64, f64, f64) {
    let (mut res, mut ox, mut oy) = (0.05, 0.0, 0.0);
    if let Ok(s) = fs::read_to_string(p) {
        for line in s.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if let Some(v) = line.strip_prefix("resolution:") {
                res = v.trim().parse().unwrap_or(res);
            } else if let Some(v) = line.strip_prefix("origin:") {
                let n: Vec<f64> = v.trim().trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if n.len() >= 2 {
                    ox = n[0];
                    oy = n[1];
                }
            }
        }
    }
    (res, ox, oy)
}

/// view.json text of the current version (cached; `None` while the file is missing or mid-write).
fn load_view(st: &State) -> Option<(String, Arc<String>)> {
    let vp = st.dir.join("view.json");
    let ver = file_ver(&vp)?;
    let mut c = st.view.lock().unwrap();
    if c.0 != ver {
        let t = fs::read_to_string(&vp).ok()?;
        if !(t.trim_start().starts_with('{') && t.trim_end().ends_with('}')) {
            return None;
        }
        *c = (ver, Arc::new(t));
    }
    Some((c.0.clone(), c.1.clone()))
}

/// Footprints (x0 y0 x1 y1) of free-standing objects from view.json: bottom below 0.4 m, not room-sized, not gone, +0.1 m.
/// Their occupied cells are furniture (sofa, table …), not walls. Wall-mounted things (frames, lamps) start higher and are left alone.
fn furniture_rects(st: &State) -> Vec<f64> {
    match load_view(st) {
        Some((_, t)) => furniture_rects_of(&t),
        None => Vec::new(),
    }
}

fn furniture_rects_of(text: &str) -> Vec<f64> {
    let v: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for o in v["objects"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        if o["state"] == "gone" {
            continue;
        }
        let (p, e) = (&o["pos"], &o["extent"]);
        let g = |a: &serde_json::Value, i: usize| a[i].as_f64().unwrap_or(0.0);
        let (hx, hy, hz) = (g(e, 0) / 2.0, g(e, 1) / 2.0, g(e, 2) / 2.0);
        if g(p, 2) - hz > 0.4 || hx * 2.0 > 5.0 || hy * 2.0 > 5.0 {
            continue;
        }
        out.extend_from_slice(&[g(p, 0) - hx - 0.1, g(p, 1) - hy - 0.1, g(p, 0) + hx + 0.1, g(p, 1) + hy + 0.1]);
    }
    out
}

/// Which robot made the run, for the trajectory label: `robot_footprint.json` next to the memory dir (run dir) or inside it.
/// An explicit `"robot"` field wins; older files only carry the simulator AABB (LIMO + OMX-F ≈ 0.35 × 0.36 × 0.38 m).
/// Unknown → null (the page says "robot").
fn robot_kind(st: &State) -> Option<String> {
    let mut cands = vec![st.dir.join("robot_footprint.json")];
    if let Some(p) = st.dir.parent() {
        cands.push(p.join("robot_footprint.json"));
    }
    for p in cands {
        let v: serde_json::Value = match std::fs::read_to_string(&p).ok().and_then(|t| serde_json::from_str(&t).ok()) {
            Some(v) => v,
            None => continue,
        };
        if let Some(r) = v["robot"].as_str() {
            return Some(r.to_string());
        }
        let e: Vec<f64> = v["robot_aabb_extent"].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default();
        if e.len() == 3 {
            let xy = e[0].max(e[1]);
            if xy < 0.5 && e[2] < 0.6 {
                return Some("limo_omx".into());
            }
        }
        return None;
    }
    None
}

/// The wall map of the current `map.pgm` (cached by mtime+size; segments are recomputed only when the map file changes).
fn wall_map(st: &State) -> Option<Arc<WallMap>> {
    let pgm_path = st.dir.join("map.pgm");
    let ver = file_ver(&pgm_path)?;
    let ig = furniture_rects(st);
    let ig_key = format!("{:?}", ig);
    let mut g = st.walls.lock().unwrap();
    if let Some(w) = g.as_ref() {
        if w.ver == ver && w.ig_key == ig_key {
            return Some(w.clone());
        }
    }
    let pgm = read_pgm(&pgm_path)?;
    let (res, ox, oy) = read_yaml(&st.dir.join("map.yaml"));
    let (w, h) = (pgm.w, pgm.h);
    // PGM: row 0 = max y, grey ≤ 90 occupied, 205 unknown, else free  →  scenemap layout (row 0 = min y)
    let mut cells = vec![0i8; w * h];
    for y in 0..h {
        let src = &pgm.px[(h - 1 - y) * w..(h - y) * w];
        let dst = &mut cells[y * w..(y + 1) * w];
        for x in 0..w {
            dst[x] = if src[x] <= 90 { 100 } else if src[x] == 205 { -1 } else { 0 };
        }
    }
    let mut buf = vec![0f64; 4 * 512];
    let mut theta = 0f64;
    let mut n = unsafe { sgv_wall_segments(cells.as_ptr(), w as i32, h as i32, res, ox, oy, ig.as_ptr(), (ig.len() / 4) as i32, buf.as_mut_ptr(), 512, &mut theta) } as usize;
    if n > 512 {
        buf = vec![0f64; 4 * n];
        n = unsafe { sgv_wall_segments(cells.as_ptr(), w as i32, h as i32, res, ox, oy, ig.as_ptr(), (ig.len() / 4) as i32, buf.as_mut_ptr(), n as i32, &mut theta) } as usize;
    }
    buf.truncate(4 * n);
    let wm = Arc::new(WallMap { ver, ig_key, w: w as i32, h: h as i32, res, ox, oy, cells, segs: buf, theta });
    *g = Some(wm.clone());
    Some(wm)
}

/// Current map pixels (version-cached; history kept for delta updates).
fn current_map(st: &State) -> Option<(String, Arc<Pgm>)> {
    let ver = file_ver(&st.dir.join("map.pgm"))?;
    let mut h = st.maps.lock().unwrap();
    if let Some((v, p)) = h.vers.last() {
        if *v == ver {
            return Some((v.clone(), p.clone()));
        }
    }
    let p = Arc::new(read_pgm(&st.dir.join("map.pgm"))?);
    h.vers.push((ver.clone(), p.clone()));
    if h.vers.len() > MAP_HIST {
        h.vers.remove(0);
    }
    Some((ver, p))
}

/// Rows [y0, y1] (PGM row order) that differ between `old` and `new`, None if the shapes differ or nothing changed.
fn changed_rows(old: &Pgm, new: &Pgm) -> Option<(usize, usize)> {
    if old.w != new.w || old.h != new.h {
        return None;
    }
    let w = new.w;
    let mut first = None;
    let mut last = 0;
    for y in 0..new.h {
        if old.px[y * w..(y + 1) * w] != new.px[y * w..(y + 1) * w] {
            if first.is_none() {
                first = Some(y);
            }
            last = y;
        }
    }
    first.map(|f| (f, last))
}

fn json_f64s(v: &[f64]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{:.4}", x));
    }
    s.push(']');
    s
}

fn walls_json(st: &State, q: &HashMap<String, String>) -> Option<String> {
    let wm = wall_map(st)?;
    let get = |k: &str| q.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    let pose = [get("x"), get("y"), get("yaw")];
    let mut out = [0f32; WALL_STATE_LEN];
    unsafe {
        sgv_wall_state(wm.cells.as_ptr(), wm.w, wm.h, wm.res, wm.ox, wm.oy, wm.segs.as_ptr(), (wm.segs.len() / 4) as i32, pose.as_ptr(), out.as_mut_ptr());
    }
    let segs: Vec<String> = wm.segs.chunks(4).map(|c| json_f64s(c)).collect();
    let state: Vec<String> = out.iter().map(|x| format!("{:.5}", x)).collect();
    Some(format!(
        "{{\"segments\":[{}],\"theta\":{:.6},\"state\":[{}],\"pose\":{},\"map_v\":\"{}\",\"n_sectors\":16,\"k_segments\":8,\"max_range\":4.0}}",
        segs.join(","),
        wm.theta,
        state.join(","),
        json_f64s(&pose),
        wm.ver
    ))
}

fn content_type(p: &str) -> &'static str {
    match p.rsplit('.').next().unwrap_or("") {
        "png" => "image/png",
        "json" => "application/json",
        "html" => "text/html; charset=utf-8",
        "js" => "application/javascript",
        "ply" | "pgm" => "application/octet-stream",
        _ => "application/octet-stream",
    }
}

fn respond(s: &mut TcpStream, code: u16, ctype: &str, extra: &str, body: &[u8]) {
    let reason = match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n{}\r\n",
        code,
        reason,
        ctype,
        body.len(),
        extra
    );
    let _ = s.write_all(head.as_bytes());
    let _ = s.write_all(body);
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() + 0 && i + 2 <= b.len() - 1 {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn parse_query(q: &str) -> HashMap<String, String> {
    q.split('&')
        .filter_map(|kv| {
            let mut it = kv.splitn(2, '=');
            Some((it.next()?.to_string(), percent_decode(it.next().unwrap_or(""))))
        })
        .collect()
}

/// `rel` below `root`, no `..`, no absolute parts
fn safe_join(root: &Path, rel: &str) -> Option<PathBuf> {
    let mut p = root.to_path_buf();
    for c in Path::new(rel).components() {
        match c {
            Component::Normal(x) => p.push(x),
            _ => return None,
        }
    }
    Some(p)
}

fn handle(mut s: TcpStream, st: Arc<State>) {
    let mut rd = BufReader::new(s.try_clone().unwrap());
    let mut line = String::new();
    if rd.read_line(&mut line).is_err() {
        return;
    }
    loop {
        let mut h = String::new();
        if rd.read_line(&mut h).unwrap_or(0) <= 2 {
            break;
        }
    }
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
    if method != "GET" {
        return respond(&mut s, 400, "text/plain", "", b"GET only");
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let q = parse_query(query);
    match path {
        "/" | "/index.html" => respond(&mut s, 200, "text/html; charset=utf-8", "", INDEX_HTML.as_bytes()),
        "/three.min.js" => respond(&mut s, 200, "application/javascript", "", THREE_JS),
        "/GLTFLoader.js" => respond(&mut s, 200, "application/javascript", "", GLTF_JS),
        _ if path.starts_with("/robot/") => match ROBOT_FILES.iter().find(|(n, _)| *n == &path["/robot/".len()..]) {
            Some((n, b)) => respond(&mut s, 200, if n.ends_with(".json") { "application/json" } else { "model/gltf-binary" }, "", b),
            None if &path["/robot/".len()..] == "robot.json" => respond(&mut s, 200, "application/json", "", b"{}"),   // no model in this build: the page keeps its box stand-in
            None => respond(&mut s, 404, "text/plain", "", b"no such model file"),
        },
        "/OrbitControls.js" => respond(&mut s, 200, "application/javascript", "", ORBIT_JS),
        "/favicon.ico" => respond(&mut s, 204, "image/x-icon", "", b""),
        "/api/mode" => respond(&mut s, 200, "application/json", "", format!("{{\"live\":{}}}", st.live_mode).as_bytes()),
        "/api/robot" => {
            let body = match robot_kind(&st) {
                Some(r) => format!("{{\"robot\":{}}}", serde_json::Value::String(r)),
                None => "{\"robot\":null}".to_string(),
            };
            respond(&mut s, 200, "application/json", "", body.as_bytes())
        }
        "/stream" => {
            let st2 = st.clone();
            return serve_stream(s, st2);
        }
        "/api/view" => {
            let vp = st.dir.join("view.json");
            let ver = file_ver(&vp).unwrap_or_default();
            if ver.is_empty() {
                return respond(&mut s, 200, "application/json", "", b"{\"waiting\":true}");
            }
            if q.get("v").map(|v| v == &ver).unwrap_or(false) {
                return respond(&mut s, 200, "application/json", "", b"{\"unchanged\":true}");
            }
            let text = match load_view(&st) {
                Some((_, t)) => t,
                None => return respond(&mut s, 200, "application/json", "", b"{\"waiting\":true}"), // mid-write: next poll
            };
            let map_v = file_ver(&st.dir.join("map.pgm")).unwrap_or_default();
            let body = format!("{{\"v\":\"{}\",\"map_v\":\"{}\",\"view\":{}}}", ver, map_v, text);
            respond(&mut s, 200, "application/json", "", body.as_bytes())
        }
        // full map, or with ?since=<map_v> only the rows changed since that version (X-Y0..X-Y1, PGM row order; X-Full: 0)
        "/api/map" => match current_map(&st) {
            Some((ver, cur)) => {
                let mut y0 = 0;
                let mut y1 = cur.h.saturating_sub(1);
                let mut full = 1;
                if let Some(since) = q.get("since") {
                    let old = st.maps.lock().unwrap().vers.iter().find(|(v, _)| v == since).map(|(_, p)| p.clone());
                    if let Some(old) = old {
                        match changed_rows(&old, &cur) {
                            Some((a, b)) => { y0 = a; y1 = b; full = 0; }
                            None if old.w == cur.w && old.h == cur.h => { full = 0; y0 = 1; y1 = 0; } // identical pixels
                            None => {}
                        }
                    }
                }
                let hdr = format!("X-W: {}\r\nX-H: {}\r\nX-Y0: {}\r\nX-Y1: {}\r\nX-Full: {}\r\nX-Ver: {}\r\n", cur.w, cur.h, y0, y1, full, ver);
                let body: &[u8] = if y0 > y1 { &[] } else { &cur.px[y0 * cur.w..(y1 + 1) * cur.w] };
                respond(&mut s, 200, "application/octet-stream", &hdr, body)
            }
            None => respond(&mut s, 404, "text/plain", "", b"no map"),
        },
        // 16-bit grey PNG (depth in mm) → little-endian u16 pixels; the page can't read 16-bit PNGs from a canvas
        "/api/depth" => {
            let rel = q.get("path").map(|s| s.as_str()).unwrap_or("");
            match safe_join(&st.dir, rel).and_then(|p| fs::read(p).ok()).and_then(|b| decode_png(&b)) {
                Some((w, h, 1, bps, px)) => {
                    let mut out = Vec::with_capacity(w * h * 2);
                    for i in 0..w * h {
                        let v = if bps == 2 { u16::from_be_bytes([px[2 * i], px[2 * i + 1]]) } else { px[i] as u16 };
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    respond(&mut s, 200, "application/octet-stream", &format!("X-W: {}\r\nX-H: {}\r\n", w, h), &out)
                }
                _ => respond(&mut s, 404, "text/plain", "", b"not a grey png"),
            }
        }
        "/api/walls" => match walls_json(&st, &q) {
            Some(j) => respond(&mut s, 200, "application/json", "", j.as_bytes()),
            None => respond(&mut s, 404, "text/plain", "", b"no map"),
        },
        _ if path.starts_with("/file/") => {
            let rel = &path["/file/".len()..];
            match safe_join(&st.dir, rel).and_then(|p| fs::read(p).ok()) {
                Some(b) => respond(&mut s, 200, content_type(rel), "", &b),
                None => respond(&mut s, 404, "text/plain", "", b"not found"),
            }
        }
        _ => respond(&mut s, 404, "text/plain", "", b"not found"),
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Live stream: C++ (sgrt) -> TCP ingest -> state + SSE broadcast -> browsers
// ---------------------------------------------------------------------------------------------------------------------
const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    out
}

/// scenemap cell (-1 unknown, 0..100 %) -> the grey the page draws (same mapping as map.pgm)
fn cell_grey(v: i8) -> u8 {
    let v = v as i32;
    if v < 0 { 205 } else if v >= 65 { 0 } else if v <= 25 { 254 } else { (254 - (v * 254) / 100) as u8 }
}

fn sse(event: &str, data: &str) -> String {
    format!("event: {}\ndata: {}\n\n", event, data)
}

fn broadcast(st: &State, msg: String) {
    let m = Arc::new(msg);
    st.clients.lock().unwrap().retain(|tx| tx.try_send(m.clone()).is_ok());   // a client that cannot keep up is dropped; it reconnects and gets the snapshot
}

fn map_event(l: &Live, x0: i32, y0: i32, x1: i32, y1: i32) -> String {
    let mut grey = Vec::with_capacity(((x1 - x0 + 1) * (y1 - y0 + 1)) as usize);
    for y in y0..=y1 {
        let row = &l.cells[(y * l.w) as usize..((y + 1) * l.w) as usize];
        grey.extend(row[x0 as usize..=x1 as usize].iter().map(|&v| cell_grey(v)));
    }
    sse("map", &format!("{{\"w\":{},\"h\":{},\"res\":{},\"ox\":{},\"oy\":{},\"x0\":{},\"y0\":{},\"x1\":{},\"y1\":{},\"b64\":\"{}\"}}", l.w, l.h, l.res, l.ox, l.oy, x0, y0, x1, y1, base64(&grey)))
}

fn pose_event(p: &[f64; 4]) -> String {
    sse("pose", &format!("{{\"t\":{:.3},\"x\":{:.4},\"y\":{:.4},\"yaw\":{:.4}}}", p[0], p[1], p[2], p[3]))
}

fn recompute_segments(l: &mut Live) {
    l.segs_dirty = false;
    if l.w <= 0 || l.h <= 0 {
        return;
    }
    let mut buf = vec![0f64; 4 * 512];
    let mut theta = 0f64;
    let mut n = unsafe { sgv_wall_segments(l.cells.as_ptr(), l.w, l.h, l.res, l.ox, l.oy, l.ig.as_ptr(), (l.ig.len() / 4) as i32, buf.as_mut_ptr(), 512, &mut theta) } as usize;
    if n > 512 {
        buf = vec![0f64; 4 * n];
        n = unsafe { sgv_wall_segments(l.cells.as_ptr(), l.w, l.h, l.res, l.ox, l.oy, l.ig.as_ptr(), (l.ig.len() / 4) as i32, buf.as_mut_ptr(), n as i32, &mut theta) } as usize;
    }
    buf.truncate(4 * n);
    if buf != l.segs || theta != l.theta {
        l.segs = buf;
        l.theta = theta;
        l.seg_ver += 1;
    }
}

fn walls_event(l: &Live, with_segments: bool) -> Option<String> {
    let p = l.pose?;
    if l.w <= 0 {
        return None;
    }
    let pose = [p[1], p[2], p[3]];
    let mut out = [0f32; WALL_STATE_LEN];
    unsafe {
        sgv_wall_state(l.cells.as_ptr(), l.w, l.h, l.res, l.ox, l.oy, l.segs.as_ptr(), (l.segs.len() / 4) as i32, pose.as_ptr(), out.as_mut_ptr());
    }
    let segs = if with_segments {
        let v: Vec<String> = l.segs.chunks(4).map(|c| json_f64s(c)).collect();
        format!("[{}],\"theta\":{:.6}", v.join(","), l.theta)
    } else {
        "null".to_string()   // unchanged: the page keeps the segments (and wall direction) it has
    };
    let state: Vec<String> = out.iter().map(|x| format!("{:.5}", x)).collect();
    Some(sse("walls", &format!("{{\"segments\":{},\"state\":[{}],\"pose\":{},\"max_range\":4.0}}", segs, state.join(","), json_f64s(&pose))))
}

/// Walls are recomputed at most every 100 ms (only when the map or the furniture changed); the state vector goes out with them.
fn maybe_walls(st: &State, l: &mut Live) {
    if l.last_walls.elapsed() < Duration::from_millis(100) {
        return;
    }
    l.last_walls = Instant::now();
    if l.segs_dirty {
        recompute_segments(l);
    }
    let with = l.seg_ver != l.seg_sent;
    if let Some(e) = walls_event(l, with) {
        l.seg_sent = l.seg_ver;
        broadcast(st, e);
    }
}

fn handle_frame(st: &State, ty: u8, pl: &[u8]) {
    let mut l = st.live.lock().unwrap();
    match ty {
        1 if pl.len() == 32 => {
            let f = |i: usize| f64::from_le_bytes(pl[i * 8..i * 8 + 8].try_into().unwrap());
            let p = [f(0), f(1), f(2), f(3)];
            l.pose = Some(p);
            broadcast(st, pose_event(&p));
            maybe_walls(st, &mut l);
        }
        2 if pl.len() >= 48 => {
            let i32at = |o: usize| i32::from_le_bytes(pl[o..o + 4].try_into().unwrap());
            let f64at = |o: usize| f64::from_le_bytes(pl[o..o + 8].try_into().unwrap());
            let (w, h, res, ox, oy) = (i32at(0), i32at(4), f64at(8), f64at(16), f64at(24));
            let (x0, y0, x1, y1) = (i32at(32), i32at(36), i32at(40), i32at(44));
            if w <= 0 || h <= 0 || x0 < 0 || y0 < 0 || x1 >= w || y1 >= h || x1 < x0 || y1 < y0 || (x1 - x0 + 1) as usize * (y1 - y0 + 1) as usize + 48 != pl.len() {
                return;
            }
            if w != l.w || h != l.h || res != l.res || ox != l.ox || oy != l.oy {
                l.w = w; l.h = h; l.res = res; l.ox = ox; l.oy = oy;
                l.cells = vec![-1i8; (w * h) as usize];
            }
            let rw = (x1 - x0 + 1) as usize;
            for y in y0..=y1 {
                let dst = (y * w + x0) as usize;
                let src = 48 + (y - y0) as usize * rw;
                for k in 0..rw {
                    l.cells[dst + k] = pl[src + k] as i8;
                }
            }
            l.segs_dirty = true;
            let e = map_event(&l, x0, y0, x1, y1);
            broadcast(st, e);
        }
        4 if pl.len() >= 12 => {
            let t = f64::from_le_bytes(pl[0..8].try_into().unwrap());
            let n = i32::from_le_bytes(pl[8..12].try_into().unwrap()) as usize;
            if n > 0 && pl.len() == 12 + n * 4 {
                let q: Vec<String> = (0..n).map(|i| format!("{:.4}", f32::from_le_bytes(pl[12 + i * 4..16 + i * 4].try_into().unwrap()))).collect();
                let e = sse("joints", &format!("{{\"t\":{:.3},\"q\":[{}]}}", t, q.join(",")));
                l.joints = Some(e.clone());
                broadcast(st, e);
            }
        }
        3 => {
            if let Ok(t) = std::str::from_utf8(pl) {
                let line = t.replace('\n', "");
                let ig = furniture_rects_of(&line);
                if ig != l.ig {
                    l.ig = ig;
                    l.segs_dirty = true;
                }
                let msg = Arc::new(line);
                l.view = Some(msg.clone());
                broadcast(st, sse("view", &msg));
            }
        }
        _ => {}
    }
}

fn ingest_conn(st: Arc<State>, mut s: TcpStream) {
    let _ = s.set_nodelay(true);
    {
        // a new simulator run (or a restart) starts from an empty state; the pages are told to clear theirs too
        let mut l = st.live.lock().unwrap();
        l.w = 0; l.h = 0; l.cells.clear(); l.view = None; l.pose = None; l.joints = None; l.ig.clear(); l.segs.clear(); l.theta = 0.0; l.seg_ver += 1; l.segs_dirty = false;
        broadcast(&st, sse("reset", "{}"));
    }
    let mut head = [0u8; 5];
    let mut pl = Vec::new();
    loop {
        if s.read_exact(&mut head).is_err() {
            return;
        }
        let len = u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as usize;
        if len > (256 << 20) {
            return;
        }
        pl.resize(len, 0);
        if s.read_exact(&mut pl).is_err() {
            return;
        }
        handle_frame(&st, head[4], &pl);
    }
}

fn ingest_loop(st: Arc<State>, addr: String) {
    let l = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("cannot listen for the stream on {}: {}", addr, e);
        std::process::exit(1);
    });
    eprintln!("sgview: stream ingest on {}", addr);
    for c in l.incoming().flatten() {
        let st = st.clone();
        std::thread::spawn(move || ingest_conn(st, c));
    }
}

/// GET /stream: server-sent events. The snapshot (map, summary, pose, walls) goes first, then every update as it arrives.
fn serve_stream(mut s: TcpStream, st: Arc<State>) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";
    if s.write_all(head.as_bytes()).is_err() {
        return;
    }
    let (tx, rx) = sync_channel::<Arc<String>>(1024);
    {
        // build the snapshot and register under the live lock so no update is missed or duplicated
        let l = st.live.lock().unwrap();
        let mut snap = String::new();
        if l.w > 0 {
            snap.push_str(&map_event(&l, 0, 0, l.w - 1, l.h - 1));
        }
        if let Some(v) = &l.view {
            snap.push_str(&sse("view", v));
        }
        if let Some(p) = &l.pose {
            snap.push_str(&pose_event(p));
        }
        if let Some(j) = &l.joints {
            snap.push_str(j);
        }
        if let Some(w) = walls_event(&l, true) {
            snap.push_str(&w);
        }
        if s.write_all(snap.as_bytes()).is_err() {
            return;
        }
        st.clients.lock().unwrap().push(tx);
    }
    let _ = s.set_nodelay(true);
    loop {
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(m) => {
                // several queued events go out in one write
                let mut buf = String::from(m.as_str());
                while let Ok(m2) = rx.try_recv() {
                    buf.push_str(&m2);
                }
                if s.write_all(buf.as_bytes()).is_err() {
                    return;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if s.write_all(b": keepalive\n\n").is_err() {
                    return;
                }
            }
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn main() {
    let mut dir: Option<String> = None;
    let (mut port, mut bind) = (8080u16, "0.0.0.0".to_string());
    let mut ingest: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => port = it.next().and_then(|v| v.parse().ok()).unwrap_or(port),
            "--bind" => bind = it.next().unwrap_or(bind),
            "--ingest" => ingest = it.next(),
            _ => dir = Some(a),
        }
    }
    let dir = match dir {
        Some(d) => PathBuf::from(d),
        None => {
            eprintln!("usage: sgview <memory_dir> [--port 8080] [--bind 0.0.0.0]");
            std::process::exit(2);
        }
    };
    let st = Arc::new(State { dir, walls: Mutex::new(None), view: Mutex::new((String::new(), Arc::new(String::new()))), maps: Mutex::new(MapHist { vers: Vec::new() }), live_mode: ingest.is_some(), live: Mutex::new(Live { w: 0, h: 0, res: 0.05, ox: 0.0, oy: 0.0, cells: Vec::new(), view: None, pose: None, joints: None, ig: Vec::new(), segs: Vec::new(), theta: 0.0, seg_ver: 0, seg_sent: 0, segs_dirty: false, last_walls: Instant::now() }), clients: Mutex::new(Vec::new()) });
    let l = TcpListener::bind((bind.as_str(), port)).unwrap_or_else(|e| {
        eprintln!("cannot listen on {}:{}: {}", bind, port, e);
        std::process::exit(1);
    });
    let _ = SystemTime::now();
    eprintln!("sgview: http://localhost:{}  (memory dir {})", port, st.dir.display());
    if let Some(a) = ingest {
        let st2 = st.clone();
        std::thread::spawn(move || ingest_loop(st2, a));
    }
    for c in l.incoming().flatten() {
        let _ = c.set_nodelay(true);
        let st = st.clone();
        std::thread::spawn(move || handle(c, st));
    }
}

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
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const INDEX_HTML: &str = include_str!("../assets/index.html");
const THREE_JS: &[u8] = include_bytes!("../assets/three.min.js");
const ORBIT_JS: &[u8] = include_bytes!("../assets/OrbitControls.js");

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
    fn sgv_wall_segments(cells: *const i8, w: i32, h: i32, res: f64, ox: f64, oy: f64, ignore: *const f64, n_ignore: i32, out: *mut f64, cap: i32) -> i32;
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
    let text = match load_view(st) {
        Some((_, t)) => t,
        None => return Vec::new(),
    };
    let v: serde_json::Value = match serde_json::from_str(&text) {
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
    let mut n = unsafe { sgv_wall_segments(cells.as_ptr(), w as i32, h as i32, res, ox, oy, ig.as_ptr(), (ig.len() / 4) as i32, buf.as_mut_ptr(), 512) } as usize;
    if n > 512 {
        buf = vec![0f64; 4 * n];
        n = unsafe { sgv_wall_segments(cells.as_ptr(), w as i32, h as i32, res, ox, oy, ig.as_ptr(), (ig.len() / 4) as i32, buf.as_mut_ptr(), n as i32) } as usize;
    }
    buf.truncate(4 * n);
    let wm = Arc::new(WallMap { ver, ig_key, w: w as i32, h: h as i32, res, ox, oy, cells, segs: buf });
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
        "{{\"segments\":[{}],\"state\":[{}],\"pose\":{},\"map_v\":\"{}\",\"n_sectors\":16,\"k_segments\":8,\"max_range\":4.0}}",
        segs.join(","),
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
        "/OrbitControls.js" => respond(&mut s, 200, "application/javascript", "", ORBIT_JS),
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

fn main() {
    let mut dir: Option<String> = None;
    let (mut port, mut bind) = (8080u16, "0.0.0.0".to_string());
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => port = it.next().and_then(|v| v.parse().ok()).unwrap_or(port),
            "--bind" => bind = it.next().unwrap_or(bind),
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
    let st = Arc::new(State { dir, walls: Mutex::new(None), view: Mutex::new((String::new(), Arc::new(String::new()))), maps: Mutex::new(MapHist { vers: Vec::new() }) });
    let l = TcpListener::bind((bind.as_str(), port)).unwrap_or_else(|e| {
        eprintln!("cannot listen on {}:{}: {}", bind, port, e);
        std::process::exit(1);
    });
    let _ = SystemTime::now();
    eprintln!("sgview: http://localhost:{}  (memory dir {})", port, st.dir.display());
    for c in l.incoming().flatten() {
        let _ = c.set_nodelay(true);
        let st = st.clone();
        std::thread::spawn(move || handle(c, st));
    }
}

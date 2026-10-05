// 요청 읽기·응답·gzip·안전한 경로 (sgview main.rs 의 respond / parse_query / percent_decode / safe_join 를 옮김)
use flate2::write::GzEncoder;
use flate2::Compression;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;

pub struct Req {
    pub path: String,
    pub q: HashMap<String, String>,
    pub gzip: bool,
    pub cookie: String,
    pub target: String,   // 요청 줄의 경로+질의 원문(sgview 로 넘길 때)
}

impl Req {
    pub fn get(&self, k: &str) -> &str {
        self.q.get(k).map(|s| s.as_str()).unwrap_or("")
    }
    pub fn num(&self, k: &str) -> Option<f64> {
        self.q.get(k).and_then(|s| s.parse().ok())
    }
}

/// 요청 줄과 머리를 읽는다. GET 만.
pub fn read_request(s: &TcpStream) -> Option<Req> {
    let mut rd = BufReader::new(s.try_clone().ok()?);
    let mut line = String::new();
    rd.read_line(&mut line).ok()?;
    let mut gzip = false;
    let mut cookie = String::new();
    loop {
        let mut h = String::new();
        if rd.read_line(&mut h).unwrap_or(0) <= 2 {
            break;
        }
        let hl = h.to_ascii_lowercase();
        if hl.starts_with("accept-encoding:") && hl.contains("gzip") {
            gzip = true;
        }
        if hl.starts_with("cookie:") {
            cookie = h[7..].trim().to_string();
        }
    }
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next()?, parts.next()?);
    if method != "GET" && method != "HEAD" {
        return None;
    }
    let (path, query) = match target.find('?') {
        Some(i) => (&target[..i], &target[i + 1..]),
        None => (target, ""),
    };
    Some(Req { path: percent_decode(path), q: parse_query(query), gzip, cookie, target: target.to_string() })
}

pub fn respond(s: &mut TcpStream, code: u16, ctype: &str, body: &[u8], gzip_ok: bool) {
    let reason = match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    // 64 KB 넘고 브라우저가 받으면 gzip(빠른 수준 1) — 5.5
    let gz;
    let (body, enc) = if gzip_ok && body.len() > 64 * 1024 {
        let mut e = GzEncoder::new(Vec::with_capacity(body.len() / 4), Compression::new(1));
        let _ = e.write_all(body);
        gz = e.finish().unwrap_or_default();
        (&gz[..], "Content-Encoding: gzip\r\n")
    } else {
        (body, "")
    };
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n{}\r\n",
        code,
        reason,
        ctype,
        body.len(),
        enc
    );
    let _ = s.write_all(head.as_bytes());
    let _ = s.write_all(body);
}

pub fn json(s: &mut TcpStream, body: &str, gzip_ok: bool) {
    respond(s, 200, "application/json; charset=utf-8", body.as_bytes(), gzip_ok)
}

pub fn not_found(s: &mut TcpStream, why: &str) {
    let b = serde_json::json!({"error": "not found", "why": why}).to_string();
    respond(s, 404, "application/json; charset=utf-8", b.as_bytes(), false)
}

pub fn bad(s: &mut TcpStream, why: &str) {
    let b = serde_json::json!({"error": "bad request", "why": why}).to_string();
    respond(s, 400, "application/json; charset=utf-8", b.as_bytes(), false)
}

pub fn content_type(p: &str) -> &'static str {
    match p.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "application/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "glb" => "model/gltf-binary",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        _ => "application/octet-stream",
    }
}

pub fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
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

pub fn parse_query(q: &str) -> HashMap<String, String> {
    q.split('&')
        .filter(|kv| !kv.is_empty())
        .filter_map(|kv| {
            let mut it = kv.splitn(2, '=');
            Some((percent_decode(it.next()?), percent_decode(it.next().unwrap_or(""))))
        })
        .collect()
}

/// JSON 숫자(NaN·무한 → null). f32 는 가장 짧은 표기.
pub fn num32(out: &mut String, v: f32) {
    use std::fmt::Write;
    if v.is_finite() {
        let _ = write!(out, "{}", v);
    } else {
        out.push_str("null");
    }
}
pub fn num64(out: &mut String, v: f64) {
    use std::fmt::Write;
    if v.is_finite() {
        let _ = write!(out, "{}", v);
    } else {
        out.push_str("null");
    }
}
pub fn jstr(out: &mut String, s: &str) {
    out.push_str(&serde_json::to_string(s).unwrap());
}

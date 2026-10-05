//! 손으로 짠 HTTP/1.1 (로컬 서버 전용): LLM·그래프 서비스 클라이언트, 가짜 LLM 서버.
//! 요청마다 새 연결(Connection: close) — 경계에서만 쓰므로 연결 재사용 이득이 없다.
//! 응답 본문은 Content-Length 또는 chunked 를 읽는다.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

#[derive(Debug)]
pub struct Resp {
    pub status: u16,
    pub body: Vec<u8>,
}

impl Resp {
    pub fn json(&self) -> Result<serde_json::Value, String> {
        serde_json::from_slice(&self.body).map_err(|e| format!("JSON 해석 실패({e}): {}", String::from_utf8_lossy(&self.body[..self.body.len().min(200)])))
    }
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// `http://host:port/path` → (host:port, path)
pub fn split_url(url: &str) -> Result<(String, String), String> {
    let rest = url.strip_prefix("http://").ok_or_else(|| format!("http:// 만 지원: {url}"))?;
    let (hp, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let hp = if hp.contains(':') { hp.to_string() } else { format!("{hp}:80") };
    Ok((hp, path.to_string()))
}

pub fn request(method: &str, url: &str, body: Option<&[u8]>, headers: &[(&str, &str)], timeout: Duration) -> Result<Resp, String> {
    let (hp, path) = split_url(url)?;
    let addr = hp.to_socket_addrs().map_err(|e| format!("{hp}: {e}"))?.next().ok_or_else(|| format!("{hp}: 주소 없음"))?;
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(5).min(timeout)).map_err(|e| format!("{hp} 접속 실패: {e}"))?;
    let _ = s.set_nodelay(true);
    s.set_read_timeout(Some(timeout)).ok();
    s.set_write_timeout(Some(timeout)).ok();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {hp}\r\nConnection: close\r\nAccept: application/json\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(b) = body {
        req.push_str(&format!("Content-Type: application/json\r\nContent-Length: {}\r\n", b.len()));
    }
    req.push_str("\r\n");
    let mut out = req.into_bytes();
    if let Some(b) = body {
        out.extend_from_slice(b);
    }
    s.write_all(&out).map_err(|e| format!("보내기 실패: {e}"))?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).map_err(|e| format!("받기 실패(시간 초과 {timeout:?}?): {e}"))?;
    parse_response(&raw)
}

pub fn parse_response(raw: &[u8]) -> Result<Resp, String> {
    let p = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or("HTTP 머리 끝 없음")?;
    let head = String::from_utf8_lossy(&raw[..p]).into_owned();
    let status: u16 = head.split_whitespace().nth(1).and_then(|s| s.parse().ok()).ok_or("상태 코드 없음")?;
    let body = &raw[p + 4..];
    let hv = |name: &str| -> Option<String> {
        head.lines().skip(1).find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim().eq_ignore_ascii_case(name).then(|| v.trim().to_string())
        })
    };
    let body = if hv("transfer-encoding").map(|v| v.eq_ignore_ascii_case("chunked")).unwrap_or(false) {
        dechunk(body)?
    } else if let Some(n) = hv("content-length").and_then(|v| v.parse::<usize>().ok()) {
        body[..n.min(body.len())].to_vec()
    } else {
        body.to_vec()
    };
    Ok(Resp { status, body })
}

fn dechunk(mut b: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let e = b.windows(2).position(|w| w == b"\r\n").ok_or("chunk 크기 줄 없음")?;
        let size_s = String::from_utf8_lossy(&b[..e]);
        let size = usize::from_str_radix(size_s.split(';').next().unwrap_or("").trim(), 16).map_err(|_| "chunk 크기 해석 실패")?;
        b = &b[e + 2..];
        if size == 0 {
            return Ok(out);
        }
        if b.len() < size {
            return Err("chunk 가 잘림".into());
        }
        out.extend_from_slice(&b[..size]);
        b = &b[(size + 2).min(b.len())..];
    }
}

pub fn post_json(url: &str, v: &serde_json::Value, headers: &[(&str, &str)], timeout: Duration) -> Result<Resp, String> {
    let body = serde_json::to_vec(v).map_err(|e| e.to_string())?;
    request("POST", url, Some(&body), headers, timeout)
}

pub fn get(url: &str, timeout: Duration) -> Result<Resp, String> {
    request("GET", url, None, &[], timeout)
}

// ---------- 서버 쪽(가짜 LLM·그래프 서비스용) ----------

pub struct Req {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

/// 요청 하나 읽기(Content-Length 본문).
pub fn read_request(s: &mut TcpStream) -> Result<Req, String> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let head_end = loop {
        let n = s.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("연결 끊김".into());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if buf.len() > 1 << 20 {
            return Err("머리가 너무 김".into());
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut it = head.split_whitespace();
    let method = it.next().unwrap_or("").to_string();
    let path = it.next().unwrap_or("/").to_string();
    let len: usize = head
        .lines()
        .skip(1)
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim().eq_ignore_ascii_case("content-length").then(|| v.trim().parse().ok())?
        })
        .unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = s.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(len);
    Ok(Req { method, path, body })
}

pub fn write_response(s: &mut TcpStream, status: u16, body: &[u8], ctype: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    let head = format!("HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
    s.write_all(head.as_bytes())?;
    s.write_all(body)?;
    s.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunked_and_length() {
        let r = parse_response(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n").unwrap();
        assert_eq!(r.body, b"Wikipedia");
        let r = parse_response(b"HTTP/1.1 404 Not Found\r\nContent-Length: 3\r\n\r\nabcdef").unwrap();
        assert_eq!((r.status, r.body.as_slice()), (404, &b"abc"[..]));
        assert_eq!(split_url("http://127.0.0.1:8081/v1").unwrap(), ("127.0.0.1:8081".into(), "/v1".into()));
    }
}

//! 웹소켓(RFC 6455) 최소 구현 — 중계기용. 외부 웹소켓 크레이트를 쓰지 않는 이유:
//! - 받은 프레임의 마스크를 풀지 않고 그대로 다시 보내려면(zero-copy) 프레임 머리·마스크 키를 직접 다뤄야 한다.
//! - 큰 페이로드를 내부 버퍼 거치지 않고 재사용 버퍼로 바로 읽고, writev(여러 조각 한 번에)로 보낸다.
//!
//! 지원: 서버 핸드셰이크(+ `/healthz` 200 응답 — 평가기 클라이언트가 먼저 부른다), 클라이언트 핸드셰이크,
//! 프레임 읽기/쓰기, ping→pong, close. 확장(permessage-deflate)은 받지 않는다(평가기는 compression=None).

use std::io::{self, IoSlice, Read, Write};
use std::net::TcpStream;

pub const OP_CONT: u8 = 0x0;
pub const OP_TEXT: u8 = 0x1;
pub const OP_BIN: u8 = 0x2;
pub const OP_CLOSE: u8 = 0x8;
pub const OP_PING: u8 = 0x9;
pub const OP_PONG: u8 = 0xA;

const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Head {
    pub fin: bool,
    pub op: u8,
    pub mask: Option<[u8; 4]>,
    pub len: usize,
}

/// 재사용 버퍼: 커지기만 하고 줄지 않는다(매 프레임 memset 방지).
#[derive(Default)]
pub struct Buf {
    v: Vec<u8>,
    n: usize,
}

impl Buf {
    pub fn data(&self) -> &[u8] {
        &self.v[..self.n]
    }
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.v[..self.n]
    }
    pub fn ensure(&mut self, n: usize) -> &mut [u8] {
        if self.v.len() < n {
            self.v.resize(n, 0);
        }
        self.n = n;
        &mut self.v[..n]
    }
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

pub struct Conn {
    pub s: TcpStream,
    rb: Vec<u8>,
    rp: usize,
    re: usize,
    /// 클라이언트 쪽(우리가 접속한 쪽)이면 보낼 때 마스크를 건다.
    pub is_client: bool,
    rng: crate::util::Rng,
    quick: bool,
}

/// 소켓 read + (켜져 있으면) TCP_QUICKACK 다시 걸기
#[inline]
fn read_q(s: &mut TcpStream, quick: bool, dst: &mut [u8]) -> io::Result<usize> {
    let n = s.read(dst)?;
    if quick {
        quickack(s);
    }
    Ok(n)
}

fn io_err(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s.into())
}

pub fn accept_key(key: &str) -> String {
    let mut v = key.trim().as_bytes().to_vec();
    v.extend_from_slice(GUID.as_bytes());
    crate::codec::b64_encode(&crate::codec::sha1(&v))
}

/// HTTP 머리(빈 줄까지)를 읽는다. 머리 뒤에 이미 들어온 바이트는 돌려준다.
fn read_http_head(s: &mut TcpStream) -> io::Result<(String, Vec<u8>)> {
    let mut buf = Vec::with_capacity(1024);
    let mut tmp = [0u8; 4096];
    loop {
        let n = s.read(&mut tmp)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "HTTP 머리 전에 연결 끊김"));
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..p]).into_owned();
            return Ok((head, buf[p + 4..].to_vec()));
        }
        if buf.len() > 64 * 1024 {
            return Err(io_err("HTTP 머리가 너무 김"));
        }
    }
}

fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().skip(1).find_map(|l| {
        let (k, v) = l.split_once(':')?;
        if k.trim().eq_ignore_ascii_case(name) {
            Some(v.trim())
        } else {
            None
        }
    })
}

pub enum Accepted {
    Ws(Conn),
    /// `/healthz` 등 일반 HTTP 요청에 응답하고 닫았다.
    Http(String),
}

pub fn set_fast(s: &TcpStream) {
    let _ = s.set_nodelay(true);
}

/// 지연 ACK 끄기(리눅스 TCP_QUICKACK). 커널이 한동안 뒤 다시 켜므로 읽을 때마다 다시 건다.
/// Windows 평가기 ↔ WSL 경계에서 보이는 약 40 ms 대기(docs/평가기_가속설계.md 4.3, 지연 ACK 추정)를 줄이려는 것.
/// `BAGENT_NO_QUICKACK=1` 이면 끈다(비교용).
#[inline]
pub fn quickack(s: &TcpStream) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let one: libc::c_int = 1;
        unsafe {
            libc::setsockopt(s.as_raw_fd(), libc::IPPROTO_TCP, libc::TCP_QUICKACK, &one as *const libc::c_int as *const libc::c_void, std::mem::size_of::<libc::c_int>() as libc::socklen_t);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = s;
}

pub fn quickack_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BAGENT_NO_QUICKACK").map(|v| v != "1").unwrap_or(true))
}

impl Conn {
    fn new(s: TcpStream, leftover: Vec<u8>, is_client: bool) -> Conn {
        set_fast(&s);
        let re = leftover.len();
        let mut rb = leftover;
        if rb.len() < 64 * 1024 {
            rb.resize(64 * 1024, 0);
        }
        let quick = quickack_enabled();
        if quick {
            quickack(&s);
        }
        Conn { s, rb, rp: 0, re, is_client, rng: crate::util::Rng::from_time(), quick }
    }

    /// 서버 쪽: 접속을 받아 핸드셰이크. `/healthz` 는 200 OK 로 응답하고 닫는다.
    pub fn accept(s: TcpStream) -> io::Result<Accepted> {
        Conn::accept_with(s, || true)
    }

    /// `/healthz` 응답을 `healthy()` 결과로 정한다(중계기: 뒤쪽 VLA 서버가 살아 있을 때만 200).
    pub fn accept_with(mut s: TcpStream, healthy: impl Fn() -> bool) -> io::Result<Accepted> {
        set_fast(&s);
        let (head, leftover) = read_http_head(&mut s)?;
        let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
        let upgrade = header(&head, "upgrade").map(|v| v.eq_ignore_ascii_case("websocket")).unwrap_or(false);
        if !upgrade {
            let (code, body) = if path == "/healthz" {
                if healthy() {
                    ("200 OK", "OK\n")
                } else {
                    ("503 Service Unavailable", "upstream down\n")
                }
            } else {
                ("404 Not Found", "not found\n")
            };
            let resp = format!(
                "HTTP/1.1 {code}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            s.write_all(resp.as_bytes())?;
            let _ = s.flush();
            return Ok(Accepted::Http(path));
        }
        let key = header(&head, "sec-websocket-key").ok_or_else(|| io_err("Sec-WebSocket-Key 없음"))?;
        let resp = format!(
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
            accept_key(key)
        );
        s.write_all(resp.as_bytes())?;
        Ok(Accepted::Ws(Conn::new(s, leftover, false)))
    }

    /// 클라이언트 쪽: `host:port` 로 접속해 핸드셰이크.
    pub fn connect(addr: &str) -> io::Result<Conn> {
        let mut s = TcpStream::connect(addr)?;
        set_fast(&s);
        let mut rng = crate::util::Rng::from_time();
        let mut k = [0u8; 16];
        for c in k.chunks_mut(4) {
            c.copy_from_slice(&rng.bytes4());
        }
        let key = crate::codec::b64_encode(&k);
        let req = format!(
            "GET / HTTP/1.1\r\nHost: {addr}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nUser-Agent: bagent-relay\r\n\r\n"
        );
        s.write_all(req.as_bytes())?;
        let (head, leftover) = read_http_head(&mut s)?;
        if !head.starts_with("HTTP/1.1 101") {
            return Err(io_err(format!("업그레이드 실패: {}", head.lines().next().unwrap_or(""))));
        }
        let acc = header(&head, "sec-websocket-accept").unwrap_or("");
        if acc != accept_key(&key) {
            return Err(io_err("Sec-WebSocket-Accept 불일치"));
        }
        Ok(Conn::new(s, leftover, true))
    }

    pub fn buffered(&self) -> usize {
        self.re - self.rp
    }

    /// 버퍼에 남은 것부터, 그다음은 소켓에서 dst 로 바로 읽는다(큰 페이로드는 중간 버퍼를 거치지 않음).
    pub fn read_exact(&mut self, dst: &mut [u8]) -> io::Result<()> {
        let have = self.buffered().min(dst.len());
        dst[..have].copy_from_slice(&self.rb[self.rp..self.rp + have]);
        self.rp += have;
        if have == dst.len() {
            return Ok(());
        }
        let rest = &mut dst[have..];
        if rest.len() >= self.rb.len() / 2 {
            // 큰 읽기: 버퍼를 거치지 않고 바로
            let mut off = 0;
            while off < rest.len() {
                match read_q(&mut self.s, self.quick, &mut rest[off..]) {
                    Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "연결 끊김")),
                    Ok(n) => off += n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
            return Ok(());
        }
        // 작은 읽기: 버퍼를 채워서 시스템 호출 수를 줄인다.
        let mut filled = 0;
        while filled < rest.len() {
            self.rp = 0;
            self.re = 0;
            let n = read_q(&mut self.s, self.quick, &mut self.rb)?;
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "연결 끊김"));
            }
            self.re = n;
            let take = (rest.len() - filled).min(n);
            rest[filled..filled + take].copy_from_slice(&self.rb[..take]);
            self.rp = take;
            filled += take;
        }
        Ok(())
    }

    /// 있는 만큼 읽는다(버퍼에 남은 것 먼저, 없으면 소켓 read 한 번). 반환: 읽은 바이트 수(> 0).
    /// 중계기의 흘려보내기(cut-through)에 쓴다: 받은 조각을 바로 다음 쪽으로 쓴다.
    pub fn read_some(&mut self, dst: &mut [u8]) -> io::Result<usize> {
        let have = self.buffered().min(dst.len());
        if have > 0 {
            dst[..have].copy_from_slice(&self.rb[self.rp..self.rp + have]);
            self.rp += have;
            return Ok(have);
        }
        loop {
            match read_q(&mut self.s, self.quick, dst) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "연결 끊김")),
                Ok(n) => return Ok(n),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
    }

    /// 프레임 머리만 쓴다(몸통은 [`Conn::write_raw`] 로 이어서).
    pub fn write_head(&mut self, fin: bool, op: u8, mask: Option<[u8; 4]>, len: usize) -> io::Result<()> {
        let (h, hl) = encode_head(fin, op, mask, len);
        self.s.write_all(&h[..hl])
    }

    pub fn write_raw(&mut self, parts: &[&[u8]]) -> io::Result<()> {
        let mut v: Vec<IoSlice<'_>> = parts.iter().filter(|p| !p.is_empty()).map(|p| IoSlice::new(p)).collect();
        write_all_vectored(&mut self.s, &mut v)
    }

    pub fn read_head(&mut self) -> io::Result<Head> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        let fin = b[0] & 0x80 != 0;
        if b[0] & 0x70 != 0 {
            return Err(io_err("RSV 비트(확장) 는 지원 안 함"));
        }
        let op = b[0] & 0x0f;
        let masked = b[1] & 0x80 != 0;
        let mut len = (b[1] & 0x7f) as u64;
        if len == 126 {
            let mut e = [0u8; 2];
            self.read_exact(&mut e)?;
            len = u16::from_be_bytes(e) as u64;
        } else if len == 127 {
            let mut e = [0u8; 8];
            self.read_exact(&mut e)?;
            len = u64::from_be_bytes(e);
        }
        if len > (1u64 << 34) {
            return Err(io_err("프레임이 너무 큼"));
        }
        let mask = if masked {
            let mut k = [0u8; 4];
            self.read_exact(&mut k)?;
            Some(k)
        } else {
            None
        };
        Ok(Head { fin, op, mask, len: len as usize })
    }

    /// 프레임 하나를 buf 로 읽는다(마스크는 풀지 않는다).
    pub fn read_frame(&mut self, buf: &mut Buf) -> io::Result<Head> {
        let h = self.read_head()?;
        let dst = buf.ensure(h.len);
        self.read_exact(dst)?;
        Ok(h)
    }

    /// 조각들을 이어 프레임 하나로 보낸다. `mask` 가 있으면 조각들은 **이미 그 키로 마스크돼 있어야** 한다.
    pub fn write_frame_raw(&mut self, fin: bool, op: u8, mask: Option<[u8; 4]>, parts: &[&[u8]]) -> io::Result<()> {
        let len: usize = parts.iter().map(|p| p.len()).sum();
        let (h, hl) = encode_head(fin, op, mask, len);
        let mut v: Vec<IoSlice<'_>> = Vec::with_capacity(parts.len() + 1);
        v.push(IoSlice::new(&h[..hl]));
        for p in parts {
            if !p.is_empty() {
                v.push(IoSlice::new(p));
            }
        }
        write_all_vectored(&mut self.s, &mut v)
    }

    /// 평문 조각들을 보낸다. 클라이언트 쪽이면 새 키로 마스크한 사본을 만든다(작은 제어 프레임·시험용).
    pub fn send(&mut self, op: u8, parts: &[&[u8]]) -> io::Result<()> {
        if self.is_client {
            let key = self.rng.bytes4();
            let mut all: Vec<u8> = parts.concat();
            mask_at(&mut all, key, 0);
            self.write_frame_raw(true, op, Some(key), &[&all])
        } else {
            self.write_frame_raw(true, op, None, parts)
        }
    }

    pub fn new_mask_key(&mut self) -> [u8; 4] {
        self.rng.bytes4()
    }

    pub fn send_close(&mut self, code: u16) -> io::Result<()> {
        self.send(OP_CLOSE, &[&code.to_be_bytes()])
    }

    /// 데이터 프레임이 올 때까지 읽는다. ping 은 pong 으로 답하고, 조각난 메시지는 이어 붙여 평문으로 돌려준다.
    /// 반환: (op, 마스크 키). 조각 모음이면 키는 None 이고 buf 는 평문이다.
    pub fn read_message(&mut self, buf: &mut Buf) -> io::Result<(u8, Option<[u8; 4]>)> {
        loop {
            let h = self.read_frame(buf)?;
            match h.op {
                OP_PING => {
                    let mut p = buf.data().to_vec();
                    if let Some(k) = h.mask {
                        mask_at(&mut p, k, 0);
                    }
                    self.send(OP_PONG, &[&p])?;
                }
                OP_PONG => {}
                OP_CLOSE => return Ok((OP_CLOSE, h.mask)),
                OP_TEXT | OP_BIN if h.fin => return Ok((h.op, h.mask)),
                OP_TEXT | OP_BIN => {
                    let op = h.op;
                    let mut all = buf.data().to_vec();
                    if let Some(k) = h.mask {
                        mask_at(&mut all, k, 0);
                    }
                    loop {
                        let h2 = self.read_frame(buf)?;
                        match h2.op {
                            OP_PING => {
                                let mut p = buf.data().to_vec();
                                if let Some(k) = h2.mask {
                                    mask_at(&mut p, k, 0);
                                }
                                self.send(OP_PONG, &[&p])?;
                            }
                            OP_PONG => {}
                            OP_CONT => {
                                let start = all.len();
                                all.extend_from_slice(buf.data());
                                if let Some(k) = h2.mask {
                                    mask_at(&mut all[start..], k, 0);
                                }
                                if h2.fin {
                                    break;
                                }
                            }
                            OP_CLOSE => return Ok((OP_CLOSE, None)),
                            o => return Err(io_err(format!("조각 사이에 이상한 op {o}"))),
                        }
                    }
                    let dst = buf.ensure(all.len());
                    dst.copy_from_slice(&all);
                    return Ok((op, None));
                }
                o => return Err(io_err(format!("알 수 없는 op {o}"))),
            }
        }
    }

    /// 제어 프레임만 처리(데이터 대기 없이). 계획 중 연결 유지용. 데이터 프레임이 오면 Err.
    pub fn service_control(&mut self, buf: &mut Buf) -> io::Result<bool> {
        let h = self.read_frame(buf)?;
        match h.op {
            OP_PING => {
                let mut p = buf.data().to_vec();
                if let Some(k) = h.mask {
                    mask_at(&mut p, k, 0);
                }
                self.send(OP_PONG, &[&p])?;
                Ok(true)
            }
            OP_PONG => Ok(true),
            OP_CLOSE => Ok(false),
            o => Err(io_err(format!("계획 중 예상 못 한 데이터 프레임 op {o}"))),
        }
    }
}

pub fn encode_head(fin: bool, op: u8, mask: Option<[u8; 4]>, len: usize) -> ([u8; 14], usize) {
    let mut h = [0u8; 14];
    h[0] = (if fin { 0x80 } else { 0 }) | (op & 0x0f);
    let mbit = if mask.is_some() { 0x80 } else { 0 };
    let mut i;
    if len < 126 {
        h[1] = mbit | len as u8;
        i = 2;
    } else if len < 65536 {
        h[1] = mbit | 126;
        h[2..4].copy_from_slice(&(len as u16).to_be_bytes());
        i = 4;
    } else {
        h[1] = mbit | 127;
        h[2..10].copy_from_slice(&(len as u64).to_be_bytes());
        i = 10;
    }
    if let Some(k) = mask {
        h[i..i + 4].copy_from_slice(&k);
        i += 4;
    }
    (h, i)
}

/// 마스크 걸기/풀기(XOR, 대칭). `start` 는 프레임 페이로드 안에서 이 조각이 시작하는 위치.
pub fn mask_at(buf: &mut [u8], key: [u8; 4], start: usize) {
    let r = start & 3;
    let k = [key[r], key[(r + 1) & 3], key[(r + 2) & 3], key[(r + 3) & 3]];
    let k8 = u64::from_ne_bytes([k[0], k[1], k[2], k[3], k[0], k[1], k[2], k[3]]);
    let mut chunks = buf.chunks_exact_mut(8);
    for c in &mut chunks {
        let v = u64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) ^ k8;
        c.copy_from_slice(&v.to_ne_bytes());
    }
    let rem = chunks.into_remainder();
    for (j, b) in rem.iter_mut().enumerate() {
        *b ^= k[j & 3];
    }
}

/// 키 회전: 원래 위치 i 의 바이트가 새 프레임에서 i + shift 에 놓일 때, 다시 마스크하지 않고 그대로 쓰도록 하는 새 키.
/// 새 키 k' 는 k'[(i + shift) % 4] == k[i % 4] 를 만족한다.
pub fn rotate_key(key: [u8; 4], shift: isize) -> [u8; 4] {
    let mut out = [0u8; 4];
    for (t, o) in out.iter_mut().enumerate() {
        let src = ((t as isize - shift).rem_euclid(4)) as usize;
        *o = key[src];
    }
    out
}

pub fn write_all_vectored(s: &mut TcpStream, mut bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
    while !bufs.is_empty() {
        match s.write_vectored(bufs) {
            Ok(0) => return Err(io::Error::new(io::ErrorKind::WriteZero, "쓰기 0")),
            Ok(n) => IoSlice::advance_slices(&mut bufs, n),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// 여러 연결 중 읽을 것이 있는지(버퍼에 남은 것 포함) 기다린다.
#[cfg(unix)]
pub fn poll_readable(conns: &[&Conn], timeout_ms: i32) -> io::Result<Vec<bool>> {
    use std::os::unix::io::AsRawFd;
    let mut ready: Vec<bool> = conns.iter().map(|c| c.buffered() > 0).collect();
    if ready.iter().any(|&r| r) {
        return Ok(ready);
    }
    let mut fds: Vec<libc::pollfd> =
        conns.iter().map(|c| libc::pollfd { fd: c.s.as_raw_fd(), events: libc::POLLIN, revents: 0 }).collect();
    let rc = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, timeout_ms) };
    if rc < 0 {
        let e = io::Error::last_os_error();
        if e.kind() == io::ErrorKind::Interrupted {
            return Ok(ready);
        }
        return Err(e);
    }
    for (r, f) in ready.iter_mut().zip(&fds) {
        *r = f.revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0;
    }
    Ok(ready)
}

#[cfg(not(unix))]
pub fn poll_readable(conns: &[&Conn], timeout_ms: i32) -> io::Result<Vec<bool>> {
    // 윈도우용 대체: 짧게 쉬고 버퍼만 본다(중계기는 WSL 에서 돈다).
    std::thread::sleep(std::time::Duration::from_millis(timeout_ms.max(0) as u64));
    Ok(conns.iter().map(|c| c.buffered() > 0).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc_accept_example() {
        // RFC 6455 1.3 의 예
        assert_eq!(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn rotate_preserves_masking() {
        let key = [1u8, 2, 3, 4];
        let plain: Vec<u8> = (0..37u8).collect();
        let mut masked = plain.clone();
        mask_at(&mut masked, key, 0);
        for shift in [-4isize, -2, -1, 0, 1, 2, 3, 5] {
            // 원래 [h..] 조각이 새 프레임 [h+shift..] 에 놓인다고 할 때, 새 키로 풀면 평문이 나와야 한다.
            let h = 5usize;
            let k2 = rotate_key(key, shift);
            let mut part = masked[h..].to_vec();
            mask_at(&mut part, k2, (h as isize + shift) as usize);
            assert_eq!(part, plain[h..].to_vec(), "shift {shift}");
        }
    }

    #[test]
    fn mask_at_offsets() {
        let key = [9u8, 8, 7, 6];
        let plain: Vec<u8> = (0..50u8).collect();
        let mut a = plain.clone();
        mask_at(&mut a, key, 0);
        let mut b = plain[13..].to_vec();
        mask_at(&mut b, key, 13);
        assert_eq!(&a[13..], &b[..]);
    }
}

// SSE /api/live (TRAIN_VIEWER.md 5.8) — sgview serve_stream 틀: 붙으면 스냅숏(클라이언트가 가진 줄 수 `from` 뒤)부터, 그다음 새 줄.
//
//   GET /api/live?runs=a,b&from=<줄>,<줄>&sig=<s>,<s>&eps=<판 수>,<판 수>&rep=<개수>,<개수>
//
// 이벤트: progress {run, data: /api/progress 와 같은 꼴(모든 키)}, reset {run, sig}(파일이 새로 쓰임·되감김 → 처음부터 다시 받을 것),
//         episodes {run, stream, total}, replay {run, stream, n}, runs {n}(목록 바뀜). 10 초마다 keepalive.
// 감시는 연결마다 한 스레드가 1 초마다 stat 하고 늘어난 바이트만 읽는다(저장소는 /api/progress 와 같이 씀).
// 클라이언트는 줄 번호(start)로 이어 붙인다 — 다시 붙어도 빠지거나 겹치는 줄이 없다. 끊기면 화면은 3 초 폴링으로 내려간다.
use crate::http::{jstr, Req};
use crate::{runs, App};
use std::io::Write;
use std::net::TcpStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn list<T: std::str::FromStr + Copy>(s: &str, n: usize, d: T) -> Vec<T> {
    let mut v: Vec<T> = s.split(',').map(|x| x.parse().unwrap_or(d)).collect();
    v.resize(n, d);
    v
}

fn ev(name: &str, data: &str) -> String {
    format!("event: {}\ndata: {}\n\n", name, data)
}

pub fn serve(mut s: TcpStream, app: Arc<App>, req: &Req) {
    let ids: Vec<String> = req.get("runs").split(',').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect();
    let n = ids.len();
    let mut from: Vec<usize> = list(req.get("from"), n, 0usize);
    let mut sig: Vec<u64> = list(req.get("sig"), n, 0u64);
    let mut eps: Vec<i64> = list(req.get("eps"), n, -1i64);
    let mut rep: Vec<i64> = list(req.get("rep"), n, -1i64);
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";
    if s.write_all(head.as_bytes()).is_err() {
        return;
    }
    let mut last_send = Instant::now();
    let mut n_runs = app.runs().len();
    let mut last_status = Instant::now() - Duration::from_secs(60);
    let mut status_sig = String::new();
    loop {
        let mut out = String::new();
        for i in 0..n {
            let Some(r) = app.find(&ids[i]) else { continue };
            let mut idq = String::new();
            jstr(&mut idq, &ids[i]);
            {
                let pa = app.progress(&r);
                let pg = pa.lock().unwrap();
                if sig[i] != 0 && pg.sig != sig[i] {
                    out.push_str(&ev("reset", &format!("{{\"run\":{},\"sig\":{}}}", idq, pg.sig)));
                    from[i] = pg.total();
                } else if pg.total() > from[i] {
                    out.push_str(&ev("progress", &format!("{{\"run\":{},\"data\":{}}}", idq, pg.rows_json(None, from[i], 0))));
                    from[i] = pg.total();
                }
                sig[i] = pg.sig;
            }
            if r.dir.join("episodes.jsonl").exists() {
                let ea = app.episodes(&r, "main");
                let t = ea.lock().unwrap().total() as i64;
                if t != eps[i] {
                    out.push_str(&ev("episodes", &format!("{{\"run\":{},\"stream\":\"main\",\"total\":{}}}", idq, t)));
                    eps[i] = t;
                }
            }
            let k = runs::count_trp(&r.dir.join("replays")) as i64;
            if k != rep[i] {
                if rep[i] >= 0 || k > 0 {
                    out.push_str(&ev("replay", &format!("{{\"run\":{},\"stream\":\"main\",\"n\":{}}}", idq, k)));
                }
                rep[i] = k;
            }
        }
        // 모든 실행의 학습 상태(초록 불·재생 기록 중) — 2 초마다, 바뀌었을 때만(나이는 서명에서 뺌, 20 초마다는 그래도 보냄)
        if last_status.elapsed() > Duration::from_secs(2) {
            let st: Vec<serde_json::Value> = app.runs().iter().map(runs::status).collect();
            let sig: String = st.iter().map(|v| format!("{}{}{}{}", v["id"], v["state"], v["recording"], v["iter"])).collect();
            if sig != status_sig || last_status.elapsed() > Duration::from_secs(20) {
                out.push_str(&ev("status", &serde_json::Value::Array(st).to_string()));
                status_sig = sig;
            }
            last_status = Instant::now();
        }
        let nr = app.runs().len();
        if nr != n_runs {
            out.push_str(&ev("runs", &format!("{{\"n\":{}}}", nr)));
            n_runs = nr;
        }
        if out.is_empty() && last_send.elapsed() > Duration::from_secs(10) {
            out.push_str(": keepalive\n\n");
        }
        if !out.is_empty() {
            if s.write_all(out.as_bytes()).is_err() || s.flush().is_err() {
                return;
            }
            last_send = Instant::now();
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

//! 실행 기록(JSONL)과 재생.
//!
//! 한 줄 = 사건 하나: `{"t_ms", "wall_ms", "env", "type", ...}`. 종류:
//! `config`, `episode_start`, `boundary`(사건 + 그 순간 그래프 노드), `llm`(목적·요청 지문·요청 메시지·응답·지연),
//! `tool`(이름·인자·결과), `decision`, `prompt_applied`, `stage_close`, `summary`, `latency`, `episode_end`.
//! 영상은 `img/` 에 JPEG 로 따로 두고 기록에는 파일 이름만 넣는다(data URL 은 길이만 남김).
//!
//! 재생: `bagent replay <trace.jsonl>` 로 시간순 요약, `--verify` 로 기록된 LLM 응답을 다시 먹여 같은 결정이 나오는지 확인,
//! `--html` 로 수업 week03 execution_player 처럼 한 파일짜리 재생기를 만든다.

use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

struct Inner {
    w: BufWriter<File>,
    dir: PathBuf,
    t0: Instant,
}

#[derive(Clone)]
pub struct Tracer {
    inner: Arc<Mutex<Inner>>,
}

impl Tracer {
    pub fn create(dir: &Path) -> std::io::Result<Tracer> {
        std::fs::create_dir_all(dir.join("img"))?;
        let f = File::create(dir.join("trace.jsonl"))?;
        Ok(Tracer { inner: Arc::new(Mutex::new(Inner { w: BufWriter::with_capacity(1 << 20, f), dir: dir.to_path_buf(), t0: Instant::now() })) })
    }

    pub fn dir(&self) -> PathBuf {
        self.inner.lock().unwrap().dir.clone()
    }

    pub fn event(&self, env: usize, kind: &str, mut data: Value) {
        let mut g = self.inner.lock().unwrap();
        let t = g.t0.elapsed().as_secs_f64() * 1000.0;
        let mut o = json!({"t_ms": (t * 1000.0).round() / 1000.0, "wall_ms": crate::util::unix_ms(), "env": env, "type": kind});
        if let (Some(m), Some(d)) = (o.as_object_mut(), data.as_object_mut()) {
            for (k, v) in std::mem::take(d) {
                m.insert(k, v);
            }
        } else {
            o["data"] = data;
        }
        let _ = writeln!(g.w, "{o}");
    }

    pub fn save_image(&self, name: &str, jpeg: &[u8]) -> String {
        let g = self.inner.lock().unwrap();
        let rel = format!("img/{name}");
        let _ = std::fs::write(g.dir.join(&rel), jpeg);
        rel
    }

    pub fn flush(&self) {
        let _ = self.inner.lock().unwrap().w.flush();
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let _ = self.w.flush();
    }
}

/// 메시지 JSON 안의 data URL 을 길이 표시로 바꾼다(기록 크기 줄이기).
pub fn strip_data_urls(v: &mut Value) {
    match v {
        Value::String(s) if s.starts_with("data:") => {
            let n = s.len();
            *s = format!("data:…({n} chars)");
        }
        Value::Array(a) => a.iter_mut().for_each(strip_data_urls),
        Value::Object(m) => m.values_mut().for_each(strip_data_urls),
        _ => {}
    }
}

pub fn read(path: &Path) -> Result<Vec<Value>, String> {
    let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (i, l) in BufReader::new(f).lines().enumerate() {
        let l = l.map_err(|e| e.to_string())?;
        if l.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&l).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?);
    }
    Ok(out)
}

/// 사람이 읽는 시간순 요약.
pub fn timeline(events: &[Value]) -> String {
    let mut out = Vec::new();
    for e in events {
        let env = e["env"].as_u64().unwrap_or(0);
        let t = e["t_ms"].as_f64().unwrap_or(0.0) / 1000.0;
        let line = match e["type"].as_str().unwrap_or("") {
            "episode_start" => format!("== env{env} 판 {} 시작: {} ({})", e["episode"], e["task"].as_str().unwrap_or("?"), e["prompt"].as_str().unwrap_or("")),
            "boundary" => format!(
                "[{t:8.2}s] env{env} step {:>5} 경계: {} (단계 {} 스텝째)",
                e["event"]["step"], e["event"]["trigger"].as_str().unwrap_or("?"), e["event"]["stage_steps"]
            ),
            "llm" => format!(
                "           LLM {} {} ms, 도구 {}",
                e["purpose"].as_str().unwrap_or(""),
                e["result"]["latency_ms"],
                e["result"]["msg"]["tool_calls"].as_array().map(|a| a.iter().map(|c| c["function"]["name"].as_str().unwrap_or("").to_string()).collect::<Vec<_>>().join(",")).unwrap_or_else(|| "(없음)".into())
            ),
            "tool" => format!("           · {}({}) -> {}", e["name"].as_str().unwrap_or(""), trunc(&e["args"].to_string(), 120), trunc(&e["result"].to_string(), 160)),
            "decision" => {
                let d = &e["decision"];
                match d["kind"].as_str().unwrap_or("") {
                    "issue" => format!("           ⇒ 지시 \"{}\" (예산 {}, 이전 단계 {}) [{}]", d["prompt"].as_str().unwrap_or(""), d["budget"], d["previous"].as_str().unwrap_or(""), d["source"].as_str().unwrap_or("")),
                    "continue" => format!("           ⇒ 계속 (+{} 스텝): {} [{}]", d["extra"], d["reason"].as_str().unwrap_or(""), d["source"].as_str().unwrap_or("")),
                    "finish" => format!("           ⇒ 끝: {} [{}]", d["reason"].as_str().unwrap_or(""), d["source"].as_str().unwrap_or("")),
                    _ => d.to_string(),
                }
            }
            "summary" => format!("           (요약 갱신 {} ms) {}", e["latency_ms"], trunc(e["summary"].as_str().unwrap_or(""), 120)),
            "episode_end" => format!("== env{env} 판 끝: {}", e["result"]),
            "latency" => format!("[{t:8.2}s] 지연 통계 {}", e["stats"]),
            _ => continue,
        };
        out.push(line);
    }
    out.join("\n")
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

/// 한 파일짜리 재생기(HTML). 기록을 안에 넣고, 영상은 기록 폴더 기준 상대 경로로 보여 준다.
pub fn export_html(events: &[Value], title: &str) -> String {
    let data = serde_json::to_string(events).unwrap_or_else(|_| "[]".into()).replace("</", "<\\/");
    let mut html = String::from(HTML_TEMPLATE);
    html = html.replace("__TITLE__", &title.replace('<', "&lt;"));
    html.replace("__DATA__", &data)
}

const HTML_TEMPLATE: &str = r#"<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>에이전트 재생</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1f;--mut:#6b6b70;--card:#fff;--line:#e2e2de;--acc:#2458d6;--ok:#1f8a4c;--bad:#c23b2a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17171a;--fg:#ececef;--mut:#9a9aa2;--card:#212126;--line:#34343b;--acc:#7aa2ff;--ok:#5cc98a;--bad:#ff8a7a}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line)}h1{font-size:16px;margin:0}
.wrap{display:grid;grid-template-columns:minmax(260px,38%) 1fr;height:calc(100vh - 50px)}
@media (max-width:760px){.wrap{grid-template-columns:1fr;height:auto}}
#list{overflow:auto;border-right:1px solid var(--line)}#detail{overflow:auto;padding:12px 16px}
.ev{padding:6px 12px;border-bottom:1px solid var(--line);cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ev:hover,.ev.sel{background:var(--card)}.t{color:var(--mut);font-variant-numeric:tabular-nums;margin-right:6px}
.k-decision{font-weight:600}.k-boundary{color:var(--acc)}.k-tool{color:var(--mut);padding-left:24px}
pre{white-space:pre-wrap;word-break:break-word;background:var(--card);border:1px solid var(--line);padding:10px;border-radius:6px;font-size:12px}
img{max-width:100%;border-radius:6px;border:1px solid var(--line)}
.nav{position:sticky;top:0;background:var(--bg);padding:6px 0}button{font:inherit;padding:4px 10px}
</style></head><body>
<header><h1>__TITLE__</h1></header>
<div class="wrap"><div id="list"></div><div id="detail"><div class="nav"><button id="prev">◀ 이전</button> <button id="next">다음 ▶</button> <span id="pos"></span></div><div id="body">왼쪽에서 사건을 고르세요. ←/→ 키로도 움직입니다.</div></div></div>
<script type="application/json" id="trace">__DATA__</script>
<script>
const ev=JSON.parse(document.getElementById('trace').textContent).filter(e=>e.type!=='latency');
const list=document.getElementById('list'),body=document.getElementById('body');let cur=-1;
function label(e){const env='env'+e.env+' ';switch(e.type){
case'boundary':return env+'step '+e.event.step+' 경계: '+e.event.trigger;
case'llm':return env+'LLM '+e.purpose+' '+e.result.latency_ms+'ms '+((e.result.msg.tool_calls||[]).map(c=>c.function.name).join(',')||'(글)');
case'tool':return env+'· '+e.name;
case'decision':{const d=e.decision;return env+'⇒ '+(d.kind==='issue'?'"'+d.prompt+'"':d.kind==='continue'?'계속':'끝')+' ['+d.source+']';}
case'episode_start':return env+'판 시작: '+e.task;case'episode_end':return env+'판 끝';
default:return env+e.type;}}
ev.forEach((e,i)=>{const d=document.createElement('div');d.className='ev k-'+e.type;d.innerHTML='<span class="t">'+(e.t_ms/1000).toFixed(2)+'s</span>';d.appendChild(document.createTextNode(label(e)));d.onclick=()=>show(i);list.appendChild(d);});
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function show(i){if(i<0||i>=ev.length)return;cur=i;[...list.children].forEach((c,j)=>c.classList.toggle('sel',j===i));list.children[i].scrollIntoView({block:'nearest'});
const e=ev[i];let h='';document.getElementById('pos').textContent=(i+1)+' / '+ev.length;
if(e.type==='boundary'){(e.event.image_files||[]).forEach(f=>h+='<p><img src="'+esc(f)+'" alt="camera"></p>');}
if(e.type==='llm'){(e.request||[]).forEach(m=>{h+='<h3>'+esc(m.role)+'</h3><pre>'+esc(typeof m.content==='string'?m.content:JSON.stringify(m.content,null,1))+(m.tool_calls?'\n'+esc(JSON.stringify(m.tool_calls,null,1)):'')+'</pre>';});h+='<h3>응답</h3>';}
h+='<pre>'+esc(JSON.stringify(e.type==='llm'?e.result:e,null,2))+'</pre>';body.innerHTML=h;}
document.getElementById('prev').onclick=()=>show(cur-1);document.getElementById('next').onclick=()=>show(cur+1);
document.addEventListener('keydown',k=>{if(k.key==='ArrowRight')show(cur+1);if(k.key==='ArrowLeft')show(cur-1);});
</script></body></html>
"#;

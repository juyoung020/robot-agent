//! `decisions-agg` — decisions.jsonl 여러 개를 실행기·모드별로 묶어 표(마크다운)로.
//! 개발용 명령이다(LLM 도구 아님 — LLM 에게 보이지 않고 판이 끝난 뒤 사람이 돌린다). 옛 스킬 explore 에서 옮김(10-06).
//!
//! ```text
//! decisions-agg <run dir 또는 decisions.jsonl>... [--by policy|mode|prompt|executor]
//! ```
//! 열: 결정 수, reached / blocked / error / timeout 비율, 평균 이동 m, 평균 새 빈칸 m², m 당 새 빈칸, 접촉 증가,
//! 표지(tags) 상위. 결정 기록 형식은 src/agent/decision_log.md.

use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Default)]
struct Agg {
    n: usize,
    status: BTreeMap<String, usize>,
    moved: f64,
    gain: f64,
    contacts: f64,
    tags: BTreeMap<String, usize>,
    latency: f64,
    lat_n: usize,
    tokens: f64,
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let by = a.iter().position(|x| x == "--by").and_then(|i| a.get(i + 1)).cloned().unwrap_or_else(|| "mode".into());
    let mut files = vec![];
    for p in a.iter().filter(|x| !x.starts_with("--") && **x != by) {
        let pb = std::path::PathBuf::from(p);
        files.push(if pb.is_dir() { pb.join("decisions.jsonl") } else { pb });
    }
    let mut groups: BTreeMap<String, Agg> = BTreeMap::new();
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else {
            eprintln!("skip {}", f.display());
            continue;
        };
        let mut last_contacts = 0.0;
        for line in text.lines() {
            let Ok(d) = serde_json::from_str::<Value>(line) else { continue };
            let key = match by.as_str() {
                "policy" => format!("{} {}", d["policy"].as_str().unwrap_or("?"), d["mode"].as_str().unwrap_or("?")),
                "prompt" => format!("{} {}", d["prompt"]["version"].as_str().unwrap_or("?"), d["policy"].as_str().unwrap_or("?")),
                "executor" => d["executor"].as_str().unwrap_or("?").to_string(),
                _ => format!("{} / {}", d["executor"].as_str().unwrap_or("?"), d["mode"].as_str().unwrap_or("?")),
            };
            let g = groups.entry(key).or_default();
            g.n += 1;
            let o = &d["outcome"];
            *g.status.entry(o["status"].as_str().unwrap_or("?").to_string()).or_default() += 1;
            g.moved += o["moved_m"].as_f64().unwrap_or(0.0);
            g.gain += o["new_free_m2"].as_f64().unwrap_or(0.0);
            let c = o["contacts"].as_f64().unwrap_or(last_contacts);
            g.contacts += (c - last_contacts).max(0.0);
            last_contacts = c;
            for t in d["label"]["tags"].as_array().cloned().unwrap_or_default() {
                *g.tags.entry(t.as_str().unwrap_or("?").to_string()).or_default() += 1;
            }
            if let Some(l) = d["llm"]["latency_ms"].as_f64() {
                g.latency += l;
                g.lat_n += 1;
                g.tokens += d["llm"]["prompt_tokens"].as_f64().unwrap_or(0.0) + d["llm"]["completion_tokens"].as_f64().unwrap_or(0.0);
            }
        }
    }
    println!("| {by} | n | reached | blocked | error | other | moved m (avg) | new free m² (avg) | m² per m | contacts | LLM ms (avg) | tokens (avg) | tags |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (k, g) in &groups {
        let pct = |s: &str| 100.0 * *g.status.get(s).unwrap_or(&0) as f64 / g.n as f64;
        let other = 100.0 - pct("reached") - pct("blocked") - pct("error");
        let tags: Vec<String> = {
            let mut v: Vec<(&String, &usize)> = g.tags.iter().collect();
            v.sort_by(|a, b| b.1.cmp(a.1));
            v.into_iter().take(4).map(|(t, n)| format!("{t} {n}")).collect()
        };
        let n = g.n as f64;
        println!(
            "| {k} | {} | {:.0}% | {:.0}% | {:.0}% | {:.0}% | {:.2} | {:.2} | {:.2} | {:.0} | {} | {} | {} |",
            g.n,
            pct("reached"),
            pct("blocked"),
            pct("error"),
            other,
            g.moved / n,
            g.gain / n,
            if g.moved > 0.0 { g.gain / g.moved } else { 0.0 },
            g.contacts,
            if g.lat_n > 0 { format!("{:.0}", g.latency / g.lat_n as f64) } else { "-".into() },
            if g.lat_n > 0 { format!("{:.0}", g.tokens / g.lat_n as f64) } else { "-".into() },
            tags.join(", ")
        );
    }
}

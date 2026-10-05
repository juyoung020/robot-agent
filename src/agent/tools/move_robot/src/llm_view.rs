//! 에이전트 런타임(`src/agent/runtime`)이 이 도구 결과를 다루는 법 — 도구 쪽 지식이라 여기 둔다.
//!
//! - [`compact`]: 결과 → LLM 에게 보일 것만(측정값 `_m`·단위·내부 상태 뺌, 16k 맥락용).
//! - [`obs_features`]·[`label`]·[`outcome`]·[`target_space`]·[`mode_key`]: 결정 기록(`src/agent/decision_log.md`) 칸.
//! - [`short_args`]: 오래된 호출 쌍을 접을 때 한 줄 요약. [`merge_map`]: 지도 없는 결과(오류)에 지난 지도·측정값 잇기.
//! - [`link_down`]: 시뮬 연결이 끊겼다는 결과인가(런타임이 판을 끝냄).

use serde_json::{json, Value};

/// 결과 → LLM 에게 보일 것만. 측정값(`_m`)·단위·내부 상태는 뺀다.
pub fn compact(r: &Value) -> Value {
    let mut o = serde_json::Map::new();
    for k in ["status", "mode", "target", "moved_m", "turned_deg", "stopped_by", "clear_m", "hint", "message", "part", "explore"] {
        if let Some(v) = r.get(k) {
            if k == "part" && v == "base" {
                continue;
            }
            o.insert(k.into(), v.clone());
        }
    }
    if r.get("mode").is_none() && r.get("part") != Some(&json!("base")) {
        if let Some(v) = r.get("state") {
            o.insert("state".into(), v.clone());
        }
    }
    if let Some(m) = r.get("map") {
        o.insert("map".into(), m.clone());
    }
    Value::Object(o)
}

/// 결정 직전 관측(compact 한 것)에서 뽑는 특징
pub fn obs_features(obs: &Value) -> Value {
    let map = &obs["map"];
    let fr = map["frontiers"].as_array().cloned().unwrap_or_default();
    let nearest = fr.iter().min_by(|a, b| a["path_m"].as_f64().unwrap_or(1e9).total_cmp(&b["path_m"].as_f64().unwrap_or(1e9)));
    json!({
        "free_m2": map["free_m2"], "n_frontiers": fr.len(),
        "best": fr.first().map(|f| json!({"id": f["id"], "path_m": f["path_m"], "new_area_m2": f["new_area_m2"]})),
        "nearest": nearest.map(|f| json!({"id": f["id"], "path_m": f["path_m"]})),
        "in_room": map.get("in_room").cloned().unwrap_or(Value::Null),
        "front": map["around"]["F"],
        "last_status": obs["status"],
    })
}

/// 사후 표지: 고른 것이 나빴나(계산할 수 있을 때만). pre = 결정 직전 관측(compact).
pub fn label(args: &Value, pre: &Value, res: &Value) -> Value {
    let mode = args["mode"].as_str().unwrap_or("");
    let status = res["status"].as_str().unwrap_or("");
    let gain = res["map"]["new_free_m2"].as_f64().unwrap_or(0.0);
    let mut l = vec![];
    if status == "error" {
        l.push("invalid_call");
        if res["message"].as_str().map_or(false, |m| m.contains("UNKNOWN")) {
            l.push("goto_into_unknown");
        }
    }
    let fr = pre["map"]["frontiers"].as_array().cloned().unwrap_or_default();
    let mut rank = Value::Null;
    if mode == "go_to" {
        if let Some(t) = args["target"].as_str() {
            if let Some(k) = fr.iter().position(|f| f["id"].as_str() == Some(&t.to_uppercase())) {
                rank = json!(k + 1);
            }
        }
    }
    if mode == "probe" {
        let fwd = args["values"].get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let moved = res["moved_m"].as_f64().unwrap_or(0.0);
        if fwd > 0.0 && status == "blocked" && moved < 0.3 {
            l.push("wasted_probe");
        }
        if fwd == 0.0 && gain < 0.5 {
            l.push("no_gain_look");
        }
    }
    if mode == "delta" && args["part"] == "base" {
        l.push("delta_for_explore");
    }
    if status == "reached" && gain < 0.2 && mode == "go_to" {
        l.push("no_gain_move");
    }
    json!({"tags": l, "rank": rank, "vla_better": Value::Null})
}

/// 결정 기록 `mode`: "base:go_to", "left_arm:absolute" …
pub fn mode_key(args: &Value) -> String {
    format!("{}:{}", args["part"].as_str().unwrap_or("?"), args["mode"].as_str().unwrap_or("?"))
}

/// go_to = 아는 빈칸 경로, probe = 모르는 곳
pub fn target_space(args: &Value) -> &'static str {
    match args["mode"].as_str() {
        Some("go_to") => "known",
        Some("probe") => "unknown",
        _ => "n/a",
    }
}

/// 결정 기록 `outcome` 중 이 도구 결과에서 오는 칸(측정값 `_m` 칸은 런타임이 스킬 설정대로 더함)
pub fn outcome(res: &Value) -> serde_json::Map<String, Value> {
    let v = json!({"status": res["status"], "moved_m": res["moved_m"], "time_s": res["time_s"], "stopped_by": res.get("stopped_by").cloned().unwrap_or(Value::Null),
        "replans": res.get("replans").cloned().unwrap_or(json!(0)), "new_free_m2": res["map"]["new_free_m2"],
        "message": res.get("message").cloned().unwrap_or(Value::Null)});
    match v {
        Value::Object(m) => m,
        _ => unreachable!(),
    }
}

/// 맥락을 접을 때 한 줄 요약 "n. go_to F1 -> reached, +5.3 m2" 의 앞부분
pub fn short_args(a: &Value) -> String {
    match a["mode"].as_str() {
        Some("go_to") => format!("go_to {}", a.get("target").and_then(|t| t.as_str()).map(|s| s.to_string()).unwrap_or_else(|| a["values"].to_string())),
        Some(m) => format!("{} {} {}", a["part"].as_str().unwrap_or("?"), m, a["values"]),
        None => a.to_string(),
    }
}

/// 한 줄 요약 전체
pub fn fold_line(n: usize, args: &Value, res: &Value) -> String {
    format!(
        "{}. {} -> {}{}, +{} m2",
        n,
        short_args(args),
        res["status"].as_str().unwrap_or("?"),
        res.get("stopped_by").and_then(|v| v.as_str()).map(|s| format!(" ({s})")).unwrap_or_default(),
        res["map"]["new_free_m2"].as_f64().unwrap_or(0.0)
    )
}

/// 다음 관측: 지도 있는 결과는 그대로, 지도 없는 결과(오류)는 지난 지도·측정값을 이어 쓴다
pub fn next_obs(res: &Value, last: &Value) -> Value {
    if res["map"].is_object() {
        res.clone()
    } else {
        merge_map(res, last)
    }
}

pub fn merge_map(r: &Value, last: &Value) -> Value {
    let mut o = r.clone();
    o["map"] = last["map"].clone();
    o["_m"] = last["_m"].clone();
    o
}

/// 시뮬 연결이 끊겼다는 결과(`TcpSim`)
pub fn link_down(res: &Value) -> bool {
    res["hint"].as_str().map_or(false, |h| h.contains("simulator link is down"))
}

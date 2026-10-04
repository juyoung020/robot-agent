// csv2run — 규약 이전에 돈 실행(ppo_run·bc_run 의 log.csv + config.json + events.txt/results.json)을 실행 폴더 규약으로 옮긴다.
// 원본 폴더는 읽기만 하고, 새 폴더(--out 밑)에 run.json · progress.jsonl · evals/ 를 쓴다. 키 이름은 ppo_run/bc_run 의 runfolder.rs 와 같다.
//
//   csv2run --out ~/trainview_work/imported <실행 폴더> [<실행 폴더> …]
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use trainfmt::RunWriter;

fn read_csv(p: &Path) -> Option<(Vec<String>, Vec<Vec<String>>)> {
    let t = fs::read_to_string(p).ok()?;
    let mut it = t.lines();
    let head: Vec<String> = it.next()?.split(',').map(|s| s.to_string()).collect();
    let rows = it.filter(|l| !l.is_empty()).map(|l| l.split(',').map(|s| s.to_string()).collect::<Vec<_>>()).filter(|r: &Vec<String>| r.len() == head.len()).collect();
    Some((head, rows))
}

fn ppo(src: &Path, dst: &Path, cfg: &Value, name: &str) -> usize {
    let (h, rows) = read_csv(&src.join("log.csv")).unwrap();
    let ix: HashMap<&str, usize> = h.iter().enumerate().map(|(i, k)| (k.as_str(), i)).collect();
    let n_env = cfg.get("n_env").and_then(|x| x.as_i64()).unwrap_or(4096);
    let hor = cfg.get("horizon").and_then(|x| x.as_i64()).unwrap_or(64);
    // events.txt 의 커리큘럼 넘어가기: "... at iter N ... -> NAME"
    let ev = fs::read_to_string(src.join("events.txt")).unwrap_or_default();
    let mut changes: Vec<(f64, String)> = vec![];
    let mut first_stage = String::new();
    for l in ev.lines() {
        if let (Some(a), Some(b)) = (l.find(" at iter "), l.rfind("-> ")) {
            let it: f64 = l[a + 9..].split_whitespace().next().unwrap_or("0").parse().unwrap_or(0.0);
            changes.push((it, l[b + 3..].trim().to_string()));
        }
        if first_stage.is_empty() {
            if let Some(p) = l.find("name: \"") {
                first_stage = l[p + 7..].split('"').next().unwrap_or("").to_string();
            }
        }
    }
    let stem = cfg.get("comment").map(|_| "").unwrap_or("");
    let _ = stem;
    let group = name.trim_end_matches(|c: char| c.is_ascii_digit()).trim_end_matches("_s").to_string();
    let meta = json!({
        "kind": "teacher", "name": name, "group": group, "seed": cfg.get("seed"),
        "trainer": "ppo_run", "imported_from": src.display().to_string(), "config": cfg,
        "ctrl_hz": 10, "n_envs": n_env, "rollout_T": hor, "env_steps_per_iter": n_env * hor, "log_envs": 0, "skills": ["approach"],
        "gamma": cfg.get("gamma"), "gae_lambda": cfg.get("lam"), "clip_range": cfg.get("clip"), "target_kl": cfg.get("kl_target").cloned().unwrap_or(json!(0.01)),
        "lr": cfg.get("lr"), "n_epochs": cfg.get("epochs"), "n_minibatch": cfg.get("minibatches"), "max_grad_norm": cfg.get("max_grad_norm").cloned().unwrap_or(json!(1.0)),
        "precision": {"fp8": cfg.get("fp8").cloned().unwrap_or(json!(0)), "default": "bf16"},
        "curriculum": {"stages": cfg.get("curriculum").and_then(|c| c.get("stages")), "start_maps": ["C0", "C1", "C2"]},
        "refs": {"train/approx_kl": cfg.get("kl_target").cloned().unwrap_or(json!(0.01)), "train/grad_norm": cfg.get("max_grad_norm").cloned().unwrap_or(json!(1.0))},
        "stage": changes.last().map(|c| c.1.clone()).unwrap_or(first_stage),
        "logged": {"progress": true, "episodes": false, "replays": false, "why": "옮긴 실행: log.csv 는 바퀴 합계만 담는다"},
    });
    let mut w = RunWriter::create(dst, meta, false, false).unwrap();
    let mt = fs::metadata(src.join("log.csv")).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    let last_wall: f64 = rows.last().and_then(|r| r[ix["wall_s"]].parse().ok()).unwrap_or(0.0);
    let t_start = mt - last_wall;
    w.meta["started"] = json!(t_start);
    w.meta["segments"] = json!([{"started": t_start}]);
    w.meta.as_object_mut().unwrap().remove("pid");
    w.meta.as_object_mut().unwrap().remove("pid_start");
    w.meta["ended"] = json!(mt);
    w.write_meta().unwrap();
    let mut last = (0.0, 0.0);
    let mut si = 0usize;
    for r in &rows {
        let g = |k: &str| ix.get(k).and_then(|&i| r[i].parse::<f64>().ok()).unwrap_or(f64::NAN);
        let wall = g("wall_s");
        let it = g("iter");
        while si < changes.len() && changes[si].0 < it {
            si += 1;
            w.flush_row(wall);
        }
        let a = &mut w.agg;
        let n = g("n_eps");
        a.last("time/iter", it);
        a.last("time/env_steps", g("env_steps"));
        a.last("time/wall", wall);
        a.mean("time/rollout_ms", g("rollout_ms"));
        a.mean("time/update_ms", g("update_ms"));
        a.mean("time/iter_ms", g("rollout_ms") + g("update_ms"));
        a.mean("time/fps_gpu", g("gpu_env_steps_per_s"));
        a.sum("rollout/n_eps", n);
        a.mean_w("rollout/ep_ret_mean", g("ep_ret"), n);
        a.mean_w("rollout/ep_len_mean", g("ep_len"), n);
        a.mean_w("rollout/success/approach", g("succ"), n);
        a.mean_w("rollout/collision_rate", g("coll"), n);
        a.mean_w("rollout/timeout_rate", g("tout"), n);
        a.mean("rollout/step_rew_mean", g("rew_mean"));
        for (i, nm) in ["C0", "C1", "C2"].iter().enumerate() {
            let nc = g(&format!("n_c{}", i));
            if nc.is_finite() {
                a.sum(&format!("curr/n_eps/{}", nm), nc);
                a.mean_w(&format!("curr/success/{}", nm), g(&format!("succ_c{}", i)), nc);
                a.mean_w(&format!("curr/collision/{}", nm), g(&format!("coll_c{}", i)), nc);
                a.mean_w(&format!("curr/stage_frac/{}", nm), nc / n.max(1.0), n);
            }
        }
        a.last("curr/stage_idx", si as f64);
        a.last("curr/env_stage", g("stage"));
        a.mean("curr/goal_known", g("goal_known"));
        a.mean("map/task_confirmed", g("map_task"));
        a.mean("train/approx_kl", g("kl"));
        a.mean("train/clip_frac", g("clipfrac"));
        a.mean("train/entropy", g("entropy"));
        a.mean("train/policy_loss", g("pg_loss"));
        a.mean("train/value_loss", g("v_loss"));
        a.mean("train/grad_norm", g("grad_norm"));
        a.mean("train/lr", g("lr"));
        a.mean("train/adv_std", g("adv_std"));
        a.mean("train/value_mean", g("value_mean"));
        a.mean("train/std/vx", g("std0"));
        a.mean("train/std/wz", g("std1"));
        if g("std0") > 0.0 && g("std1") > 0.0 {
            a.mean("train/log_std_mean", (g("std0").ln() + g("std1").ln()) / 2.0);
        }
        if w.due(wall) {
            if last.0 > 0.0 && wall > last.0 {
                w.agg.last("time/fps_env", (g("env_steps") - last.1) / (wall - last.0));
            }
            last = (wall, g("env_steps"));
        }
        w.end_update(wall);
    }
    w.finish(last_wall, json!({"ended": mt}));
    w.rows_written() as usize
}

fn bc(src: &Path, dst: &Path, cfg: &Value, name: &str) -> usize {
    let (h, rows) = read_csv(&src.join("log.csv")).unwrap();
    let ix: HashMap<&str, usize> = h.iter().enumerate().map(|(i, k)| (k.as_str(), i)).collect();
    let n_env = cfg.get("n_env").and_then(|x| x.as_f64()).unwrap_or(4096.0);
    let hor = cfg.get("horizon").and_then(|x| x.as_f64()).unwrap_or(64.0);
    let dagger = cfg.get("dagger").and_then(|x| x.as_bool()).unwrap_or(true) && cfg.get("dagger_iters").and_then(|x| x.as_i64()).unwrap_or(4) > 0;
    let group = name.trim_end_matches(|c: char| c.is_ascii_digit()).trim_end_matches("_s").to_string();
    let meta = json!({
        "kind": if dagger { "dagger" } else { "bc" }, "name": name, "group": group, "seed": cfg.get("seed"),
        "trainer": "bc_run", "imported_from": src.display().to_string(), "config": cfg, "teacher": cfg.get("teacher"),
        "ctrl_hz": 10, "n_envs": n_env, "rollout_T": hor, "log_envs": 0, "skills": ["approach"],
        "lr": cfg.get("lr"), "batch": cfg.get("mb"), "chunk_H": cfg.get("chunk"), "denoise_steps": cfg.get("flow_steps"),
        "rounds": cfg.get("dagger_iters"), "dagger": dagger,
        "precision": {"fp8": cfg.get("fp8").cloned().unwrap_or(json!(0)), "vit_prec": cfg.get("vit_prec"), "default": "bf16"},
        "refs": {"train/grad_norm": cfg.get("max_grad_norm").cloned().unwrap_or(json!(1.0))},
        "x_default": "time/iter",
        "logged": {"progress": true, "episodes": false, "replays": false, "why": "옮긴 실행: log.csv 는 롤아웃·갱신 합계만 담는다"},
    });
    let mut w = RunWriter::create(dst, meta, false, false).unwrap();
    let mt = fs::metadata(src.join("log.csv")).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    let last_wall: f64 = rows.last().and_then(|r| r[ix["wall_s"]].parse().ok()).unwrap_or(0.0);
    w.meta["started"] = json!(mt - last_wall);
    w.meta["segments"] = json!([{"started": mt - last_wall}]);
    w.meta.as_object_mut().unwrap().remove("pid");
    w.meta.as_object_mut().unwrap().remove("pid_start");
    w.meta["ended"] = json!(mt);
    w.write_meta().unwrap();
    let (mut steps, mut adam_t) = (0.0, 0.0);
    let mut phase = String::new();
    for r in &rows {
        let g = |k: &str| ix.get(k).and_then(|&i| r[i].parse::<f64>().ok()).unwrap_or(f64::NAN);
        let ph = &r[ix["phase"]];
        let wall = g("wall_s");
        if *ph != phase {
            w.flush_row(wall);
            phase = ph.clone();
        }
        let kind = g("kind") as i32;
        if kind == 0 {
            steps += n_env * hor;
        } else {
            adam_t = g("adam_t");
        }
        let a = &mut w.agg;
        a.last("time/iter", adam_t);
        a.last("time/env_steps", steps);
        a.last("time/wall", wall);
        let d: String = ph.chars().filter(|c| c.is_ascii_digit()).collect();
        a.last("dagger/round", d.parse().unwrap_or(0.0));
        let n = g("n_eps");
        if kind == 1 {
            a.mean("bc/loss", g("loss"));
            a.mean("train/grad_norm", g("grad_norm"));
            a.mean("time/update_gpu_ms", g("gpu_ms"));
            a.last("data/samples", g("count"));
        } else if g("record") as i32 == 1 {
            let p = if g("actor") as i32 == 1 { "rollout" } else { "rollout_teacher" };
            a.sum(&format!("{}/n_eps", p), n);
            a.mean_w(&format!("{}/success/approach", p), g("succ"), n);
            a.mean_w(&format!("{}/collision_rate", p), g("coll"), n);
            a.mean_w(&format!("{}/timeout_rate", p), g("tout"), n);
            for (i, nm) in ["C0", "C1", "C2"].iter().enumerate() {
                a.mean_w(&format!("{}/success_start/{}", p, nm), g(&format!("succ_c{}", i)), g(&format!("n_c{}", i)));
            }
            a.mean("dagger/disagree", g("disagree"));
            a.last("data/samples", g("count"));
            a.mean("time/rollout_gpu_ms", g("gpu_ms"));
        } else {
            a.mean("time/eval_rollout_gpu_ms", g("gpu_ms"));
        }
        w.end_update(wall);
        // 평가 단계 끝: results.json 의 그 표를 한 줄로(bc_run runfolder 와 같은 키)
    }
    w.flush_row(last_wall);
    if let Some(res) = fs::read_to_string(src.join("results.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
        if let Some(o) = res.as_object() {
            for (k, tab) in o {
                let who = if k == "teacher" { "teacher" } else { "student" };
                let all = &tab["all"];
                let mut rows_t = vec![json!({"eval": "all", "split": "all", "n": all["episodes"], "success": all["success"], "collision": all["collision"], "timeout": all["timeout"], "steps": all["steps"]})];
                for grp in ["by_stage", "by_completeness"] {
                    if let Some(o2) = tab.get(grp).and_then(|x| x.as_object()) {
                        for (s, r) in o2 {
                            rows_t.push(json!({"eval": grp, "split": s, "n": r["episodes"], "success": r["success"], "collision": r["collision"], "timeout": r["timeout"], "steps": r["steps"]}));
                        }
                    }
                }
                w.eval(&format!("{}", k), &json!({"name": k, "driver": who, "rows": rows_t, "imported": true}));
            }
        }
    }
    w.finish(last_wall, json!({"ended": mt}));
    w.rows_written() as usize
}

fn main() {
    let mut out: Option<PathBuf> = None;
    let mut srcs: Vec<PathBuf> = vec![];
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == "--out" {
            out = it.next().map(PathBuf::from);
        } else {
            srcs.push(PathBuf::from(a));
        }
    }
    let out = out.unwrap_or_else(|| {
        eprintln!("usage: csv2run --out DIR <run dir with log.csv> …");
        std::process::exit(2)
    });
    for s in srcs {
        let cfg: Value = fs::read_to_string(s.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(json!({}));
        let Some((h, _)) = read_csv(&s.join("log.csv")) else {
            eprintln!("skip {}: no log.csv", s.display());
            continue;
        };
        // 이름: 마지막 두 폴더(g6/bf16_s1 → g6_bf16_s1)
        let comps: Vec<String> = s.components().rev().take(2).map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
        let name = format!("{}_{}", comps.get(1).cloned().unwrap_or_default(), comps[0]);
        let dst = out.join(&name);
        let _ = fs::remove_dir_all(&dst);
        let n = if h.first().map(|x| x == "phase").unwrap_or(false) { bc(&s, &dst, &cfg, &name) } else { ppo(&s, &dst, &cfg, &name) };
        eprintln!("csv2run: {} -> {} ({} progress lines)", s.display(), dst.display(), n);
    }
}

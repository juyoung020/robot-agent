// 학습 뷰어 실행 폴더 규약(docs/map_vla/TRAIN_VIEWER.md 4절) 쓰기 — bc_run 의 기록 경로(Run::drain, 끝난 기록을 꺼낼 때)에서만 부른다.
// log.csv·results.json 은 그대로 두고 같은 기록을 run.json + progress.jsonl(≤ 1 줄/s 로 합침, 단계가 바뀌면 끊음) + evals/<이름>.json 으로도 낸다.
//   갱신 기록  → time/iter = Adam 스텝, bc/loss, train/grad_norm
//   학생이 몬 기록 롤아웃(DAgger 모으기) → rollout/* (본 줄기 = 학생 판), dagger/disagree
//   교사가 몬 기록 롤아웃(교사 기록, "dagger": false 대조) → rollout_teacher/* (3.1: 학생 집계에 섞지 않음)
//   평가 롤아웃 → 롤아웃마다는 안 씀. 평가가 끝나면 표(앞 eval_drop 롤아웃 뺀 모든 판)를 eval/<교사|학생>/* 한 줄 + evals/<이름>.json
// 판마다 줄·궤적은 내지 않는다(장치 표가 합계만 담음 — 남은 일).
use crate::{BcConfig, BcLog};
use serde_json::{json, Value};
use std::path::Path;
use trainfmt::RunWriter;

#[allow(dead_code)]
pub struct RunFolder {
    w: RunWriter,
    phase: String,
    env_steps: f64,
    steps_per_rollout: f64,
    skill: String,
    adam_t: f64,
}

fn round_of(phase: &str) -> f64 {
    let d: String = phase.chars().filter(|c| c.is_ascii_digit()).collect();
    d.parse().unwrap_or(0.0)
}

#[allow(clippy::too_many_arguments)]
pub fn open(out: &Path, cfg_path: &str, v: &Value, c: &BcConfig, teacher: &str, dagger: bool, nd: usize, params: i64, dev_bytes: i64) -> Option<RunFolder> {
    if v.get("runfolder").and_then(|x| x.as_bool()) == Some(false) {
        return None;
    }
    let stem = Path::new(cfg_path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let skill = v.get("skill").and_then(|x| x.as_str()).unwrap_or("approach").to_string();
    let kind = v.get("kind").and_then(|x| x.as_str()).unwrap_or(if dagger && nd > 0 { "dagger" } else { "bc" });
    let meta = json!({
        "kind": kind,
        "group": v.get("group").and_then(|x| x.as_str()).unwrap_or(&stem),
        "seed": c.seed,
        "trainer": "bc_run",
        "config_path": std::fs::canonicalize(cfg_path).map(|p| p.display().to_string()).unwrap_or(cfg_path.to_string()),
        "config": v,
        "git": trainfmt::git_info(Path::new(env!("CARGO_MANIFEST_DIR"))),
        "teacher": teacher,
        "init_from": v.get("init_student"),
        "ctrl_hz": 10,
        "n_envs": c.n_env,
        "rollout_T": c.horizon,
        "env_steps_per_rollout": c.n_env as i64 * c.horizon as i64,
        "log_envs": 0,
        "skills": [skill],
        "lr": c.lr, "max_grad_norm": c.max_grad_norm,
        "batch": c.mb, "upd_steps": c.upd_steps, "chunk_H": c.chunk, "denoise_steps": c.flow_steps,
        "head": if c.head != 0 { "flow" } else { "mse" },
        "vision": c.vision, "text": c.text, "use_map": c.use_map,
        "rounds": nd, "dagger": dagger,
        "beta_schedule": if dagger { "student drives every collect round (beta 0)" } else { "teacher drives (control)" },
        "precision": {"fp8": v.get("fp8").and_then(|x| x.as_i64()).unwrap_or(0), "vit_prec": v.get("vit_prec").and_then(|x| x.as_i64()), "default": "bf16"},
        "num_params": params,
        "device_bytes": dev_bytes,
        "curriculum": {"start_maps": ["C0", "C1", "C2"], "map": [c.map_p0, c.map_p1]},
        "refs": {"train/grad_norm": c.max_grad_norm},
        "x_default": "time/iterations",
        "logged": {
            "progress": true,
            "episodes": false,
            "replays": false,
            "why": "bc_run 의 장치 표·기록 링은 롤아웃 합계만 담는다. 판마다 줄·궤적은 장치 쪽 기록 버퍼가 필요하다(CUDA 변경, TRAIN_VIEWER 남은 일)."
        },
    });
    let mut w = match RunWriter::create(out, meta, false, true) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("runfolder: {} — progress.jsonl 을 쓰지 않음", e);
            return None;
        }
    };
    w.every_s = v.get("progress_every_s").and_then(|x| x.as_f64()).unwrap_or(1.0);
    Some(RunFolder { w, phase: String::new(), env_steps: 0.0, steps_per_rollout: c.n_env as f64 * c.horizon as f64, skill, adam_t: 0.0 })
}

impl RunFolder {
    pub fn log(&mut self, phase: &str, l: &BcLog, wall: f64) {
        if phase != self.phase {
            self.w.flush_row(wall); // 단계가 바뀌면 줄을 끊는다(평가·모으기·갱신을 한 줄에 섞지 않음)
            self.phase = phase.to_string();
        }
        let a = &mut self.w.agg;
        if l.kind == 0 {
            self.env_steps += self.steps_per_rollout;
        } else {
            self.adam_t = l.adam_t as f64;
        }
        a.last("time/iterations", self.adam_t);
        a.last("time/total_timesteps", self.env_steps);
        a.last("time/time_elapsed", wall);
        a.last("dagger/round", round_of(phase));
        let n = l.n_eps as f64;
        if l.kind == 1 {
            a.mean("train/loss", l.loss as f64);
            a.mean("train/grad_norm", l.grad_norm as f64);
            a.mean("time/update_gpu_ms", l.gpu_ms as f64);
            a.last("dagger/dataset_size", l.count as f64);
        } else if l.record == 1 {
            let p = if l.actor == 1 { "rollout" } else { "rollout_teacher" };
            a.sum(&format!("{}/n_episodes", p), n);
            a.mean_w(&format!("{}/success_rate", p), l.succ as f64, n);
            a.mean_w(&format!("{}/collision_rate", p), l.coll as f64, n);
            a.mean_w(&format!("{}/timeout_rate", p), l.tout as f64, n);
            for (i, nm) in ["C0", "C1", "C2"].iter().enumerate() {
                a.mean_w(&format!("{}/success_rate_by_start_map/{}", p, nm), l.s_c[i] as f64, l.n_c[i] as f64);
            }
            a.mean("dagger/action_mse", l.disagree as f64);
            a.last("dagger/dataset_size", l.count as f64);
            a.mean("time/rollout_gpu_ms", l.gpu_ms as f64);
        } else {
            a.mean("time/eval_rollout_gpu_ms", l.gpu_ms as f64);
        }
        self.w.end_update(wall);
    }

    /// 평가 하나 끝: 표 → eval/<actor>/* 한 줄 + evals/<이름>.json (4.5)
    pub fn eval(&mut self, name: &str, actor: i32, tab: &Value, ckpt: Option<&Path>, wall: f64) {
        self.w.flush_row(wall);
        let who = if actor == 0 { "teacher" } else { "student" };
        let ns = if actor == 0 { "eval_teacher" } else { "eval" };   // 학생(이 실행의 정책) = eval/*, 교사 기준 = eval_teacher/*
        let a = &mut self.w.agg;
        a.last("time/iterations", self.adam_t);
        a.last("time/total_timesteps", self.env_steps);
        a.last("time/time_elapsed", wall);
        let all = &tab["all"];
        let f = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
        a.last(&format!("{}/n_episodes", ns), f(all, "episodes"));
        if f(all, "episodes") > 0.0 {
            a.last(&format!("{}/success_rate", ns), f(all, "success"));
            a.last(&format!("{}/collision_rate", ns), f(all, "collision"));
            a.last(&format!("{}/timeout_rate", ns), f(all, "timeout"));
            a.last(&format!("{}/mean_ep_length", ns), f(all, "steps"));
        }
        if let Some(bc) = tab.get("by_completeness").and_then(|x| x.as_object()) {
            for (k, r) in bc {
                if f(r, "episodes") > 0.0 {
                    a.last(&format!("{}/success_rate_by_completion/{}", ns, k.replace(' ', "")), f(r, "success"));
                }
            }
        }
        self.w.flush_row(wall);
        let mut rows = vec![];
        let row = |grp: &str, split: &str, r: &Value| json!({"eval": grp, "split": split, "n": r["episodes"], "success": r["success"], "collision": r["collision"], "timeout": r["timeout"], "steps": r["steps"], "steps_succ": r["steps_succ"]});
        rows.push(row("all", "all", all));
        for grp in ["by_stage", "by_completeness"] {
            if let Some(o) = tab.get(grp).and_then(|x| x.as_object()) {
                for (k, r) in o {
                    rows.push(row(grp, k, r));
                }
            }
        }
        self.w.eval(
            &format!("it{:07}_{}", self.adam_t as i64, name),
            &json!({"iter": self.adam_t, "env_steps": self.env_steps, "name": name, "driver": who, "ckpt": ckpt.map(|p| p.display().to_string()), "rows": rows}),
        );
    }

    pub fn finish(&mut self, wall: f64) {
        self.w.finish(wall, json!({}));
    }
}

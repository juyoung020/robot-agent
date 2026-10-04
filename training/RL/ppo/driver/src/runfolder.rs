// 학습 뷰어 실행 폴더 규약(docs/map_vla/TRAIN_VIEWER.md 4절) 쓰기 — ppo_run 의 기록 스레드에서만 부른다(학습 고리·GPU 와 무관).
// log.csv·events.txt 는 그대로 두고, 같은 바퀴 기록을 run.json + progress.jsonl(≤ 1 줄/s 로 합침)로도 낸다.
// 판마다 줄(episodes.jsonl)·궤적(replays/*.trp)은 내지 않는다: 장치 기록 링이 바퀴 합계만 담기 때문(판마다 기록은 CUDA 쪽 변경 — 남은 일).
use crate::{PpoConfig, PpoLog, Stage};
use serde_json::{json, Value};
use std::path::Path;
use trainfmt::RunWriter;

pub struct RunFolder {
    w: RunWriter,
    stage: usize,
    stages: Vec<String>,
    skill: String,
    mem_mb: f64,
    act_dims: i32,
    last: (f64, f64), // 마지막으로 줄을 쓴 (벽시계, 환경 스텝) — 구간 처리량
}

#[allow(clippy::too_many_arguments)]
pub fn open(out: &Path, cfg_path: &str, v: &Value, c: &PpoConfig, stages: &[Stage], window: usize, resume: Option<&str>, params: i64, dev_bytes: i64) -> Option<RunFolder> {
    if v.get("runfolder").and_then(|x| x.as_bool()) == Some(false) {
        return None;
    }
    let stem = Path::new(cfg_path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let skill = v.get("skill").and_then(|x| x.as_str()).unwrap_or("approach").to_string();
    let meta = json!({
        "kind": v.get("kind").and_then(|x| x.as_str()).unwrap_or("teacher"),
        "group": v.get("group").and_then(|x| x.as_str()).unwrap_or(&stem),
        "seed": c.seed,
        "trainer": "ppo_run",
        "config_path": std::fs::canonicalize(cfg_path).map(|p| p.display().to_string()).unwrap_or(cfg_path.to_string()),
        "config": v,
        "git": trainfmt::git_info(Path::new(env!("CARGO_MANIFEST_DIR"))),
        "init_from": resume,
        "ctrl_hz": 10,
        "n_envs": c.n_env,
        "rollout_T": c.horizon,
        "env_steps_per_iter": c.n_env as i64 * c.horizon as i64,
        "log_envs": 0,
        "skills": [skill],
        "gamma": c.gamma, "gae_lambda": c.lam, "clip_range": c.clip, "target_kl": c.kl_target, "lr": c.lr,
        "n_epochs": c.epochs, "n_minibatch": c.minibatches, "max_grad_norm": c.max_grad_norm, "log_std_init": c.init_logstd,
        "ent_coef": c.ent_coef, "vf_coef": c.vf_coef, "adaptive_lr": c.adaptive_lr, "act_dims": c.act_dims,
        "use_map": c.use_map, "goal_from_map": c.goal_from_map,
        "precision": {"fp8": v.get("fp8").and_then(|x| x.as_i64()).unwrap_or(0), "default": "bf16"},
        "stage": stages.first().map(|s| s.name.clone()).unwrap_or_default(),
        "num_params": params,
        "device_bytes": dev_bytes,
        "curriculum": {
            "stages": stages.iter().map(|s| json!({"name": s.name, "env": s.env, "map": [s.p0, s.p1], "promote": s.promote, "metric": s.metric})).collect::<Vec<_>>(),
            "window": window,
            "start_maps": ["C0", "C1", "C2"],
        },
        // 화면 기준선(8절 4번: 뷰어에 숫자를 박지 않는다)
        "refs": {"train/approx_kl": c.kl_target, "train/grad_norm": c.max_grad_norm},
        "logged": {
            "progress": true,
            "episodes": false,
            "replays": false,
            "why": "ppo_run 의 장치 기록 링은 바퀴 합계(성공·충돌·처음 지도별 판 수)만 담는다. 판마다 줄·궤적은 장치 쪽 기록 버퍼가 필요하다(CUDA 변경, TRAIN_VIEWER 남은 일)."
        },
    });
    let mut w = match RunWriter::create(out, meta, resume.is_some(), true) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("runfolder: {} — progress.jsonl 을 쓰지 않음", e);
            return None;
        }
    };
    w.every_s = v.get("progress_every_s").and_then(|x| x.as_f64()).unwrap_or(1.0);
    Some(RunFolder {
        w,
        stage: 0,
        stages: stages.iter().map(|s| s.name.clone()).collect(),
        skill,
        mem_mb: dev_bytes as f64 / 1e6,
        act_dims: c.act_dims,
        last: (0.0, 0.0),
    })
}

impl RunFolder {
    pub fn log(&mut self, l: &PpoLog, wall: f64, gpu_sps: f64) {
        let a = &mut self.w.agg;
        let n = l.n_eps as f64;
        a.last("time/iter", l.iter as f64);
        a.last("time/env_steps", l.env_steps as f64);
        a.last("time/wall", wall);
        a.mean("time/rollout_ms", l.rollout_ms as f64);
        a.mean("time/update_ms", l.update_ms as f64);
        a.mean("time/iter_ms", (l.rollout_ms + l.update_ms) as f64);
        a.mean("time/fps_gpu", gpu_sps);
        a.sum("rollout/n_eps", n);
        a.mean_w("rollout/ep_ret_mean", l.ep_ret as f64, n);
        a.mean_w("rollout/ep_len_mean", l.ep_len as f64, n);
        a.mean_w(&format!("rollout/success/{}", self.skill), l.succ as f64, n);
        a.mean_w("rollout/collision_rate", l.coll as f64, n);
        a.mean_w("rollout/timeout_rate", l.tout as f64, n);
        a.mean("rollout/step_rew_mean", l.rew_mean as f64);
        for (i, nm) in ["C0", "C1", "C2"].iter().enumerate() {
            let nc = l.n_c[i] as f64;
            a.sum(&format!("curr/n_eps/{}", nm), nc);
            a.mean_w(&format!("curr/success/{}", nm), l.s_c[i] as f64, nc);
            a.mean_w(&format!("curr/collision/{}", nm), l.k_c[i] as f64, nc);
            a.mean_w(&format!("curr/stage_frac/{}", nm), nc / n.max(1.0), n);
        }
        a.last("curr/stage_idx", self.stage as f64);
        a.last("curr/env_stage", l.stage as f64);
        a.mean("curr/goal_known", l.goal_known as f64);
        a.mean("map/task_confirmed", l.map_task as f64);
        a.mean("train/approx_kl", l.kl as f64);
        a.mean("train/clip_frac", l.clipfrac as f64);
        a.mean("train/entropy", l.entropy as f64);
        a.mean("train/policy_loss", l.pg_loss as f64);
        a.mean("train/value_loss", l.v_loss as f64);
        a.mean("train/grad_norm", l.grad_norm as f64);
        a.mean("train/lr", l.lr as f64);
        a.mean("train/adv_mean", l.adv_mean as f64);
        a.mean("train/adv_std", l.adv_std as f64);
        a.mean("train/value_mean", l.value_mean as f64);
        a.mean("train/std/vx", l.std0 as f64);
        a.mean("train/std/wz", l.std1 as f64);
        if l.std0 > 0.0 && l.std1 > 0.0 {
            // 학습하는 행동은 앞 act_dims 개(approach 는 2: vx, wz). σ 둘의 log 평균
            a.mean("train/log_std_mean", ((l.std0 as f64).ln() + (l.std1 as f64).ln()) / 2.0);
        }
        a.last("gpu/mem_used_mb", self.mem_mb);
        let _ = self.act_dims;
        if self.w.due(wall) {
            let (w0, s0) = self.last;
            if wall > w0 && self.last.0 > 0.0 {
                self.w.agg.last("time/fps_env", (l.env_steps as f64 - s0) / (wall - w0));
            }
            self.last = (wall, l.env_steps as f64);
        }
        self.w.end_update(wall);
    }

    pub fn stage(&mut self, si: usize, wall: f64) {
        self.w.flush_row(wall); // 단계 경계에서 줄을 끊는다(두 단계를 한 줄에 섞지 않음)
        self.stage = si;
        let name = self.stages.get(si).cloned().unwrap_or_default();
        self.w.set_meta("stage", json!(name));
    }

    pub fn ckpt(&mut self, path: &Path) {
        self.w.set_meta("last_ckpt", json!(path.display().to_string()));
    }

    pub fn finish(&mut self, wall: f64) {
        self.w.finish(wall, json!({}));
    }
}

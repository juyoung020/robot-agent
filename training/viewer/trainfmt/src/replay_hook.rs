//! 체크포인트마다 재생 판 자동 기록 — 학습기(ppo_run·bc_run)의 실행 폴더 쓰개가 체크포인트를 쓴 뒤 부른다.
//! 따로 된 기록 도구(training/viewer/tools/record_replay 의 record_ppo / record_bc)를 **낮은 우선순위 뒷 프로세스**로 띄운다:
//!   `setpriv --pdeathsig KILL -- nice -n 19 <도구> … --tag it<이터> --episodes K --keep-fail F --n-env N`
//! - 학습을 막지 않는다: 띄우기만 하고 기다리지 않음(끝 체크포인트만 끝에 잠깐 기다림). 하나씩만 — 앞 것이 아직 돌면 이번 것은 건너뜀(셈).
//! - 학습기가 죽으면 같이 죽는다(pdeathsig — 띄운 스레드가 끝나도). GPU 메모리는 판 수 N 으로 묶는다(기본 64 판, 수백 MB).
//! - 결과는 실행 폴더 s_eval/replays/ep_<n>_it<이터>_<스킬>_<결과>.trp + s_eval/episodes_eval.jsonl, 도구 출력은 s_eval/record.log.
//! 설정(학습 설정 JSON 의 "replays"): false 면 끔. {"episodes": 4, "failures": 2, "n_env": 64, "recorder": 경로, "every": 1(체크포인트 몇 개마다)}.
//! 도구 경로: 설정 → 환경 변수 TRAINVIEW_RECORD_PPO / TRAINVIEW_RECORD_BC → ~/ra_recbuild/record_{ppo,bc}. 없으면 조용히 끔(로그 한 줄).
//! 도구는 학습기와 **같은 소스 나무**로 빌드해야 한다(신경망·관측 배치가 같아야 체크포인트가 읽힘) — README "재생 기록".
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub struct ReplayHook {
    bin: PathBuf,
    run: PathBuf,
    kind: &'static str,
    episodes: i64,
    failures: i64,
    n_env: i64,
    every: i64,
    extra: Vec<String>,
    child: Option<(Child, String)>,
    seen: i64,
    pub launched: u32,
    pub skipped: u32,
    pub done: u32,
}

impl ReplayHook {
    /// kind: "ppo" | "bc"
    pub fn from_config(v: &Value, kind: &'static str, run: &Path) -> Option<ReplayHook> {
        let c = v.get("replays").cloned().unwrap_or(Value::Null);
        if c == Value::Bool(false) || std::env::var("TRAINVIEW_REPLAYS").map(|x| x == "0").unwrap_or(false) {
            return None;
        }
        let g = |k: &str, d: i64| c.get(k).and_then(|x| x.as_i64()).unwrap_or(d);
        let env_key = if kind == "ppo" { "TRAINVIEW_RECORD_PPO" } else { "TRAINVIEW_RECORD_BC" };
        let home = std::env::var("HOME").unwrap_or_default();
        let bin = c
            .get("recorder")
            .and_then(|x| x.as_str())
            .map(|s| s.replace('~', &home))
            .or_else(|| std::env::var(env_key).ok())
            .unwrap_or_else(|| format!("{}/ra_recbuild/record_{}", home, kind));
        let bin = PathBuf::from(bin);
        if !bin.is_file() {
            eprintln!("replays: recorder {} not found — automatic replays off (build training/viewer/tools/record_replay)", bin.display());
            return None;
        }
        Some(ReplayHook {
            bin,
            run: run.to_path_buf(),
            kind,
            episodes: g("episodes", 4),
            failures: g("failures", 2),
            n_env: g("n_env", 64),
            every: g("every", 1).max(1),
            extra: vec![],
            child: None,
            seen: 0,
            launched: 0,
            skipped: 0,
            done: 0,
        })
    }
    /// 도구에 더 줄 인자(예: --stage 1 --map 0.2 0.6). 다음 띄우기부터
    pub fn set_extra(&mut self, a: Vec<String>) {
        self.extra = a;
    }
    fn running(&mut self) -> bool {
        if let Some((c, _)) = &mut self.child {
            match c.try_wait() {
                Ok(None) => return true,
                _ => {
                    self.child = None;
                    self.done += 1;
                    let _ = fs::remove_file(self.run.join("s_eval/recording.json"));
                }
            }
        }
        false
    }
    fn spawn(&mut self, ckpt: &Path, tag: &str) {
        let d = self.run.join("s_eval");
        let _ = fs::create_dir_all(&d);
        let log = fs::OpenOptions::new().create(true).append(true).open(d.join("record.log"));
        let (o, e) = match log {
            Ok(f) => (Stdio::from(f.try_clone().unwrap()), Stdio::from(f)),
            Err(_) => (Stdio::null(), Stdio::null()),
        };
        let mut cmd = Command::new("setpriv");
        cmd.args(["--pdeathsig", "KILL", "--", "nice", "-n", "19"]).arg(&self.bin);
        cmd.arg(if self.kind == "ppo" { "--ckpt" } else { "--student" }).arg(ckpt);
        cmd.arg("--out").arg(&self.run).args(["--split", "eval", "--tag", tag]);
        cmd.args(["--episodes", &self.episodes.to_string(), "--keep-fail", &self.failures.to_string()]);
        cmd.args(["--n-env", &self.n_env.to_string(), "--track", &self.n_env.min(16).to_string()]);
        if self.kind == "bc" {
            cmd.arg("--no-images");   // 영상 학생 카메라 칸은 큼 — 자동 기록에서는 끔(손으로 record_bc 하면 켜짐)
        }
        cmd.args(&self.extra);
        cmd.stdin(Stdio::null()).stdout(o).stderr(e);
        match cmd.spawn() {
            Ok(c) => {
                // 뷰어의 "재생 기록 중" 표시: pid·시작 시각·체크포인트(끝나면 지움; 학습기가 죽어 남아도 pid 가 없으면 안 켜짐)
                let st = crate::pid_start(c.id()).unwrap_or(0);
                let _ = fs::write(d.join("recording.json"), serde_json::json!({"pid": c.id(), "pid_start": st, "tag": tag, "started": crate::now_ts()}).to_string());
                self.child = Some((c, tag.to_string()));
                self.launched += 1;
            }
            Err(err) => eprintln!("replays: cannot start {}: {}", self.bin.display(), err),
        }
    }
    /// 체크포인트 하나를 썼다. 앞 기록이 아직 돌면 건너뜀(학습을 막지 않음)
    pub fn on_ckpt(&mut self, ckpt: &Path, tag: &str) {
        self.seen += 1;
        if (self.seen - 1) % self.every != 0 {
            return;
        }
        if self.running() {
            self.skipped += 1;
            return;
        }
        self.spawn(ckpt, tag);
    }
    /// 학습 끝: 돌던 기록을 잠깐 기다리고, 끝 체크포인트를 기록하고 기다린다(학습은 이미 끝 — 막는 것 없음). 넘으면 죽임
    pub fn finish(&mut self, ckpt: Option<&Path>, tag: &str, wait: Duration) {
        let t0 = Instant::now();
        while self.running() && t0.elapsed() < wait {
            std::thread::sleep(Duration::from_millis(100));
        }
        if self.running() {
            if let Some((c, _)) = &mut self.child {
                let _ = c.kill();
                let _ = c.wait();
            }
            self.child = None;
        }
        if let Some(p) = ckpt {
            self.spawn(p, tag);
            let t1 = Instant::now();
            while self.running() && t1.elapsed() < wait {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        if let Some((c, _)) = &mut self.child {
            let _ = c.kill();
            let _ = c.wait();
            self.child = None;
        }
        let _ = fs::remove_file(self.run.join("s_eval/recording.json"));
    }
}

impl Drop for ReplayHook {
    fn drop(&mut self) {
        if let Some((c, _)) = &mut self.child {
            let _ = c.kill();
            let _ = c.wait();
        }
        let _ = fs::remove_file(self.run.join("s_eval/recording.json"));
    }
}

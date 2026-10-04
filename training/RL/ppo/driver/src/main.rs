// RL 교사 PPO 실행기(계획서 GPU_TRAINING.md 1.1: Rust = 설정·실행 순서·커리큘럼 값·로그·체크포인트).
// 계산은 모두 libppo(C++/CUDA, 그래프 둘). 이 프로그램은 바퀴를 띄우고, 끝난 바퀴의 기록을 이벤트로 확인해 꺼내고(기다리지 않음),
// 기록·체크포인트 파일 쓰기는 따로 된 스레드가 한다.
//
//   ppo_run <config.json> [--out DIR] [--minutes M] [--resume CKPT]
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PpoConfig {
    n_env: i32,
    horizon: i32,
    epochs: i32,
    minibatches: i32,
    stage: i32,
    use_map: i32,
    adaptive_lr: i32,
    use_graphs: i32,
    log_ring: i32,
    dw_chunk: i32,
    seed: u64,
    gamma: f32,
    lam: f32,
    clip: f32,
    vclip: f32,
    vf_coef: f32,
    ent_coef: f32,
    lr: f32,
    lr_min: f32,
    lr_max: f32,
    kl_target: f32,
    max_grad_norm: f32,
    adam_b1: f32,
    adam_b2: f32,
    adam_eps: f32,
    init_logstd: f32,
    reward_scale: f32,
    act_dims: i32,
    shape_coef: f32,
    shape_near: f32,
    shape_aim: f32,
    shape_zone: f32,
    shape_v: f32,
    shape_w: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct PpoLog {
    iter: i64,
    env_steps: i64,
    rollout_ms: f32,
    update_ms: f32,
    rew_mean: f32,
    ep_ret: f32,
    ep_len: f32,
    n_eps: f32,
    succ: f32,
    coll: f32,
    tout: f32,
    kl: f32,
    clipfrac: f32,
    entropy: f32,
    pg_loss: f32,
    v_loss: f32,
    grad_norm: f32,
    lr: f32,
    adv_mean: f32,
    adv_std: f32,
    value_mean: f32,
    std0: f32,
    std1: f32,
    map_task: f32,
    stage: i32,
    pad: i32,
}

#[allow(dead_code)]
extern "C" {
    fn ppo_create(cfg: *const PpoConfig) -> *mut std::ffi::c_void;
    fn ppo_destroy(h: *mut std::ffi::c_void);
    fn ppo_iterate(h: *mut std::ffi::c_void) -> i32;
    fn ppo_poll(h: *mut std::ffi::c_void, out: *mut PpoLog) -> i32;
    fn ppo_inflight(h: *mut std::ffi::c_void) -> i32;
    fn ppo_set_stage(h: *mut std::ffi::c_void, stage: i32) -> i32;
    fn ppo_ckpt_begin(h: *mut std::ffi::c_void) -> i32;
    fn ppo_ckpt_poll(h: *mut std::ffi::c_void, data: *mut *const u8, nbytes: *mut i64) -> i32;
    fn ppo_load(h: *mut std::ffi::c_void, data: *const u8, nbytes: i64) -> i32;
    fn ppo_num_params(h: *mut std::ffi::c_void) -> i64;
    fn ppo_device_bytes(h: *mut std::ffi::c_void) -> i64;
}

fn gi(v: &Value, k: &str, d: i64) -> i64 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(d)
}
fn gf(v: &Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(d)
}

fn make_config(v: &Value) -> PpoConfig {
    PpoConfig {
        n_env: gi(v, "n_env", 4096) as i32,
        horizon: gi(v, "horizon", 64) as i32,
        epochs: gi(v, "epochs", 5) as i32,
        minibatches: gi(v, "minibatches", 4) as i32,
        stage: gi(v, "stage", 0) as i32,
        use_map: gi(v, "use_map", 1) as i32,
        adaptive_lr: gi(v, "adaptive_lr", 1) as i32,
        use_graphs: gi(v, "use_graphs", 1) as i32,
        log_ring: gi(v, "log_ring", 16) as i32,
        dw_chunk: gi(v, "dw_chunk", 1024) as i32,
        seed: gi(v, "seed", 1) as u64,
        gamma: gf(v, "gamma", 0.99) as f32,
        lam: gf(v, "lam", 0.95) as f32,
        clip: gf(v, "clip", 0.2) as f32,
        vclip: gf(v, "vclip", 0.2) as f32,
        vf_coef: gf(v, "vf_coef", 0.5) as f32,
        ent_coef: gf(v, "ent_coef", 0.0) as f32,
        lr: gf(v, "lr", 3e-4) as f32,
        lr_min: gf(v, "lr_min", 1e-5) as f32,
        lr_max: gf(v, "lr_max", 1e-3) as f32,
        kl_target: gf(v, "kl_target", 0.01) as f32,
        max_grad_norm: gf(v, "max_grad_norm", 1.0) as f32,
        adam_b1: gf(v, "adam_b1", 0.9) as f32,
        adam_b2: gf(v, "adam_b2", 0.999) as f32,
        adam_eps: gf(v, "adam_eps", 1e-8) as f32,
        init_logstd: gf(v, "init_logstd", -0.5) as f32,
        reward_scale: gf(v, "reward_scale", 1.0) as f32,
        act_dims: gi(v, "act_dims", 8) as i32,
        shape_coef: gf(v.get("shaping").unwrap_or(&Value::Null), "coef", 0.0) as f32,
        shape_near: gf(v.get("shaping").unwrap_or(&Value::Null), "near", 8.0) as f32,
        shape_aim: gf(v.get("shaping").unwrap_or(&Value::Null), "aim", 1.0) as f32,
        shape_zone: gf(v.get("shaping").unwrap_or(&Value::Null), "zone", 1.0) as f32,
        shape_v: gf(v.get("shaping").unwrap_or(&Value::Null), "v", 4.0) as f32,
        shape_w: gf(v.get("shaping").unwrap_or(&Value::Null), "w", 2.0) as f32,
    }
}

enum Msg {
    Log(PpoLog, f64),
    Ckpt(Vec<u8>, PathBuf),
    Note(String),
}

// 기록 스레드: CSV 한 줄씩 + print_every 마다 화면 요약, 체크포인트 파일 쓰기(임시 파일 → 이름 바꾸기)
fn writer(rx: mpsc::Receiver<Msg>, out: PathBuf, n_per_iter: f64, print_every: i64) {
    let mut csv = fs::File::create(out.join("log.csv")).expect("log.csv");
    writeln!(
        csv,
        "iter,env_steps,wall_s,stage,succ,coll,tout,n_eps,ep_ret,ep_len,rew_mean,kl,clipfrac,entropy,pg_loss,v_loss,grad_norm,lr,adv_std,value_mean,std0,std1,map_task,rollout_ms,update_ms,gpu_env_steps_per_s"
    )
    .unwrap();
    let mut notes = fs::File::create(out.join("events.txt")).expect("events.txt");
    for m in rx {
        match m {
            Msg::Log(l, wall) => {
                let gpu_sps = n_per_iter / ((l.rollout_ms + l.update_ms) as f64 * 1e-3);
                writeln!(
                    csv,
                    "{},{},{:.3},{},{:.4},{:.4},{:.4},{},{:.3},{:.1},{:.5},{:.5},{:.4},{:.4},{:.5},{:.5},{:.4},{:.3e},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.3},{:.4e}",
                    l.iter, l.env_steps, wall, l.stage, l.succ, l.coll, l.tout, l.n_eps as i64, l.ep_ret, l.ep_len, l.rew_mean, l.kl, l.clipfrac,
                    l.entropy, l.pg_loss, l.v_loss, l.grad_norm, l.lr, l.adv_std, l.value_mean, l.std0, l.std1, l.map_task, l.rollout_ms,
                    l.update_ms, gpu_sps
                )
                .unwrap();
                if l.iter % print_every == 0 || l.iter == 1 {
                    println!(
                        "it {:5}  steps {:10.3e}  t {:6.0}s  A{}  succ {:.3} coll {:.3} tout {:.3}  ret {:6.2} len {:5.1}  kl {:.4} ent {:6.3} lr {:.1e} σ {:.2}/{:.2}  map_task {:.2}  roll {:.1}+upd {:.1} ms ({:.2e} st/s)",
                        l.iter, l.env_steps as f64, wall, l.stage, l.succ, l.coll, l.tout, l.ep_ret, l.ep_len, l.kl, l.entropy, l.lr, l.std0,
                        l.std1, l.map_task, l.rollout_ms, l.update_ms, gpu_sps
                    );
                }
            }
            Msg::Ckpt(b, p) => {
                let tmp = p.with_extension("tmp");
                fs::write(&tmp, &b).expect("ckpt write");
                fs::rename(&tmp, &p).expect("ckpt rename");
                writeln!(notes, "checkpoint {}", p.display()).unwrap();
            }
            Msg::Note(s) => {
                println!("{}", s);
                writeln!(notes, "{}", s).unwrap();
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: ppo_run <config.json> [--out DIR] [--minutes M] [--resume CKPT]");
        std::process::exit(2);
    }
    let text = fs::read_to_string(&args[1]).expect("config");
    let v: Value = serde_json::from_str(&text).expect("config json");
    let mut out = PathBuf::from(v.get("out").and_then(|x| x.as_str()).unwrap_or("runs/ppo"));
    let mut minutes = gf(&v, "budget_minutes", 60.0);
    let mut resume: Option<String> = None;
    let mut a = 2;
    while a < args.len() {
        match args[a].as_str() {
            "--out" => { out = PathBuf::from(&args[a + 1]); a += 1; }
            "--minutes" => { minutes = args[a + 1].parse().unwrap(); a += 1; }
            "--resume" => { resume = Some(args[a + 1].clone()); a += 1; }
            _ => {}
        }
        a += 1;
    }
    fs::create_dir_all(&out).expect("out dir");
    fs::write(out.join("config.json"), &text).unwrap();
    let cfg = make_config(&v);
    let cur = v.get("curriculum").cloned().unwrap_or(Value::Null);
    let stages: Vec<i32> = cur.get("stages").and_then(|x| x.as_array()).map(|a| a.iter().map(|s| s.as_i64().unwrap() as i32).collect()).unwrap_or(vec![cfg.stage]);
    let promote = gf(&cur, "promote_success", 0.8);
    let window = gi(&cur, "window", 20) as usize;
    let stop_success = gf(&cur, "stop_success", 2.0);
    let ckpt_every = gi(&v, "ckpt_every", 200);
    let print_every = gi(&v, "print_every", 10);
    let depth = gi(&v, "inflight", 3) as i32;

    let mut c = cfg;
    c.stage = stages[0];
    let h = unsafe { ppo_create(&c) };
    let n_per_iter = (c.n_env as f64) * (c.horizon as f64);
    let (tx, rx) = mpsc::channel::<Msg>();
    let out_w = out.clone();
    let wt = std::thread::spawn(move || writer(rx, out_w, n_per_iter, print_every));
    unsafe {
        tx.send(Msg::Note(format!(
            "ppo_run: N={} T={} epochs={} minibatches={} params={} device {:.2} GB, stages {:?}, budget {} min, out {}",
            c.n_env, c.horizon, c.epochs, c.minibatches, ppo_num_params(h), ppo_device_bytes(h) as f64 / 1e9, stages, minutes, out.display()
        )))
        .unwrap();
    }
    if let Some(p) = &resume {
        let b = fs::read(p).expect("resume file");
        let r = unsafe { ppo_load(h, b.as_ptr(), b.len() as i64) };
        assert_eq!(r, 0, "resume failed (size/format)");
        tx.send(Msg::Note(format!("resumed from {}", p))).unwrap();
    }

    let t0 = Instant::now();
    let budget = Duration::from_secs_f64(minutes * 60.0);
    let mut si = 0usize;
    let mut win: Vec<f32> = Vec::new();
    let mut pending_stage = false;
    let mut ckpt_pending = false;
    let mut last_iter = 0i64;
    let mut stop = false;
    let mut log = PpoLog::default();
    let ckpt_path = |it: i64| out.join(format!("ckpt_{:06}.bin", it));
    loop {
        let timeup = t0.elapsed() > budget;
        if !timeup && !stop && !pending_stage {
            while unsafe { ppo_inflight(h) } < depth {
                if unsafe { ppo_iterate(h) } != 0 {
                    break;
                }
            }
        }
        let mut got = false;
        while unsafe { ppo_poll(h, &mut log) } == 1 {
            got = true;
            last_iter = log.iter;
            tx.send(Msg::Log(log, t0.elapsed().as_secs_f64())).unwrap();
            if log.n_eps > 0.0 {
                win.push(log.succ);
                if win.len() > window {
                    win.remove(0);
                }
            }
            let avg = if win.len() == window { win.iter().sum::<f32>() / window as f32 } else { 0.0 };
            if si + 1 < stages.len() && avg as f64 >= promote && !pending_stage {
                pending_stage = true;
                tx.send(Msg::Note(format!(
                    "curriculum: stage A{} success {:.3} (window {}) >= {} at iter {} / {} env-steps -> A{}",
                    stages[si], avg, window, promote, log.iter, log.env_steps, stages[si + 1]
                )))
                .unwrap();
            }
            if si + 1 == stages.len() && avg as f64 >= stop_success && !stop {
                stop = true;
                tx.send(Msg::Note(format!("stop: final stage success {:.3} >= {} at iter {}", avg, stop_success, log.iter))).unwrap();
            }
            if ckpt_every > 0 && log.iter % ckpt_every == 0 && !ckpt_pending {
                unsafe { ppo_ckpt_begin(h) };
                ckpt_pending = true;
            }
        }
        if ckpt_pending {
            let mut p: *const u8 = std::ptr::null();
            let mut n: i64 = 0;
            if unsafe { ppo_ckpt_poll(h, &mut p, &mut n) } == 1 {
                let b = unsafe { std::slice::from_raw_parts(p, n as usize) }.to_vec();
                tx.send(Msg::Ckpt(b, ckpt_path(last_iter))).unwrap();
                ckpt_pending = false;
            }
        }
        let idle = unsafe { ppo_inflight(h) } == 0;
        if pending_stage && idle && !ckpt_pending {
            si += 1;
            let r = unsafe { ppo_set_stage(h, stages[si]) };
            assert_eq!(r, 0);
            win.clear();
            pending_stage = false;
            tx.send(Msg::Note(format!("curriculum: now stage A{} (env + map recreated, rollout graph recaptured)", stages[si]))).unwrap();
        }
        if (timeup || stop) && idle && !ckpt_pending {
            break;
        }
        if !got {
            std::thread::sleep(Duration::from_micros(300));
        }
    }
    // 마지막 체크포인트
    unsafe { ppo_ckpt_begin(h) };
    loop {
        let mut p: *const u8 = std::ptr::null();
        let mut n: i64 = 0;
        if unsafe { ppo_ckpt_poll(h, &mut p, &mut n) } == 1 {
            let b = unsafe { std::slice::from_raw_parts(p, n as usize) }.to_vec();
            tx.send(Msg::Ckpt(b, out.join("ckpt_final.bin"))).unwrap();
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    tx.send(Msg::Note(format!("done: {} iterations, {:.1} s wall", last_iter, t0.elapsed().as_secs_f64()))).unwrap();
    drop(tx);
    wt.join().unwrap();
    unsafe { ppo_destroy(h) };
    let _ = Path::new(".");
}

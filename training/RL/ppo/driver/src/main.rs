// RL 교사 PPO 실행기(계획서 GPU_TRAINING.md 1.1: Rust = 설정·실행 순서·커리큘럼 값·로그·체크포인트).
// 계산은 모두 libppo(C++/CUDA, 그래프 둘). 이 프로그램은 바퀴를 띄우고, 끝난 바퀴의 기록을 이벤트로 확인해 꺼내고(기다리지 않음),
// 기록·체크포인트 파일 쓰기는 따로 된 스레드가 한다.
//
//   ppo_run <config.json> [--out DIR] [--minutes M] [--steps S] [--resume CKPT]   (--steps: 환경 스텝 예산, 넘으면 새 바퀴를 띄우지 않음 — G6 BF16 대 FP8 같은 예산 비교)
//
// 커리큘럼(G4, 계획서 5.5): 단계마다 G1 환경 단계(env)와 처음 지도 비율(map: [C0, C1], 나머지 C2)을 둔다. 넘어가기 = 그 단계가 재는
// 처음 지도(metric: 0 C0, 1 C1, 2 C2, -1 전체)의 에피소드 성공률(최근 window 바퀴 에피소드 가중) ≥ promote.
// 지도 비율만 바뀌면 장치 값 복사 하나(ppo_set_map_curriculum — 동기·그래프 다시 잡기 없음), 환경 단계가 바뀌면 예전처럼 ppo_set_stage.
use serde_json::Value;
mod runfolder; // 학습 뷰어 실행 폴더(run.json · progress.jsonl, TRAIN_VIEWER.md 4절) — 기록 스레드에서만
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
    goal_from_map: i32,
    map_p0: f32,
    map_p1: f32,
    map_kmin: i32,
    map_kmax: i32,
    map_reveal_r: f32,
    bound_coef: f32,
    coll_extra: f32,
    fp8: i32,
    pad_fp8: i32,
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
    n_c: [f32; 3],
    s_c: [f32; 3],
    k_c: [f32; 3],
    goal_known: f32,
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
    fn ppo_set_map_curriculum(h: *mut std::ffi::c_void, p0: f32, p1: f32, kmin: i32, kmax: i32, reveal_r: f32) -> i32;
    fn ppo_issued(h: *mut std::ffi::c_void) -> i64;
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
        goal_from_map: gi(v, "goal_from_map", 0) as i32,
        map_p0: 0.0,
        map_p1: 0.0,
        map_kmin: gi(v, "map_kmin", 3) as i32,
        map_kmax: gi(v, "map_kmax", 6) as i32,
        map_reveal_r: gf(v, "map_reveal_r", 1.5) as f32,
        bound_coef: gf(v, "bound_coef", 0.0) as f32,
        coll_extra: gf(v.get("shaping").unwrap_or(&Value::Null), "coll", 0.0) as f32,
        fp8: gi(v, "fp8", 0) as i32,
        pad_fp8: 0,
    }
}

// 커리큘럼 한 단계
#[derive(Clone, Debug)]
struct Stage {
    name: String,
    env: i32,
    p0: f32,
    p1: f32,
    promote: f64,
    metric: i32,
}

fn parse_stages(cur: &Value, default_env: i32, default_promote: f64) -> Vec<Stage> {
    let arr = match cur.get("stages").and_then(|x| x.as_array()) {
        Some(a) => a.clone(),
        None => return vec![Stage { name: format!("A{}", default_env), env: default_env, p0: 0.0, p1: 0.0, promote: default_promote, metric: -1 }],
    };
    arr.iter()
        .map(|e| {
            if let Some(k) = e.as_i64() {
                // 예전 꼴(G3): 정수 = 환경 단계, 지도는 빈 지도
                Stage { name: format!("A{}", k), env: k as i32, p0: 0.0, p1: 0.0, promote: default_promote, metric: -1 }
            } else {
                let m = e.get("map").and_then(|x| x.as_array()).cloned().unwrap_or_default();
                let p = |i: usize| m.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                Stage {
                    name: e.get("name").and_then(|x| x.as_str()).unwrap_or("?").to_string(),
                    env: gi(e, "env", default_env as i64) as i32,
                    p0: p(0),
                    p1: p(1),
                    promote: gf(e, "promote", default_promote),
                    metric: gi(e, "metric", -1) as i32,
                }
            }
        })
        .collect()
}

enum Msg {
    Log(PpoLog, f64),
    Ckpt(Vec<u8>, PathBuf),
    Note(String),
    Stage(usize, f64), // 커리큘럼 단계 번호(실행 폴더 기록용)
}

// 기록 스레드: CSV 한 줄씩 + print_every 마다 화면 요약, 체크포인트 파일 쓰기(임시 파일 → 이름 바꾸기)
fn writer(rx: mpsc::Receiver<Msg>, out: PathBuf, n_per_iter: f64, print_every: i64, mut rf: Option<runfolder::RunFolder>) {
    let mut csv = fs::File::create(out.join("log.csv")).expect("log.csv");
    writeln!(
        csv,
        "iter,env_steps,wall_s,stage,succ,coll,tout,n_eps,ep_ret,ep_len,rew_mean,kl,clipfrac,entropy,pg_loss,v_loss,grad_norm,lr,adv_std,value_mean,std0,std1,map_task,rollout_ms,update_ms,gpu_env_steps_per_s,n_c0,n_c1,n_c2,succ_c0,succ_c1,succ_c2,coll_c0,coll_c1,coll_c2,goal_known"
    )
    .unwrap();
    let mut notes = fs::File::create(out.join("events.txt")).expect("events.txt");
    for m in rx {
        match m {
            Msg::Log(l, wall) => {
                let gpu_sps = n_per_iter / ((l.rollout_ms + l.update_ms) as f64 * 1e-3);
                writeln!(
                    csv,
                    "{},{},{:.3},{},{:.4},{:.4},{:.4},{},{:.3},{:.1},{:.5},{:.5},{:.4},{:.4},{:.5},{:.5},{:.4},{:.3e},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.3},{:.4e},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    l.iter, l.env_steps, wall, l.stage, l.succ, l.coll, l.tout, l.n_eps as i64, l.ep_ret, l.ep_len, l.rew_mean, l.kl, l.clipfrac,
                    l.entropy, l.pg_loss, l.v_loss, l.grad_norm, l.lr, l.adv_std, l.value_mean, l.std0, l.std1, l.map_task, l.rollout_ms,
                    l.update_ms, gpu_sps, l.n_c[0] as i64, l.n_c[1] as i64, l.n_c[2] as i64, l.s_c[0], l.s_c[1], l.s_c[2], l.k_c[0], l.k_c[1], l.k_c[2],
                    l.goal_known
                )
                .unwrap();
                if let Some(r) = rf.as_mut() { r.log(&l, wall, gpu_sps); }
                if l.iter % print_every == 0 || l.iter == 1 {
                    println!(
                        "it {:5}  steps {:10.3e}  t {:6.0}s  A{}  succ {:.3} coll {:.3} tout {:.3}  C0/1/2 {:.2}/{:.2}/{:.2} (n {}/{}/{})  goal {:.2}  len {:5.1}  kl {:.4} lr {:.1e} σ {:.2}/{:.2}  roll {:.1}+upd {:.1} ms ({:.2e} st/s)",
                        l.iter, l.env_steps as f64, wall, l.stage, l.succ, l.coll, l.tout, l.s_c[0], l.s_c[1], l.s_c[2], l.n_c[0] as i64,
                        l.n_c[1] as i64, l.n_c[2] as i64, l.goal_known, l.ep_len, l.kl, l.lr, l.std0, l.std1, l.rollout_ms, l.update_ms, gpu_sps
                    );
                }
            }
            Msg::Ckpt(b, p) => {
                let tmp = p.with_extension("tmp");
                fs::write(&tmp, &b).expect("ckpt write");
                fs::rename(&tmp, &p).expect("ckpt rename");
                if let Some(r) = rf.as_mut() { r.ckpt(&p); }
                writeln!(notes, "checkpoint {}", p.display()).unwrap();
            }
            Msg::Note(s) => {
                println!("{}", s);
                writeln!(notes, "{}", s).unwrap();
            }
            Msg::Stage(si, wall) => if let Some(r) = rf.as_mut() { r.stage(si, wall); },
        }
    }
    if let Some(r) = rf.as_mut() { r.finish(0.0); }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: ppo_run <config.json> [--out DIR] [--minutes M] [--resume CKPT] [--seed S]");
        std::process::exit(2);
    }
    let text = fs::read_to_string(&args[1]).expect("config");
    let v: Value = serde_json::from_str(&text).expect("config json");
    let mut out = PathBuf::from(v.get("out").and_then(|x| x.as_str()).unwrap_or("runs/ppo"));
    let mut minutes = gf(&v, "budget_minutes", 60.0);
    let mut resume: Option<String> = None;
    let mut seed: Option<u64> = None;   // 설정의 seed 를 덮어씀(같은 설정으로 씨앗 여럿)
    let mut max_steps: i64 = gi(&v, "budget_steps", 0);   // 0 = 없음
    let mut a = 2;
    while a < args.len() {
        match args[a].as_str() {
            "--out" => { out = PathBuf::from(&args[a + 1]); a += 1; }
            "--minutes" => { minutes = args[a + 1].parse().unwrap(); a += 1; }
            "--resume" => { resume = Some(args[a + 1].clone()); a += 1; }
            "--seed" => { seed = Some(args[a + 1].parse().unwrap()); a += 1; }
            "--steps" => { max_steps = args[a + 1].parse::<f64>().unwrap() as i64; a += 1; }
            _ => {}
        }
        a += 1;
    }
    fs::create_dir_all(&out).expect("out dir");
    fs::write(out.join("config.json"), &text).unwrap();
    let cfg = make_config(&v);
    let cur = v.get("curriculum").cloned().unwrap_or(Value::Null);
    let promote_default = gf(&cur, "promote_success", 0.8);
    let stages = parse_stages(&cur, cfg.stage, promote_default);
    let window = gi(&cur, "window", 20) as usize;
    let stop_success = gf(&cur, "stop_success", 2.0);
    let ckpt_every = gi(&v, "ckpt_every", 200);
    let print_every = gi(&v, "print_every", 10);
    let depth = gi(&v, "inflight", 3) as i32;

    let mut c = cfg;
    if let Some(s) = seed { c.seed = s; }
    c.stage = stages[0].env;
    c.map_p0 = stages[0].p0;
    c.map_p1 = stages[0].p1;
    let h = unsafe { ppo_create(&c) };
    let n_per_iter = (c.n_env as f64) * (c.horizon as f64);
    let (tx, rx) = mpsc::channel::<Msg>();
    let out_w = out.clone();
    let rf = runfolder::open(&out, &args[1], &v, &c, &stages, window, resume.as_deref(), unsafe { ppo_num_params(h) }, unsafe { ppo_device_bytes(h) });
    let wt = std::thread::spawn(move || writer(rx, out_w, n_per_iter, print_every, rf));
    unsafe {
        tx.send(Msg::Note(format!(
            "ppo_run: N={} T={} epochs={} minibatches={} params={} device {:.2} GB, goal_from_map {}, use_map {}, budget {} min, out {}\n  stages {:?}",
            c.n_env, c.horizon, c.epochs, c.minibatches, ppo_num_params(h), ppo_device_bytes(h) as f64 / 1e9, c.goal_from_map, c.use_map, minutes,
            out.display(), stages
        )))
        .unwrap();
    }
    let mut si = gi(&v, "start_stage", 0) as usize;   // 이어 하기 때 시작 단계
    if let Some(p) = &resume {
        let b = fs::read(p).expect("resume file");
        let r = unsafe { ppo_load(h, b.as_ptr(), b.len() as i64) };
        assert_eq!(r, 0, "resume failed (size/format)");
        tx.send(Msg::Note(format!("resumed from {}", p))).unwrap();
    }
    if si > 0 {
        unsafe {
            if stages[si].env != stages[0].env {
                assert_eq!(ppo_set_stage(h, stages[si].env), 0);
            }
            ppo_set_map_curriculum(h, stages[si].p0, stages[si].p1, c.map_kmin, c.map_kmax, c.map_reveal_r);
        }
        tx.send(Msg::Note(format!("start at stage {} ({:?})", si, stages[si]))).unwrap();
        tx.send(Msg::Stage(si, 0.0)).unwrap();
    }

    let t0 = Instant::now();
    let budget = Duration::from_secs_f64(minutes * 60.0);
    let mut win: Vec<(f64, f64)> = Vec::new();   // (성공 수, 에피소드 수) — 바퀴마다
    let mut pending_env = false;
    let mut ignore_upto = 0i64;   // 지도 비율을 바꾼 뒤 이미 띄워 둔 바퀴(예전 비율)의 기록은 넘어가기 판단에 쓰지 않음
    let mut ckpt_pending = false;
    let mut last_iter = 0i64;
    let mut stop = false;
    let mut log = PpoLog::default();
    let ckpt_path = |it: i64| out.join(format!("ckpt_{:06}.bin", it));
    loop {
        let timeup = t0.elapsed() > budget || (max_steps > 0 && log.env_steps >= max_steps);
        if !timeup && !stop && !pending_env {
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
            if log.iter <= ignore_upto || pending_env {
                continue;
            }
            let st = &stages[si];
            let (ns, ne) = if st.metric < 0 {
                ((log.succ * log.n_eps) as f64, log.n_eps as f64)
            } else {
                let m = st.metric as usize;
                ((log.s_c[m] * log.n_c[m]) as f64, log.n_c[m] as f64)
            };
            if ne > 0.0 {
                win.push((ns, ne));
                if win.len() > window {
                    win.remove(0);
                }
            }
            let avg = if win.len() == window { win.iter().map(|x| x.0).sum::<f64>() / win.iter().map(|x| x.1).sum::<f64>().max(1.0) } else { 0.0 };
            if si + 1 < stages.len() && avg >= st.promote {
                let nx = stages[si + 1].clone();
                tx.send(Msg::Note(format!(
                    "curriculum: stage {} success {:.3} (metric C{}, window {}) >= {} at iter {} / {} env-steps / {:.0} s -> {}",
                    st.name, avg, st.metric, window, st.promote, log.iter, log.env_steps, t0.elapsed().as_secs_f64(), nx.name
                )))
                .unwrap();
                win.clear();
                if nx.env != st.env {
                    pending_env = true;   // 환경을 새로 만들어야 함: 띄운 바퀴가 다 끝난 뒤
                } else {
                    si += 1;
                    tx.send(Msg::Stage(si, t0.elapsed().as_secs_f64())).unwrap();
                    unsafe { ppo_set_map_curriculum(h, nx.p0, nx.p1, c.map_kmin, c.map_kmax, c.map_reveal_r) };
                    ignore_upto = unsafe { ppo_issued(h) };
                    tx.send(Msg::Note(format!("curriculum: now {} (first map C0 {:.2} C1 {:.2} C2 {:.2}: device value, no sync, no recapture; from iter {})",
                        nx.name, nx.p0, nx.p1, 1.0 - nx.p0 - nx.p1, ignore_upto + 1))).unwrap();
                }
            }
            if si + 1 == stages.len() && avg >= stop_success && !stop {
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
        if pending_env && idle && !ckpt_pending {
            si += 1;
            tx.send(Msg::Stage(si, t0.elapsed().as_secs_f64())).unwrap();
            let nx = &stages[si];
            let r = unsafe { ppo_set_stage(h, nx.env) };
            assert_eq!(r, 0);
            unsafe { ppo_set_map_curriculum(h, nx.p0, nx.p1, c.map_kmin, c.map_kmax, c.map_reveal_r) };
            win.clear();
            pending_env = false;
            tx.send(Msg::Note(format!("curriculum: now {} (env A{} recreated, rollout graph recaptured; first map C0 {:.2} C1 {:.2})", nx.name, nx.env, nx.p0, nx.p1))).unwrap();
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

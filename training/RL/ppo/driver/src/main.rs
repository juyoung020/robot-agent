// RL 교사 PPO 실행기(계획서 GPU_TRAINING.md 1.1: Rust = 설정·실행 순서·커리큘럼 값·로그·체크포인트).
// 계산은 모두 libppo(C++/CUDA, 그래프 둘). 이 프로그램은 바퀴를 띄우고, 끝난 바퀴의 기록을 이벤트로 확인해 꺼내고(기다리지 않음),
// 기록·체크포인트 파일 쓰기는 따로 된 스레드가 한다.
//
//   ppo_run <config.json> [--out DIR] [--minutes M] [--steps S] [--resume CKPT]   (--steps: 환경 스텝 예산, 넘으면 새 바퀴를 띄우지 않음 — G6 BF16 대 FP8 같은 예산 비교)
//
// 커리큘럼(G4, 계획서 5.5): 단계마다 G1 환경 단계(env)와 처음 지도 비율(map: [C0, C1], 나머지 C2)을 둔다. 넘어가기 = 그 단계가 재는
// 처음 지도(metric: 0 C0, 1 C1, 2 C2, -1 전체)의 에피소드 성공률(최근 window 바퀴 에피소드 가중) ≥ promote.
// 넘어가기 판단은 장치가 한다(ppo_curr_*: 갱신 그래프 끝 커널이 창·문턱을 보고 처음 지도 비율·행동 비트·BEHAVIOR 값을 바로 바꿈 —
// 호스트 왕복·지연 없음, 띄운 바퀴 수와 무관하게 결정적). 환경 단계가 바뀌어도 장치가 다음 바퀴 롤아웃 앞에서 모든 판을 새로 시작한다(evt 3,
// 다시 만들기·그래프 다시 잡기 없음). 예전 판(evt 1: 호스트가 ppo_set_stage → ppo_curr_ack)도 처리는 남김.
use serde_json::Value;
mod runfolder; // 학습 뷰어 실행 폴더(run.json · progress.jsonl, TRAIN_VIEWER.md 4절) — 기록 스레드에서만
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

// BEHAVIOR 커리큘럼 장치 값(ppo_capi.h PpoBCurr = env bsc::BCurr)
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct PpoBCurr {
    p1: f32,
    p2: f32,
    scene_mask: u32,
    split: i32,
    yaw_jit: f32,
    strict: i32,
    nofilter: i32,
    eval_instr: i32,
    p_point: f32,
    p_goto: f32,
    // 잡기 물리(E6): B4·B5·B6 비율, 실패 판(미끄러짐·막힌 자리), 물리 끄기 비트
    p4: f32,
    p5: f32,
    p6: f32,
    p_slip: f32,
    p_occ: f32,
    phys: i32,
    p_cov: f32,   // 지도 쌓기(커리큘럼 1단계) 섞음
}

#[repr(C)]
#[derive(Clone, Copy)]
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
    // VLA_INPUT 5·6절: 행동 가림(장치 값)과 학습 때 흔들기(observation/obs.h ObsAug)
    act_mask: u32,
    aug_on: i32,
    aug_vel_sigma: f32,
    aug_prev_drop: f32,
    aug_prev_sigma: f32,
    aug_p_erase: f32,
    aug_p_syn: f32,
    aug_p_hyper: f32,
    aug_p_wrong: f32,
    aug_p_slot_drop: f32,
    aug_p_map_off: f32,
    aug_eval_unseen: i32,
    // E2 BEHAVIOR(env 단계 3): 장면 묶음·커리큘럼 시작 값
    beh: i32,
    map_nav_k: i32,
    bcurr: PpoBCurr,
    b_scenes: [u8; 256],
    b_rasc_dir: [u8; 256],
    env_stages: u32,
    pad_es: i32,
    bc_coef: f32,    // 대본 교사 모방 보조 손실 λ0(잡기 판)
    bc_decay: i32,   // λ 를 0 까지 줄이는 바퀴 수
    teach_drive: f32,   // 잡기 판 중 대본 교사가 모는 비율(판 단위, bc_decay 로 줄임)
    pad_td: i32,
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
    n_b: [f32; 3],
    s_b: [f32; 3],
    k_b: [f32; 3],
    n_p: [f32; 3],
    s_p: [f32; 3],
    k_p: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PpoCurrStage {
    env: i32,
    p0: f32,
    p1: f32,
    promote: f32,
    metric: i32,
    act_mask: u32,
    b_set: i32,
    bcurr: PpoBCurr,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct PpoCurrLog {
    si: i32,
    req: i32,
    evt: i32,
    n_win: i32,
    avg: f32,
    pad: [f32; 3],
}

#[allow(dead_code)]
extern "C" {
    fn ppo_curr_set(h: *mut std::ffi::c_void, st: *const PpoCurrStage, n: i32, window: i32, start: i32) -> i32;
    fn ppo_curr_ack(h: *mut std::ffi::c_void, si: i32) -> i32;
    fn ppo_curr_log(h: *mut std::ffi::c_void, out: *mut PpoCurrLog) -> i32;
    fn ppo_create(cfg: *const PpoConfig) -> *mut std::ffi::c_void;
    fn ppo_destroy(h: *mut std::ffi::c_void);
    fn ppo_iterate(h: *mut std::ffi::c_void) -> i32;
    fn ppo_poll(h: *mut std::ffi::c_void, out: *mut PpoLog) -> i32;
    fn ppo_inflight(h: *mut std::ffi::c_void) -> i32;
    fn ppo_set_stage(h: *mut std::ffi::c_void, stage: i32) -> i32;
    fn ppo_ckpt_begin(h: *mut std::ffi::c_void) -> i32;
    fn ppo_ckpt_iter(h: *mut std::ffi::c_void) -> i64;
    fn ppo_ckpt_poll(h: *mut std::ffi::c_void, data: *mut *const u8, nbytes: *mut i64) -> i32;
    fn ppo_load(h: *mut std::ffi::c_void, data: *const u8, nbytes: i64) -> i32;
    fn ppo_set_map_curriculum(h: *mut std::ffi::c_void, p0: f32, p1: f32, kmin: i32, kmax: i32, reveal_r: f32) -> i32;
    fn ppo_set_act_mask(h: *mut std::ffi::c_void, mask: u32) -> i32;
    fn ppo_issued(h: *mut std::ffi::c_void) -> i64;
    fn ppo_num_params(h: *mut std::ffi::c_void) -> i64;
    fn ppo_device_bytes(h: *mut std::ffi::c_void) -> i64;
    fn ppo_set_bcurr(h: *mut std::ffi::c_void, b: *const PpoBCurr) -> i32;
    fn ppo_struct_size(which: i32) -> i64;
    fn ppo_scene_mask(h: *mut std::ffi::c_void, names: *const std::os::raw::c_char) -> u32;
    fn ppo_scene_name(h: *mut std::ffi::c_void, i: i32) -> *const std::os::raw::c_char;
}

fn cstr256(s: &str) -> [u8; 256] {
    let mut b = [0u8; 256];
    let n = s.len().min(255);
    b[..n].copy_from_slice(&s.as_bytes()[..n]);
    b
}

// BEHAVIOR 커리큘럼 값(설정): 전역 "beh" 기본 + 단계 "b" 덮어쓰기. 장면은 이름 목록(비면 장면 묶음 전부) — 비트는 학습기를 만든 뒤 ppo_scene_mask 로
#[derive(Clone, Debug, Default)]
struct BSpec {
    p1: f32,
    p2: f32,
    scenes: Vec<String>,
    split: i32,
    yaw_jit: f32,
    strict: i32,
    eval_instr: i32,
    p_point: f32,
    p_goto: f32,
    pnp: [f32; 3],
    fail: [f32; 2],
    feas: i32,
    p_cov: f32,
}
fn parse_bspec(v: &Value, base: &BSpec) -> BSpec {
    let mut b = base.clone();
    if let Some(m) = v.get("mix").and_then(|x| x.as_array()) {
        b.p1 = m.first().and_then(|x| x.as_f64()).unwrap_or(b.p1 as f64) as f32;
        b.p2 = m.get(1).and_then(|x| x.as_f64()).unwrap_or(b.p2 as f64) as f32;
    }
    if let Some(a) = v.get("scenes").and_then(|x| x.as_array()) {
        b.scenes = a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect();
    }
    b.split = gi(v, "split", b.split as i64) as i32;
    b.yaw_jit = gf(v, "yaw_jit", b.yaw_jit as f64) as f32;
    b.strict = gi(v, "strict", b.strict as i64) as i32;
    b.eval_instr = gi(v, "eval_instr", b.eval_instr as i64) as i32;
    b.p_point = gf(v, "p_point", b.p_point as f64) as f32;   // 목표 점(VLA_INPUT 2.1): 놓을 곳 = 점 섞음
    b.p_goto = gf(v, "p_goto", b.p_goto as f64) as f32;     // 점으로 가기(B1·B3 변형) 섞음
    if let Some(m) = v.get("pnp").and_then(|x| x.as_array()) {   // 잡기 물리(E6): [B4, B5, B6] 비율(B3 몫에서)
        for k in 0..3 { b.pnp[k] = m.get(k).and_then(|x| x.as_f64()).unwrap_or(b.pnp[k] as f64) as f32; }
    }
    if let Some(m) = v.get("fail").and_then(|x| x.as_array()) {  // 실패 판: [p_slip, p_occ]
        for k in 0..2 { b.fail[k] = m.get(k).and_then(|x| x.as_f64()).unwrap_or(b.fail[k] as f64) as f32; }
    }
    b.p_cov = gf(v, "p_cov", b.p_cov as f64) as f32;   // 지도 쌓기(B1 변형, 목표 없음) 섞음
    b.feas = gi(v, "feas", b.feas as i64) as i32;   // 1 = B4–B6 를 잡기 가능 짝에서만(env PF_FEAS = phys 8, 학습기가 시작 때 표를 만듦)
    b
}
fn bcurr_of(h: *mut std::ffi::c_void, b: &BSpec, all: u32) -> PpoBCurr {
    let mask = if b.scenes.is_empty() {
        all
    } else {
        let names = std::ffi::CString::new(b.scenes.join(",")).unwrap();
        let m = unsafe { ppo_scene_mask(h, names.as_ptr()) };
        assert!(m != 0, "beh: none of the scenes {:?} is in the scene set", b.scenes);
        m
    };
    PpoBCurr {
        p1: b.p1, p2: b.p2, scene_mask: mask, split: b.split, yaw_jit: b.yaw_jit, strict: b.strict, nofilter: 0, eval_instr: b.eval_instr, p_point: b.p_point, p_goto: b.p_goto,
        p4: b.pnp[0], p5: b.pnp[1], p6: b.pnp[2], p_slip: b.fail[0], p_occ: b.fail[1], phys: if b.feas != 0 { 8 } else { 0 }, p_cov: b.p_cov,
    }
}

fn env_name(e: i32) -> String {
    if e >= 3 { "B".to_string() } else { format!("A{}", e) }   // 3 = BEHAVIOR 집(B1–B3), 0–2 = 상자 방 A0–A2(= B0)
}
fn metric_name(m: i32) -> String {
    match m {
        0..=2 => format!("C{}", m),
        3..=5 => format!("B{}", m - 2),
        _ => "all".to_string(),
    }
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
        act_mask: gi(v, "act_mask", 0) as u32,
        aug_on: gi(v.get("aug").unwrap_or(&Value::Null), "on", 0) as i32,
        aug_vel_sigma: gf(v.get("aug").unwrap_or(&Value::Null), "vel_sigma", 0.0) as f32,
        aug_prev_drop: gf(v.get("aug").unwrap_or(&Value::Null), "prev_drop", 0.0) as f32,
        aug_prev_sigma: gf(v.get("aug").unwrap_or(&Value::Null), "prev_sigma", 0.0) as f32,
        aug_p_erase: gf(v.get("aug").unwrap_or(&Value::Null), "p_erase", 0.0) as f32,
        aug_p_syn: gf(v.get("aug").unwrap_or(&Value::Null), "p_syn", 0.0) as f32,
        aug_p_hyper: gf(v.get("aug").unwrap_or(&Value::Null), "p_hyper", 0.0) as f32,
        aug_p_wrong: gf(v.get("aug").unwrap_or(&Value::Null), "p_wrong", 0.0) as f32,
        aug_p_slot_drop: gf(v.get("aug").unwrap_or(&Value::Null), "p_slot_drop", 0.0) as f32,
        aug_p_map_off: gf(v.get("aug").unwrap_or(&Value::Null), "p_map_off", 0.0) as f32,
        aug_eval_unseen: gi(v.get("aug").unwrap_or(&Value::Null), "eval_unseen", 0) as i32,
        beh: 0,
        map_nav_k: gi(v.get("beh").unwrap_or(&Value::Null), "nav_k", 0) as i32,
        bcurr: PpoBCurr::default(),
        b_scenes: cstr256(&v.get("beh").and_then(|b| b.get("build")).and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(",")).unwrap_or_default()),
        b_rasc_dir: cstr256(v.get("beh").and_then(|b| b.get("rasc_dir")).and_then(|x| x.as_str()).unwrap_or("")),
        env_stages: 0,
        pad_es: 0,
        bc_coef: gf(v, "bc_coef", 0.0) as f32,
        bc_decay: gi(v, "bc_decay", 1000) as i32,
        teach_drive: gf(v, "teach_drive", 0.0) as f32,
        pad_td: 0,
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
    act_mask: u32, // 0 = 바꾸지 않음(설정 act_mask/act_dims 그대로)
    b: Option<BSpec>, // env 3(BEHAVIOR) 단계만: 그 단계의 B1/B2 비율·장면·split·엄격·지시문
}

fn parse_stages(cur: &Value, default_env: i32, default_promote: f64, bbase: &BSpec) -> Vec<Stage> {
    let bfor = |env: i32, e: &Value| if env >= 3 { Some(parse_bspec(e.get("b").unwrap_or(&Value::Null), bbase)) } else { None };
    let arr = match cur.get("stages").and_then(|x| x.as_array()) {
        Some(a) => a.clone(),
        None => return vec![Stage { name: format!("A{}", default_env), env: default_env, p0: 0.0, p1: 0.0, promote: default_promote, metric: -1, act_mask: 0, b: bfor(default_env, &Value::Null) }],
    };
    arr.iter()
        .map(|e| {
            if let Some(k) = e.as_i64() {
                // 예전 꼴(G3): 정수 = 환경 단계, 지도는 빈 지도
                Stage { name: format!("A{}", k), env: k as i32, p0: 0.0, p1: 0.0, promote: default_promote, metric: -1, act_mask: 0, b: bfor(k as i32, &Value::Null) }
            } else {
                let m = e.get("map").and_then(|x| x.as_array()).cloned().unwrap_or_default();
                let p = |i: usize| m.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                let env = gi(e, "env", default_env as i64) as i32;
                Stage {
                    name: e.get("name").and_then(|x| x.as_str()).unwrap_or("?").to_string(),
                    env: gi(e, "env", default_env as i64) as i32,
                    p0: p(0),
                    p1: p(1),
                    promote: gf(e, "promote", default_promote),
                    metric: gi(e, "metric", -1) as i32,
                    act_mask: gi(e, "act_mask", 0) as u32,
                    b: bfor(env, e),
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
        "iter,env_steps,wall_s,stage,succ,coll,tout,n_eps,ep_ret,ep_len,rew_mean,kl,clipfrac,entropy,pg_loss,v_loss,grad_norm,lr,adv_std,value_mean,std0,std1,map_task,rollout_ms,update_ms,gpu_env_steps_per_s,n_c0,n_c1,n_c2,succ_c0,succ_c1,succ_c2,coll_c0,coll_c1,coll_c2,goal_known,n_b1,n_b2,n_b3,succ_b1,succ_b2,succ_b3,coll_b1,coll_b2,coll_b3,n_b4,n_b5,n_b6,succ_b4,succ_b5,succ_b6,coll_b4,coll_b5,coll_b6"
    )
    .unwrap();
    let mut notes = fs::File::create(out.join("events.txt")).expect("events.txt");
    for m in rx {
        match m {
            Msg::Log(l, wall) => {
                let gpu_sps = n_per_iter / ((l.rollout_ms + l.update_ms) as f64 * 1e-3);
                writeln!(
                    csv,
                    "{},{},{:.3},{},{:.4},{:.4},{:.4},{},{:.3},{:.1},{:.5},{:.5},{:.4},{:.4},{:.5},{:.5},{:.4},{:.3e},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.3},{:.4e},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    l.iter, l.env_steps, wall, l.stage, l.succ, l.coll, l.tout, l.n_eps as i64, l.ep_ret, l.ep_len, l.rew_mean, l.kl, l.clipfrac,
                    l.entropy, l.pg_loss, l.v_loss, l.grad_norm, l.lr, l.adv_std, l.value_mean, l.std0, l.std1, l.map_task, l.rollout_ms,
                    l.update_ms, gpu_sps, l.n_c[0] as i64, l.n_c[1] as i64, l.n_c[2] as i64, l.s_c[0], l.s_c[1], l.s_c[2], l.k_c[0], l.k_c[1], l.k_c[2],
                    l.goal_known, l.n_b[0] as i64, l.n_b[1] as i64, l.n_b[2] as i64, l.s_b[0], l.s_b[1], l.s_b[2], l.k_b[0], l.k_b[1], l.k_b[2],
                    l.n_p[0] as i64, l.n_p[1] as i64, l.n_p[2] as i64, l.s_p[0], l.s_p[1], l.s_p[2], l.k_p[0], l.k_p[1], l.k_p[2]
                )
                .unwrap();
                if let Some(r) = rf.as_mut() { r.log(&l, wall, gpu_sps); }
                if l.iter % print_every == 0 || l.iter == 1 {
                    println!(
                        "it {:5}  steps {:10.3e}  t {:6.0}s  {}  succ {:.3} coll {:.3} tout {:.3}  C0/1/2 {:.2}/{:.2}/{:.2} (n {}/{}/{})  goal {:.2}  len {:5.1}  kl {:.4} lr {:.1e} σ {:.2}/{:.2}  roll {:.1}+upd {:.1} ms ({:.2e} st/s)",
                        l.iter, l.env_steps as f64, wall, env_name(l.stage), l.succ, l.coll, l.tout, l.s_c[0], l.s_c[1], l.s_c[2], l.n_c[0] as i64,
                        l.n_c[1] as i64, l.n_c[2] as i64, l.goal_known, l.ep_len, l.kl, l.lr, l.std0, l.std1, l.rollout_ms, l.update_ms, gpu_sps
                    );
                    if l.n_b.iter().sum::<f32>() > 0.0 {
                        println!(
                            "          BEHAVIOR B1/B2/B3 succ {:.3}/{:.3}/{:.3} coll {:.3}/{:.3}/{:.3} (n {}/{}/{})",
                            l.s_b[0], l.s_b[1], l.s_b[2], l.k_b[0], l.k_b[1], l.k_b[2], l.n_b[0] as i64, l.n_b[1] as i64, l.n_b[2] as i64
                        );
                    }
                    if l.n_p.iter().sum::<f32>() > 0.0 {
                        println!(
                            "          PICK/PLACE B4/B5/B6 succ {:.3}/{:.3}/{:.3} coll {:.3}/{:.3}/{:.3} (n {}/{}/{})",
                            l.s_p[0], l.s_p[1], l.s_p[2], l.k_p[0], l.k_p[1], l.k_p[2], l.n_p[0] as i64, l.n_p[1] as i64, l.n_p[2] as i64
                        );
                    }
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
    unsafe {   // C ABI 구조체 배치 확인(ppo_capi.h 와 이 파일이 어긋나면 바로 멈춤)
        use std::mem::size_of;
        for (k, n, name) in [(0, size_of::<PpoConfig>(), "PpoConfig"), (1, size_of::<PpoLog>(), "PpoLog"), (2, size_of::<PpoCurrStage>(), "PpoCurrStage"),
                             (3, size_of::<PpoCurrLog>(), "PpoCurrLog"), (4, size_of::<PpoBCurr>(), "PpoBCurr")] {
            assert_eq!(ppo_struct_size(k) as usize, n, "C ABI size mismatch: {}", name);
        }
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
    // BEHAVIOR(env 3) 기본값: CURRICULUM_BEHAVIOR2026 5.4 의 env_verify 기본 섞음(B1 0.34·B2 0.33·B3 나머지), 학습 인스턴스, 느슨 거르개
    let bbase = parse_bspec(v.get("beh").unwrap_or(&Value::Null), &BSpec { p1: 0.34, p2: 0.33, ..Default::default() });
    let stages = parse_stages(&cur, cfg.stage, promote_default, &bbase);
    // GT 지도 막기(2026-10-06 사용자 결정 — 학습·평가는 빈 지도 C2 에서 자라는 지도만): 처음 지도 C0/C1(미리 채움)은 --debug-gt-map 일 때만
    if stages.iter().any(|s| s.p0 > 0.0 || s.p1 > 0.0) && !args.iter().any(|x| x == "--debug-gt-map") {
        eprintln!("ppo_run: config asks for a prefilled (GT) start map (stage map [p0, p1] > 0). Training uses the growing map only (C2, CURRICULUM_BEHAVIOR2026 5.7). Pass --debug-gt-map to run it anyway (debug only).");
        std::process::exit(2);
    }
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
    c.beh = if stages.iter().any(|s| s.env >= 3) || gi(v.get("beh").unwrap_or(&Value::Null), "on", 0) != 0 { 1 } else { 0 };
    c.env_stages = stages.iter().fold(0u32, |m, s| m | (1u32 << s.env));   // 장치 단계 바꾸기가 띄울 환경 커널 무리
    let h = unsafe { ppo_create(&c) };
    // BEHAVIOR 단계마다 장치 커리큘럼 값(장면 이름 → 장면 묶음 비트는 학습기가 묶음을 만든 뒤에야 앎)
    let mut nsc = 0;
    while !unsafe { ppo_scene_name(h, nsc) }.is_null() { nsc += 1; }
    let all_mask: u32 = if nsc >= 32 { u32::MAX } else { (1u32 << nsc) - 1 };
    let b_of: Vec<Option<PpoBCurr>> = stages.iter().map(|s| s.b.as_ref().map(|b| bcurr_of(h, b, all_mask))).collect();
    if let Some(b) = &b_of[0] { unsafe { ppo_set_bcurr(h, b) }; }
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
            if stages[si].act_mask != 0 { ppo_set_act_mask(h, stages[si].act_mask); }
            if let Some(b) = &b_of[si] { ppo_set_bcurr(h, b); }
        }
        tx.send(Msg::Note(format!("start at stage {} ({:?})", si, stages[si]))).unwrap();
        tx.send(Msg::Stage(si, 0.0)).unwrap();
    }

    {
        // 장치 커리큘럼: 단계 표·창을 장치에(시작 때 한 번). 판단은 갱신 그래프 끝 커널이 바퀴마다
        let tab: Vec<PpoCurrStage> = stages
            .iter()
            .zip(b_of.iter())
            .map(|(s, b)| PpoCurrStage {
                env: s.env, p0: s.p0, p1: s.p1, promote: s.promote as f32, metric: s.metric, act_mask: s.act_mask,
                b_set: b.is_some() as i32, bcurr: b.unwrap_or_default(),
            })
            .collect();
        let r = unsafe { ppo_curr_set(h, tab.as_ptr(), tab.len() as i32, window as i32, si as i32) };
        assert_eq!(r, 0, "device curriculum: at most 16 stages, window 1..64");
    }
    let t0 = Instant::now();
    let budget = Duration::from_secs_f64(minutes * 60.0);
    let mut pending_env: Option<usize> = None;   // 장치가 요청한 환경 바꾸기(단계 번호) — 띄운 바퀴가 다 끝난 뒤 처리
    let mut ckpt_pending = false;
    let mut last_iter = 0i64;
    let mut stop = false;
    let mut log = PpoLog::default();
    let ckpt_path = |it: i64| out.join(format!("ckpt_{:06}.bin", it));
    loop {
        let timeup = t0.elapsed() > budget || (max_steps > 0 && log.env_steps >= max_steps);
        if !timeup && !stop && pending_env.is_none() {
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
            let mut cl = PpoCurrLog::default();
            unsafe { ppo_curr_log(h, &mut cl) };
            if cl.evt != 0 {
                let st = &stages[si];
                let nx = &stages[si + 1];
                tx.send(Msg::Note(format!(
                    "curriculum: stage {} success {:.3} (metric {}, window {}) >= {} at iter {} / {} env-steps / {:.0} s -> {} (decided on device)",
                    st.name, cl.avg, metric_name(st.metric), window, st.promote, log.iter, log.env_steps, t0.elapsed().as_secs_f64(), nx.name
                )))
                .unwrap();
                if cl.evt == 2 || cl.evt == 3 {
                    si = cl.si as usize;
                    tx.send(Msg::Stage(si, t0.elapsed().as_secs_f64())).unwrap();
                    let nx = &stages[si];
                    if cl.evt == 3 {
                        tx.send(Msg::Note(format!("curriculum: env {} restarted on device before iter {} (no host sync, no graph recapture)", env_name(nx.env), log.iter + 1))).unwrap();
                    }
                    tx.send(Msg::Note(format!("curriculum: now {} (first map C0 {:.2} C1 {:.2} C2 {:.2}{}: switched on device from iter {}, no host round trip)",
                        nx.name, nx.p0, nx.p1, 1.0 - nx.p0 - nx.p1,
                        b_of[si].map(|b| format!(", B1 {:.2} B2 {:.2} B3 {:.2} scenes {:#x}", b.p1, b.p2, 1.0 - b.p1 - b.p2, b.scene_mask)).unwrap_or_default(),
                        log.iter + 1))).unwrap();
                } else {
                    pending_env = Some(cl.req as usize);   // 환경을 새로 만들어야 함: 띄운 바퀴가 다 끝난 뒤
                }
            }
            if si + 1 == stages.len() && pending_env.is_none() && cl.n_win as usize == window && cl.avg as f64 >= stop_success && !stop {
                stop = true;
                tx.send(Msg::Note(format!("stop: final stage success {:.3} >= {} at iter {}", cl.avg, stop_success, log.iter))).unwrap();
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
                // 이름 = 사본에 든 바퀴 수(뜰 때 띄운 바퀴 수 — 예전에는 꺼낸 바퀴 번호라 1–3 바퀴 어긋났음)
                tx.send(Msg::Ckpt(b, ckpt_path(unsafe { ppo_ckpt_iter(h) }))).unwrap();
                ckpt_pending = false;
            }
        }
        let idle = unsafe { ppo_inflight(h) } == 0;
        if let (Some(req), true, false) = (pending_env, idle, ckpt_pending) {
            si = req;
            tx.send(Msg::Stage(si, t0.elapsed().as_secs_f64())).unwrap();
            let nx = &stages[si];
            let r = unsafe { ppo_set_stage(h, nx.env) };
            assert_eq!(r, 0);
            unsafe { ppo_set_map_curriculum(h, nx.p0, nx.p1, c.map_kmin, c.map_kmax, c.map_reveal_r) };
            if nx.act_mask != 0 { unsafe { ppo_set_act_mask(h, nx.act_mask) }; }
            if let Some(b) = &b_of[si] { unsafe { ppo_set_bcurr(h, b) }; }
            unsafe { ppo_curr_ack(h, si as i32) };
            pending_env = None;
            tx.send(Msg::Note(format!("curriculum: now {} (env {} recreated, rollout graph recaptured; first map C0 {:.2} C1 {:.2}{})", nx.name, env_name(nx.env), nx.p0, nx.p1,
                b_of[si].map(|b| format!(", B1 {:.2} B2 {:.2} scenes {:#x}", b.p1, b.p2, b.scene_mask)).unwrap_or_default()))).unwrap();
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

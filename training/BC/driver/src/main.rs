#![recursion_limit = "256"]
// BC 학생 실행기(계획서 GPU_TRAINING.md 1.1: Rust = 설정·실행 순서·로그). 계산은 모두 libbc(C++/CUDA, 그래프 둘).
//
//   bc_run <config.json> [--out DIR]
//
// 순서: (교사 평가) → 교사 기록 R0 롤아웃 → BC 갱신 U0 번 → 학생 평가 → DAgger i = 1..D [학생이 움직이며 교사가 라벨 Rd 롤아웃 → 모은 자료로 갱신 Ud 번 → 평가].
// "dagger": false 면 같은 횟수를 교사가 움직이며 기록한다(자료 양·학습 양이 같은 대조 = BC + 교사 자료 더).
// 단계 안에서는 그래프를 띄우고 끝난 기록을 이벤트로 꺼내기만 한다(기다리지 않음). 동기는 단계 경계(환경 다시 만들기, 표 읽기, 저장)뿐.
use serde_json::{json, Value};
mod runfolder; // 학습 뷰어 실행 폴더(run.json · progress.jsonl · evals, TRAIN_VIEWER.md 4절) — 기록 꺼낼 때만
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct BcConfig {
    n_env: i32,
    horizon: i32,
    stage: i32,
    use_map: i32,
    teacher_use_map: i32,
    student_goal: i32,
    use_graphs: i32,
    log_ring: i32,
    mb: i32,
    upd_steps: i32,
    dw_chunk: i32,
    store_render: i32,
    cap: i64,
    seed: u64,
    env_seed: u64,
    lr: f32,
    adam_b1: f32,
    adam_b2: f32,
    adam_eps: f32,
    max_grad_norm: f32,
    map_p0: f32,
    map_p1: f32,
    map_kmin: i32,
    map_kmax: i32,
    map_reveal_r: f32,
    fp8: i32,
    vision: i32,
    head: i32,
    chunk: i32,
    flow_steps: i32,
    text: i32,
    render_profile: i32,
    render_batch: i32,
    img_dim: i32,
    sample_render: i32,
    vit_prec: i32,
    // v2(VLA_INPUT 1–7절): 토큰마다 학생, 행동 가림, 지시 과제, 흔들기, 렌더 흔들기
    arch: i32,
    tf_d: i32,
    tf_layers: i32,
    tf_heads: i32,
    tf_mlp: i32,
    tf_elayers: i32,
    act_mask: u32,
    task: i32,
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
    render_aug: i32,
    ra_color: f32,
    ra_light: f32,
    ra_expo: f32,
    render_team_mix: f32,
    // E2 BEHAVIOR(stage 3): 장면 묶음·커리큘럼 값(bc_capi.h)
    beh: i32,
    map_nav_k: i32,
    b_p1: f32,
    b_p2: f32,
    b_scene_mask: u32,
    b_split: i32,
    b_yaw_jit: f32,
    b_strict: i32,
    b_nofilter: i32,
    b_eval_instr: i32,
    b_p_point: f32,
    b_p_goto: f32,
    goal_drop: f32,
    // 잡기 물리(E6): B4·B5·B6 비율, 실패 판, 대본 교사 라벨
    b_p4: f32,
    b_p5: f32,
    b_p6: f32,
    b_p_slip: f32,
    b_p_occ: f32,
    teacher_script: i32,
    b_feas: i32,
    b_gcand: i32,
    b_sltol: i32,
    mlp_w: i32,
    b_house_split: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct BcLog {
    seq: i64,
    kind: i32,
    actor: i32,
    record: i32,
    pad: i32,
    count: i64,
    adam_t: i64,
    loss: f32,
    grad_norm: f32,
    disagree: f32,
    n_eps: f32,
    succ: f32,
    coll: f32,
    tout: f32,
    n_c: [f32; 3],
    s_c: [f32; 3],
    k_c: [f32; 3],
    gpu_ms: f32,
    g_n: f32,
    g_ok: f32,
    p_n: f32,
    p_succ: f32,
}

type H = *mut std::ffi::c_void;
extern "C" {
    fn bc_struct_size(which: i32) -> i64;
    fn bc_set_goal_drop(h: H, p: f32) -> i32;
    fn bc_create(cfg: *const BcConfig) -> H;
    fn bc_destroy(h: H);
    fn bc_load_teacher(h: H, path: *const std::os::raw::c_char) -> i32;
    fn bc_reset_env(h: H, seed: u64) -> i32;
    fn bc_set_mode(h: H, actor: i32, record: i32) -> i32;
    fn bc_set_lr(h: H, lr: f32) -> i32;
    fn bc_rollout(h: H) -> i32;
    fn bc_update(h: H) -> i32;
    fn bc_poll(h: H, out: *mut BcLog) -> i32;
    fn bc_inflight(h: H) -> i32;
    fn bc_table(h: H, out: *mut u64) -> i32;
    fn bc_clear_table(h: H) -> i32;
    fn bc_save_student(h: H, path: *const std::os::raw::c_char) -> i32;
    fn bc_load_student(h: H, path: *const std::os::raw::c_char) -> i32;
    fn bc_num_params(h: H) -> i64;
    fn bc_device_bytes(h: H) -> i64;
    fn bc_load_text_table(h: H, path: *const std::os::raw::c_char) -> i32;
    fn bc_sldiag(h: H, tag: *const std::os::raw::c_char) -> i32;
    fn bc_set_beta(h: H, beta: f32) -> i32;
    fn bc_set_houses(h: H, which: i32) -> i32;
    fn bc_set_keep(h: H, keep: i64) -> i32;
    fn bc_set_demo_frac(h: H, frac: f32) -> i32;
}

fn gi(v: &Value, k: &str, d: i64) -> i64 { v.get(k).and_then(|x| x.as_i64()).unwrap_or(d) }
fn gf(v: &Value, k: &str, d: f64) -> f64 { v.get(k).and_then(|x| x.as_f64()).unwrap_or(d) }
fn cstr(s: &str) -> std::ffi::CString { std::ffi::CString::new(s).unwrap() }

struct Run {
    h: H,
    csv: fs::File,
    phase: String,
    t0: Instant,
    logs: Vec<BcLog>,
    rf: Option<runfolder::RunFolder>,
}
impl Run {
    fn drain(&mut self) {
        let mut l = BcLog::default();
        while unsafe { bc_poll(self.h, &mut l) } == 1 {
            writeln!(self.csv, "{},{:.2},{},{},{},{},{},{},{:.6},{:.4},{:.6},{},{:.4},{:.4},{:.4},{},{},{},{:.4},{:.4},{:.4},{:.3}",
                self.phase, self.t0.elapsed().as_secs_f64(), l.seq, l.kind, l.actor, l.record, l.count, l.adam_t, l.loss, l.grad_norm, l.disagree,
                l.n_eps as i64, l.succ, l.coll, l.tout, l.n_c[0] as i64, l.n_c[1] as i64, l.n_c[2] as i64, l.s_c[0], l.s_c[1], l.s_c[2], l.gpu_ms).unwrap();
            if let Some(r) = self.rf.as_mut() { r.log(&self.phase, &l, self.t0.elapsed().as_secs_f64()); }
            self.logs.push(l);
        }
    }
    // kind 0 = 롤아웃, 1 = 갱신. after(k) 는 k 번째를 띄운 뒤 부름(표 지우기 등 스트림 순서 작업)
    fn launch(&mut self, kind: i32, n: usize, mut after: impl FnMut(H, usize)) -> Vec<BcLog> {
        let start = self.logs.len();
        let mut done = 0usize;
        while done < n {
            let r = unsafe { if kind == 0 { bc_rollout(self.h) } else { bc_update(self.h) } };
            if r == 0 {
                done += 1;
                after(self.h, done);
            } else {
                std::thread::sleep(Duration::from_micros(200));
            }
            self.drain();
        }
        while unsafe { bc_inflight(self.h) } > 0 {
            std::thread::sleep(Duration::from_micros(200));
            self.drain();
        }
        self.logs[start..].to_vec()
    }
}

// 처음 완성도 칸(ppo_verify eval 과 같은 칸): 미리 확정 수 0 | 1–2 | 3–6 | 7–8 | 9
fn table_json(t: &[u64]) -> Value {
    let bucket = |c: usize| if c == 0 { 0 } else if c <= 2 { 1 } else if c <= 6 { 2 } else if c <= 8 { 3 } else { 4 };
    let names = ["0 %", "0-30 %", "30-70 %", "70-100 %", "100 %"];
    let mut b = [[0f64; 6]; 5];
    let mut s = [[0f64; 6]; 3];
    let mut all = [0f64; 6];
    for st in 0..3 {
        for g in 0..2 {
            for c in 0..10 {
                for q in 0..6 {
                    let v = t[((st * 2 + g) * 10 + c) * 6 + q] as f64;
                    b[bucket(c)][q] += v;
                    s[st][q] += v;
                    all[q] += v;
                }
            }
        }
    }
    let row = |a: &[f64; 6]| {
        let n = a[0].max(1.0);
        json!({"episodes": a[0], "success": a[1] / n, "collision": a[2] / n, "timeout": a[3] / n, "steps": a[4] / n, "steps_succ": a[5] / a[1].max(1.0)})
    };
    json!({
        "all": row(&all),
        "by_stage": {"C0": row(&s[0]), "C1": row(&s[1]), "C2": row(&s[2])},
        "by_completeness": names.iter().enumerate().map(|(i, n)| (n.to_string(), row(&b[i]))).collect::<serde_json::Map<String, Value>>(),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: bc_run <config.json> [--out DIR]");
        std::process::exit(2);
    }
    let v: Value = serde_json::from_str(&fs::read_to_string(&args[1]).expect("config")).expect("json");
    let mut out = PathBuf::from(v.get("out").and_then(|x| x.as_str()).unwrap_or("runs/bc"));
    let mut a = 2;
    while a < args.len() {
        if args[a] == "--out" { out = PathBuf::from(&args[a + 1]); a += 1; }
        a += 1;
    }
    fs::create_dir_all(&out).unwrap();
    fs::write(out.join("config.json"), serde_json::to_string_pretty(&v).unwrap()).unwrap();

    let home = std::env::var("HOME").unwrap_or_default();   // 설정에 적은 ~ 펼치기용
    // 교사 체크포인트: 설정 "teacher" > $RA_CHECKPOINTS/ppo/teacher.bin (기본 training/runs)
    let ckpts = std::env::var("RA_CHECKPOINTS").ok().filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{}/../../training/runs", env!("CARGO_MANIFEST_DIR")));
    let teacher = v.get("teacher").and_then(|x| x.as_str()).map(|s| s.replace('~', &home)).unwrap_or_else(|| format!("{}/ppo/teacher.bin", ckpts));
    unsafe {   // C ABI 구조체 배치 확인(bc_capi.h 와 어긋나면 바로 멈춤)
        assert_eq!(bc_struct_size(0) as usize, std::mem::size_of::<BcConfig>(), "C ABI size mismatch: BcConfig");
        assert_eq!(bc_struct_size(1) as usize, std::mem::size_of::<BcLog>(), "C ABI size mismatch: BcLog");
    }
    let c = BcConfig {
        n_env: gi(&v, "n_env", 4096) as i32,
        horizon: gi(&v, "horizon", 64) as i32,
        stage: gi(&v, "stage", 2) as i32,
        use_map: gi(&v, "use_map", 2) as i32,
        teacher_use_map: gi(&v, "teacher_use_map", 2) as i32,
        student_goal: gi(&v, "student_goal", 0) as i32,
        use_graphs: 1,
        log_ring: 16,
        mb: gi(&v, "mb", 8192) as i32,
        upd_steps: gi(&v, "upd_steps", 50) as i32,
        dw_chunk: gi(&v, "dw_chunk", 1024) as i32,
        store_render: gi(&v, "store_render", 1) as i32,
        cap: gi(&v, "cap", 3_000_000),
        seed: gi(&v, "seed", 1) as u64,
        env_seed: gi(&v, "data_seed", 1000) as u64,
        lr: gf(&v, "lr", 3e-4) as f32,
        adam_b1: 0.9,
        adam_b2: 0.999,
        adam_eps: 1e-8,
        max_grad_norm: gf(&v, "max_grad_norm", 1.0) as f32,
        map_p0: gf(&v, "map_p0", 0.0) as f32,
        map_p1: gf(&v, "map_p1", 0.0) as f32,
        map_kmin: gi(&v, "map_kmin", 1) as i32,
        map_kmax: gi(&v, "map_kmax", 8) as i32,
        map_reveal_r: gf(&v, "map_reveal_r", 1.5) as f32,
        fp8: gi(&v, "fp8", 0) as i32,
        vision: gi(&v, "vision", 0) as i32,
        head: gi(&v, "head", 0) as i32,
        chunk: gi(&v, "chunk", 16) as i32,
        flow_steps: gi(&v, "flow_steps", 10) as i32,
        text: gi(&v, "text", 0) as i32,
        render_profile: gi(&v, "render_profile", 1) as i32,
        render_batch: gi(&v, "render_batch", 256) as i32,
        img_dim: 16,
        sample_render: 1,
        vit_prec: gi(&v, "vit_prec", 1) as i32,
        arch: gi(&v, "arch", 0) as i32,
        tf_d: gi(&v, "tf_d", 0) as i32,
        tf_layers: gi(&v, "tf_layers", 0) as i32,
        tf_heads: gi(&v, "tf_heads", 0) as i32,
        tf_mlp: gi(&v, "tf_mlp", 0) as i32,
        tf_elayers: gi(&v, "tf_elayers", 0) as i32,
        act_mask: gi(&v, "act_mask", 0) as u32,
        task: gi(&v, "task", 0) as i32,
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
        render_aug: gi(v.get("render_aug").unwrap_or(&Value::Null), "on", 0) as i32,
        ra_color: gf(v.get("render_aug").unwrap_or(&Value::Null), "color", 0.0) as f32,
        ra_light: gf(v.get("render_aug").unwrap_or(&Value::Null), "light", 0.0) as f32,
        ra_expo: gf(v.get("render_aug").unwrap_or(&Value::Null), "expo", 0.0) as f32,
        render_team_mix: gf(v.get("render_aug").unwrap_or(&Value::Null), "team_mix", 0.0) as f32,
        beh: gi(v.get("beh").unwrap_or(&Value::Null), "on", 0) as i32,
        map_nav_k: gi(v.get("beh").unwrap_or(&Value::Null), "nav_k", 0) as i32,
        b_p1: v.get("beh").and_then(|b| b.get("mix")).and_then(|m| m.get(0)).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
        b_p2: v.get("beh").and_then(|b| b.get("mix")).and_then(|m| m.get(1)).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
        b_scene_mask: gi(v.get("beh").unwrap_or(&Value::Null), "scene_mask", 0) as u32,
        b_split: gi(v.get("beh").unwrap_or(&Value::Null), "split", 0) as i32,
        b_yaw_jit: gf(v.get("beh").unwrap_or(&Value::Null), "yaw_jit", 0.0) as f32,
        b_strict: gi(v.get("beh").unwrap_or(&Value::Null), "strict", 0) as i32,
        b_nofilter: 0,
        b_eval_instr: gi(v.get("beh").unwrap_or(&Value::Null), "eval_instr", 0) as i32,
        b_p_point: gf(v.get("beh").unwrap_or(&Value::Null), "p_point", 0.0) as f32,   // 목표 점 섞음(VLA_INPUT 2.1)
        b_p_goto: gf(v.get("beh").unwrap_or(&Value::Null), "p_goto", 0.0) as f32,
        goal_drop: gf(&v, "goal_drop", 0.5) as f32,   // (가정) 학생 목표 표시 감추기 확률 — README "지시문·목표 표시 감추기"
        b_p4: v.get("beh").and_then(|b| b.get("pnp")).and_then(|m| m.get(0)).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
        b_p5: v.get("beh").and_then(|b| b.get("pnp")).and_then(|m| m.get(1)).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
        b_p6: v.get("beh").and_then(|b| b.get("pnp")).and_then(|m| m.get(2)).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
        b_p_slip: v.get("beh").and_then(|b| b.get("fail")).and_then(|m| m.get(0)).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
        b_p_occ: v.get("beh").and_then(|b| b.get("fail")).and_then(|m| m.get(1)).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
        teacher_script: gi(&v, "teacher_script", 0) as i32,   // E6: B4–B6 라벨 = 대본 특권 교사
        b_feas: v.get("beh").and_then(|b| b.get("feas")).and_then(|x| x.as_i64()).unwrap_or(0) as i32,   // 잡기 가능 짝만(PF_FEAS)
        b_gcand: v.get("beh").and_then(|b| b.get("gcand")).and_then(|x| x.as_i64()).unwrap_or(0) as i32,   // 잡기 서는 자리 후보(상태 없는 교사)
        b_sltol: v.get("beh").and_then(|b| b.get("sltol")).and_then(|x| x.as_i64()).unwrap_or(0) as i32,   // 상태 없는 교사 단계 문턱(1 = 배울 수 있는 값)
        mlp_w: gi(&v, "mlp_w", 0) as i32,   // MLP 학생 몸통 폭(0 = 256)
        b_house_split: v.get("beh").and_then(|b| b.get("house_split")).and_then(|x| x.as_i64()).unwrap_or(1) as i32,   // 학습 집 / 평가 집 나누기(기본 켬)
    };
    // GT 지도 막기(2026-10-06 사용자 결정): 처음 지도 C0/C1(미리 채운 참 지도)은 --debug-gt-map 일 때만. 학습·평가는 빈 지도 C2 에서 자라는 지도
    if (c.map_p0 > 0.0 || c.map_p1 > 0.0) && !args.iter().any(|x| x == "--debug-gt-map") {
        eprintln!("bc_run: config asks for a prefilled (GT) start map (map_p0 {} map_p1 {}). Training uses the growing map only (C2, CURRICULUM_BEHAVIOR2026 5.7). Pass --debug-gt-map to run it anyway (debug only).", c.map_p0, c.map_p1);
        std::process::exit(2);
    }
    let r0 = gi(&v, "record_rollouts", 4) as usize;
    let u0 = gi(&v, "bc_updates", 100) as usize;
    let nd = gi(&v, "dagger_iters", 4) as usize;
    let rd = gi(&v, "dagger_rollouts", 2) as usize;
    let ud = gi(&v, "dagger_updates", 50) as usize;
    let dagger = v.get("dagger").and_then(|x| x.as_bool()).unwrap_or(true);
    let eval_iters = gi(&v, "eval_iters", 40) as usize;
    let eval_drop = gi(&v, "eval_drop", 3) as usize;
    let eval_seed = gi(&v, "eval_seed", 777) as u64;
    let eval_teacher = v.get("eval_teacher").and_then(|x| x.as_bool()).unwrap_or(true);
    let lr_decay = gf(&v, "lr_dagger", c.lr as f64) as f32;

    let h = unsafe { bc_create(&c) };
    if !(c.teacher_script != 0 && teacher.is_empty()) {   // 대본 교사만 쓰는 판(B4–B6)은 체크포인트 없이도 됨
        assert_eq!(unsafe { bc_load_teacher(h, cstr(&teacher).as_ptr()) }, 0, "teacher {}", teacher);
    }
    if c.text != 0 {
        // 지시 문장 글 벡터 표(기본: 설정 파일 옆 ../data/instr_a2.f32)
        let def = PathBuf::from(&args[1]).parent().map(|p| p.join("../data/instr_a2.f32")).unwrap_or_default();
        let tp = v.get("text_table").and_then(|x| x.as_str()).map(|s| s.replace('~', &home)).unwrap_or(def.to_string_lossy().to_string());
        let k = unsafe { bc_load_text_table(h, cstr(&tp).as_ptr()) };
        assert!(k > 0, "text table {} ({})", tp, k);
        println!("text table {}: {} sentences", tp, k);
    }
    if let Some(p) = v.get("init_student").and_then(|x| x.as_str()) {
        assert_eq!(unsafe { bc_load_student(h, cstr(&p.replace('~', &home)).as_ptr()) }, 0);
    }
    let mut csv = fs::File::create(out.join("log.csv")).unwrap();
    writeln!(csv, "phase,wall_s,seq,kind,actor,record,count,adam_t,loss,grad_norm,disagree,n_eps,succ,coll,tout,n_c0,n_c1,n_c2,succ_c0,succ_c1,succ_c2,gpu_ms").unwrap();
    let rf = runfolder::open(&out, &args[1], &v, &c, &teacher, dagger, nd, unsafe { bc_num_params(h) }, unsafe { bc_device_bytes(h) });
    let mut run = Run { h, csv, phase: String::new(), t0: Instant::now(), logs: vec![], rf };
    println!("bc_run: N {} T {} mb {} K {} cap {} student params {} device {:.2} GB, teacher {}, dagger {} | vision {} text {} head {} (chunk {}, flow steps {}) render profile {}",
        c.n_env, c.horizon, c.mb, c.upd_steps, c.cap, unsafe { bc_num_params(h) }, unsafe { bc_device_bytes(h) } as f64 / 1e9, teacher, dagger,
        c.vision, c.text, if c.head != 0 { "flow" } else { "mse" }, c.chunk, c.flow_steps, c.render_profile);
    let mut results = serde_json::Map::new();

    let eval_one = |run: &mut Run, name: &str, actor: i32, results: &mut serde_json::Map<String, Value>| {
        run.phase = format!("eval_{}", name);
        let t = Instant::now();
        unsafe { bc_reset_env(run.h, eval_seed); bc_set_mode(run.h, actor, 0); }
        let lg = run.launch(0, eval_iters, |h, k| if k == eval_drop { unsafe { bc_clear_table(h); } });
        // 접지(BEHAVIOR B2·B3): 평가 바퀴(앞 eval_drop 버림)의 끝 스텝 가장 가까운 과제 물체 = 목표 비율, B2·B3 성공
        let (mut gn, mut gk, mut pn, mut ps) = (0f64, 0f64, 0f64, 0f64);
        for l in lg.iter().skip(eval_drop) { gn += l.g_n as f64; gk += (l.g_ok * l.g_n) as f64; pn += l.p_n as f64; ps += (l.p_succ * l.p_n) as f64; }
        let mut tab = vec![0u64; 3 * 2 * 10 * 6];
        unsafe { bc_table(run.h, tab.as_mut_ptr()); bc_sldiag(run.h, cstr(name).as_ptr()); }
        let tj = table_json(&tab);
        let a = &tj["all"];
        let st = &tj["by_stage"];
        println!("eval {:<10} ({}): success {:.4} collision {:.4} timeout {:.4} ({} episodes) | C0 {:.3}/{:.3} C1 {:.3}/{:.3} C2 {:.3}/{:.3} | {:.1} s",
            name, if actor == 0 { "teacher" } else { "student" }, a["success"].as_f64().unwrap(), a["collision"].as_f64().unwrap(), a["timeout"].as_f64().unwrap(),
            a["episodes"].as_f64().unwrap(), st["C0"]["success"].as_f64().unwrap(), st["C0"]["collision"].as_f64().unwrap(), st["C1"]["success"].as_f64().unwrap(),
            st["C1"]["collision"].as_f64().unwrap(), st["C2"]["success"].as_f64().unwrap(), st["C2"]["collision"].as_f64().unwrap(), t.elapsed().as_secs_f64());
        if pn > 0.0 {
            println!("     grounding {:<10}: nearest task object is the target in {:.4} of {} B2/B3 episodes (>= 2 task objects); B2/B3 success {:.4} of {}",
                name, if gn > 0.0 { gk / gn } else { 0.0 }, gn, ps / pn, pn);
        }
        if let Some(r) = run.rf.as_mut() {
            let ck = if actor == 1 { Some(out.join(format!("student_{}.bin", name))) } else { None };
            r.eval(name, actor, &tj, ck.as_deref(), run.t0.elapsed().as_secs_f64());
        }
        let mut tj = tj;
        if pn > 0.0 { tj["grounding"] = json!({"episodes": gn, "nearest_is_target": if gn > 0.0 { gk / gn } else { 0.0 }, "pnp_episodes": pn, "pnp_success": ps / pn}); }
        results.insert(name.to_string(), tj);
        fs::write(out.join("results.json"), serde_json::to_string_pretty(&Value::Object(results.clone())).unwrap()).unwrap();
    };
    let house_split = c.b_house_split != 0 && (c.beh != 0 || c.stage >= 3);
    let eval = |run: &mut Run, name: &str, actor: i32, results: &mut serde_json::Map<String, Value>| {
        if house_split {   // 평가는 평가 집(이름 그대로)과 학습 집(<이름>_trainhouses) 따로 — 학습 롤아웃은 학습 집으로 되돌림
            unsafe { bc_set_houses(run.h, 0) };
            eval_one(run, &format!("{}_trainhouses", name), actor, results);
            unsafe { bc_set_houses(run.h, 1) };
            eval_one(run, name, actor, results);
            unsafe { bc_set_houses(run.h, 0) };
        } else {
            eval_one(run, name, actor, results);
        }
    };
    // 학습률 일정(갱신 그래프마다, 비동기 장치 값): "lr_sched": "cos" 면 앞 warmup 몫은 선형으로 올리고 그 뒤 cos 로 lr → lr_min (BC + DAgger 모든 갱신 그래프를 한 줄로)
    let sched_cos = v.get("lr_sched").and_then(|x| x.as_str()) == Some("cos");
    let lr_min = gf(&v, "lr_min", 0.0) as f32;
    let warm = gf(&v, "lr_warmup", 0.02);
    let total_upd = (u0 + nd * ud).max(1) as f64;
    let lr0 = c.lr;
    let lr_at = move |g: f64| -> f32 {
        let x = g / total_upd;
        if x < warm { lr0 * ((x / warm.max(1e-9)) as f32).max(0.02) } else { lr_min + 0.5 * (lr0 - lr_min) * (1.0 + (std::f64::consts::PI * ((x - warm) / (1.0 - warm)).min(1.0)).cos() as f32) }
    };
    let mut upd_done = 0usize;
    if sched_cos { unsafe { bc_set_lr(h, lr_at(0.0)); } }
    let mut train = |run: &mut Run, n: usize| -> (f32, f32, f64) {
        let t = Instant::now();
        let base = upd_done;
        let l = run.launch(1, n, |hh, k| if sched_cos { unsafe { bc_set_lr(hh, lr_at((base + k) as f64)); } });
        upd_done += n;
        let k = l.len().saturating_sub(5);
        let last = l[k..].iter().map(|x| x.loss).sum::<f32>() / (l.len() - k).max(1) as f32;
        let gms: f32 = l.iter().map(|x| x.gpu_ms).sum();
        (l.first().map(|x| x.loss).unwrap_or(0.0), last, (gms as f64) / 1e3 + 0.0 * t.elapsed().as_secs_f64())
    };
    let record = |run: &mut Run, actor: i32, n: usize, seed: u64| -> (f32, i64, f64) {
        unsafe { bc_reset_env(run.h, seed); bc_set_mode(run.h, actor, 1); }
        let l = run.launch(0, n, |_, _| {});
        unsafe { bc_sldiag(run.h, cstr(&run.phase.clone()).as_ptr()); }
        let dis = l.iter().map(|x| x.disagree).sum::<f32>() / l.len().max(1) as f32;
        let gms: f32 = l.iter().map(|x| x.gpu_ms).sum();
        (dis, l.last().map(|x| x.count).unwrap_or(0), gms as f64 / 1e3)
    };

    // 학생 평가 + BEHAVIOR 면 접지 평가(목표 표시를 늘 끔 — 지시문과 칸 이름 벡터만으로 맞는 물체로 가나), 뒤에 설정 값으로 되돌림
    let grounding = c.stage >= 3 || c.beh != 0;
    let eval_noflag = v.get("eval_noflag").and_then(|x| x.as_bool()).unwrap_or(true);
    let eval_tdrive = v.get("eval_tdrive").and_then(|x| x.as_bool()).unwrap_or(false);
    let eval_s = |run: &mut Run, name: &str, results: &mut serde_json::Map<String, Value>| {
        eval(run, name, 1, results);
        if eval_tdrive {   // 진단: 교사가 몰고(β 1) 학생은 앞 계산만 — 교사가 간 상태에서 학생 어긋남(BC_SLDIAG 표)
            unsafe { bc_set_beta(run.h, 1.0) };
            eval(run, &format!("{}_tdrive", name), 1, results);
            unsafe { bc_set_beta(run.h, 0.0) };
        }
        if grounding && eval_noflag {
            unsafe { bc_set_goal_drop(run.h, 1.0) };
            eval(run, &format!("{}_noflag", name), 1, results);
            unsafe { bc_set_goal_drop(run.h, c.goal_drop) };
        }
    };
    if eval_teacher {
        eval(&mut run, "teacher", 0, &mut results);
    }
    run.phase = "record".into();
    let (_, cnt, gs) = record(&mut run, 0, r0, c.env_seed);
    println!("record: teacher {} rollouts -> {} samples ({:.2} s GPU)", r0, cnt, gs);
    if v.get("keep_demos").and_then(|x| x.as_bool()).unwrap_or(false) {   // 교사 시연은 DAgger 자료에 밀려나지 않게(고리 앞 cnt 표본)
        let r = unsafe { bc_set_keep(h, cnt) };
        println!("keep_demos: first {} samples protected ({})", cnt, if r == 0 { "ok" } else { "buffer already full — not protected" });
        let df = gf(&v, "demo_frac", 0.0) as f32;   // DAgger 갱신 미니배치의 시연 몫(0 = 균등)
        if df > 0.0 && r == 0 { unsafe { bc_set_demo_frac(h, df) }; println!("demo_frac: {:.2} of each minibatch from the protected demos", df); }
    }
    run.phase = "bc".into();
    let (l0, l1, gs) = train(&mut run, u0);
    println!("bc: {} updates x {} steps, loss {:.5} -> {:.5} ({:.2} s GPU)", u0, c.upd_steps, l0, l1, gs);
    unsafe { bc_save_student(h, cstr(out.join("student_bc.bin").to_str().unwrap()).as_ptr()); }
    if v.get("eval_bc").and_then(|x| x.as_bool()).unwrap_or(true) {
        eval_s(&mut run, "bc", &mut results);
    }
    if lr_decay != c.lr { unsafe { bc_set_lr(h, lr_decay); } }
    for i in 1..=nd {
        run.phase = format!("collect{}", i);
        let beta = v.get("dagger_beta").and_then(|x| x.as_array()).and_then(|a| a.get(i - 1).or(a.last())).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
        if dagger { unsafe { bc_set_beta(h, beta) }; }
        let (dis, cnt, gs) = record(&mut run, if dagger { 1 } else { 0 }, rd, c.env_seed + i as u64);
        unsafe { bc_set_beta(h, 0.0) };
        if dagger { println!("dagger {}: beta {:.2} (teacher drives that share of episodes)", i, beta); }
        println!("{} {}: {} rollouts, student-vs-teacher disagreement {:.5} on visited states, data {} ({:.2} s GPU)",
            if dagger { "dagger" } else { "teacher-more" }, i, rd, dis, cnt, gs);
        run.phase = format!("train{}", i);
        let (l0, l1, gs) = train(&mut run, ud);
        println!("  train: loss {:.5} -> {:.5} ({:.2} s GPU)", l0, l1, gs);
        let nm = format!("{}{}", if dagger { "dagger" } else { "more" }, i);
        unsafe { bc_save_student(h, cstr(out.join(format!("student_{}.bin", nm)).to_str().unwrap()).as_ptr()); }
        eval_s(&mut run, &nm, &mut results);
    }
    println!("done: {:.1} s wall", run.t0.elapsed().as_secs_f64());
    if let Some(r) = run.rf.as_mut() { r.finish(run.t0.elapsed().as_secs_f64()); }
    unsafe { bc_destroy(h) };
}

// BC 학생 실행기(계획서 GPU_TRAINING.md 1.1: Rust = 설정·실행 순서·로그). 계산은 모두 libbc(C++/CUDA, 그래프 둘).
//
//   bc_run <config.json> [--out DIR]
//
// 순서: (교사 평가) → 교사 기록 R0 롤아웃 → BC 갱신 U0 번 → 학생 평가 → DAgger i = 1..D [학생이 움직이며 교사가 라벨 Rd 롤아웃 → 모은 자료로 갱신 Ud 번 → 평가].
// "dagger": false 면 같은 횟수를 교사가 움직이며 기록한다(자료 양·학습 양이 같은 대조 = BC + 교사 자료 더).
// 단계 안에서는 그래프를 띄우고 끝난 기록을 이벤트로 꺼내기만 한다(기다리지 않음). 동기는 단계 경계(환경 다시 만들기, 표 읽기, 저장)뿐.
use serde_json::{json, Value};
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
    pad: i32,
    vision: i32,
    head: i32,
    chunk: i32,
    flow_steps: i32,
    text: i32,
    render_profile: i32,
    render_batch: i32,
    img_dim: i32,
    sample_render: i32,
    pad2: i32,
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
}

type H = *mut std::ffi::c_void;
extern "C" {
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
}
impl Run {
    fn drain(&mut self) {
        let mut l = BcLog::default();
        while unsafe { bc_poll(self.h, &mut l) } == 1 {
            writeln!(self.csv, "{},{:.2},{},{},{},{},{},{},{:.6},{:.4},{:.6},{},{:.4},{:.4},{:.4},{},{},{},{:.4},{:.4},{:.4},{:.3}",
                self.phase, self.t0.elapsed().as_secs_f64(), l.seq, l.kind, l.actor, l.record, l.count, l.adam_t, l.loss, l.grad_norm, l.disagree,
                l.n_eps as i64, l.succ, l.coll, l.tout, l.n_c[0] as i64, l.n_c[1] as i64, l.n_c[2] as i64, l.s_c[0], l.s_c[1], l.s_c[2], l.gpu_ms).unwrap();
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

    let home = std::env::var("HOME").unwrap_or_default();
    let teacher = v.get("teacher").and_then(|x| x.as_str()).unwrap_or("~/ra_ppoout/g5/t4/on_s1/ckpt_final.bin").replace('~', &home);
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
        map_p0: gf(&v, "map_p0", 0.2) as f32,
        map_p1: gf(&v, "map_p1", 0.6) as f32,
        map_kmin: gi(&v, "map_kmin", 1) as i32,
        map_kmax: gi(&v, "map_kmax", 8) as i32,
        map_reveal_r: gf(&v, "map_reveal_r", 1.5) as f32,
        pad: 0,
        vision: gi(&v, "vision", 0) as i32,
        head: gi(&v, "head", 0) as i32,
        chunk: gi(&v, "chunk", 16) as i32,
        flow_steps: gi(&v, "flow_steps", 10) as i32,
        text: gi(&v, "text", 0) as i32,
        render_profile: gi(&v, "render_profile", 1) as i32,
        render_batch: gi(&v, "render_batch", 256) as i32,
        img_dim: 16,
        sample_render: 1,
        pad2: 0,
    };
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
    assert_eq!(unsafe { bc_load_teacher(h, cstr(&teacher).as_ptr()) }, 0, "teacher {}", teacher);
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
    let mut run = Run { h, csv, phase: String::new(), t0: Instant::now(), logs: vec![] };
    println!("bc_run: N {} T {} mb {} K {} cap {} student params {} device {:.2} GB, teacher {}, dagger {} | vision {} text {} head {} (chunk {}, flow steps {}) render profile {}",
        c.n_env, c.horizon, c.mb, c.upd_steps, c.cap, unsafe { bc_num_params(h) }, unsafe { bc_device_bytes(h) } as f64 / 1e9, teacher, dagger,
        c.vision, c.text, if c.head != 0 { "flow" } else { "mse" }, c.chunk, c.flow_steps, c.render_profile);
    let mut results = serde_json::Map::new();

    let eval = |run: &mut Run, name: &str, actor: i32, results: &mut serde_json::Map<String, Value>| {
        run.phase = format!("eval_{}", name);
        let t = Instant::now();
        unsafe { bc_reset_env(run.h, eval_seed); bc_set_mode(run.h, actor, 0); }
        run.launch(0, eval_iters, |h, k| if k == eval_drop { unsafe { bc_clear_table(h); } });
        let mut tab = vec![0u64; 3 * 2 * 10 * 6];
        unsafe { bc_table(run.h, tab.as_mut_ptr()); }
        let tj = table_json(&tab);
        let a = &tj["all"];
        let st = &tj["by_stage"];
        println!("eval {:<10} ({}): success {:.4} collision {:.4} timeout {:.4} ({} episodes) | C0 {:.3}/{:.3} C1 {:.3}/{:.3} C2 {:.3}/{:.3} | {:.1} s",
            name, if actor == 0 { "teacher" } else { "student" }, a["success"].as_f64().unwrap(), a["collision"].as_f64().unwrap(), a["timeout"].as_f64().unwrap(),
            a["episodes"].as_f64().unwrap(), st["C0"]["success"].as_f64().unwrap(), st["C0"]["collision"].as_f64().unwrap(), st["C1"]["success"].as_f64().unwrap(),
            st["C1"]["collision"].as_f64().unwrap(), st["C2"]["success"].as_f64().unwrap(), st["C2"]["collision"].as_f64().unwrap(), t.elapsed().as_secs_f64());
        results.insert(name.to_string(), tj);
        fs::write(out.join("results.json"), serde_json::to_string_pretty(&Value::Object(results.clone())).unwrap()).unwrap();
    };
    let train = |run: &mut Run, n: usize| -> (f32, f32, f64) {
        let t = Instant::now();
        let l = run.launch(1, n, |_, _| {});
        let k = l.len().saturating_sub(5);
        let last = l[k..].iter().map(|x| x.loss).sum::<f32>() / (l.len() - k).max(1) as f32;
        let gms: f32 = l.iter().map(|x| x.gpu_ms).sum();
        (l.first().map(|x| x.loss).unwrap_or(0.0), last, (gms as f64) / 1e3 + 0.0 * t.elapsed().as_secs_f64())
    };
    let record = |run: &mut Run, actor: i32, n: usize, seed: u64| -> (f32, i64, f64) {
        unsafe { bc_reset_env(run.h, seed); bc_set_mode(run.h, actor, 1); }
        let l = run.launch(0, n, |_, _| {});
        let dis = l.iter().map(|x| x.disagree).sum::<f32>() / l.len().max(1) as f32;
        let gms: f32 = l.iter().map(|x| x.gpu_ms).sum();
        (dis, l.last().map(|x| x.count).unwrap_or(0), gms as f64 / 1e3)
    };

    if eval_teacher {
        eval(&mut run, "teacher", 0, &mut results);
    }
    run.phase = "record".into();
    let (_, cnt, gs) = record(&mut run, 0, r0, c.env_seed);
    println!("record: teacher {} rollouts -> {} samples ({:.2} s GPU)", r0, cnt, gs);
    run.phase = "bc".into();
    let (l0, l1, gs) = train(&mut run, u0);
    println!("bc: {} updates x {} steps, loss {:.5} -> {:.5} ({:.2} s GPU)", u0, c.upd_steps, l0, l1, gs);
    unsafe { bc_save_student(h, cstr(out.join("student_bc.bin").to_str().unwrap()).as_ptr()); }
    eval(&mut run, "bc", 1, &mut results);
    if lr_decay != c.lr { unsafe { bc_set_lr(h, lr_decay); } }
    for i in 1..=nd {
        run.phase = format!("collect{}", i);
        let (dis, cnt, gs) = record(&mut run, if dagger { 1 } else { 0 }, rd, c.env_seed + i as u64);
        println!("{} {}: {} rollouts, student-vs-teacher disagreement {:.5} on visited states, data {} ({:.2} s GPU)",
            if dagger { "dagger" } else { "teacher-more" }, i, rd, dis, cnt, gs);
        run.phase = format!("train{}", i);
        let (l0, l1, gs) = train(&mut run, ud);
        println!("  train: loss {:.5} -> {:.5} ({:.2} s GPU)", l0, l1, gs);
        let nm = format!("{}{}", if dagger { "dagger" } else { "more" }, i);
        unsafe { bc_save_student(h, cstr(out.join(format!("student_{}.bin", nm)).to_str().unwrap()).as_ptr()); }
        eval(&mut run, &nm, 1, &mut results);
    }
    println!("done: {:.1} s wall", run.t0.elapsed().as_secs_f64());
    unsafe { bc_destroy(h) };
}

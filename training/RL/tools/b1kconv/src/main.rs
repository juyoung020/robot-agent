//! b1kconv — BEHAVIOR 2026 scenes + tasks -> RASC v1 device files (CURRICULUM_BEHAVIOR2026 E1).
//!
//!   b1kconv header [--out cpp/rasc_format.h] [--check cpp/rasc_format.h]
//!   b1kconv convert --out DIR [--scene NAME]... [--b1k PATH] [--doc PATH]
//!   b1kconv verify  --out DIR [--scene NAME]...
//!   b1kconv negative [--b1k PATH]          (fault injection: must fail; exit 0 = all faults caught)

mod bddl;
mod feas;
mod format;
mod png;
mod pnp;
mod scene;
mod task;
mod util;
mod verify;
mod write;

use format::*;
use scene::{Paths, SCENES};
use std::collections::HashMap;
use std::fmt::Write as _;

#[cfg(not(target_endian = "little"))]
compile_error!("RASC is little endian");

struct Args {
    cmd: String,
    out: String,
    scenes: Vec<String>,
    b1k: String,
    doc: String,
    check: Option<String>,
    outer: String,
    inner: String,
}

fn args() -> Args {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../..");
    let mut a = Args {
        cmd: String::new(),
        out: String::new(),
        scenes: vec![],
        b1k: format!("{root}/src/behavior-2026/BEHAVIOR-1K"),
        doc: format!("{root}/docs/map_vla/CURRICULUM_BEHAVIOR2026.md"),
        check: None,
        outer: String::new(),
        inner: String::new(),
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let next = |i: &mut usize| -> String {
            *i += 1;
            v.get(*i).cloned().unwrap_or_else(|| {
                eprintln!("missing value after {}", v[*i - 1]);
                std::process::exit(2)
            })
        };
        match v[i].as_str() {
            "--out" => a.out = next(&mut i),
            "--scene" => a.scenes.push(next(&mut i)),
            "--b1k" => a.b1k = next(&mut i),
            "--doc" => a.doc = next(&mut i),
            "--check" => a.check = Some(next(&mut i)),
            "--outer" => a.outer = next(&mut i),
            "--inner" => a.inner = next(&mut i),
            s if a.cmd.is_empty() => a.cmd = s.to_string(),
            s => {
                eprintln!("unknown argument {s}");
                std::process::exit(2)
            }
        }
        i += 1;
    }
    if a.scenes.is_empty() {
        a.scenes = SCENES.iter().map(|s| s.to_string()).collect();
    }
    a
}

struct SceneTasks {
    feas: Vec<task::FeasTask>,
    summary: String,
}

fn convert_scene(p: &Paths, c: &mut scene::Common, name: &str, out_dir: Option<&str>, fault: task::Fault, lim: [LimitsRec; 2]) -> Result<SceneTasks, String> {
    let t0 = std::time::Instant::now();
    let mut sc = scene::build(c, name)?;
    let metas = task::task_list(p)?;
    let rooms = task::rooms_csv(p)?;
    let mut t = write::Tables { tasks: vec![], tobjs: vec![], lits: vec![], vars: vec![], cands: vec![], removed: vec![], insts: vec![], poses: vec![], pnp: None };
    let mut feas = vec![];
    let mut pose_src = [0usize; 4];
    let mut extra = 0;
    let mut not_open = vec![];
    for m in &metas {
        let (s, doc_t, full_t) = task::find_scene(p, &m.name)?;
        if s != name {
            continue;
        }
        let ri = rooms.get(&m.name).cloned().unwrap_or_default();
        let mut o = task::Out {
            tasks: &mut t.tasks,
            tobjs: &mut t.tobjs,
            lits: &mut t.lits,
            vars: &mut t.vars,
            cands: &mut t.cands,
            removed: &mut t.removed,
            insts: &mut t.insts,
            poses: &mut t.poses,
        };
        let b = task::build_task(p, c, &mut sc, m, &ri, &doc_t, &full_t, &mut o, fault)?;
        for k in 0..4 {
            pose_src[k] += b.pose_src[k];
        }
        extra += b.extra_keys;
        for x in &b.not_open_inst {
            not_open.push((m.index, x.0, x.1));
        }
        // feasibility into the task record
        let tr = t.tasks.last_mut().unwrap();
        tr.q_strict = feas::run(&b.feas, &feas::STRICT, feas::Geo::Doc).q() as f32;
        tr.q_loose = feas::run(&b.feas, &feas::LOOSE, feas::Geo::Doc).q() as f32;
        tr.skills = feas::cols(&b.feas).skills;
        feas.push(b.feas);
    }
    t.pnp = Some(pnp::build(&sc, &t, lim));
    let pnp_txt = pnp_summary(&sc, &t);
    let bytes = write::assemble(&mut sc, &t);
    let mut s = String::new();
    let robot_on_free = t.insts.iter().filter(|r| match sc.cell(r.robot_pos[0] as f64, r.robot_pos[1] as f64) {
        Some((rr, cc)) => { let i = rr * sc.hdr.grid_w as usize + cc; sc.trav[0][i >> 3] >> (i & 7) & 1 == 1 }
        None => false,
    }).count();
    let _ = writeln!(
        s,
        "{name}: {} B ({:.2} MB) objs {} boxes {} doors {} rooms {} (png {}) cats {} grid {}x{} tasks {} task_objs {} lits {} vars {} insts {} poses {} joints {} | pose src: file {} scene {} template {} | extra instance keys {} | robot start on TRAV cell {}/{} | {:.1}s",
        bytes.len(), bytes.len() as f64 / 1e6, sc.objs.len(), sc.boxes.len(), sc.doors.len(), sc.rooms.len(), sc.hdr.n_rooms_png, sc.cats.len(),
        sc.hdr.grid_w, sc.hdr.grid_h, t.tasks.len(), t.tobjs.len(), t.lits.len(), t.vars.len(), t.insts.len(), t.poses.len(), sc.joints.len(),
        pose_src[1], pose_src[2], pose_src[3], extra, robot_on_free, t.insts.len(), t0.elapsed().as_secs_f64()
    );
    let mut sec_sizes = String::new();
    for (i, sn) in SEC_NAMES.iter().enumerate() {
        if sc.hdr.sections[i].bytes > 0 {
            let _ = write!(sec_sizes, " {sn}={}", sc.hdr.sections[i].bytes);
        }
    }
    let _ = writeln!(s, "  sections:{sec_sizes}");
    s.push_str(&pnp_txt.0);
    for w in &sc.warnings {
        let _ = writeln!(s, "  warning: {w}");
    }
    // doc 1.2 scene numbers
    let n_art = sc.objs.iter().filter(|o| o.flags & flag("ARTICULATED") != 0).count();
    let n_wall = sc.objs.iter().filter(|o| o.flags & flag("WALL") != 0).count();
    let n_stairs = sc.objs.iter().filter(|o| o.flags & flag("STAIRS") != 0).count();
    let _ = writeln!(s, "  doc1.2: objects {} rooms {} doors {} articulated {} walls {} trav_m2 {:.0} stairs {}",
        sc.objs.len(), sc.hdr.n_rooms_png, sc.doors.len(), n_art, n_wall, sc.hdr.trav_area_m2, n_stairs);
    let mut it: Vec<String> = vec![];
    for tr in &t.tasks {
        for l in &t.lits[tr.goal_off as usize..(tr.goal_off + tr.n_goal as u32) as usize] {
            if l.flags & 3 == 3 {
                it.push(format!("{}:{}{}", tr.task_index, if l.neg != 0 { "not " } else { "" }, PREDS[l.pred as usize]));
            }
        }
    }
    let _ = writeln!(s, "  init-true (BDDL :init, closed world) doc goal literals: {} [{}]", it.len(), it.join(", "));
    for (ti, a, n) in &not_open {
        let _ = writeln!(s, "  not-open task {ti}: closed at start in {a}/{n} instances");
    }
    if let Some(dir) = out_dir {
        std::fs::write(format!("{dir}/{name}.rasc"), &bytes).map_err(|e| e.to_string())?;
        let mut e = String::new();
        let _ = writeln!(e, "scene {name}");
        let _ = writeln!(e, "layout_hash {}", layout_hash());
        let _ = writeln!(e, "grid {} {}", sc.hdr.grid_w, sc.hdr.grid_h);
        for (i, sn) in SEC_NAMES.iter().enumerate() {
            let sh = &sc.hdr.sections[i];
            let b = &bytes[sh.offset as usize..(sh.offset + sh.bytes) as usize];
            let _ = writeln!(e, "sec {sn} {} {:016x}", sh.count, write::fnv64(b));
        }
        let _ = writeln!(e, "n_doc_lits {}", t.tasks.iter().map(|x| x.n_doc_lits as u32).sum::<u32>());
        std::fs::write(format!("{dir}/{name}.pnp.tsv"), &pnp_txt.1).map_err(|e| e.to_string())?;
        std::fs::write(format!("{dir}/{name}.expect"), e).map_err(|e| e.to_string())?;
    }
    Ok(SceneTasks { feas, summary: s })
}

// ---------------------------------------------------------------- doc comparison
const ABBR: [(&str, &str); 7] = [
    ("house_single_floor", "HSF"),
    ("house_double_floor_lower", "HDL"),
    ("house_double_floor_upper", "HDU"),
    ("restaurant_diner", "RD"),
    ("Rs_int", "RS"),
    ("hotel_suite_large", "HSL"),
    ("office_cubicles_right", "OCR"),
];

fn table_and_compare(all: &mut [task::FeasTask], doc: &str, out_dir: &str) -> String {
    all.sort_by_key(|t| t.idx);
    let mut table = String::from("| # | 과제 | 장면 | 사람 s | 목표 조건 수 | 목표 술어 | 옮길 물체 → 목적지 | 스킬 | q 상한 엄격 / 느슨 | 엄격에서 막는 것(문자 수) |\n|---:|---|---|---:|---:|---|---|---|---|---|\n");
    let mut tsv = String::from("idx\ttask\tscene\tn_lits\tok_strict\tq_strict\tok_loose\tq_loose\tq_strict_world\tq_loose_world\twhy_strict\twhy_loose\n");
    let mut rows: HashMap<usize, Vec<String>> = HashMap::new();
    let mut tops: HashMap<usize, Vec<(String, usize)>> = HashMap::new();
    let (mut ss, mut sl, mut sw, mut slw) = (vec![], vec![], vec![], vec![]);
    let mut kinds: Vec<(String, usize)> = vec![];
    let mut block_tasks: Vec<(String, usize)> = vec![];
    let mut skill_tasks = [0usize; 10];
    let (mut not_open_lits, mut not_open_tasks) = (0, 0);
    for t in all.iter() {
        let s = feas::run(t, &feas::STRICT, feas::Geo::Doc);
        let l = feas::run(t, &feas::LOOSE, feas::Geo::Doc);
        let sw_ = feas::run(t, &feas::STRICT, feas::Geo::World);
        let lw_ = feas::run(t, &feas::LOOSE, feas::Geo::World);
        let cl = feas::cols(t);
        for (k, v) in &s.kinds {
            match kinds.iter_mut().find(|(a, _)| a == k) {
                Some((_, n)) => *n += v,
                None => kinds.push((k.clone(), *v)),
            }
        }
        for (k, _) in &s.why {
            match block_tasks.iter_mut().find(|(a, _)| a == k) {
                Some((_, n)) => *n += 1,
                None => block_tasks.push((k.clone(), 1)),
            }
        }
        for (i, sk) in skill_tasks.iter_mut().enumerate() {
            if cl.skills & (1 << i) != 0 {
                *sk += 1;
            }
        }
        let no = s.kinds.iter().find(|k| k.0 == "not open").map_or(0, |k| k.1);
        not_open_lits += no;
        not_open_tasks += (no > 0) as usize;
        let mut kk: Vec<&String> = s.kinds.iter().map(|k| &k.0).collect();
        kk.sort();
        let preds = kk.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
        let md = if cl.movers.is_empty() && cl.dests.is_empty() {
            "—".to_string()
        } else {
            let m = if cl.movers.is_empty() { "—".to_string() } else { cl.movers.join(", ") };
            if cl.dests.is_empty() { m } else { format!("{m} → {}", cl.dests.join(", ")) }
        };
        let top = s.top3();
        let block = if top.is_empty() { "—".to_string() } else { top.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", ") };
        let ab = ABBR.iter().find(|a| a.0 == t.scene).unwrap().1;
        let row = vec![
            t.idx.to_string(), format!("`{}`", t.task), ab.to_string(), format!("{:.0}", t.len_s), s.n.to_string(), preds, md,
            feas::skill_letters(cl.skills), format!("{} / {}", feas::fmt_q(s.q()), feas::fmt_q(l.q())), block,
        ];
        let _ = writeln!(table, "| {} |", row.join(" | "));
        let why = |f: &feas::TaskFeas| f.why.iter().map(|(k, n)| format!("{k}:{n}")).collect::<Vec<_>>().join(",");
        let _ = writeln!(tsv, "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}", t.idx, t.task, t.scene, s.n, s.ok, s.q(), l.ok, l.q(), sw_.q(), lw_.q(), why(&s), why(&l));
        rows.insert(t.idx, row);
        tops.insert(t.idx, s.why.clone());
        ss.push((t.idx, s.q()));
        sl.push((t.idx, l.q()));
        sw.push((t.idx, sw_.q()));
        slw.push((t.idx, lw_.q()));
    }
    let _ = std::fs::write(format!("{out_dir}/task_table.md"), &table);
    let _ = std::fs::write(format!("{out_dir}/feasibility.tsv"), &tsv);

    let mut r = String::new();
    let summ = |v: &[(usize, f64)]| -> (Vec<usize>, Vec<usize>, f64) {
        (v.iter().filter(|x| x.1 > 0.0).map(|x| x.0).collect(), v.iter().filter(|x| x.1 >= 1.0).map(|x| x.0).collect(), v.iter().map(|x| x.1).sum::<f64>() / v.len().max(1) as f64)
    };
    let n_lits: usize = kinds.iter().map(|k| k.1).sum();
    let (a, b, m) = summ(&ss);
    let _ = writeln!(r, "strict: nonzero {} {:?}  q=1 {} {:?}  mean {:.4} ({:.3})   [doc: 7 [26,27,30,54,79,80,89] / 1 [80] / 0.027]", a.len(), a, b.len(), b, m, m);
    let doc_ok_s = a == vec![26, 27, 30, 54, 79, 80, 89] && b == vec![80] && format!("{m:.3}") == "0.027";
    let (a, b, m) = summ(&sl);
    let _ = writeln!(r, "loose:  nonzero {} {:?}  q=1 {} {:?}  mean {:.4} ({:.3})   [doc: 14 [0,1,9,18,26,27,30,52,54,77,79,80,87,89] / 4 [0,1,80,87] / 0.087]", a.len(), a, b.len(), b, m, m);
    let doc_ok_l = a == vec![0, 1, 9, 18, 26, 27, 30, 52, 54, 77, 79, 80, 87, 89] && b == vec![0, 1, 80, 87] && format!("{m:.3}") == "0.087";
    let (a, b, m) = summ(&sw);
    let _ = writeln!(r, "strict, world box (rotation + bbox offset): nonzero {} {:?} q=1 {:?} mean {:.4}", a.len(), a, b, m);
    let (a, b, m) = summ(&slw);
    let _ = writeln!(r, "loose,  world box (rotation + bbox offset): nonzero {} {:?} q=1 {:?} mean {:.4}", a.len(), a, b, m);
    let _ = writeln!(r, "goal literals (doc grounding): {n_lits}   [doc: 550]");
    kinds.sort_by(|a, b| b.1.cmp(&a.1));
    let _ = writeln!(r, "  kinds: {}", kinds.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", "));
    let _ = writeln!(r, "'not open' goal literals (doc: initially true): {not_open_lits} in {not_open_tasks} tasks   [doc: 31 / 22]");
    block_tasks.sort_by(|a, b| b.1.cmp(&a.1));
    let _ = writeln!(r, "strict blockers (tasks): {}", block_tasks.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", "));
    let _ = writeln!(r, "  [doc: 폭 66, 무게 65, 높이 51, 놓을높이 35, 열기 25, 처음부터 참인 닫힘 22, 닫힌 곳 안 13, 시각 입자 13, 자르기 9, 요리 6, 토글 높이 5, 붙이기 4]");
    let _ = writeln!(r, "skills (tasks): {}", SKILLS.iter().zip(skill_tasks).map(|(s, n)| format!("{} {n}", s.1)).collect::<Vec<_>>().join(", "));

    // row-by-row comparison with the doc table
    let mut diffs = vec![];
    let mut nrows = 0;
    if let Ok(d) = std::fs::read_to_string(doc) {
        for line in d.lines() {
            let f: Vec<&str> = line.split('|').map(|x| x.trim()).collect();
            if f.len() < 12 || !f[2].starts_with('`') {
                continue;
            }
            let Ok(idx) = f[1].parse::<usize>() else { continue };
            nrows += 1;
            let Some(row) = rows.get(&idx) else {
                diffs.push(format!("{idx}: not produced"));
                continue;
            };
            let names = ["#", "과제", "장면", "사람 s", "목표 조건 수", "목표 술어", "옮길 물체 → 목적지", "스킬", "q 상한"];
            for (k, nm) in names.iter().enumerate() {
                if row[k] != f[k + 1] {
                    diffs.push(format!("{idx} {nm}: tool '{}' vs doc '{}'", row[k], f[k + 1]));
                }
            }
            // blockers: every doc item must match the tool's count; top-3 count multiset must match
            let why = &tops[&idx];
            let doc_items: Vec<(String, usize)> = if f[10] == "—" {
                vec![]
            } else {
                f[10].split(", ").filter_map(|x| x.rsplit_once(' ').map(|(a, b)| (a.to_string(), b.parse().unwrap_or(0)))).collect()
            };
            let mut ok = doc_items.iter().all(|(k, n)| why.iter().any(|(a, b)| a == k && b == n));
            let mut tc: Vec<usize> = feas::TaskFeas { n: 0, ok: 0, why: why.clone(), kinds: vec![] }.top3().iter().map(|x| x.1).collect();
            let mut dc: Vec<usize> = doc_items.iter().map(|x| x.1).collect();
            tc.sort();
            dc.sort();
            ok &= tc == dc;
            if !ok {
                diffs.push(format!("{idx} 막는 것: tool '{}' vs doc '{}'", row[9], f[10]));
            }
        }
    }
    let _ = writeln!(r, "doc 1.3 table: {nrows} rows compared, {} differences", diffs.len());
    for d in &diffs {
        let _ = writeln!(r, "  {d}");
    }
    let pass = doc_ok_s && doc_ok_l && n_lits == 550 && not_open_lits == 31 && not_open_tasks == 22 && nrows == 100 && diffs.is_empty();
    let _ = writeln!(r, "E1 criterion 1 (re-derive doc numbers): {}", if pass { "PASS" } else { "FAIL" });
    r
}

fn main() {
    let a = args();
    let p = Paths { b1k: a.b1k.clone() };
    let res: Result<i32, String> = (|| match a.cmd.as_str() {
        "header" => {
            let h = header_text();
            if let Some(c) = &a.check {
                let cur = std::fs::read_to_string(c).map_err(|e| format!("{c}: {e}"))?;
                if cur != h {
                    return Err(format!("{c} is stale: regenerate with `b1kconv header --out {c}`"));
                }
                println!("{c} up to date (layout hash {:08x})", layout_hash());
            } else if a.out.is_empty() {
                print!("{h}");
            } else {
                std::fs::write(&a.out, h).map_err(|e| e.to_string())?;
                println!("wrote {} (layout hash {:08x})", a.out, layout_hash());
            }
            Ok(0)
        }
        "convert" => {
            if a.out.is_empty() {
                return Err("--out DIR required".into());
            }
            std::fs::create_dir_all(&a.out).map_err(|e| e.to_string())?;
            let mut lim = pnp::limits_default();
            pnp::parse_limits(&mut lim[0], &a.outer)?;
            pnp::parse_limits(&mut lim[1], &a.inner)?;
            let mut c = scene::Common::load(&p)?;
            let mut all = vec![];
            let mut summary = String::new();
            for s in &a.scenes {
                let st = convert_scene(&p, &mut c, s, Some(&a.out), task::Fault::None, lim)?;
                print!("{}", st.summary);
                summary.push_str(&st.summary);
                all.extend(st.feas);
                c.meta.shrink_to_fit();
            }
            if all.len() == 100 {
                let r = table_and_compare(&mut all, &a.doc, &a.out);
                print!("{r}");
                summary.push_str(&r);
            } else {
                println!("({} tasks converted; the doc comparison needs all 7 scenes)", all.len());
            }
            std::fs::write(format!("{}/summary.txt", a.out), summary).map_err(|e| e.to_string())?;
            Ok(0)
        }
        "verify" => {
            let mut bad = 0;
            for s in &a.scenes {
                let path = format!("{}/{s}.rasc", a.out);
                let r = verify::verify(&p, &path)?;
                println!("{s}: {} values compared, {} differences", r.n, r.diffs.len());
                for d in r.diffs.iter().filter(|d| !d.is_empty()) {
                    println!("  {d}");
                }
                bad += r.diffs.len();
            }
            Ok(if bad == 0 { 0 } else { 1 })
        }
        "negative" => {
            let mut c = scene::Common::load(&p)?;
            let mut caught = 0;
            let faults = [(task::Fault::DropInstanceObject, "instance file lacks a task object"), (task::Fault::BadGoalObject, "goal object not in :objects")];
            for (f, what) in faults {
                match convert_scene(&p, &mut c, "Rs_int", None, f, pnp::limits_default()) {
                    Err(e) => {
                        caught += 1;
                        println!("fault '{what}': stopped with error (expected): {}", e.lines().next().unwrap_or(""));
                    }
                    Ok(_) => println!("fault '{what}': NOT detected"),
                }
            }
            // corrupted file must be rejected by the reader
            let dir = std::env::temp_dir().join(format!("b1kconv_neg_{}", std::process::id()));
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            convert_scene(&p, &mut c, "Rs_int", dir.to_str(), task::Fault::None, pnp::limits_default())?;
            let tmpf = dir.join("Rs_int.rasc");
            let tmp = dir.join("bad.rasc");
            let mut b = std::fs::read(&tmpf).map_err(|e| e.to_string())?;
            b[12] ^= 1; // layout hash
            std::fs::write(&tmp, &b).map_err(|e| e.to_string())?;
            match verify::File::open(tmp.to_str().unwrap()) {
                Err(e) => {
                    caught += 1;
                    println!("fault 'layout hash flipped': rejected (expected): {e}");
                }
                Ok(_) => println!("fault 'layout hash flipped': NOT detected"),
            }
            let _ = std::fs::remove_dir_all(&dir);
            println!("negative control: {caught}/3 faults caught");
            Ok(if caught == 3 { 0 } else { 1 })
        }
        _ => {
            eprintln!("usage: b1kconv header|convert|verify|negative  (see src/main.rs)");
            Ok(2)
        }
    })();
    match res {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    }
}

/// per-scene pick-and-place counts (text) + per-task table (TSV)
fn pnp_summary(sc: &scene::Scene, t: &write::Tables) -> (String, String) {
    let pn = t.pnp.as_ref().unwrap();
    let sr = pn.ranges.last().unwrap();
    let mut s = String::new();
    let sp = &pn.picks[..sr.n_pick as usize];
    let spl = &pn.places[..sr.n_place as usize];
    let kinds = |v: &[PlaceRec], k: u16| v.iter().filter(|p| p.kind == k).count();
    let spairs = &pn.pairs[..sr.n_pair as usize];
    let _ = writeln!(
        s,
        "  pnp scene: picks {} (inner {}, approachable {}) | supports ontop {} inside {} floors {} | pairs {} reachable {} reachable+inner {} | TRAV components {}",
        sp.len(), sp.iter().filter(|p| p.flags & 1 != 0).count(), sp.iter().filter(|p| p.n_approach > 0).count(),
        kinds(spl, 1), kinds(spl, 2), kinds(spl, 3), spairs.len(), spairs.iter().filter(|p| p.reachable & 1 != 0).count(),
        spairs.iter().filter(|p| p.reachable & 3 == 3).count(), pn.n_comp
    );
    let mut tsv = String::from("task_index\ttask\tinstances\twith_pick\twith_pick_inner\twith_reachable_pair\twith_robot_reachable_pair\twith_robot_reachable_inner_pair\tpicks_per_inst\tpick_categories\n");
    let (mut ip, mut ipi, mut ir, mut irr, mut irri, mut npk, mut npl, mut npr) = (0, 0, 0, 0, 0, 0usize, 0usize, 0usize);
    let ni = t.insts.len();
    for tr in &t.tasks {
        let (mut a, mut b, mut c, mut d, mut e, mut np) = (0, 0, 0, 0, 0, 0);
        let mut cats: Vec<String> = vec![];
        for ii in tr.inst_off..tr.inst_off + tr.n_inst {
            let r = &pn.ranges[ii as usize];
            let pk = &pn.picks[r.pick_off as usize..r.pick_off as usize + r.n_pick as usize];
            let pr = &pn.pairs[r.pair_off as usize..(r.pair_off + r.n_pair) as usize];
            np += pk.len();
            npl += r.n_place as usize;
            npr += pr.len();
            a += !pk.is_empty() as usize;
            b += pk.iter().any(|p| p.flags & 1 != 0) as usize;
            c += pr.iter().any(|p| p.reachable & 1 != 0) as usize;
            d += pr.iter().any(|p| p.reachable & 5 == 5) as usize;
            e += pr.iter().any(|p| p.reachable & 7 == 7) as usize;
            for p in pk {
                let o = &t.tobjs[(p.obj & !PNP_TASKOBJ) as usize];
                let cn = &sc.cat_names[o.cat as usize];
                if !cats.contains(cn) {
                    cats.push(cn.clone());
                }
            }
        }
        cats.sort();
        npk += np;
        ip += a;
        ipi += b;
        ir += c;
        irr += d;
        irri += e;
        let _ = writeln!(tsv, "{}\t{}\t{}\t{a}\t{b}\t{c}\t{d}\t{e}\t{:.2}\t{}", tr.task_index, util::cstr(&sc.strings.bytes, tr.name), tr.n_inst, np as f64 / tr.n_inst.max(1) as f64, cats.join(","));
    }
    let _ = writeln!(
        s,
        "  pnp instances {ni}: picks {npk} supports {npl} pairs {npr} | instances with a graspable task object {ip} (inner {ipi}), with a reachable pair {ir}, robot start in its component {irr}, + inner limits {irri}"
    );
    (s, tsv)
}

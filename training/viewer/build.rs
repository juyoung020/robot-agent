// 화면 파일(assets/*)과 sgview 의 three.js·로봇 모델(src/scene_graph/sgview/assets/{three.min.js, OrbitControls.js, GLTFLoader.js, robot/*})을
// 바이너리에 넣는다(sgview build.rs 방식). sgview 파일은 읽기만 한다 — 두 벌 두지 않음(TRAIN_VIEWER.md 7절).
use std::path::Path;

fn list(dir: &Path, prefix: &str, code: &mut String) {
    let mut names: Vec<_> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.flatten().filter(|e| e.path().is_file()).map(|e| e.file_name().to_string_lossy().to_string()).collect(),
        Err(_) => return,
    };
    names.sort();
    for n in names {
        let abs = std::fs::canonicalize(dir.join(&n)).unwrap();
        code.push_str(&format!("    ({:?}, include_bytes!({:?})),\n", format!("{}{}", prefix, n), abs.to_string_lossy()));
    }
    println!("cargo:rerun-if-changed={}", dir.display());
}

fn main() {
    let sg = Path::new("../../src/scene_graph/sgview/assets");
    let mut code = String::from("pub const FILES: &[(&str, &[u8])] = &[\n");
    list(Path::new("assets"), "", &mut code);
    for f in ["three.min.js", "OrbitControls.js", "GLTFLoader.js"] {
        let p = sg.join(f);
        if p.exists() {
            let abs = std::fs::canonicalize(&p).unwrap();
            code.push_str(&format!("    ({:?}, include_bytes!({:?})),\n", f, abs.to_string_lossy()));
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }
    list(&sg.join("robot"), "robot/", &mut code);
    code.push_str("];\n");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("assets.rs");
    std::fs::write(out, code).unwrap();
}

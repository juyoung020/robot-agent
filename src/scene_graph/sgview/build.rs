// 벽 2D 계산은 scenemap 의 C++ (walls.cpp) 를 그대로 컴파일해서 쓴다 — 파이썬 복사본도, Rust 로 다시 짠 복사본도 두지 않는다.
fn main() {
    // assets/robot/* (URDF model: GLB meshes + robot.json) are embedded into the binary: the page needs no files next to it
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("robot_assets.rs");
    let dir = std::path::Path::new("assets/robot");
    let mut code = String::from("pub const ROBOT_FILES: &[(&str, &[u8])] = &[\n");
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut names: Vec<_> = rd.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
        names.sort();
        for n in names {
            let abs = std::fs::canonicalize(dir.join(&n)).unwrap();
            code.push_str(&format!("    ({:?}, include_bytes!({:?})),\n", n, abs.to_string_lossy()));
        }
    }
    code.push_str("];\n");
    std::fs::write(out, code).unwrap();
    println!("cargo:rerun-if-changed=assets/robot");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .include("../scenemap/include")
        .file("../scenemap/src/walls.cpp")
        .file("src/walls_ffi.cpp")
        .compile("sgview_walls");
    println!("cargo:rerun-if-changed=../scenemap/src/walls.cpp");
    println!("cargo:rerun-if-changed=../scenemap/include/scenemap/walls.hpp");
    println!("cargo:rerun-if-changed=src/walls_ffi.cpp");
}

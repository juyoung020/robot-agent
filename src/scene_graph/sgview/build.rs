// 벽 2D 계산은 scenemap 의 C++ (walls.cpp) 를 그대로 컴파일해서 쓴다 — 파이썬 복사본도, Rust 로 다시 짠 복사본도 두지 않는다.
fn main() {
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

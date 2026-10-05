//! 공용 물체 색인(src/scene_graph/clip, C++)의 공유 라이브러리 libsgclip_c.so 에 링크한다.
//! 위치: 환경 변수 SGCLIP_LIB_DIR, 없으면 ~/sgclip_build (README "만들기"). 실행 파일에 rpath 를 박아 LD_LIBRARY_PATH 없이 돈다.
fn main() {
    let dir = std::env::var("SGCLIP_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/sgclip_build", std::env::var("HOME").unwrap_or_else(|_| ".".into())));
    println!("cargo:rerun-if-env-changed=SGCLIP_LIB_DIR");
    println!("cargo:rerun-if-changed={dir}/libsgclip_c.so");
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=sgclip_c");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
}

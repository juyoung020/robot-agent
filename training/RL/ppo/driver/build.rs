// libppo(C++/CUDA)를 CMake 로 빌드하고 동적 링크한다. PPO_BUILD_DIR 를 주면 그 빌드 폴더를 그대로 쓴다(빌드하지 않음).
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    println!("cargo:rerun-if-env-changed=PPO_BUILD_DIR");
    for d in ["src", "include", "CMakeLists.txt", "../network", "../observation", "../env/src", "../env/include", "../map/src", "../map/include"] {
        println!("cargo:rerun-if-changed={}", src.join(d).display());
    }
    let build = match std::env::var("PPO_BUILD_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            let b = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("cmake");
            let ok = Command::new("cmake").arg("-S").arg(&src).arg("-B").arg(&b).arg("-DCMAKE_BUILD_TYPE=Release").status().expect("cmake");
            assert!(ok.success(), "cmake configure failed");
            let ok = Command::new("cmake").arg("--build").arg(&b).arg("--target").arg("ppo").arg("-j").arg("16").status().expect("cmake --build");
            assert!(ok.success(), "cmake build failed");
            b
        }
    };
    println!("cargo:rustc-link-search=native={}", build.display());
    println!("cargo:rustc-link-lib=dylib=ppo");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build.display());
}

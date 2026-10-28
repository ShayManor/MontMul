//! With feature `gpu`, compiles the kernel crate `kernels/` to PTX through `cargo oxide` and
//! writes it to OUT_DIR, where `src/gpu/device.rs` embeds it. The kernel build runs from the
//! kernel crate's directory, so its toolchain pin and `RUSTC_BOOTSTRAP` never reach this crate.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }
    let kernels = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels");
    for f in ["src", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml"] {
        println!("cargo:rerun-if-changed={}", kernels.join(f).display());
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // OUT_DIR is <target>/<profile>/build/<package>-<hash>/out; one kernel target directory
    // per host target directory keeps the kernel dependencies compiled once.
    let target = out.ancestors().nth(4).unwrap().join("kernels");

    let mut cmd = Command::new("cargo");
    cmd.args(["oxide", "build", "--", "--release", "--locked"]).current_dir(&kernels);
    // Drop what Cargo sets for this build script, above all RUSTUP_TOOLCHAIN, which would
    // override the kernel crate's toolchain pin.
    for (key, _) in std::env::vars_os() {
        let k = key.to_string_lossy();
        let host_only = (k.starts_with("CARGO") && k != "CARGO_HOME")
            || k.starts_with("RUSTC")
            || k.starts_with("RUSTUP_TOOLCHAIN")
            || k.starts_with("RUSTDOC")
            || ["RUSTFLAGS", "OUT_DIR", "TARGET", "HOST", "PROFILE", "OPT_LEVEL", "DEBUG", "NUM_JOBS"].contains(&&*k);
        if host_only {
            cmd.env_remove(&key);
        }
    }
    cmd.env("CARGO_TARGET_DIR", &target).env("CUDA_OXIDE_PTX_DIR", &out);
    // A stale PTX must not pass the existence check below.
    let ptx = out.join("generalmodulo_kernels.ptx");
    let _ = std::fs::remove_file(&ptx);
    let res = cmd.output().unwrap_or_else(|e| panic!("cannot run cargo: {e}"));
    let stderr = String::from_utf8_lossy(&res.stderr);
    if !res.status.success() {
        if stderr.contains("no such command") {
            panic!("feature `gpu` needs cargo-oxide (`cargo install --path cuda-oxide/crates/cargo-oxide`)");
        }
        panic!(
            "kernel build failed ({}); the CUDA 13 / LLVM 21 environment of cuda-oxide must be set up\n{}\n{stderr}",
            res.status,
            String::from_utf8_lossy(&res.stdout),
        );
    }
    assert!(ptx.exists(), "kernel build succeeded but wrote no {}\n{stderr}", ptx.display());
}

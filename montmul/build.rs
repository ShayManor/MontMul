//! With feature `gpu`, compiles the kernel crate `kernels/` to PTX through `cargo oxide` and
//! writes it to OUT_DIR, where `src/gpu/device.rs` embeds it. The kernel build runs from the
//! kernel crate's directory, so its toolchain pin and `RUSTC_BOOTSTRAP` never reach this crate.
//! The published crate has no `kernels/` (Cargo leaves out nested packages) and embeds the
//! PTX shipped in `ptx/` instead.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let (kernels, shipped) = (manifest.join("kernels"), manifest.join("ptx/montmul_kernels.ptx"));
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx = out.join("montmul_kernels.ptx");
    println!("cargo:rerun-if-changed={}", shipped.display());
    if !kernels.join("Cargo.toml").exists() {
        std::fs::copy(&shipped, &ptx).unwrap_or_else(|e| panic!("cannot copy {}: {e}", shipped.display()));
        return;
    }
    for f in ["src", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml"] {
        println!("cargo:rerun-if-changed={}", kernels.join(f).display());
    }
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
    // cargo oxide writes the PTX only when it recompiles the kernels, so it goes to a directory
    // that outlives this OUT_DIR and is copied from there.
    let built = target.join("ptx").join("montmul_kernels.ptx");
    cmd.env("CARGO_TARGET_DIR", &target).env("CUDA_OXIDE_PTX_DIR", built.parent().unwrap());
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
    std::fs::copy(&built, &ptx).unwrap_or_else(|e| panic!("kernel build succeeded but wrote no {} ({e}); `cargo clean` forces a rebuild\n{stderr}", built.display()));
    if std::fs::read(&built).ok() != std::fs::read(&shipped).ok() {
        println!("cargo:warning=ptx/montmul_kernels.ptx differs from the kernels just built; copy {} there before publishing", built.display());
    }
}

#[test]
fn bench_cpu_prints_one_result_line() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_montmul"))
        .args(["bench", "--backend", "cpu", "--bits", "1000", "--slots", "2", "--steps", "20"])
        .output().unwrap();
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(out.status.success());
    assert!(s.contains("backend=cpu") && s.contains("L=64") && s.contains("bitsteps_per_s="));
}
#[test]
fn prp_t802_is_prp() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_montmul"))
        .args(["prp", "--index", "802", "--backend", "cpu"]).output().unwrap();
    assert!(String::from_utf8(out.stdout).unwrap().contains("prp=true"));
}
#[test]
fn bad_options_are_rejected() {
    for args in [
        &["score", "--backend", "cpu", "--slot", "8"][..],
        &["score", "--backend", "cpu", "--label", "--ncu"],
        &["score", "--backend", "cpu", "--bare"],
        &["bench", "--backend", "cpu", "--bits", "1000", "--bits", "2000", "--slots", "1", "--steps", "20"],
    ] {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_montmul")).args(args).output().unwrap();
        assert!(!out.status.success(), "{args:?} was accepted");
    }
}

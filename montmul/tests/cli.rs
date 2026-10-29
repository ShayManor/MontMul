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
#[test]
fn score_appends_to_csv() {
    let path = std::env::temp_dir().join(format!("montmul-score-{}.csv", std::process::id()));
    let _ = std::fs::remove_file(&path);
    for _ in 0..2 {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_montmul"))
            .args(["score", "--backend", "gmp", "--slots", "1", "--csv", path.to_str().unwrap()])
            .output().unwrap();
        assert!(out.status.success());
    }
    let csv = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 3, "{csv}");
    assert!(lines[0].starts_with("timestamp,label,device") && lines[1].contains(",gmp,1,"));
}

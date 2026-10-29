//! Command-line front end:
//!
//!   montmul bench --backend cpu|gpu|gmp --bits B --slots S --steps K [--bare] [--csv PATH]
//!   montmul prp --index N --backend cpu|gpu|gmp
//!   montmul score --backend cpu|gpu|gmp [--slots S] [--label TEXT] [--ncu] [--csv PATH]

use anyhow::{Context, Result, bail, ensure};
use montmul::score::{GMP_M5_SECONDS_PER_MODSQ_100K, Score, ScoreConfig, arm, best_slots, score};
use montmul::seq::tribonacci;
use montmul::{Backend, Cpu, Gmp, ModBatch, fermat};
use rug::Integer;
use rug::rand::RandState;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "gpu")]
use montmul::{Multi, gpu::Gpu};

/// Per subcommand: the `--key value` options, then the bare `--flag`s it accepts.
fn allowed(cmd: &str) -> (&'static [&'static str], &'static [&'static str]) {
    match cmd {
        "bench" => (&["backend", "bits", "slots", "steps", "csv"], &["bare"]),
        "prp" => (&["index", "backend"], &[]),
        "score" => (&["backend", "slots", "label", "csv"], &["ncu"]),
        "ncu-launch" => (&["slots"], &[]),
        _ => (&[], &[]),
    }
}

/// `--key value` options and bare `--flag`s.
struct Args {
    opts: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    /// Rejects unknown or repeated keys and a value that is itself an option.
    fn parse(cmd: &str, argv: &[String]) -> Result<Args> {
        let (keys, flag_keys) = allowed(cmd);
        let (mut opts, mut flags) = (HashMap::new(), Vec::new());
        let mut it = argv.iter();
        while let Some(a) = it.next() {
            let Some(key) = a.strip_prefix("--") else { bail!("unexpected argument {a}") };
            ensure!(!opts.contains_key(key) && !flags.iter().any(|f| f == key), "--{key} given twice");
            if flag_keys.contains(&key) {
                flags.push(key.to_string());
            } else if keys.contains(&key) {
                let v = it.next().filter(|v| !v.starts_with("--")).with_context(|| format!("--{key} needs a value"))?;
                opts.insert(key.to_string(), v.clone());
            } else {
                bail!("{cmd} does not take --{key}");
            }
        }
        Ok(Args { opts, flags })
    }
    fn get(&self, key: &str) -> Option<&str> {
        self.opts.get(key).map(String::as_str)
    }
    fn num<T: FromStr>(&self, key: &str) -> Result<T> {
        let v = self.get(key).with_context(|| format!("missing --{key}"))?;
        v.parse().map_err(|_| anyhow::anyhow!("--{key}: cannot parse {v}"))
    }
    fn flag(&self, key: &str) -> bool {
        self.flags.iter().any(|f| f == key)
    }
}

/// Calls `$f(factory, args...)` with the factory of the backend named `$name`.
macro_rules! with_backend {
    ($name:expr, $f:ident($($arg:expr),*)) => {
        match $name {
            "cpu" => $f(Cpu::default, $($arg),*),
            "gmp" => $f(Gmp::default, $($arg),*),
            #[cfg(feature = "gpu")]
            "gpu" => $f(|| Multi::new(Gpu::all().expect("CUDA device")), $($arg),*),
            other => bail!("unknown or unavailable backend {other}"),
        }
    };
}

/// Device name with whitespace replaced by `_`, so it fits a `key=value` field and a file name;
/// prefixed with the device count (`2x `) when the gpu backend uses several devices (`all`).
#[allow(unused_variables)]
fn device(backend: &str, all: bool) -> Result<String> {
    let name = match backend {
        #[cfg(feature = "gpu")]
        "gpu" => match if all { Gpu::device_count()? } else { 1 } {
            1 => Gpu::new()?.device_name(),
            g => format!("{g}x {}", Gpu::new()?.device_name()),
        },
        _ => cpu_model(),
    };
    Ok(name.split_whitespace().collect::<Vec<_>>().join("_").replace(',', ""))
}

fn cpu_model() -> String {
    let mac = std::process::Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]).output();
    if let Some(out) = mac.ok().filter(|o| o.status.success()) {
        return String::from_utf8_lossy(&out.stdout).trim().to_string();
    }
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split(':').nth(1)).map(|m| m.trim().to_string()))
        .unwrap_or_else(|| "cpu".to_string())
}

/// Appends one CSV row, writing the header first if the file is new or empty.
fn append_csv(path: &Path, header: &[&str], row: &[String]) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let fresh = std::fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if fresh {
        writeln!(f, "{}", header.join(","))?;
    }
    writeln!(f, "{}", row.join(","))?;
    Ok(())
}

struct BenchOpts {
    backend: String,
    device: String,
    bits: u32,
    slots: usize,
    steps: usize,
    bare: bool,
    csv: Option<PathBuf>,
}

/// Random odd moduli and `steps`-bit exponents, one warm-up `run(16)`, then one timed `run(steps)`.
fn bench<B: Backend>(make: impl Fn() -> B, o: &BenchOpts, run: impl Fn(&mut ModBatch<B>, usize)) -> Result<()> {
    ensure!(o.bits >= 2 && o.slots > 0 && o.steps > 0, "bits ≥ 2, slots > 0 and steps > 0 required");
    let mut rng = RandState::new();
    let moduli: Vec<Integer> = (0..o.slots)
        .map(|_| {
            let mut n = Integer::from(Integer::random_bits(o.bits, &mut rng));
            n.set_bit(o.bits - 1, true);
            n.set_bit(0, true);
            n
        })
        .collect();
    let exps: Vec<Integer> = (0..o.slots)
        .map(|_| {
            let mut e = Integer::from(Integer::random_bits(o.steps as u32, &mut rng));
            e.set_bit(o.steps as u32 - 1, true);
            e
        })
        .collect();
    let mut batch = ModBatch::new(make(), &moduli, 3)?;
    arm(&mut batch, &exps);
    run(&mut batch, 16);
    arm(&mut batch, &exps);
    let t = Instant::now();
    run(&mut batch, o.steps);
    let secs = t.elapsed().as_secs_f64();
    let us_per_step_slot = secs * 1e6 / (o.steps * o.slots) as f64;
    let bitsteps_per_s = o.bits as f64 * o.slots as f64 * o.steps as f64 / secs;
    let fields = [
        ("backend", o.backend.clone()),
        ("device", o.device.clone()),
        ("bits", o.bits.to_string()),
        ("L", batch.params().len.to_string()),
        ("slots", o.slots.to_string()),
        ("steps", o.steps.to_string()),
        ("bare", o.bare.to_string()),
        ("us_per_step_slot", format!("{us_per_step_slot:.3}")),
        ("bitsteps_per_s", format!("{bitsteps_per_s:.4e}")),
    ];
    println!("{}", fields.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" "));
    if let Some(path) = &o.csv {
        let header: Vec<&str> = fields.iter().map(|f| f.0).collect();
        append_csv(path, &header, &fields.map(|f| f.1))?;
    }
    Ok(())
}

/// Fermat base 3 on T(index).
fn prp<B: Backend>(make: impl Fn() -> B, index: usize) -> Result<()> {
    let t = tribonacci(index);
    let start = Instant::now();
    let got = fermat(make(), &[t.clone()], 3, 1 << 16)?;
    println!("T({index}) bits={} prp={} seconds={:.3}", t.significant_bits(), got[0], start.elapsed().as_secs_f64());
    Ok(())
}

/// The standard score at `slots`, or at the best of `candidates` when `slots` is `None`.
fn score_at<B: Backend>(make: impl Fn() -> B, slots: Option<usize>, candidates: &[usize]) -> Result<Score> {
    let slots = match slots {
        Some(s) => s,
        None => best_slots(&make, 100_000, candidates)?,
    };
    score(&make, &ScoreConfig::standard(slots))
}

/// Candidate slot counts for `best_slots`: logical cores × {1, 2} on the CPU; SMs over all devices
/// × {1, 2, 4, 8, 16} on the GPU, keeping only batches that fit in device memory.
fn candidates(backend: &str) -> Result<Vec<usize>> {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    match backend {
        #[cfg(feature = "gpu")]
        "gpu" => {
            let sms: usize = Gpu::all()?.iter().map(Gpu::sm_count).sum();
            let (moduli, _) = montmul::score::workload(&ScoreConfig::standard(1))?;
            Ok([1, 2, 4, 8, 16]
                .map(|f| f * sms)
                .into_iter()
                .filter(|&s| Gpu::all().is_ok_and(|g| ModBatch::new(Multi::new(g), &vec![moduli[0].clone(); s], 3).is_ok()))
                .collect())
        }
        _ => Ok(vec![cores, 2 * cores]),
    }
}

/// Runs one 16-step launch of the score workload under Nsight Compute and returns
/// (SM throughput %, memory throughput %) of the last `run_steps` kernel.
#[cfg(feature = "gpu")]
fn ncu_metrics(slots: usize) -> Result<(f64, f64)> {
    const SM: &str = "sm__throughput.avg.pct_of_peak_sustained_elapsed";
    const MEM: &str = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed";
    let out = std::process::Command::new("ncu")
        .args(["--csv", "--kernel-name", "regex:run_steps", "--metrics", &format!("{SM},{MEM}")])
        .arg(std::env::current_exe()?)
        .args(["ncu-launch", "--slots", &slots.to_string()])
        .output()
        .context("running ncu")?;
    ensure!(out.status.success(), "ncu failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    let rows: Vec<Vec<String>> = text
        .lines()
        .filter(|l| l.starts_with('"'))
        .map(|l| l.trim().trim_matches('"').split("\",\"").map(str::to_string).collect())
        .collect();
    let header = rows.first().context("no ncu output")?;
    let col = |name: &str| header.iter().position(|h| h == name).with_context(|| format!("ncu column {name}"));
    let (name, value) = (col("Metric Name")?, col("Metric Value")?);
    let last = |metric: &str| -> Result<f64> {
        let v = rows.iter().rev().find(|r| r.get(name).is_some_and(|n| n == metric)).with_context(|| format!("ncu metric {metric}"))?;
        Ok(v[value].replace(',', "").parse()?)
    };
    Ok((last(SM)?, last(MEM)?))
}

fn run_score(args: &Args) -> Result<()> {
    let backend = args.get("backend").context("missing --backend")?;
    let slots = args.get("slots").map(|_| args.num("slots")).transpose()?;
    let label = args.get("label").unwrap_or("");
    ensure!(backend == "gpu" || !args.flag("ncu"), "--ncu needs the gpu backend");
    let cands = if slots.is_none() { candidates(backend)? } else { vec![] };
    #[allow(unused_mut)]
    let mut s = with_backend!(backend, score_at(slots, &cands))?;
    #[cfg(feature = "gpu")]
    let (sm_pct, mem_pct) = if backend == "gpu" {
        // Device 0 alone, at the slot count it ran in the score.
        let per_device = s.slots.div_ceil(Gpu::device_count()?);
        let cfg = ScoreConfig::standard(per_device);
        s.overhead = Some(montmul::score::gpu_overhead(|| Gpu::new().expect("CUDA device"), &cfg)?);
        // A failed profile (e.g. no counter access) leaves the columns empty and keeps the score.
        match args.flag("ncu").then(|| ncu_metrics(per_device)) {
            Some(Ok((sm, mem))) => (Some(sm), Some(mem)),
            Some(Err(e)) => {
                eprintln!("warning: ncu profile failed, sm_pct and mem_pct left empty: {e:#}");
                (None, None)
            }
            None => (None, None),
        }
    } else {
        (None, None)
    };
    #[cfg(not(feature = "gpu"))]
    let (sm_pct, mem_pct) = (None, None);
    let device = device(backend, true)?;
    let opt = |v: Option<f64>| v.map(|x| format!("{x:.4}")).unwrap_or_default();
    let gmp_core_equiv = Some(s.modsq_per_s * GMP_M5_SECONDS_PER_MODSQ_100K);
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let fields = [
        ("timestamp", timestamp.to_string()),
        ("label", format!("\"{}\"", label.replace('"', "\"\""))),
        ("device", device.clone()),
        ("backend", backend.to_string()),
        ("slots", s.slots.to_string()),
        ("modsq_per_s", format!("{:.2}", s.modsq_per_s)),
        ("mad", format!("{:.2}", s.mad)),
        ("fermat_tests_per_hour", format!("{:.4}", s.fermat_tests_per_hour)),
        ("overhead", opt(s.overhead)),
        ("sm_pct", opt(sm_pct)),
        ("mem_pct", opt(mem_pct)),
        ("gmp_core_equiv", opt(gmp_core_equiv)),
    ];
    println!("{}", fields.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" "));
    if let Some(path) = args.get("csv") {
        let header: Vec<&str> = fields.iter().map(|f| f.0).collect();
        append_csv(Path::new(path), &header, &fields.map(|f| f.1))?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = argv.split_first() else { bail!("usage: montmul bench|prp|score [options]") };
    let args = Args::parse(cmd, rest)?;
    match cmd.as_str() {
        "bench" => {
            let backend = args.get("backend").context("missing --backend")?;
            let o = BenchOpts {
                backend: backend.to_string(),
                device: device(backend, !args.flag("bare"))?,
                bits: args.num("bits")?,
                slots: args.num("slots")?,
                steps: args.num("steps")?,
                bare: args.flag("bare"),
                csv: args.get("csv").map(PathBuf::from),
            };
            match (backend, o.bare) {
                #[cfg(feature = "gpu")]
                ("gpu", true) => bench(
                    || Gpu::new().expect("CUDA device"),
                    &o,
                    |b: &mut ModBatch<Gpu>, k| {
                        let (gpu, st) = b.backend_and_state();
                        gpu.run_bare(st, k);
                    },
                ),
                (_, true) => bail!("--bare needs the gpu backend"),
                _ => with_backend!(backend, bench(&o, |b, k| b.run(k))),
            }
        }
        "prp" => with_backend!(args.get("backend").context("missing --backend")?, prp(args.num("index")?)),
        "score" => run_score(&args),
        // Internal: one 16-step launch of the score workload, profiled by `score --ncu`.
        #[cfg(feature = "gpu")]
        "ncu-launch" => {
            let cfg = ScoreConfig::standard(args.num("slots")?);
            let (moduli, exps) = montmul::score::workload(&cfg)?;
            let mut batch = ModBatch::new(Gpu::new()?, &moduli, 3)?;
            arm(&mut batch, &exps);
            batch.run(16);
            Ok(())
        }
        other => bail!("unknown command {other}"),
    }
}

use generalmodulo::score::{ScoreConfig, score};
use generalmodulo::transform::BatchSetup;
use generalmodulo::{Backend, Cpu, Limits};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Delegates to `B` but flips bit 0 of limb 0 when reading slot `.1`.
struct CorruptRead<B>(B, usize);

impl<B: Limits> Limits for CorruptRead<B> {
    fn coefficient_bound(&self) -> u128 {
        self.0.coefficient_bound()
    }
    fn supported_lengths(&self) -> &[usize] {
        self.0.supported_lengths()
    }
}

impl<B: Backend> Backend for CorruptRead<B> {
    type State = B::State;

    fn create(&self, setup: &BatchSetup) -> anyhow::Result<B::State> {
        self.0.create(setup)
    }
    fn load(&self, st: &mut B::State, i: usize, limbs: &[u32]) {
        self.0.load(st, i, limbs)
    }
    fn set_exponent(&self, st: &mut B::State, i: usize, words: &[u32], nbits: usize) {
        self.0.set_exponent(st, i, words, nbits)
    }
    fn run(&self, st: &mut B::State, bits: usize) {
        self.0.run(st, bits)
    }
    fn mul(&self, st: &mut B::State, out: usize, a: usize, b: usize) {
        self.0.mul(st, out, a, b)
    }
    fn read(&self, st: &B::State, i: usize) -> Vec<u32> {
        let mut v = self.0.read(st, i);
        if i == self.1 {
            v[0] ^= 1;
        }
        v
    }
    fn remaining(&self, st: &B::State, i: usize) -> usize {
        self.0.remaining(st, i)
    }
}

#[test]
fn score_is_positive_and_checked() {
    let cfg = ScoreConfig { bits: 1_000, slots: 4, ebits: 100, repeats: 3, seed: 1 };
    let s = score(Cpu::default, &cfg).unwrap();
    assert!(s.modsq_per_s > 0.0 && s.mad >= 0.0);
    assert!((s.fermat_tests_per_hour - s.modsq_per_s * 3600.0 / 1000.0).abs() < 1e-6 * s.modsq_per_s);
}

#[test]
fn wrong_backend_gets_no_score() {
    // Corrupts only the last slot: the correctness check must still reject it.
    let cfg = ScoreConfig { bits: 1_000, slots: 4, ebits: 100, repeats: 1, seed: 1 };
    let err = score(|| CorruptRead(Cpu::default(), 3), &cfg).unwrap_err();
    assert!(err.to_string().contains("slot 3 disagrees with pow_mod"), "{err}");
}

/// Delegates to `Cpu`; each `run` that finds work left sleeps for the next entry of `.0`
/// (the last entry repeats) and counts itself in `.1`.
struct SlowRun(Vec<u64>, Arc<AtomicUsize>, Cpu);

impl Limits for SlowRun {
    fn coefficient_bound(&self) -> u128 {
        self.2.coefficient_bound()
    }
    fn supported_lengths(&self) -> &[usize] {
        self.2.supported_lengths()
    }
}

impl Backend for SlowRun {
    type State = <Cpu as Backend>::State;

    fn create(&self, setup: &BatchSetup) -> anyhow::Result<Self::State> {
        self.2.create(setup)
    }
    fn load(&self, st: &mut Self::State, i: usize, limbs: &[u32]) {
        self.2.load(st, i, limbs)
    }
    fn set_exponent(&self, st: &mut Self::State, i: usize, words: &[u32], nbits: usize) {
        self.2.set_exponent(st, i, words, nbits)
    }
    fn run(&self, st: &mut Self::State, bits: usize) {
        if self.2.remaining(st, 0) > 0 {
            let n = self.1.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(self.0[n.min(self.0.len() - 1)]));
        }
        self.2.run(st, bits)
    }
    fn mul(&self, st: &mut Self::State, out: usize, a: usize, b: usize) {
        self.2.mul(st, out, a, b)
    }
    fn read(&self, st: &Self::State, i: usize) -> Vec<u32> {
        self.2.read(st, i)
    }
    fn remaining(&self, st: &Self::State, i: usize) -> usize {
        self.2.remaining(st, i)
    }
}

#[test]
fn score_is_median_rate_over_reloaded_repeats() {
    // Warm-up, then timed repeats of 200, 200 and 20 ms: the median rate is slots · ebits / 200 ms
    // and the MAD is near 0, while the mean rate would be 4 times higher.
    let cfg = ScoreConfig { bits: 200, slots: 4, ebits: 100, repeats: 3, seed: 1 };
    let calls = Arc::new(AtomicUsize::new(0));
    let s = score(|| SlowRun(vec![0, 200, 200, 20], calls.clone(), Cpu::default()), &cfg).unwrap();
    let want = (cfg.slots * cfg.ebits) as f64 / 0.200;
    assert!((s.modsq_per_s - want).abs() < 0.3 * want, "rate {} vs {want}", s.modsq_per_s);
    assert!(s.mad < 0.1 * want, "mad {}", s.mad);
    // One warm-up plus one per repeat: each repeat must reload the slots to have work.
    assert_eq!(calls.load(Ordering::SeqCst), cfg.repeats + 1);
}

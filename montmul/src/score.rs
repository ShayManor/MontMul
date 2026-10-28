//! Efficiency score: sustained modular squarings per second on one fixed workload, reported
//! only after the results match GMP.

use crate::batch::ModBatch;
use crate::transform::Backend;
use anyhow::{Result, ensure};
use rug::Integer;
use rug::rand::RandState;
use std::time::Instant;

/// Median seconds per modular squaring of `Gmp` with one slot on one M5 performance core at
/// 100,000 bits (`score --backend gmp --slots 1`; median of three runs, 3,691 squarings per second).
pub const GMP_M5_SECONDS_PER_MODSQ_100K: f64 = 2.709e-4;

#[derive(Clone, Copy, Debug)]
pub struct ScoreConfig {
    pub bits: u32,
    pub slots: usize,
    pub ebits: usize,
    pub repeats: usize,
    pub seed: u64,
}

impl ScoreConfig {
    /// The score workload: 100,000-bit moduli (L = 8,192), 2,000-bit exponents, 5 repeats.
    pub fn standard(slots: usize) -> Self {
        ScoreConfig { bits: 100_000, slots, ebits: 2_000, repeats: 5, seed: 2026 }
    }
}

#[derive(Clone, Debug)]
pub struct Score {
    pub slots: usize,
    /// Median over repeats of slots · ebits / seconds.
    pub modsq_per_s: f64,
    /// Median absolute deviation of `modsq_per_s`.
    pub mad: f64,
    /// modsq_per_s · 3600 / bits.
    pub fermat_tests_per_hour: f64,
    /// Spec section 11.1: per-step time of `run` over `run_bare` (GPU only).
    pub overhead: Option<f64>,
}

/// Random odd moduli with the top bit set, and per modulus the top `ebits` bits of N − 1.
pub fn workload(cfg: &ScoreConfig) -> Result<(Vec<Integer>, Vec<Integer>)> {
    ensure!(cfg.slots > 0 && cfg.repeats > 0, "slots and repeats must be positive");
    ensure!(cfg.ebits > 0 && cfg.ebits <= cfg.bits as usize, "ebits must be in 1..=bits");
    let mut rng = RandState::new();
    rng.seed(&Integer::from(cfg.seed));
    let moduli: Vec<Integer> = (0..cfg.slots)
        .map(|_| {
            let mut n = Integer::from(Integer::random_bits(cfg.bits, &mut rng));
            n.set_bit(cfg.bits - 1, true);
            n.set_bit(0, true);
            n
        })
        .collect();
    let shift = cfg.bits - cfg.ebits as u32;
    let exps = moduli.iter().map(|n| (n.clone() - 1u32) >> shift).collect();
    Ok((moduli, exps))
}

/// Loads 1 and the exponent into every slot.
pub fn arm<B: Backend>(batch: &mut ModBatch<B>, exps: &[Integer]) {
    for (i, e) in exps.iter().enumerate() {
        batch.load(i, &Integer::from(1));
        batch.set_exponent(i, e);
    }
}

pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let m = s.len() / 2;
    if s.len() % 2 == 1 { s[m] } else { (s[m - 1] + s[m]) / 2.0 }
}

/// Median absolute deviation from the median.
pub fn mad(v: &[f64]) -> f64 {
    let m = median(v);
    median(&v.iter().map(|x| (x - m).abs()).collect::<Vec<_>>())
}

/// Acceptance rule for the hill climb: `new` is kept only if its median beats `old` by at least
/// 2 % and by more than twice the larger MAD.
pub fn accept(old: &Score, new: &Score) -> bool {
    let gain = new.modsq_per_s - old.modsq_per_s;
    gain >= 0.02 * old.modsq_per_s && gain > 2.0 * old.mad.max(new.mad)
}

/// One warm-up `run(64)`, then `repeats` timed `run(ebits)` on freshly loaded slots. Fails if
/// slot 0, the last slot or two seeded random slots disagree with `pow_mod`.
pub fn score<B: Backend>(make: impl Fn() -> B, cfg: &ScoreConfig) -> Result<Score> {
    let (moduli, exps) = workload(cfg)?;
    let mut batch = ModBatch::new(make(), &moduli, 3)?;
    arm(&mut batch, &exps);
    batch.run(64);
    let mut rates = Vec::with_capacity(cfg.repeats);
    for _ in 0..cfg.repeats {
        arm(&mut batch, &exps);
        let t = Instant::now();
        batch.run(cfg.ebits);
        rates.push((cfg.slots * cfg.ebits) as f64 / t.elapsed().as_secs_f64());
    }
    let mut rng = RandState::new();
    rng.seed(&Integer::from(cfg.seed + 1));
    let mut checked = vec![0, cfg.slots - 1];
    for _ in 0..2 {
        checked.push(Integer::from(cfg.slots).random_below(&mut rng).to_usize().unwrap());
    }
    for i in checked {
        let want = Integer::from(3).pow_mod(&exps[i], &moduli[i]).unwrap();
        ensure!(batch.remaining(i) == 0 && batch.read(i) == want, "slot {i} disagrees with pow_mod");
    }
    let modsq_per_s = median(&rates);
    Ok(Score {
        slots: cfg.slots,
        modsq_per_s,
        mad: mad(&rates),
        fermat_tests_per_hour: modsq_per_s * 3600.0 / cfg.bits as f64,
        overhead: None,
    })
}

/// The candidate slot count with the highest `score` at `ebits` = 200 and one repeat.
pub fn best_slots<B: Backend>(make: impl Fn() -> B, bits: u32, candidates: &[usize]) -> Result<usize> {
    ensure!(!candidates.is_empty(), "no candidate slot counts");
    let mut best = (candidates[0], 0.0);
    for &slots in candidates {
        let s = score(&make, &ScoreConfig { bits, slots, ebits: 200, repeats: 1, seed: 2026 })?;
        if s.modsq_per_s > best.1 {
            best = (slots, s.modsq_per_s);
        }
    }
    Ok(best.0)
}

/// Spec section 11.1 on the score workload: median per-step time of `run` over that of `run_bare`.
#[cfg(feature = "gpu")]
pub fn gpu_overhead(make: impl Fn() -> crate::gpu::Gpu, cfg: &ScoreConfig) -> Result<f64> {
    let (moduli, exps) = workload(cfg)?;
    let mut batch = ModBatch::new(make(), &moduli, 3)?;
    let mut time = |bare: bool| {
        arm(&mut batch, &exps);
        let t = Instant::now();
        if bare {
            let (gpu, st) = batch.backend_and_state();
            gpu.run_bare(st, cfg.ebits);
        } else {
            batch.run(cfg.ebits);
        }
        t.elapsed().as_secs_f64()
    };
    time(false);
    time(true);
    let full: Vec<f64> = (0..cfg.repeats).map(|_| time(false)).collect();
    let bare: Vec<f64> = (0..cfg.repeats).map(|_| time(true)).collect();
    Ok(median(&full) / median(&bare))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(modsq_per_s: f64, mad: f64) -> Score {
        Score { slots: 1, modsq_per_s, mad, fermat_tests_per_hour: 0.0, overhead: None }
    }

    #[test]
    fn median_and_mad() {
        assert_eq!(median(&[5.0, 1.0, 3.0]), 3.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 10.0]), 3.5);
        // Deviations from 4: 3, 1, 1, 0, 96 → median 1 (the mean deviation would be 20.2).
        assert_eq!(mad(&[1.0, 5.0, 3.0, 4.0, 100.0]), 1.0);
        // Deviations from 3.5: 2.5, 0.5, 0.5, 6.5 → median 1.5.
        assert_eq!(mad(&[1.0, 4.0, 3.0, 10.0]), 1.5);
    }

    #[test]
    fn acceptance_rule() {
        assert!(accept(&at(100.0, 0.5), &at(103.0, 0.5)));
        assert!(!accept(&at(100.0, 0.1), &at(101.9, 0.1)), "below 2 %");
        assert!(!accept(&at(100.0, 2.0), &at(103.0, 0.5)), "within 2 · MAD_old");
        assert!(!accept(&at(100.0, 0.5), &at(103.0, 1.5)), "within 2 · MAD_new");
        assert!(!accept(&at(100.0, 0.5), &at(97.0, 0.5)), "slower");
    }
}

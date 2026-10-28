mod common;
use common::*;
use generalmodulo::{Cpu, ModBatch, fermat};
use rug::Integer;

#[test]
fn exactness_bound() {
    exactness(&Cpu::default(), 1 << 18);
}

#[test]
fn transform_batched_all_lengths() {
    transform_batched(&Cpu::default(), 1 << 14);
}

#[test]
fn random_modmul_sizes() {
    for (bits, cases) in [(64, 300), (100, 300), (1000, 200), (10_000, 50), (100_000, 8), (1 << 20, 2), (1 << 22, 1)] {
        for base in [3, 4] {
            random_modmul(Cpu::default, bits, cases, bits, base);
        }
    }
}

#[test]
fn edge_values() {
    for base in [3, 4] {
        edges(Cpu::default, base);
    }
}

#[test]
fn independence() {
    batch_independence(Cpu::default, 3000, &[1, 7, 64]);
}

#[test]
fn powmod_mul() {
    powmod_and_mul(Cpu::default, 2000, 9);
}

#[test]
fn fermat_tribonacci() {
    let prps = A092835_SMALL;
    let mut ns: Vec<usize> = prps.to_vec();
    ns.extend([100, 200, 1001, 2000, 4000]);
    end_to_end(Cpu::default, &ns);
}

#[test]
fn api() {
    api_contract(Cpu::default);
}

#[test]
fn carries() {
    carry_stress(Cpu::default, 1 << 13);
}

#[test]
fn tribonacci_prps() {
    known_prps(Cpu::default, &A092835_SMALL, &[8, 9, 10, 12, 13, 20, 86, 88, 803]);
}

#[test]
#[ignore = "slow: T(4202), 3,600 bits"]
fn tribonacci_prps_large() {
    known_prps(Cpu::default, &[4202], &[4203]);
}

#[test]
fn same_raw_across_batches() {
    for bits in [100, 5000] {
        assert_same_raw(Cpu::default, Cpu::default, bits, 4, 50);
    }
}

#[test]
fn long_run_matches_powmod() {
    long_run(Cpu::default, 3000, 4, 2000);
}

#[test]
fn is_one_accepts_upper_lazy_representative() {
    let mut rng = rug::rand::RandState::new();
    let n = rand_odd(&mut rng, 1000);
    let mut b = ModBatch::new(Cpu::default(), &[n.clone()], 3).unwrap();
    let r = (Integer::from(1) << (32 * b.params().k as u32)).modulo(&n);
    b.load_raw(0, &(r.clone() + &n));
    assert!(b.is_one(0));
    b.load_raw(0, &r);
    assert!(b.is_one(0));
    b.load_raw(0, &(r + 1u32));
    assert!(!b.is_one(0));
}

#[test]
#[should_panic(expected = "raw value outside [0, 2N)")]
fn load_raw_rejects_out_of_range() {
    let n = Integer::from(1_000_003);
    let mut b = ModBatch::new(Cpu::default(), &[n.clone()], 3).unwrap();
    b.load_raw(0, &(n * 2u32));
}

#[test]
fn fermat_chunk_zero_is_error() {
    let n = Integer::from(1_000_003);
    assert!(fermat(Cpu::default(), &[n], 3, 0).is_err());
}

#[test]
fn wraparound_matches_full_product() {
    // Same k = L/2 on both sides, so the raw Montgomery limbs must agree exactly.
    for bits in [64, 1000, 87_914, 1 << 20] {
        assert_same_raw(|| Cpu::default().with_full_k(true), || Cpu::with_wraparound(true), bits, 8, 100);
    }
    long_run(|| Cpu::with_wraparound(true), 3000, 4, 2000);
    powmod_and_mul(|| Cpu::with_wraparound(true), 2000, 9);
    random_modmul(|| Cpu::with_wraparound(true), 1000, 100, 1000, 3);
    edges(|| Cpu::with_wraparound(true), 3);
}

mod common;
use common::*;
use montmul::{Cpu, Gmp};

#[test]
fn random_modmul_sizes() {
    for (bits, cases) in [(64, 300), (100, 300), (1000, 200), (10_000, 50), (100_000, 8), (1 << 20, 2), (1 << 22, 1)] {
        for base in [3, 4] {
            random_modmul(Gmp::default, bits, cases, bits, base);
        }
    }
}

#[test]
fn edge_values() {
    for base in [3, 4] {
        edges(Gmp::default, base);
    }
}

#[test]
fn independence() {
    batch_independence(Gmp::default, 3000, &[1, 7, 64]);
}

#[test]
fn powmod_mul() {
    powmod_and_mul(Gmp::default, 2000, 9);
}

#[test]
fn fermat_tribonacci() {
    let mut ns: Vec<usize> = A092835_SMALL.to_vec();
    ns.extend([100, 200, 1001, 2000, 4000]);
    end_to_end(Gmp::default, &ns);
}

#[test]
fn api() {
    api_contract(Gmp::default);
}

#[test]
fn carries() {
    carry_stress(Gmp::default, 1 << 13);
}

#[test]
fn tribonacci_prps() {
    known_prps(Gmp::default, &A092835_SMALL, &[8, 9, 10, 12, 13, 20, 86, 88, 803]);
}

#[test]
fn tribonacci_prps_large() {
    known_prps(Gmp::default, &[4202], &[4203]);
}

#[test]
fn long_run_matches_powmod() {
    long_run(Gmp::default, 3000, 4, 2000);
}

#[test]
fn same_raw_as_cpu() {
    assert_same_raw(Gmp::default, Cpu::default, 87_914, 4, 100);
}

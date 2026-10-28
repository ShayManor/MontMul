mod common;
use common::*;
use generalmodulo::Cpu;

#[test]
fn exactness_bound() {
    exactness(&Cpu::default(), 1 << 14);
}

#[test]
fn random_modmul_sizes() {
    for (bits, cases) in [(64, 300), (100, 300), (1000, 200), (10_000, 50), (100_000, 8), (1 << 20, 2)] {
        random_modmul(Cpu::default, bits, cases, bits);
    }
}

#[test]
fn edge_values() {
    edges(Cpu::default);
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

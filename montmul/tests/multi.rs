mod common;
use common::*;
use montmul::{Cpu, Gmp, Limits, ModBatch, Multi, Params, fermat};
use rug::Integer;
use rug::rand::RandState;

fn multi_cpu(parts: usize) -> Multi<Cpu> {
    Multi::new((0..parts).map(|_| Cpu::with_wraparound(true)).collect())
}

fn multi_gmp(parts: usize) -> Multi<Gmp> {
    Multi::new((0..parts).map(|_| Gmp::default()).collect())
}

#[test]
fn same_raw_as_one_backend() {
    // Uneven splits and fewer slots than parts.
    for parts in 1..=3 {
        for slots in [1, 2, 3, 5, 7] {
            for bits in [100, 3000] {
                assert_same_raw(|| multi_cpu(parts), || Cpu::with_wraparound(true), bits, slots, 50);
            }
        }
    }
}

#[test]
fn mul_within_and_across_shards() {
    for parts in 1..=3 {
        powmod_and_mul(|| multi_cpu(parts), 2000, 7);
    }
    // Five slots over three parts: shards {0}, {1, 2}, {3, 4}.
    let mut rng = RandState::new();
    let n = rand_odd(&mut rng, 3000);
    let mut b = ModBatch::new(multi_cpu(3), &vec![n.clone(); 5], 3).unwrap();
    let xs: Vec<Integer> = (0..5).map(|_| Integer::from(n.random_below_ref(&mut rng))).collect();
    for (out, a, bb) in [(2, 1, 2), (0, 3, 4), (4, 0, 1), (1, 1, 3), (3, 3, 3)] {
        for (i, x) in xs.iter().enumerate() {
            b.load(i, x);
        }
        b.mul(out, a, bb);
        assert_eq!(b.read(out), (xs[a].clone() * &xs[bb]).modulo(&n), "mul({out}, {a}, {bb})");
        assert!(b.read_raw(out) < n.clone() * 2u32, "lazy range");
        for i in (0..5).filter(|&i| i != out) {
            assert_eq!(b.read(i), xs[i], "slot {i} unchanged by mul({out}, {a}, {bb})");
        }
    }
}

#[test]
fn fermat_matches_gmp() {
    let mut ns: Vec<usize> = A092835_SMALL.to_vec();
    ns.extend([100, 200, 1001, 2000, 4000]);
    let lengths = Gmp::default().supported_lengths().to_vec();
    let mut groups: std::collections::BTreeMap<usize, Vec<Integer>> = Default::default();
    for t in ns.iter().map(|&n| tribonacci(n)).filter(|t| t.is_odd() && *t > 3) {
        groups.entry(Params::for_bits(t.significant_bits(), &lengths).unwrap().len).or_default().push(t);
    }
    for ms in groups.values() {
        let got = fermat(multi_gmp(3), ms, 3, 500).unwrap();
        assert_eq!(got, fermat(Gmp::default(), ms, 3, 500).unwrap(), "{} moduli", ms.len());
    }
    end_to_end(|| multi_gmp(3), &ns);
    known_prps(|| multi_gmp(3), &A092835_SMALL, &[8, 9, 10, 12, 13, 20, 86, 88, 803]);
}

#[test]
fn api() {
    api_contract(|| multi_cpu(3));
}

#[test]
#[should_panic(expected = "part 1")]
fn mismatched_parts_rejected() {
    Multi::new(vec![Cpu::with_wraparound(true), Cpu::with_wraparound(false)]);
}

#[test]
#[should_panic(expected = "no parts")]
fn empty_rejected() {
    Multi::<Cpu>::new(vec![]);
}

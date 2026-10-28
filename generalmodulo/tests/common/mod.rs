//! Backend-generic correctness tests (spec section 10); every check compares against GMP.
#![allow(dead_code)]

use generalmodulo::batch::to_limbs;
use generalmodulo::{Backend, ModBatch, Params};
use rug::rand::RandState;
use rug::{Assign, Integer};

pub fn rand_odd(rng: &mut RandState, bits: u32) -> Integer {
    let mut n = Integer::from(Integer::random_bits(bits, rng));
    n.set_bit(bits - 1, true);
    n.set_bit(0, true);
    n
}

/// x² · c · R⁻¹ mod N, the expected step result.
fn expected(x: &Integer, c: u32, n: &Integer, k: usize) -> Integer {
    let rinv = (Integer::from(1) << (32 * k as u32)).invert(n).unwrap();
    (x.clone() * x * c * rinv).modulo(n)
}

/// 1. All-ones limbs at maximum k for every L up to `max_len`: the convolution is exact.
pub fn exactness<B: Backend>(b: &B, max_len: usize) {
    for &len in b.supported_lengths().iter().filter(|&&l| l <= max_len) {
        let k = len / 2;
        let c = 4u32;
        Params { k, len }.check_exact(c, b.coefficient_bound()).unwrap();
        let mut coeffs = vec![0u32; len];
        coeffs[..k].fill(u32::MAX);
        let mut x = b.upload(&coeffs, len, 1);
        b.forward(&mut x, len, 1);
        let mut y = b.upload(&coeffs, len, 1);
        b.forward(&mut y, len, 1);
        let s = b.upload_scales(&[c]);
        let mut out = b.upload(&vec![0; len], len, 1);
        b.pointwise(&mut out, &x, &y, Some(&s), len, 1);
        b.inverse(&mut out, len, 1);
        let got = b.download(&out);
        let mut val = Integer::new();
        for (q, &v) in got.iter().enumerate() {
            val += Integer::from(v) << (32 * q as u32);
        }
        let a = (Integer::from(1) << (32 * k as u32)) - 1u32;
        assert_eq!(val, a.clone() * &a * c, "len {len}");
        // Each coefficient must also match its exact value c · (2^32 − 1)² · min(q + 1, 2k − 1 − q).
        let sq = (1u128 << 32) - 1;
        for q in 0..len {
            let cnt = if q < k { q + 1 } else if q < 2 * k - 1 { 2 * k - 1 - q } else { 0 } as u128;
            assert_eq!(got[q], c as u128 * sq * sq * cnt, "len {len} coeff {q}");
        }
    }
}

/// 2. Random odd N of `bits`, random x in [0, 2N), one step each with c = 1 and c = base.
pub fn random_modmul<B: Backend>(make: impl Fn() -> B, bits: u32, cases: usize, seed: u32) {
    let mut rng = RandState::new();
    rng.seed(&Integer::from(seed));
    let moduli: Vec<Integer> = (0..cases).map(|_| rand_odd(&mut rng, bits)).collect();
    let base = 3;
    let mut batch = ModBatch::new(make(), &moduli, base).unwrap();
    let k = batch.params().k;
    let xs: Vec<Integer> = moduli
        .iter()
        .map(|n| Integer::from((n.clone() * 2u32).random_below_ref(&mut rng)))
        .collect();
    // Exponent 0b10: the first step uses c = base, the second c = 1.
    for (i, x) in xs.iter().enumerate() {
        batch.load_raw(i, x);
        batch.set_exponent(i, &Integer::from(2));
    }
    let mut prev = xs.clone();
    for c in [base, 1] {
        batch.run(1);
        for (i, x) in prev.iter_mut().enumerate() {
            let n = &moduli[i];
            let raw = batch.read_raw(i);
            assert!(raw < n.clone() * 2u32, "lazy range, bits {bits} slot {i}");
            assert_eq!(raw.clone().modulo(n), expected(x, c, n, k), "bits {bits} slot {i} c {c}");
            *x = raw;
        }
    }
}

/// 3. Edge values x = 0, N, 2N − 1, and N with k · w = b + 4 exactly.
pub fn edges<B: Backend>(make: impl Fn() -> B) {
    let mut rng = RandState::new();
    // b = 32k − 4 gives k · w = b + 4; also the largest and smallest b for several L.
    let mut sizes = vec![28u32, 60, 124, 252, 1020, 4092, 32764];
    for len in [64usize, 1024, 8192] {
        let kmax = len / 2;
        sizes.push(32 * kmax as u32 - 4); // top of the bucket
        sizes.push(32 * (kmax / 2) as u32 - 3); // bottom of the bucket
    }
    for bits in sizes {
        let moduli = vec![rand_odd(&mut rng, bits), (Integer::from(1) << bits) - 1u32];
        let mut batch = ModBatch::new(make(), &moduli, 3).unwrap();
        let k = batch.params().k;
        for which in 0..3 {
            for (i, n) in moduli.iter().enumerate() {
                let x = match which {
                    0 => Integer::new(),
                    1 => n.clone(),
                    _ => n.clone() * 2u32 - 1u32,
                };
                batch.load_raw(i, &x);
                batch.set_exponent(i, &Integer::from(1));
            }
            batch.run(1);
            for (i, n) in moduli.iter().enumerate() {
                let x = match which {
                    0 => Integer::new(),
                    1 => n.clone(),
                    _ => n.clone() * 2u32 - 1u32,
                };
                let raw = batch.read_raw(i);
                assert!(raw < n.clone() * 2u32, "lazy range bits {bits}");
                assert_eq!(raw.modulo(n), expected(&x, 3, n, k), "bits {bits} case {which}");
            }
        }
    }
}

/// 5. One modulus gives identical results in every slot position and batch size.
pub fn batch_independence<B: Backend>(make: impl Fn() -> B, bits: u32, sizes: &[usize]) {
    let mut rng = RandState::new();
    let target = rand_odd(&mut rng, bits);
    let e = Integer::from(Integer::random_bits(64, &mut rng)) | 1u32;
    let mut reference: Option<Vec<u32>> = None;
    for &size in sizes {
        let mut moduli: Vec<Integer> = (0..size).map(|_| rand_odd(&mut rng, bits)).collect();
        let pos = (size * 5 / 7).min(size - 1);
        for p in [0, pos, size - 1] {
            moduli[p].assign(&target);
        }
        let mut batch = ModBatch::new(make(), &moduli, 3).unwrap();
        for i in 0..size {
            batch.load(i, &Integer::from(1));
            batch.set_exponent(i, &e);
        }
        batch.run(1000);
        for p in [0, pos, size - 1] {
            let raw = to_limbs(&batch.read_raw(p), batch.params().k);
            match &reference {
                None => reference = Some(raw),
                Some(r) => assert_eq!(&raw, r, "size {size} slot {p}"),
            }
            assert_eq!(batch.read(p), Integer::from(3).pow_mod(&e, &target).unwrap());
        }
    }
}

/// Full exponentiation against pow_mod, plus `mul`.
pub fn powmod_and_mul<B: Backend>(make: impl Fn() -> B, bits: u32, count: usize) {
    let mut rng = RandState::new();
    let moduli: Vec<Integer> = (0..count).map(|_| rand_odd(&mut rng, bits)).collect();
    let mut batch = ModBatch::new(make(), &moduli, 3).unwrap();
    let mut exps = Vec::new();
    for (i, n) in moduli.iter().enumerate() {
        let e = Integer::from(Integer::random_bits(32 + i as u32 * 7, &mut rng)) | 1u32;
        batch.load(i, &Integer::from(1));
        batch.set_exponent(i, &e);
        exps.push(e);
        let _ = n;
    }
    while (0..count).any(|i| batch.remaining(i) > 0) {
        batch.run(37);
    }
    for (i, n) in moduli.iter().enumerate() {
        assert_eq!(batch.read(i), Integer::from(3).pow_mod(&exps[i], n).unwrap(), "slot {i}");
    }
    // mul: slots sharing one modulus.
    let n = rand_odd(&mut rng, bits);
    let mut batch = ModBatch::new(make(), &[n.clone(), n.clone(), n.clone()], 3).unwrap();
    let a = Integer::from(n.random_below_ref(&mut rng));
    let b = Integer::from(n.random_below_ref(&mut rng));
    batch.load(0, &a);
    batch.load(1, &b);
    batch.mul(2, 0, 1);
    assert_eq!(batch.read(2), (a.clone() * &b).modulo(&n));
    batch.mul(0, 0, 0);
    assert_eq!(batch.read(0), (a.clone() * &a).modulo(&n));
}

/// Tribonacci T(n) with T(0) = T(1) = 0, T(2) = 1 (OEIS A000073).
pub fn tribonacci(n: usize) -> Integer {
    let (mut a, mut b, mut c) = (Integer::from(0), Integer::from(0), Integer::from(1));
    for _ in 0..n {
        let d = Integer::from(&a + &b) + &c;
        a = b;
        b = c;
        c = d;
    }
    a
}

/// 6. Fermat base 3 on T(n) matches pow_mod; listed tribonacci PRPs report PRP.
pub fn end_to_end<B: Backend>(make: impl Fn() -> B, ns: &[usize]) {
    let moduli: Vec<Integer> = ns.iter().map(|&n| tribonacci(n)).filter(|t| t.is_odd() && *t > 3).collect();
    // Group by transform length.
    let lengths = make().supported_lengths().to_vec();
    let mut groups: std::collections::BTreeMap<usize, Vec<Integer>> = Default::default();
    for m in moduli {
        groups.entry(Params::for_bits(m.significant_bits(), &lengths).unwrap().len).or_default().push(m);
    }
    for (_, ms) in groups {
        let got = generalmodulo::fermat(make(), &ms, 3, 500).unwrap();
        for (m, g) in ms.iter().zip(got) {
            let want = Integer::from(3).pow_mod(&(m.clone() - 1u32), m).unwrap() == 1;
            assert_eq!(g, want, "T with {} bits", m.significant_bits());
        }
    }
}

/// OEIS A092835 lists n − 1 for prime T(n) (A000073 indexing): 3, 5, 6, 10, 86, 97, 214,
/// 801, 4201, 18698, 96878. T(4) = 2 is even, so these are the odd ones below 1000 bits.
pub const A092835_SMALL: [usize; 7] = [6, 7, 11, 87, 98, 215, 802];
/// The larger entries: T(4202), T(18699), T(96879).
pub const A092835_LARGE: [usize; 3] = [4202, 18699, 96879];

/// Known tribonacci primes must report PRP; composites T(n) must not.
pub fn known_prps<B: Backend>(make: impl Fn() -> B, prime_ns: &[usize], composite_ns: &[usize]) {
    for (&n, want) in prime_ns.iter().map(|n| (n, true)).chain(composite_ns.iter().map(|n| (n, false))) {
        let t = tribonacci(n);
        if t.is_even() {
            continue;
        }
        let got = generalmodulo::fermat(make(), &[t.clone()], 3, 1 << 20).unwrap();
        assert_eq!(got[0], want, "T({n})");
    }
}

/// API contract: parameter rules, rejection of bad input, canonical read, is_one, chunked runs.
pub fn api_contract<B: Backend>(make: impl Fn() -> B) {
    let lengths = make().supported_lengths().to_vec();
    assert!(lengths.iter().all(|l| l.is_power_of_two()));
    // Rejections.
    assert!(ModBatch::new(make(), &[], 3).is_err(), "empty batch");
    assert!(ModBatch::new(make(), &[Integer::from(10)], 3).is_err(), "even modulus");
    assert!(ModBatch::new(make(), &[Integer::from(1)], 3).is_err(), "modulus 1");
    assert!(ModBatch::new(make(), &[Integer::from(-7)], 3).is_err(), "negative modulus");
    assert!(ModBatch::new(make(), &[Integer::from(11)], 5).is_err(), "base above lazy range");
    assert!(ModBatch::new(make(), &[Integer::from(11)], 0).is_err(), "base 0");
    let small = Integer::from(101);
    let big = (Integer::from(1) << 5000u32) + 1u32;
    assert!(ModBatch::new(make(), &[small.clone(), big], 3).is_err(), "mixed transform lengths");
    // Parameter selection: k · 32 ≥ b + 4 and the smallest L ≥ 2k.
    for bits in [2u32, 28, 29, 60, 61, 1000, 87_914] {
        let p = Params::for_bits(bits, &lengths).unwrap();
        assert!(p.k * 32 >= bits as usize + 4 && (p.k - 1) * 32 < bits as usize + 4);
        assert!(p.len >= 2 * p.k && (p.len == lengths[0] || p.len / 2 < 2 * p.k));
    }
    assert!(Params { k: 1 << 30, len: 1 << 31 }.check_exact(4, make().coefficient_bound()).is_err());
    // Tiny moduli, load of values ≥ N and negative values, canonical read, is_one.
    let mut rng = RandState::new();
    for n in [3u32, 5, 7, 9, 15, 101, 65_537, 4_294_967_291, u32::MAX] {
        let n = Integer::from(n);
        let mut b = ModBatch::new(make(), &[n.clone()], 3).unwrap();
        for x in [Integer::from(0), Integer::from(1), n.clone(), n.clone() * 5u32 + 2u32, Integer::from(-4)] {
            b.load(0, &x);
            assert_eq!(b.read(0), x.clone().modulo(&n), "read after load, n {n}");
            assert_eq!(b.is_one(0), x.clone().modulo(&n) == 1, "is_one, n {n}");
            let e = Integer::from(Integer::random_bits(40, &mut rng)) + 1u32;
            b.load(0, &x);
            b.set_exponent(0, &e);
            b.run(1 << 20);
            assert_eq!(b.remaining(0), 0);
            // x^(2^bits(e)) · 3^e: the loaded value is squared once per exponent bit.
            let sq = Integer::from(1) << e.significant_bits();
            let want = (x.clone().modulo(&n).pow_mod(&sq, &n).unwrap() * Integer::from(3).pow_mod(&e, &n).unwrap()).modulo(&n);
            assert_eq!(b.read(0), want, "pow, n {n}");
        }
    }
    // Chunked runs equal one long run; run(0) is a no-op; finished slots stay frozen.
    let moduli: Vec<Integer> = (0..5).map(|_| rand_odd(&mut rng, 777)).collect();
    let exps: Vec<Integer> = (0..5).map(|i| Integer::from(Integer::random_bits(10 + 60 * i, &mut rng)) | 1u32).collect();
    let mut a = ModBatch::new(make(), &moduli, 3).unwrap();
    let mut b = ModBatch::new(make(), &moduli, 3).unwrap();
    for i in 0..5 {
        for m in [&mut a, &mut b] {
            m.load(i, &Integer::from(1));
            m.set_exponent(i, &exps[i]);
        }
    }
    a.run(0);
    assert_eq!(a.remaining(4), exps[4].significant_bits() as usize);
    a.run(1000);
    for chunk in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 1000] {
        b.run(chunk);
    }
    for i in 0..5 {
        assert_eq!(a.read(i), Integer::from(3).pow_mod(&exps[i], &moduli[i]).unwrap());
        assert_eq!(a.read_raw(i), b.read_raw(i), "chunked run slot {i}");
    }
    a.run(100);
    assert_eq!(a.read(0), Integer::from(3).pow_mod(&exps[0], &moduli[0]).unwrap(), "frozen after finish");
    // Re-arming a slot restarts it: 2^(2^3) · 3^5 for e = 0b101.
    a.load(0, &Integer::from(2));
    a.set_exponent(0, &Integer::from(5));
    a.run(10);
    assert_eq!(a.read(0), (Integer::from(256) * 243u32).modulo(&moduli[0]));
}

/// Inputs that maximize carries: x = 2N − 1 with N = 2^b − 1, repeated squaring, every bucket.
pub fn carry_stress<B: Backend>(make: impl Fn() -> B, max_len: usize) {
    for &len in make().supported_lengths().iter().filter(|&&l| l >= 8 && l <= max_len) {
        let bits = 32 * (len / 2) as u32 - 4;
        let n = (Integer::from(1) << bits) - 1u32;
        let n2 = (Integer::from(1) << bits) - (Integer::from(1) << (bits / 2)) + 1u32;
        let mut b = ModBatch::new(make(), &[n.clone(), n2.clone()], 3).unwrap();
        let e = Integer::from(0xffff_ffffu32);
        for (i, m) in [&n, &n2].into_iter().enumerate() {
            b.load_raw(i, &(m.clone() * 2u32 - 1u32));
            b.set_exponent(i, &e);
        }
        b.run(32);
        let k = b.params().k as u32;
        for (i, m) in [&n, &n2].into_iter().enumerate() {
            // From x̃ = 2N − 1 ≡ −1: the canonical value is (−R⁻¹)^(2^32) · 3^e · ... computed directly.
            let rinv = (Integer::from(1) << (32 * k)).invert(m).unwrap();
            let x = (m.clone() - rinv).modulo(m);
            let want = (x.pow_mod(&(Integer::from(1) << 32u32), m).unwrap() * Integer::from(3).pow_mod(&e, m).unwrap()).modulo(m);
            assert_eq!(b.read(i), want, "len {len} slot {i}");
        }
    }
}

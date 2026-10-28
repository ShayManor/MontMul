//! Arithmetic in GF(M31²) and GF(M61²) with i² = −1, roots of unity, twiddle tables and CRT.
//!
//! Both primes are ≡ 3 mod 4, so −1 is a non-residue and GF(p²) = GF(p)[i]. The 2-Sylow
//! subgroup of GF(p²)* has order 2^(e) with e = 32 for M31 and e = 62 for M61, which gives
//! power-of-two transforms of any length used here.

pub const P1: u32 = 0x7fff_ffff;
pub const P2: u64 = 0x1fff_ffff_ffff_ffff;

/// Product of the transform moduli, Q = M31 · M61.
pub const Q: u128 = P1 as u128 * P2 as u128;

#[inline(always)]
pub fn add1(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= P1 { s - P1 } else { s }
}
#[inline(always)]
pub fn sub1(a: u32, b: u32) -> u32 {
    if a >= b { a - b } else { a + P1 - b }
}
#[inline(always)]
pub fn mul1(a: u32, b: u32) -> u32 {
    let t = a as u64 * b as u64;
    let s = ((t & P1 as u64) + (t >> 31)) as u32;
    if s >= P1 { s - P1 } else { s }
}
#[inline(always)]
pub fn add2(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= P2 { s - P2 } else { s }
}
#[inline(always)]
pub fn sub2(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { a + P2 - b }
}
#[inline(always)]
pub fn mul2(a: u64, b: u64) -> u64 {
    let t = a as u128 * b as u128;
    let s = (t as u64 & P2) + (t >> 61) as u64;
    if s >= P2 { s - P2 } else { s }
}

/// An element of GF(p²) for one of the two primes.
pub trait Elem: Copy + Default + PartialEq + std::fmt::Debug + Send + Sync {
    type Base: Copy;
    const ORDER_LOG2: u32;
    fn new(re: Self::Base, im: Self::Base) -> Self;
    fn from_u32(v: u32) -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    /// Multiply by a base-field scalar given as a small integer.
    fn scale(self, c: u32) -> Self;
    fn inv(self) -> Self;
    fn one() -> Self;
    fn i() -> Self;
    /// Generator of the 2-Sylow subgroup, order 2^ORDER_LOG2.
    fn sylow2() -> Self;
    fn pow(self, mut e: u128) -> Self {
        let mut b = self;
        let mut r = Self::one();
        while e > 0 {
            if e & 1 == 1 {
                r = r.mul(b);
            }
            b = b.mul(b);
            e >>= 1;
        }
        r
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
#[repr(C, align(8))]
pub struct F1 {
    pub re: u32,
    pub im: u32,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
#[repr(C, align(16))]
pub struct F2 {
    pub re: u64,
    pub im: u64,
}

fn base_pow1(mut b: u32, mut e: u64) -> u32 {
    let mut r = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = mul1(r, b);
        }
        b = mul1(b, b);
        e >>= 1;
    }
    r
}
fn base_pow2(mut b: u64, mut e: u64) -> u64 {
    let mut r = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = mul2(r, b);
        }
        b = mul2(b, b);
        e >>= 1;
    }
    r
}

/// Smallest a + i with a ≥ 1 whose (p²−1)/2^e power has order exactly 2^e.
fn find_sylow2<F: Elem>(cofactor: u128) -> F {
    for a in 1u32.. {
        let g = F::new_from_parts(a, 1).pow(cofactor);
        if g.pow(1u128 << (F::ORDER_LOG2 - 1)) != F::one() {
            return g;
        }
    }
    unreachable!()
}

trait FromParts {
    fn new_from_parts(re: u32, im: u32) -> Self;
}
impl<F: Elem> FromParts for F {
    fn new_from_parts(re: u32, im: u32) -> Self {
        F::from_u32(re).add(F::i().scale(im))
    }
}

impl Elem for F1 {
    type Base = u32;
    const ORDER_LOG2: u32 = 32;
    #[inline(always)]
    fn new(re: u32, im: u32) -> Self {
        F1 { re, im }
    }
    #[inline(always)]
    fn from_u32(v: u32) -> Self {
        F1 { re: if v >= P1 { v - P1 } else { v }, im: 0 }
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        F1 { re: add1(self.re, o.re), im: add1(self.im, o.im) }
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        F1 { re: sub1(self.re, o.re), im: sub1(self.im, o.im) }
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        F1 {
            re: sub1(mul1(self.re, o.re), mul1(self.im, o.im)),
            im: add1(mul1(self.re, o.im), mul1(self.im, o.re)),
        }
    }
    #[inline(always)]
    fn scale(self, c: u32) -> Self {
        let c = if c >= P1 { c - P1 } else { c };
        F1 { re: mul1(self.re, c), im: mul1(self.im, c) }
    }
    fn inv(self) -> Self {
        let n = add1(mul1(self.re, self.re), mul1(self.im, self.im));
        let ni = base_pow1(n, P1 as u64 - 2);
        F1 { re: mul1(self.re, ni), im: mul1(sub1(0, self.im), ni) }
    }
    fn one() -> Self {
        F1 { re: 1, im: 0 }
    }
    fn i() -> Self {
        F1 { re: 0, im: 1 }
    }
    fn sylow2() -> Self {
        find_sylow2::<F1>((1u128 << 30) - 1)
    }
}

impl Elem for F2 {
    type Base = u64;
    const ORDER_LOG2: u32 = 62;
    #[inline(always)]
    fn new(re: u64, im: u64) -> Self {
        F2 { re, im }
    }
    #[inline(always)]
    fn from_u32(v: u32) -> Self {
        F2 { re: v as u64, im: 0 }
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        F2 { re: add2(self.re, o.re), im: add2(self.im, o.im) }
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        F2 { re: sub2(self.re, o.re), im: sub2(self.im, o.im) }
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        F2 {
            re: sub2(mul2(self.re, o.re), mul2(self.im, o.im)),
            im: add2(mul2(self.re, o.im), mul2(self.im, o.re)),
        }
    }
    #[inline(always)]
    fn scale(self, c: u32) -> Self {
        F2 { re: mul2(self.re, c as u64), im: mul2(self.im, c as u64) }
    }
    fn inv(self) -> Self {
        let n = add2(mul2(self.re, self.re), mul2(self.im, self.im));
        let ni = base_pow2(n, P2 - 2);
        F2 { re: mul2(self.re, ni), im: mul2(sub2(0, self.im), ni) }
    }
    fn one() -> Self {
        F2 { re: 1, im: 0 }
    }
    fn i() -> Self {
        F2 { re: 0, im: 1 }
    }
    fn sylow2() -> Self {
        find_sylow2::<F2>((1u128 << 60) - 1)
    }
}

/// Tables for one complex transform length n (= L/2) in one field.
///
/// `tw[t] = ω^t` and `itw[t] = ω^−t` for t < n/2, with ω a primitive n-th root.
/// `w[j] = ζ^j` and `iw[j] = ζ^−j / n` for j < n, with ζ^n = i (right-angle weights:
/// a length-n weighted cyclic transform computes products modulo X^(2n) + 1).
#[derive(Clone, Debug)]
pub struct FieldTables<F: Elem> {
    pub tw: Vec<F>,
    pub itw: Vec<F>,
    pub w: Vec<F>,
    pub iw: Vec<F>,
}

impl<F: Elem> FieldTables<F> {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 2);
        let log = n.trailing_zeros();
        assert!(log + 2 <= F::ORDER_LOG2);
        let g = F::sylow2();
        let omega = g.pow(1u128 << (F::ORDER_LOG2 - log));
        let mut zeta = g.pow(1u128 << (F::ORDER_LOG2 - log - 2));
        if zeta.pow(n as u128) != F::i() {
            zeta = zeta.inv();
        }
        assert_eq!(zeta.pow(n as u128), F::i());
        let powers = |r: F, m: usize| {
            let mut v = Vec::with_capacity(m);
            let mut x = F::one();
            for _ in 0..m {
                v.push(x);
                x = x.mul(r);
            }
            v
        };
        let ninv = F::from_u32(n as u32).inv();
        let iw = powers(zeta.inv(), n).into_iter().map(|x| x.mul(ninv)).collect();
        FieldTables {
            tw: powers(omega, n / 2),
            itw: powers(omega.inv(), n / 2),
            w: powers(zeta, n),
            iw,
        }
    }
}

/// Tables for both fields at one transform length.
#[derive(Clone, Debug)]
pub struct Tables {
    pub n: usize,
    pub t1: FieldTables<F1>,
    pub t2: FieldTables<F2>,
}

impl Tables {
    pub fn new(n: usize) -> Self {
        Tables { n, t1: FieldTables::new(n), t2: FieldTables::new(n) }
    }
}

/// (M61 mod M31)⁻¹ mod M31, used by CRT.
pub fn crt_inv() -> u32 {
    base_pow1((P2 % P1 as u64) as u32, P1 as u64 - 2)
}

/// The unique C < Q with C ≡ c1 (mod M31) and C ≡ c2 (mod M61).
#[inline(always)]
pub fn crt(c1: u32, c2: u64, inv: u32) -> u128 {
    let c2m = (c2 % P1 as u64) as u32;
    let h = mul1(sub1(c1, c2m), inv);
    c2 as u128 + P2 as u128 * h as u128
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sylow_orders() {
        let g1 = F1::sylow2();
        assert_eq!(g1.pow(1 << 32), F1::one());
        assert_ne!(g1.pow(1 << 31), F1::one());
        let g2 = F2::sylow2();
        assert_eq!(g2.pow(1 << 62), F2::one());
        assert_ne!(g2.pow(1 << 61), F2::one());
    }

    #[test]
    fn crt_roundtrip() {
        let inv = crt_inv();
        for &c in &[0u128, 1, Q - 1, Q / 3, 123456789012345678901234567u128 % Q] {
            assert_eq!(crt((c % P1 as u128) as u32, (c % P2 as u128) as u64, inv), c);
        }
    }
}

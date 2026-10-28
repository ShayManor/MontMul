//! Host API (spec section 8): parameter selection, Montgomery conversion and checking with GMP.

use crate::transform::{Backend, BatchSetup};
use anyhow::{Result, bail, ensure};
use rug::Integer;

/// Limb width w.
pub const W: u32 = 32;

/// Shared parameters of a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub k: usize,
    pub len: usize,
}

impl Params {
    /// Smallest k with k · w ≥ b + 4 and the smallest supported L ≥ 2k (section 4).
    pub fn for_bits(bits: u32, lengths: &[usize]) -> Result<Params> {
        let k = (bits as usize + 4).div_ceil(W as usize);
        let Some(&len) = lengths.iter().find(|&&l| l >= 2 * k) else { bail!("no transform length for {bits} bits") };
        Ok(Params { k, len })
    }

    /// The exactness rule c · k · (2^w − 1)² < Q.
    pub fn check_exact(&self, c: u32, bound: u128) -> Result<()> {
        let lhs = c.max(1) as u128 * self.k as u128 * (((1u128 << W) - 1) * ((1u128 << W) - 1));
        ensure!(lhs < bound, "c · k · (2^w − 1)² = {lhs} exceeds the coefficient bound {bound}");
        Ok(())
    }
}

pub fn to_limbs(x: &Integer, k: usize) -> Vec<u32> {
    let mut v = x.to_digits::<u32>(rug::integer::Order::Lsf);
    assert!(v.len() <= k, "value does not fit {k} limbs");
    v.resize(k, 0);
    v
}

pub fn from_limbs(v: &[u32]) -> Integer {
    Integer::from_digits(v, rug::integer::Order::Lsf)
}

struct Modulus {
    n: Integer,
    /// R mod N.
    r: Integer,
    /// R⁻¹ mod N.
    rinv: Integer,
}

pub struct ModBatch<B: Backend> {
    backend: B,
    state: B::State,
    params: Params,
    moduli: Vec<Modulus>,
}

impl<B: Backend> ModBatch<B> {
    /// Registers moduli; fails if any violates the rules in section 4 or they need different L.
    pub fn new(backend: B, moduli: &[Integer], base: u32) -> Result<Self> {
        ensure!(!moduli.is_empty(), "empty batch");
        ensure!((1..=4).contains(&base), "base {base} does not fit the lazy range (c ≤ 4)");
        let lengths = backend.supported_lengths().to_vec();
        let mut params: Option<Params> = None;
        for n in moduli {
            ensure!(n.is_odd() && *n > 1, "modulus must be odd and > 1");
            let p = Params::for_bits(n.significant_bits(), &lengths)?;
            params = Some(match params {
                None => p,
                Some(q) => {
                    ensure!(q.len == p.len, "moduli need different transform lengths ({} vs {})", q.len, p.len);
                    Params { k: q.k.max(p.k), len: q.len }
                }
            });
        }
        let params = params.unwrap();
        params.check_exact(base, backend.coefficient_bound())?;
        let k = params.k;
        let r = Integer::from(1) << (k as u32 * W);
        let mut ms = Vec::new();
        let (mut nl, mut npl, mut tnl) = (Vec::new(), Vec::new(), Vec::new());
        for n in moduli {
            let ninv = n.clone().invert(&r).unwrap();
            let nprime = (r.clone() - ninv) % &r;
            nl.push(to_limbs(n, k));
            npl.push(to_limbs(&nprime, k));
            tnl.push(to_limbs(&(n.clone() * 2u32), k));
            ms.push(Modulus { n: n.clone(), r: r.clone() % n, rinv: r.clone().invert(n).unwrap() });
        }
        let setup = BatchSetup { k, len: params.len, base, n: &nl, nprime: &npl, two_n: &tnl };
        let state = backend.create(&setup)?;
        Ok(ModBatch { backend, state, params, moduli: ms })
    }

    pub fn params(&self) -> Params {
        self.params
    }
    pub fn slots(&self) -> usize {
        self.moduli.len()
    }
    pub fn modulus(&self, i: usize) -> &Integer {
        &self.moduli[i].n
    }
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Uploads x (any integer) to slot `i` in Montgomery form.
    pub fn load(&mut self, i: usize, x: &Integer) {
        let m = &self.moduli[i];
        let xm = (x.clone() * &m.r).modulo(&m.n);
        let limbs = to_limbs(&xm, self.params.k);
        self.backend.load(&mut self.state, i, &limbs);
    }

    /// Writes a raw Montgomery-form value in [0, 2N), for tests of the lazy range.
    pub fn load_raw(&mut self, i: usize, xm: &Integer) {
        let limbs = to_limbs(xm, self.params.k);
        self.backend.load(&mut self.state, i, &limbs);
    }

    /// Uploads the exponent whose bits drive `run`, most significant bit first.
    pub fn set_exponent(&mut self, i: usize, e: &Integer) {
        let nbits = e.significant_bits() as usize;
        let mut words = vec![0u32; nbits.div_ceil(32)];
        for p in 0..nbits {
            if e.get_bit((nbits - 1 - p) as u32) {
                words[p / 32] |= 1 << (31 - p % 32);
            }
        }
        self.backend.set_exponent(&mut self.state, i, &words, nbits);
    }

    /// Runs up to `bits` exponent bits on every active slot.
    pub fn run(&mut self, bits: usize) {
        self.backend.run(&mut self.state, bits);
    }

    /// Exponent bits still to run on slot `i`.
    pub fn remaining(&self, i: usize) -> usize {
        self.backend.remaining(&self.state, i)
    }

    /// a · b mod N for slots a, b into slot `out` (same modulus); used by error checks.
    pub fn mul(&mut self, out: usize, a: usize, b: usize) {
        assert!(self.moduli[out].n == self.moduli[a].n && self.moduli[a].n == self.moduli[b].n);
        self.backend.mul(&mut self.state, out, a, b);
    }

    /// Montgomery-form value in [0, 2N) as stored on the device.
    pub fn read_raw(&self, i: usize) -> Integer {
        from_limbs(&self.backend.read(&self.state, i))
    }

    /// Canonical value in [0, N), out of Montgomery form.
    pub fn read(&self, i: usize) -> Integer {
        let m = &self.moduli[i];
        (self.read_raw(i) * &m.rinv).modulo(&m.n)
    }

    /// True if slot i holds 1 (compares against R mod N).
    pub fn is_one(&self, i: usize) -> bool {
        let m = &self.moduli[i];
        let v = self.read_raw(i);
        v == m.r || v == m.r.clone() + &m.n
    }
}

/// Fermat base-`base` PRP test of every modulus: loads 1, runs N − 1, checks for 1.
pub fn fermat<B: Backend>(backend: B, moduli: &[Integer], base: u32, chunk: usize) -> Result<Vec<bool>> {
    let mut b = ModBatch::new(backend, moduli, base)?;
    for (i, n) in moduli.iter().enumerate() {
        b.load(i, &Integer::from(1));
        b.set_exponent(i, &(n.clone() - 1u32));
    }
    while (0..moduli.len()).any(|i| b.remaining(i) > 0) {
        b.run(chunk);
    }
    Ok((0..moduli.len()).map(|i| b.is_one(i)).collect())
}

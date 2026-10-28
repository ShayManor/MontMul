//! GMP backend: the Montgomery step of section 3 on `rug` integers, threaded over slots. It is the
//! CPU baseline for the score and needs no transform, so every length is exact.

use crate::batch::{from_limbs, to_limbs};
use crate::transform::{Backend, BatchSetup, Limits};
use rayon::prelude::*;
use rug::{Assign, Integer};

pub struct Gmp {
    lengths: Vec<usize>,
}

impl Default for Gmp {
    fn default() -> Self {
        Gmp { lengths: (3..=24).map(|e| 1usize << e).collect() }
    }
}

impl Limits for Gmp {
    /// GMP products are exact.
    fn coefficient_bound(&self) -> u128 {
        u128::MAX
    }
    fn supported_lengths(&self) -> &[usize] {
        &self.lengths
    }
}

struct Slot {
    n: Integer,
    /// N′ = −N⁻¹ mod R.
    nprime: Integer,
    two_n: Integer,
    x: Integer,
    /// Scratch for T and m, kept to reuse their allocations.
    t: Integer,
    m: Integer,
    exp: Vec<u32>,
    nbits: usize,
    pos: usize,
}

impl Slot {
    /// x ← T · R⁻¹ mod N in [0, 2N) for T in `t`: m = (T mod R) · N′ mod R, x = (T + m · N) / R.
    fn redc(&mut self, rbits: u32) {
        self.m.assign(self.t.keep_bits_ref(rbits));
        self.m *= &self.nprime;
        self.m.keep_bits_mut(rbits);
        self.m *= &self.n;
        self.t += &self.m;
        self.t >>= rbits;
        std::mem::swap(&mut self.x, &mut self.t);
        debug_assert!(self.x < self.two_n, "lazy range violated");
    }

    fn square_step(&mut self, c: u32, rbits: u32) {
        self.t.assign(self.x.square_ref());
        self.t *= c;
        self.redc(rbits);
    }
}

pub struct GmpState {
    k: usize,
    base: u32,
    slots: Vec<Slot>,
}

impl Backend for Gmp {
    type State = GmpState;

    fn create(&self, st: &BatchSetup) -> anyhow::Result<GmpState> {
        let slots = (0..st.n.len())
            .map(|i| Slot {
                n: from_limbs(&st.n[i]),
                nprime: from_limbs(&st.nprime[i]),
                two_n: from_limbs(&st.two_n[i]),
                x: Integer::new(),
                t: Integer::new(),
                m: Integer::new(),
                exp: vec![],
                nbits: 0,
                pos: 0,
            })
            .collect();
        Ok(GmpState { k: st.k, base: st.base, slots })
    }
    fn load(&self, st: &mut GmpState, i: usize, limbs: &[u32]) {
        st.slots[i].x = from_limbs(limbs);
    }
    fn set_exponent(&self, st: &mut GmpState, i: usize, words: &[u32], nbits: usize) {
        let s = &mut st.slots[i];
        s.exp = words.to_vec();
        s.nbits = nbits;
        s.pos = 0;
    }
    fn run(&self, st: &mut GmpState, bits: usize) {
        let (rbits, base) = (32 * st.k as u32, st.base);
        st.slots.par_iter_mut().for_each(|s| {
            for _ in 0..bits {
                if s.pos >= s.nbits {
                    break;
                }
                let bit = (s.exp[s.pos / 32] >> (31 - s.pos % 32)) & 1;
                s.square_step(if bit == 1 { base } else { 1 }, rbits);
                s.pos += 1;
            }
        });
    }
    fn mul(&self, st: &mut GmpState, out: usize, a: usize, b: usize) {
        let t = Integer::from(&st.slots[a].x * &st.slots[b].x);
        let s = &mut st.slots[out];
        s.t = t;
        s.redc(32 * st.k as u32);
    }
    fn read(&self, st: &GmpState, i: usize) -> Vec<u32> {
        to_limbs(&st.slots[i].x, st.k)
    }
    fn remaining(&self, st: &GmpState, i: usize) -> usize {
        st.slots[i].nbits - st.slots[i].pos
    }
}

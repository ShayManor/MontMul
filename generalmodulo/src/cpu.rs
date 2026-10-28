//! CPU backend: the same M31 × M61 right-angle transform in plain Rust, threaded over slots.

use crate::field::*;
use crate::transform::{Backend, BatchSetup, Transform};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct Cpu {
    lengths: Vec<usize>,
    tables: Mutex<HashMap<usize, Arc<Tables>>>,
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu { lengths: (3..=24).map(|e| 1usize << e).collect(), tables: Mutex::new(HashMap::new()) }
    }
}

impl Cpu {
    /// Tables for transform length `len` (complex length len / 2).
    pub fn tables(&self, len: usize) -> Arc<Tables> {
        self.tables.lock().unwrap().entry(len).or_insert_with(|| Arc::new(Tables::new(len / 2))).clone()
    }
}

/// Decimation in frequency, natural order in, bit-reversed order out.
fn dif<F: Elem>(a: &mut [F], tw: &[F]) {
    let n = a.len();
    let (mut h, mut s) = (n / 2, 0);
    while h >= 1 {
        for g in (0..n).step_by(2 * h) {
            for j in 0..h {
                let (u, v) = (a[g + j], a[g + j + h]);
                a[g + j] = u.add(v);
                a[g + j + h] = u.sub(v).mul(tw[j << s]);
            }
        }
        h /= 2;
        s += 1;
    }
}

/// Decimation in time, bit-reversed order in, natural order out (unscaled).
fn dit<F: Elem>(a: &mut [F], itw: &[F]) {
    let n = a.len();
    let mut h = 1;
    let mut s = n.trailing_zeros() as usize;
    while h < n {
        s -= 1;
        for g in (0..n).step_by(2 * h) {
            for j in 0..h {
                let u = a[g + j];
                let v = a[g + j + h].mul(itw[j << s]);
                a[g + j] = u.add(v);
                a[g + j + h] = u.sub(v);
            }
        }
        h *= 2;
    }
}

/// Weights and forward-transforms one vector in place.
pub(crate) fn forward1(a1: &mut [F1], a2: &mut [F2], t: &Tables) {
    for j in 0..t.n {
        a1[j] = a1[j].mul(t.t1.w[j]);
        a2[j] = a2[j].mul(t.t2.w[j]);
    }
    dif(a1, &t.t1.tw);
    dif(a2, &t.t2.tw);
}

/// Inverse-transforms and unweights one vector in place.
pub(crate) fn inverse1(a1: &mut [F1], a2: &mut [F2], t: &Tables) {
    dit(a1, &t.t1.itw);
    dit(a2, &t.t2.itw);
    for j in 0..t.n {
        a1[j] = a1[j].mul(t.t1.iw[j]);
        a2[j] = a2[j].mul(t.t2.iw[j]);
    }
}

/// Packs real coefficients (at most 2n) into a1, a2 and forward-transforms.
fn forward_limbs(c: &[u32], a1: &mut [F1], a2: &mut [F2], t: &Tables) {
    let n = t.n;
    let get = |q: usize| if q < c.len() { c[q] } else { 0 };
    for j in 0..n {
        let (re, im) = (get(j), get(j + n));
        a1[j] = F1 { re: re % P1, im: im % P1 };
        a2[j] = F2 { re: re as u64, im: im as u64 };
    }
    forward1(a1, a2, t);
}

/// CRT and carry: writes 2n limbs of Σ c_q 2^(32q); returns the carry out of the top.
fn carry(a1: &[F1], a2: &[F2], out: &mut [u32], inv: u32) -> u128 {
    let n = a1.len();
    let mut c: u128 = 0;
    for q in 0..2 * n {
        let j = q % n;
        let v = if q < n { crt(a1[j].re, a2[j].re, inv) } else { crt(a1[j].im, a2[j].im, inv) };
        c += v;
        out[q] = c as u32;
        c >>= 32;
    }
    c
}

fn pointwise1(a1: &mut [F1], a2: &mut [F2], b1: &[F1], b2: &[F2], c: u32) {
    for j in 0..a1.len() {
        a1[j] = a1[j].mul(b1[j]);
        a2[j] = a2[j].mul(b2[j]);
        if c != 1 {
            a1[j] = a1[j].scale(c);
            a2[j] = a2[j].scale(c);
        }
    }
}

pub struct CpuBuf {
    pub len: usize,
    pub count: usize,
    pub f1: Vec<F1>,
    pub f2: Vec<F2>,
}

impl Transform for Cpu {
    type Buf = CpuBuf;
    type Scales = Vec<u32>;

    fn coefficient_bound(&self) -> u128 {
        Q
    }
    fn supported_lengths(&self) -> &[usize] {
        &self.lengths
    }
    fn upload(&self, coeffs: &[u32], len: usize, count: usize) -> CpuBuf {
        let n = len / 2;
        let mut b = CpuBuf { len, count, f1: vec![F1::default(); n * count], f2: vec![F2::default(); n * count] };
        for v in 0..count {
            let c = &coeffs[v * len..(v + 1) * len];
            for j in 0..n {
                b.f1[v * n + j] = F1 { re: c[j] % P1, im: c[j + n] % P1 };
                b.f2[v * n + j] = F2 { re: c[j] as u64, im: c[j + n] as u64 };
            }
        }
        b
    }
    fn download(&self, d: &CpuBuf) -> Vec<u128> {
        let n = d.len / 2;
        let inv = crt_inv();
        let mut out = vec![0u128; d.len * d.count];
        for v in 0..d.count {
            for j in 0..n {
                let (x1, x2) = (d.f1[v * n + j], d.f2[v * n + j]);
                out[v * d.len + j] = crt(x1.re, x2.re, inv);
                out[v * d.len + j + n] = crt(x1.im, x2.im, inv);
            }
        }
        out
    }
    fn upload_scales(&self, scales: &[u32]) -> Vec<u32> {
        scales.to_vec()
    }
    fn forward(&self, d: &mut CpuBuf, len: usize, count: usize) {
        let t = self.tables(len);
        let n = len / 2;
        d.f1[..n * count]
            .par_chunks_mut(n)
            .zip(d.f2[..n * count].par_chunks_mut(n))
            .for_each(|(a1, a2)| forward1(a1, a2, &t));
    }
    fn inverse(&self, d: &mut CpuBuf, len: usize, count: usize) {
        let t = self.tables(len);
        let n = len / 2;
        d.f1[..n * count]
            .par_chunks_mut(n)
            .zip(d.f2[..n * count].par_chunks_mut(n))
            .for_each(|(a1, a2)| inverse1(a1, a2, &t));
    }
    fn pointwise(&self, out: &mut CpuBuf, a: &CpuBuf, b: &CpuBuf, scale: Option<&Vec<u32>>, len: usize, count: usize) {
        let n = len / 2;
        for v in 0..count {
            let c = scale.map_or(1, |s| s[v]);
            let r = v * n..(v + 1) * n;
            out.f1[r.clone()].copy_from_slice(&a.f1[r.clone()]);
            out.f2[r.clone()].copy_from_slice(&a.f2[r.clone()]);
            pointwise1(&mut out.f1[r.clone()], &mut out.f2[r.clone()], &b.f1[r.clone()], &b.f2[r], c);
        }
    }
}

struct Slot {
    nhat1: Vec<F1>,
    nhat2: Vec<F2>,
    nphat1: Vec<F1>,
    nphat2: Vec<F2>,
    two_n: Vec<u32>,
    x: Vec<u32>,
    exp: Vec<u32>,
    nbits: usize,
    pos: usize,
}

pub struct CpuState {
    k: usize,
    base: u32,
    tab: Arc<Tables>,
    slots: Vec<Slot>,
}

struct Scratch {
    a1: Vec<F1>,
    a2: Vec<F2>,
    t: Vec<u32>,
    s: Vec<u32>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Scratch { a1: vec![F1::default(); n], a2: vec![F2::default(); n], t: vec![0; 2 * n], s: vec![0; 2 * n] }
    }
}

/// a < b for equal-length little-endian limb vectors.
fn limbs_lt(a: &[u32], b: &[u32]) -> bool {
    for j in (0..a.len()).rev() {
        if a[j] != b[j] {
            return a[j] < b[j];
        }
    }
    false
}

/// Steps 2 to 4 of section 3: from T in `sc.t`, writes T · R⁻¹ mod N (lazy) to `x`.
fn redc(k: usize, tab: &Tables, s: &Slot, sc: &mut Scratch, x: &mut [u32], inv: u32) {
    let flag = sc.t[..k].iter().any(|&v| v != 0);
    forward_limbs(&sc.t[..k], &mut sc.a1, &mut sc.a2, tab);
    pointwise1(&mut sc.a1, &mut sc.a2, &s.nphat1, &s.nphat2, 1);
    inverse1(&mut sc.a1, &mut sc.a2, tab);
    carry(&sc.a1, &sc.a2, &mut sc.s, inv);
    let m = sc.s[..k].to_vec();
    forward_limbs(&m, &mut sc.a1, &mut sc.a2, tab);
    pointwise1(&mut sc.a1, &mut sc.a2, &s.nhat1, &s.nhat2, 1);
    inverse1(&mut sc.a1, &mut sc.a2, tab);
    let top = carry(&sc.a1, &sc.a2, &mut sc.s, inv);
    debug_assert_eq!(top, 0);
    let mut c = flag as u64;
    for j in 0..k {
        c += sc.t[k + j] as u64 + sc.s[k + j] as u64;
        x[j] = c as u32;
        c >>= 32;
    }
    debug_assert_eq!(c, 0);
    debug_assert!(limbs_lt(x, &s.two_n), "lazy range violated");
}

fn square_step(k: usize, tab: &Tables, s: &mut Slot, sc: &mut Scratch, c: u32, inv: u32) {
    forward_limbs(&s.x, &mut sc.a1, &mut sc.a2, tab);
    for j in 0..tab.n {
        sc.a1[j] = sc.a1[j].mul(sc.a1[j]).scale(c);
        sc.a2[j] = sc.a2[j].mul(sc.a2[j]).scale(c);
    }
    inverse1(&mut sc.a1, &mut sc.a2, tab);
    carry(&sc.a1, &sc.a2, &mut sc.t, inv);
    let mut x = std::mem::take(&mut s.x);
    redc(k, tab, s, sc, &mut x, inv);
    s.x = x;
}

impl Backend for Cpu {
    type State = CpuState;

    fn create(&self, st: &BatchSetup) -> anyhow::Result<CpuState> {
        let tab = self.tables(st.len);
        let n = tab.n;
        let slots = (0..st.n.len())
            .into_par_iter()
            .map(|i| {
                let mut sc = Scratch::new(n);
                forward_limbs(&st.n[i], &mut sc.a1, &mut sc.a2, &tab);
                let (nhat1, nhat2) = (sc.a1.clone(), sc.a2.clone());
                forward_limbs(&st.nprime[i], &mut sc.a1, &mut sc.a2, &tab);
                Slot {
                    nhat1,
                    nhat2,
                    nphat1: sc.a1,
                    nphat2: sc.a2,
                    two_n: st.two_n[i].clone(),
                    x: vec![0; st.k],
                    exp: vec![],
                    nbits: 0,
                    pos: 0,
                }
            })
            .collect();
        Ok(CpuState { k: st.k, base: st.base, tab, slots })
    }
    fn load(&self, st: &mut CpuState, i: usize, limbs: &[u32]) {
        st.slots[i].x.copy_from_slice(limbs);
    }
    fn set_exponent(&self, st: &mut CpuState, i: usize, words: &[u32], nbits: usize) {
        let s = &mut st.slots[i];
        s.exp = words.to_vec();
        s.nbits = nbits;
        s.pos = 0;
    }
    fn run(&self, st: &mut CpuState, bits: usize) {
        let (k, base, tab) = (st.k, st.base, st.tab.clone());
        let inv = crt_inv();
        st.slots.par_iter_mut().for_each(|s| {
            let mut sc = Scratch::new(tab.n);
            for _ in 0..bits {
                if s.pos >= s.nbits {
                    break;
                }
                let bit = (s.exp[s.pos / 32] >> (31 - s.pos % 32)) & 1;
                square_step(k, &tab, s, &mut sc, if bit == 1 { base } else { 1 }, inv);
                s.pos += 1;
            }
        });
    }
    fn mul(&self, st: &mut CpuState, out: usize, a: usize, b: usize) {
        let tab = st.tab.clone();
        let inv = crt_inv();
        let mut sc = Scratch::new(tab.n);
        let (mut b1, mut b2) = (vec![F1::default(); tab.n], vec![F2::default(); tab.n]);
        forward_limbs(&st.slots[b].x, &mut b1, &mut b2, &tab);
        forward_limbs(&st.slots[a].x, &mut sc.a1, &mut sc.a2, &tab);
        pointwise1(&mut sc.a1, &mut sc.a2, &b1, &b2, 1);
        inverse1(&mut sc.a1, &mut sc.a2, &tab);
        carry(&sc.a1, &sc.a2, &mut sc.t, inv);
        let mut x = vec![0; st.k];
        redc(st.k, &tab, &st.slots[out], &mut sc, &mut x, inv);
        st.slots[out].x = x;
    }
    fn read(&self, st: &CpuState, i: usize) -> Vec<u32> {
        st.slots[i].x.clone()
    }
    fn remaining(&self, st: &CpuState, i: usize) -> usize {
        st.slots[i].nbits - st.slots[i].pos
    }
}

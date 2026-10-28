//! CPU backend: the same M31 × M61 right-angle transform in plain Rust, threaded over slots.

use crate::field::*;
use crate::transform::{Backend, BatchSetup, Limits, Transform};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct Cpu {
    lengths: Vec<usize>,
    tables: Mutex<HashMap<usize, Arc<Tables>>>,
    full_k: bool,
    wraparound: bool,
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu {
            lengths: (3..=24).map(|e| 1usize << e).collect(),
            tables: Mutex::new(HashMap::new()),
            full_k: false,
            wraparound: false,
        }
    }
}

impl Cpu {
    /// Step 3 as a length-L/2 cyclic product with the wraparound correction (spec section 9); implies k = L/2.
    pub fn with_wraparound(on: bool) -> Cpu {
        Cpu { wraparound: on, ..Cpu::default() }
    }

    /// k = L/2 without the wraparound correction, so the two step-3 paths can be compared bit for bit.
    pub fn with_full_k(self, on: bool) -> Cpu {
        Cpu { full_k: on, ..self }
    }

    /// Tables for transform length `len` (complex length len / 2).
    pub fn tables(&self, len: usize) -> Arc<Tables> {
        self.tables.lock().unwrap().entry(len).or_insert_with(|| Arc::new(Tables::new(len / 2))).clone()
    }

    /// Transform-domain data without CRT.
    pub fn download_raw(&self, d: &CpuBuf) -> (Vec<F1>, Vec<F2>) {
        (d.f1.clone(), d.f2.clone())
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

/// Conjugation (the Frobenius map) and the two real parts of an element, for the real-to-complex split.
trait Conj: Elem {
    fn conj(self) -> Self;
    /// (re, im) as two elements of the base field.
    fn parts(self) -> (Self, Self);
}
impl Conj for F1 {
    fn conj(self) -> Self {
        F1 { re: self.re, im: sub1(0, self.im) }
    }
    fn parts(self) -> (Self, Self) {
        (F1 { re: self.re, im: 0 }, F1 { re: self.im, im: 0 })
    }
}
impl Conj for F2 {
    fn conj(self) -> Self {
        F2 { re: self.re, im: sub2(0, self.im) }
    }
    fn parts(self) -> (Self, Self) {
        (F2 { re: self.re, im: 0 }, F2 { re: self.im, im: 0 })
    }
}

/// Unpack twiddles for one field: `u[p] = w^rev(p)`, `iu[p] = w^−rev(p)`, with w a primitive
/// 2h-th root and w² the ω of the length-h tables.
pub(crate) struct Unpack<F: Elem> {
    pub(crate) u: Vec<F>,
    pub(crate) iu: Vec<F>,
}

impl<F: Elem> Unpack<F> {
    fn new(h: usize, rev: impl Fn(usize) -> usize, omega: Option<F>) -> Self {
        let w = F::sylow2().pow(1u128 << (F::ORDER_LOG2 - (2 * h).trailing_zeros()));
        assert!(omega.is_none_or(|o| w.mul(w) == o));
        let (u, iu) = (0..h).map(|p| (w.pow(rev(p) as u128), w.inv().pow(rev(p) as u128))).unzip();
        Unpack { u, iu }
    }
}

/// Real cyclic products of length 2h (= k = L/2) through complex transforms of length h.
///
/// A vector of 2h real coefficients a is packed as z_j = a_2j + i a_2j+1 and transformed
/// unweighted by `dif`, so position p holds Z at index rev(p) (bit reversal of log2 h bits).
pub(crate) struct Half {
    pub(crate) tab: Arc<Tables>,
    /// Position of index (h − rev(p)) mod h.
    partner: Vec<usize>,
    pub(crate) w1: Unpack<F1>,
    pub(crate) w2: Unpack<F2>,
}

impl Half {
    pub(crate) fn new(tab: Arc<Tables>) -> Self {
        let h = tab.n;
        let bits = h.trailing_zeros();
        let rev = |p: usize| p.reverse_bits() >> (usize::BITS - bits);
        let partner = (0..h).map(|p| rev((h - rev(p)) % h)).collect();
        let (o1, o2) = if h >= 4 { (Some(tab.t1.tw[1]), Some(tab.t2.tw[1])) } else { (None, None) };
        Half { w1: Unpack::new(h, rev, o1), w2: Unpack::new(h, rev, o2), partner, tab }
    }
}

/// One output of the split, pointwise product and repack at position p (see `half_pointwise`).
#[inline(always)]
fn half_one<F: Conj>(zk: F, zj: F, nx: F, ny: F, u: F, iu: F) -> F {
    let (x, y) = half_spectrum(zk, zj, u);
    let (px, py) = (x.mul(nx), y.mul(ny));
    px.add(py).add(px.sub(py).mul(iu).mul(F::i()))
}

/// The real spectrum (2 A_k, 2 A_k+h) at position p, from Z at p and at its partner q.
#[inline(always)]
fn half_spectrum<F: Conj>(zk: F, zj: F, u: F) -> (F, F) {
    let mi = F::default().sub(F::i());
    let e = zk.add(zj.conj());
    let wo = zk.sub(zj.conj()).mul(mi).mul(u);
    (e.add(wo), e.sub(wo))
}

/// In the packed domain: z ← repack(unpack(z) ⊙ unpack(N)), where `s` is N̂ as stored by `half_nhat`.
/// Unscaled `dit` of the result gives the cyclic product packed like the inputs.
fn half_pointwise<F: Conj>(z: &mut [F], s: &[F], partner: &[usize], w: &Unpack<F>) {
    let (x0, y0) = s[0].parts();
    for p in 0..z.len() {
        let q = partner[p];
        if q < p {
            continue;
        }
        let (zp, zq) = (z[p], z[q]);
        let (nx, ny) = if p == 0 { (x0, y0) } else { (s[p], s[q].conj()) };
        z[p] = half_one(zp, zq, nx, ny, w.u[p], w.iu[p]);
        if q != p {
            z[q] = half_one(zq, zp, s[q], s[p].conj(), w.u[q], w.iu[q]);
        }
    }
}

/// N̂ for the half path: position p ≠ 0 holds A_rev(p) / 2k, position 0 holds A_0 / 2k + i A_h / 2k,
/// where A is the length-k real DFT of N with root w. The 1/2k folds every split and inverse scale.
fn half_nhat<F: Conj>(z: &mut [F], partner: &[usize], w: &Unpack<F>) {
    let h = z.len();
    let c = F::from_u32(8 * h as u32).inv();
    let src = z.to_vec();
    for p in 0..h {
        let (x, y) = half_spectrum(src[p], src[partner[p]], w.u[p]);
        z[p] = if p == 0 { x.mul(c).add(y.mul(c).mul(F::i())) } else { x.mul(c) };
    }
}

/// N̂ of the half path (`half_nhat`) for the k limbs of N, h elements per field.
pub(crate) fn half_nhat_of(n: &[u32], hf: &Half) -> (Vec<F1>, Vec<F2>) {
    let h = hf.tab.n;
    let (mut a1, mut a2) = (vec![F1::default(); h], vec![F2::default(); h]);
    half_forward(n, &mut a1, &mut a2, hf);
    half_nhat(&mut a1, &hf.partner, &hf.w1);
    half_nhat(&mut a2, &hf.partner, &hf.w2);
    (a1, a2)
}

/// The GPU's form of the half path's pointwise step: `half_one` at p equals e · A_p + d · B_p
/// with e = z_p + conj z_q, d = z_p − conj z_q (q = partner p), A = X + Y and B = −i u (X − Y),
/// where X = n_x (1 + i iu) and Y = n_y (1 − i iu). A at p and B at h + p, from N̂ as stored by
/// `half_nhat`.
#[cfg(feature = "gpu")]
fn half_factors<F: Conj>(s: &[F], partner: &[usize], w: &Unpack<F>) -> Vec<F> {
    let (h, i) = (s.len(), F::i());
    let mi = F::default().sub(i);
    let (a, b) = (0..h)
        .map(|p| {
            let (nx, ny) = if p == 0 { s[0].parts() } else { (s[p], s[partner[p]].conj()) };
            let (x, y) = (nx.mul(F::one().add(i.mul(w.iu[p]))), ny.mul(F::one().sub(i.mul(w.iu[p]))));
            (x.add(y), x.sub(y).mul(mi).mul(w.u[p]))
        })
        .unzip::<_, _, Vec<F>, Vec<F>>();
    [a, b].concat()
}

/// `half_factors` for the k limbs of N, 2h elements per field.
#[cfg(feature = "gpu")]
pub(crate) fn half_factors_of(n: &[u32], hf: &Half) -> (Vec<F1>, Vec<F2>) {
    let (s1, s2) = half_nhat_of(n, hf);
    (half_factors(&s1, &hf.partner, &hf.w1), half_factors(&s2, &hf.partner, &hf.w2))
}

/// Packs k real coefficients as z_j = c_2j + i c_2j+1 and forward-transforms (no weights).
fn half_forward(c: &[u32], a1: &mut [F1], a2: &mut [F2], hf: &Half) {
    for j in 0..hf.tab.n {
        let (re, im) = (c[2 * j], c[2 * j + 1]);
        a1[j] = F1 { re: re % P1, im: im % P1 };
        a2[j] = F2 { re: re as u64, im: im as u64 };
    }
    dif(a1, &hf.tab.t1.tw);
    dif(a2, &hf.tab.t2.tw);
}

/// x = ⌊T / R⌋ + ((cyclic + T mod R) mod (R − 1)), from the packed cyclic product m · N in a1, a2.
/// The second term equals ⌊S / R⌋ + [T mod R ≠ 0] (spec section 9) and is below N < R − 1.
fn wrap_combine(a1: &[F1], a2: &[F2], t: &[u32], x: &mut [u32], inv: u32) {
    let k = x.len();
    let mut c: u128 = 0;
    for q in 0..k {
        let (e1, e2) = (a1[q / 2], a2[q / 2]);
        c += if q % 2 == 0 { crt(e1.re, e2.re, inv) } else { crt(e1.im, e2.im, inv) };
        c += t[q] as u128;
        x[q] = c as u32;
        c >>= 32;
    }
    // End-around carry: 2^(32k) ≡ 1 mod R − 1.
    while c != 0 {
        for v in x.iter_mut() {
            c += *v as u128;
            *v = c as u32;
            c >>= 32;
            if c == 0 {
                break;
            }
        }
    }
    // R − 1 itself cannot appear: it would need H + flag = 0, hence T mod R = 0, m = 0 and a zero sum.
    debug_assert!(x.iter().any(|&v| v != u32::MAX));
    for q in 0..k {
        c += x[q] as u128 + t[k + q] as u128;
        x[q] = c as u32;
        c >>= 32;
    }
    debug_assert_eq!(c, 0);
}

pub struct CpuBuf {
    pub len: usize,
    pub count: usize,
    pub f1: Vec<F1>,
    pub f2: Vec<F2>,
}

impl Limits for Cpu {
    fn coefficient_bound(&self) -> u128 {
        Q
    }
    fn supported_lengths(&self) -> &[usize] {
        &self.lengths
    }
    fn full_k(&self) -> bool {
        self.full_k || self.wraparound
    }
}

impl Transform for Cpu {
    type Buf = CpuBuf;
    type Scales = Vec<u32>;

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
    half: Option<Arc<Half>>,
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
fn redc(k: usize, tab: &Tables, half: Option<&Half>, s: &Slot, sc: &mut Scratch, x: &mut [u32], inv: u32) {
    let flag = sc.t[..k].iter().any(|&v| v != 0);
    forward_limbs(&sc.t[..k], &mut sc.a1, &mut sc.a2, tab);
    pointwise1(&mut sc.a1, &mut sc.a2, &s.nphat1, &s.nphat2, 1);
    inverse1(&mut sc.a1, &mut sc.a2, tab);
    carry(&sc.a1, &sc.a2, &mut sc.s, inv);
    let m = sc.s[..k].to_vec();
    if let Some(hf) = half {
        let h = hf.tab.n;
        let (a1, a2) = (&mut sc.a1[..h], &mut sc.a2[..h]);
        half_forward(&m, a1, a2, hf);
        half_pointwise(a1, &s.nhat1, &hf.partner, &hf.w1);
        half_pointwise(a2, &s.nhat2, &hf.partner, &hf.w2);
        dit(a1, &hf.tab.t1.itw);
        dit(a2, &hf.tab.t2.itw);
        wrap_combine(a1, a2, &sc.t, x, inv);
    } else {
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
    }
    debug_assert!(limbs_lt(x, &s.two_n), "lazy range violated");
}

fn square_step(k: usize, tab: &Tables, half: Option<&Half>, s: &mut Slot, sc: &mut Scratch, c: u32, inv: u32) {
    forward_limbs(&s.x, &mut sc.a1, &mut sc.a2, tab);
    for j in 0..tab.n {
        sc.a1[j] = sc.a1[j].mul(sc.a1[j]).scale(c);
        sc.a2[j] = sc.a2[j].mul(sc.a2[j]).scale(c);
    }
    inverse1(&mut sc.a1, &mut sc.a2, tab);
    carry(&sc.a1, &sc.a2, &mut sc.t, inv);
    let mut x = std::mem::take(&mut s.x);
    redc(k, tab, half, s, sc, &mut x, inv);
    s.x = x;
}

impl Backend for Cpu {
    type State = CpuState;

    fn create(&self, st: &BatchSetup) -> anyhow::Result<CpuState> {
        let tab = self.tables(st.len);
        let n = tab.n;
        let half = self.wraparound.then(|| Arc::new(Half::new(self.tables(st.len / 2))));
        assert!(half.is_none() || st.k == n, "wraparound needs k = L/2");
        let slots = (0..st.n.len())
            .into_par_iter()
            .map(|i| {
                let mut sc = Scratch::new(n);
                let (nhat1, nhat2) = match &half {
                    Some(hf) => half_nhat_of(&st.n[i], hf),
                    None => {
                        forward_limbs(&st.n[i], &mut sc.a1, &mut sc.a2, &tab);
                        (sc.a1.clone(), sc.a2.clone())
                    }
                };
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
        Ok(CpuState { k: st.k, base: st.base, tab, half, slots })
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
        let (k, base, tab, half) = (st.k, st.base, st.tab.clone(), st.half.clone());
        let inv = crt_inv();
        st.slots.par_iter_mut().for_each(|s| {
            let mut sc = Scratch::new(tab.n);
            for _ in 0..bits {
                if s.pos >= s.nbits {
                    break;
                }
                let bit = (s.exp[s.pos / 32] >> (31 - s.pos % 32)) & 1;
                square_step(k, &tab, half.as_deref(), s, &mut sc, if bit == 1 { base } else { 1 }, inv);
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
        redc(st.k, &tab, st.half.as_deref(), &st.slots[out], &mut sc, &mut x, inv);
        st.slots[out].x = x;
    }
    fn read(&self, st: &CpuState, i: usize) -> Vec<u32> {
        st.slots[i].x.clone()
    }
    fn remaining(&self, st: &CpuState, i: usize) -> usize {
        st.slots[i].nbits - st.slots[i].pos
    }
}

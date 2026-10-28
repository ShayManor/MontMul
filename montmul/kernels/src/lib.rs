//! CUDA kernels for the M31 × M61 right-angle transform, compiled to PTX by cuda-oxide and
//! loaded by the host crate's `gpu::device`.
//!
//! Results match `src/cpu.rs` and `src/field.rs` of the host crate bit for bit. One thread
//! block owns one vector. Vectors are raw pointers into shared or global memory, never
//! references, because a reference held across a barrier is undefined behaviour.

use cuda_device::{DynamicSharedArray, kernel, launch_bounds, ptx_asm, thread, warp};

pub const P1: u32 = 0x7fff_ffff;
pub const P2: u64 = 0x1fff_ffff_ffff_ffff;
/// Coefficients per thread in one carry chunk; at least 2, so that adding a neighbour's carry
/// (below 2^64) leaves a carry of 0 or 1 (see `block_carry`).
const CARRY_E: u32 = 2;
const _: () = assert!(CARRY_E >= 2);

/// An element of GF(M31²), laid out as the host's `field::F1`.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct F1 {
    pub re: u32,
    pub im: u32,
}

/// An element of GF(M61²), laid out as the host's `field::F2`.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct F2 {
    pub re: u64,
    pub im: u64,
}

/// Tables for one complex length n in device memory (see `field::FieldTables`; tw and itw are
/// followed by per-pass runs, see `gpu::device::radix4_twiddles`), passed by value. The host
/// mirrors this layout in `gpu::device::Tab`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Tab {
    pub tw1: *const F1,
    pub itw1: *const F1,
    pub w1: *const F1,
    pub iw1: *const F1,
    pub tw2: *const F2,
    pub itw2: *const F2,
    pub w2: *const F2,
    pub iw2: *const F2,
    pub n: u32,
    pub logn: u32,
    pub crt_inv: u32,
    /// Nonzero: spectra stay in swz order between `dif` and `dit` (see `at` and `swizzled`);
    /// `steps` sets it, the host passes 0 (`mul_slots` relies on that: its `pointwise_const`
    /// operand b is a natural-order spectrum).
    pub swz: u32,
}

/// Tables of the wraparound path (spec section 9) for transform length h = L/4: `tab` at length
/// h (its weights unused). `tab.n` = 0 when the path is off. The host mirrors this layout in
/// `gpu::device::HalfTab`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HalfTab {
    pub tab: Tab,
}

// ---- GF(M31) and GF(M61) ----

#[inline(always)]
fn add1(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= P1 { s - P1 } else { s }
}
#[inline(always)]
fn sub1(a: u32, b: u32) -> u32 {
    if a >= b { a - b } else { a + P1 - b }
}
#[inline(always)]
fn mul1(a: u32, b: u32) -> u32 {
    let t = a as u64 * b as u64;
    let s = (t as u32 & P1) + (t >> 31) as u32;
    if s >= P1 { s - P1 } else { s }
}
/// t mod M31, canonical, for t < 2^63 (so t >> 31 < 2^32 and the folded sum is at most
/// 2 · M31 + 1, which two subtractions reduce).
#[inline(always)]
fn red1(t: u64) -> u32 {
    let hi = (t >> 31) as u32;
    let s = (t as u32 & P1) + (hi & P1) + (hi >> 31);
    let s = if s >= P1 { s - P1 } else { s };
    if s >= P1 { s - P1 } else { s }
}
#[inline(always)]
fn add2(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= P2 { s - P2 } else { s }
}
#[inline(always)]
fn sub2(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { a + P2 - b }
}
#[inline(always)]
fn mul2(a: u64, b: u64) -> u64 {
    let lo = a.wrapping_mul(b);
    let hi = ((a as u128 * b as u128) >> 64) as u64; // one mul.hi.u64
    let s = (lo & P2) + ((hi << 3) | (lo >> 61));
    if s >= P2 { s - P2 } else { s }
}
// ---- GF(p²), i² = −1 ----

#[inline(always)]
fn cadd1(a: F1, b: F1) -> F1 {
    F1 { re: add1(a.re, b.re), im: add1(a.im, b.im) }
}
#[inline(always)]
fn csub1(a: F1, b: F1) -> F1 {
    F1 { re: sub1(a.re, b.re), im: sub1(a.im, b.im) }
}
#[inline(always)]
fn cmul1(a: F1, b: F1) -> F1 {
    // Each part is a sum of two products of canonical factors (P1 − b.im at most M31),
    // reduced once.
    let m = |x: u32, y: u32| x as u64 * y as u64;
    F1 { re: red1(m(a.re, b.re) + m(a.im, P1 - b.im)), im: red1(m(a.re, b.im) + m(a.im, b.re)) }
}
#[inline(always)]
fn csqr1(a: F1) -> F1 {
    let m = mul1(a.re, a.im);
    F1 { re: mul1(add1(a.re, a.im), sub1(a.re, a.im)), im: add1(m, m) }
}
#[inline(always)]
fn cscale1(a: F1, c: u32) -> F1 {
    let c = if c >= P1 { c - P1 } else { c };
    F1 { re: mul1(a.re, c), im: mul1(a.im, c) }
}

#[inline(always)]
fn cadd2(a: F2, b: F2) -> F2 {
    F2 { re: add2(a.re, b.re), im: add2(a.im, b.im) }
}
#[inline(always)]
fn csub2(a: F2, b: F2) -> F2 {
    F2 { re: sub2(a.re, b.re), im: sub2(a.im, b.im) }
}
#[inline(always)]
fn cmul2(a: F2, b: F2) -> F2 {
    // Karatsuba: three base multiplications.
    let ac = mul2(a.re, b.re);
    let bd = mul2(a.im, b.im);
    let m = mul2(add2(a.re, a.im), add2(b.re, b.im));
    F2 { re: sub2(ac, bd), im: sub2(m, add2(ac, bd)) }
}
#[inline(always)]
fn csqr2(a: F2) -> F2 {
    let m = mul2(a.re, a.im);
    F2 { re: mul2(add2(a.re, a.im), sub2(a.re, a.im)), im: add2(m, m) }
}
#[inline(always)]
fn cscale2(a: F2, c: u32) -> F2 {
    F2 { re: mul2(a.re, c as u64), im: mul2(a.im, c as u64) }
}

// ---- Block-level transform pieces (a1, a2 may point to shared or global memory) ----

/// Coefficients `lim[0..cnt)` (cnt ≤ n) as real parts, weighted by ζ^j.
#[inline(always)]
unsafe fn load_limbs(lim: *const u32, cnt: u32, a1: *mut F1, a2: *mut F2, t: &Tab) {
    let mut j = thread::threadIdx_x();
    while j < t.n {
        let i = j as usize;
        let v = if j < cnt { unsafe { *lim.add(i) } } else { 0 };
        let v1 = v % P1;
        unsafe {
            let (w1, w2) = (*t.w1.add(i), *t.w2.add(i));
            *a1.add(i) = F1 { re: mul1(w1.re, v1), im: mul1(w1.im, v1) };
            *a2.add(i) = F2 { re: mul2(w2.re, v as u64), im: mul2(w2.im, v as u64) };
        }
        j += thread::blockDim_x();
    }
}

/// GF(p²) operations shared by the butterflies of both fields.
trait Gf: Copy {
    fn add(self, b: Self) -> Self;
    fn sub(self, b: Self) -> Self;
    fn mul(self, b: Self) -> Self;
    fn conj(self) -> Self;
    /// self + i · b.
    fn add_i(self, b: Self) -> Self;
    /// self − i · b.
    fn sub_i(self, b: Self) -> Self;
}

impl Gf for F1 {
    #[inline(always)]
    fn add(self, b: F1) -> F1 {
        cadd1(self, b)
    }
    #[inline(always)]
    fn sub(self, b: F1) -> F1 {
        csub1(self, b)
    }
    #[inline(always)]
    fn mul(self, b: F1) -> F1 {
        cmul1(self, b)
    }
    #[inline(always)]
    fn conj(self) -> F1 {
        F1 { re: self.re, im: sub1(0, self.im) }
    }
    #[inline(always)]
    fn add_i(self, b: F1) -> F1 {
        F1 { re: sub1(self.re, b.im), im: add1(self.im, b.re) }
    }
    #[inline(always)]
    fn sub_i(self, b: F1) -> F1 {
        F1 { re: add1(self.re, b.im), im: sub1(self.im, b.re) }
    }
}

impl Gf for F2 {
    #[inline(always)]
    fn add(self, b: F2) -> F2 {
        cadd2(self, b)
    }
    #[inline(always)]
    fn sub(self, b: F2) -> F2 {
        csub2(self, b)
    }
    #[inline(always)]
    fn mul(self, b: F2) -> F2 {
        cmul2(self, b)
    }
    #[inline(always)]
    fn conj(self) -> F2 {
        F2 { re: self.re, im: sub2(0, self.im) }
    }
    #[inline(always)]
    fn add_i(self, b: F2) -> F2 {
        F2 { re: sub2(self.re, b.im), im: add2(self.im, b.re) }
    }
    #[inline(always)]
    fn sub_i(self, b: F2) -> F2 {
        F2 { re: add2(self.re, b.im), im: sub2(self.im, b.re) }
    }
}

/// Two DIF stages on a[i0 + {0, q, 2q, 3q}] (spans 2q and q) as one radix-4 butterfly, with
/// twiddles w1 = W^j, w2 = W^2j and w3 = W^3j of the block's 4q-th root W. W^q = i in both
/// fields, so the result equals two radix-2 stages with three multiplications instead of four.
/// With `UNIT` (q = 1, so j = 0) every twiddle is 1 and w1, w2, w3 are unused. `SIN` and
/// `SOUT` read and write element i at swz(i) (see `at`).
#[inline(always)]
unsafe fn dif4<T: Gf, const UNIT: bool, const SIN: bool, const SOUT: bool>(a: *mut T, i0: usize, q: usize, w1: T, w2: T, w3: T) {
    let mul = |y: T, w: T| if UNIT { y } else { y.mul(w) };
    let (r, w) = (|k: usize| at::<SIN>(i0 + k * q), |k: usize| at::<SOUT>(i0 + k * q));
    unsafe {
        let (x0, x1, x2, x3) = (*a.add(r(0)), *a.add(r(1)), *a.add(r(2)), *a.add(r(3)));
        let (y0, y1, y2, y3) = (x0.add(x2), x1.add(x3), x0.sub(x2), x1.sub(x3));
        fence::<SIN, SOUT>();
        *a.add(w(0)) = y0.add(y1);
        *a.add(w(1)) = mul(y0.sub(y1), w2);
        *a.add(w(2)) = mul(y2.add_i(y3), w1);
        *a.add(w(3)) = mul(y2.sub_i(y3), w3);
    }
}

/// Two DIT stages on a[i0 + {0, q, 2q, 3q}] (spans q and 2q) as one radix-4 butterfly, with
/// twiddles w1 = V^j, w2 = V^2j and w3 = V^3j of the block's inverse root V, V^q = −i; `UNIT`,
/// `SIN` and `SOUT` as for `dif4`.
#[inline(always)]
unsafe fn dit4<T: Gf, const UNIT: bool, const SIN: bool, const SOUT: bool>(a: *mut T, i0: usize, q: usize, w1: T, w2: T, w3: T) {
    let mul = |y: T, w: T| if UNIT { y } else { y.mul(w) };
    let (r, w) = (|k: usize| at::<SIN>(i0 + k * q), |k: usize| at::<SOUT>(i0 + k * q));
    unsafe {
        let (x0, x1, x2, x3) = (*a.add(r(0)), mul(*a.add(r(1)), w2), mul(*a.add(r(2)), w1), mul(*a.add(r(3)), w3));
        let (y0, y1, y2, y3) = (x0.add(x1), x0.sub(x1), x2.add(x3), x2.sub(x3));
        fence::<SIN, SOUT>();
        *a.add(w(0)) = y0.add(y2);
        *a.add(w(2)) = y0.sub(y2);
        *a.add(w(1)) = y1.sub_i(y3);
        *a.add(w(3)) = y1.add_i(y3);
    }
}

/// swz(i) = i ^ (((i >> 3) ^ (i >> 4)) & 7) when `S`, else i. swz permutes each aligned block
/// of 8 elements and is its own inverse. In that order the span-1 radix-4 pass (each thread's
/// 4 elements contiguous) reads and writes shared memory without bank conflicts, and the span-4
/// pass with half as many. With an even number of stages these are the last DIF and the first
/// DIT passes, so when `swizzled` the spectrum stays in swz order between `dif` and `dit`.
#[inline(always)]
fn at<const S: bool>(i: usize) -> usize {
    if S { i ^ (((i >> 3) ^ (i >> 4)) & 7) } else { i }
}

/// Spectra of `t` in swz order: `Tab::swz` set and log₂ n even and at least 4 (every DIF and
/// DIT stage range that ends or starts at log₂ n then holds both passes).
#[inline(always)]
fn swizzled(t: &Tab) -> bool {
    t.swz != 0 && t.logn % 2 == 0 && t.logn >= 4
}

/// Between a radix-4 group's reads and writes when their orders differ: the writes can land on
/// elements that the group's other lanes (span 4: four consecutive lanes share 16 elements)
/// have yet to read.
#[inline(always)]
fn fence<const SIN: bool, const SOUT: bool>() {
    if SIN != SOUT {
        warp::sync_mask(0xf << (thread::threadIdx_x() % 32 & !3));
    }
}

/// One radix-4 pass over the first `len` elements: DIF stages with second span q = 2^lq, or DIT
/// stages with first span q; `UNIT`, `SIN` and `SOUT` as for `dif4`.
#[inline(always)]
unsafe fn pass4<const DIF: bool, const UNIT: bool, const SIN: bool, const SOUT: bool>(a1: *mut F1, a2: *mut F2, t: &Tab, lq: u32, len: u32) {
    thread::sync_threads();
    let q = 1u32 << lq;
    let mut b = thread::threadIdx_x();
    while b < len >> 2 {
        let j = b & (q - 1);
        let i0 = (((b >> lq) << (lq + 2)) + j) as usize;
        let w1 = (2 * t.n - 6 * q + j) as usize;
        let (w2, w3) = (w1 + q as usize, w1 + 2 * q as usize);
        unsafe {
            if DIF {
                dif4::<_, UNIT, SIN, SOUT>(a1, i0, q as usize, *t.tw1.add(w1), *t.tw1.add(w2), *t.tw1.add(w3));
                dif4::<_, UNIT, SIN, SOUT>(a2, i0, q as usize, *t.tw2.add(w1), *t.tw2.add(w2), *t.tw2.add(w3));
            } else {
                dit4::<_, UNIT, SIN, SOUT>(a1, i0, q as usize, *t.itw1.add(w1), *t.itw1.add(w2), *t.itw1.add(w3));
                dit4::<_, UNIT, SIN, SOUT>(a2, i0, q as usize, *t.itw2.add(w1), *t.itw2.add(w2), *t.itw2.add(w3));
            }
        }
        b += thread::blockDim_x();
    }
}

/// Decimation in frequency, natural order in, bit-reversed order out (swz order when
/// `swizzled`, see `at`). With `TILED` (global mode) the stages that act within blocks of at
/// most 2^TILE_LOG elements run on tiles in shared memory (see `tiles`).
#[inline(always)]
unsafe fn dif<const TILED: bool>(a1: *mut F1, a2: *mut F2, t: &Tab) {
    unsafe {
        if TILED { tiles::<true>(a1, a2, t) } else { dif_stages(a1, a2, t, 0, t.logn, t.n) }
    }
}

/// DIF stages s_from..s_to on the first `len` elements, which hold whole blocks of the stages'
/// span. Radix 4 (stages s and s + 1 per barrier), with a trailing radix-2 stage when the
/// count is odd, which must then be stage log₂ n − 1.
#[inline(always)]
unsafe fn dif_stages(a1: *mut F1, a2: *mut F2, t: &Tab, s_from: u32, s_to: u32, len: u32) {
    // When `swizzled` and the stages end at log₂ n, the last two passes (spans 4 and 1, the
    // latter with unit twiddles) leave swz order.
    let swz = s_to == t.logn && s_to - s_from >= 4 && swizzled(t);
    let end = if swz { s_to - 4 } else { s_to };
    let mut s = s_from;
    while s + 2 <= end {
        unsafe { pass4::<true, false, false, false>(a1, a2, t, t.logn - 2 - s, len) };
        s += 2;
    }
    if swz {
        unsafe {
            pass4::<true, false, false, true>(a1, a2, t, 2, len);
            pass4::<true, true, true, true>(a1, a2, t, 0, len);
        }
        s = s_to;
    }
    if s < s_to {
        // Stage log₂ n − 1: span 1, twiddle tw[0] = 1.
        thread::sync_threads();
        let mut b = thread::threadIdx_x();
        while b < len >> 1 {
            let (i0, i1) = (2 * b as usize, 2 * b as usize + 1);
            unsafe {
                let (u1, v1) = (*a1.add(i0), *a1.add(i1));
                *a1.add(i0) = cadd1(u1, v1);
                *a1.add(i1) = csub1(u1, v1);
                let (u2, v2) = (*a2.add(i0), *a2.add(i1));
                *a2.add(i0) = cadd2(u2, v2);
                *a2.add(i1) = csub2(u2, v2);
            }
            b += thread::blockDim_x();
        }
    }
    thread::sync_threads();
}

/// Decimation in time, bit-reversed order in (swz order when `swizzled`, see `at`), natural
/// order out, then unweighting by ζ^−j / n.
#[inline(always)]
unsafe fn dit<const TILED: bool>(a1: *mut F1, a2: *mut F2, t: &Tab) {
    unsafe { dit_raw::<TILED>(a1, a2, t) };
    let mut j = thread::threadIdx_x();
    while j < t.n {
        let i = j as usize;
        unsafe {
            *a1.add(i) = cmul1(*a1.add(i), *t.iw1.add(i));
            *a2.add(i) = cmul2(*a2.add(i), *t.iw2.add(i));
        }
        j += thread::blockDim_x();
    }
    thread::sync_threads();
}

/// Decimation in time without the unweighting; `TILED` as for `dif`.
#[inline(always)]
unsafe fn dit_raw<const TILED: bool>(a1: *mut F1, a2: *mut F2, t: &Tab) {
    unsafe {
        if TILED { tiles::<false>(a1, a2, t) } else { dit_stages(a1, a2, t, t.logn, 0, t.n) }
    }
}

/// DIT stages s_hi − 1 down to s_lo on the first `len` elements, which hold whole blocks of
/// the stages' span. Radix 4 (stages s − 1 and s − 2 per barrier), with a trailing radix-2
/// stage s_lo when the count is odd.
#[inline(always)]
unsafe fn dit_stages(a1: *mut F1, a2: *mut F2, t: &Tab, s_hi: u32, s_lo: u32, len: u32) {
    // When `swizzled` and the stages start at log₂ n, the first two passes take swz order (see
    // `dif_stages`).
    let mut s = s_hi;
    if s_hi == t.logn && s_hi - s_lo >= 4 && swizzled(t) {
        unsafe {
            pass4::<false, true, true, true>(a1, a2, t, 0, len);
            pass4::<false, false, true, false>(a1, a2, t, 2, len);
        }
        s -= 4;
    }
    while s >= s_lo + 2 {
        unsafe { pass4::<false, false, false, false>(a1, a2, t, t.logn - s, len) };
        s -= 2;
    }
    if s == s_lo + 1 {
        // Stage s_lo: span h = 2^(log₂ n − 1 − s_lo), twiddle index j << s_lo.
        let lh = t.logn - 1 - s_lo;
        let h = 1u32 << lh;
        thread::sync_threads();
        let mut b = thread::threadIdx_x();
        while b < len >> 1 {
            let j = b & (h - 1);
            let i0 = ((b >> lh) << (lh + 1)) + j;
            let (i0, i1, w) = (i0 as usize, (i0 + h) as usize, (j << s_lo) as usize);
            unsafe {
                let (u1, v1) = (*a1.add(i0), cmul1(*a1.add(i1), *t.itw1.add(w)));
                *a1.add(i0) = cadd1(u1, v1);
                *a1.add(i1) = csub1(u1, v1);
                let (u2, v2) = (*a2.add(i0), cmul2(*a2.add(i1), *t.itw2.add(w)));
                *a2.add(i0) = cadd2(u2, v2);
                *a2.add(i1) = csub2(u2, v2);
            }
            b += thread::blockDim_x();
        }
    }
    thread::sync_threads();
}

/// log₂ of the largest global-mode tile, in elements of both fields (24 bytes each).
const TILE_LOG: u32 = 11;
/// Byte offset of the global-mode tiles in dynamic shared memory, after the carry scratch of
/// up to 1,024 threads (`block_carry`). The F2 tile comes first, then the F1 tile.
const TILE_OFFSET: usize = 528;

/// First stage of length 2^logn that runs in tiles: stages s ≥ s0 act within blocks of
/// 2^(logn − s0) ≤ 2^TILE_LOG elements. Even, so the stages before it pair into radix-4 passes.
#[inline(always)]
const fn tile_stage(logn: u32) -> u32 {
    let s = logn.saturating_sub(TILE_LOG);
    s + (s & 1)
}

/// A whole transform in global memory (DIF, or DIT without the unweighting): the stages below
/// s0 = `tile_stage` in place, the others on each block of 2^(logn − s0) elements copied
/// into a shared-memory tile and back, so they cost one global round trip instead of one per
/// radix-4 pass.
#[inline(always)]
unsafe fn tiles<const DIF: bool>(a1: *mut F1, a2: *mut F2, t: &Tab) {
    let s0 = tile_stage(t.logn);
    let m = t.n >> s0;
    let b2 = DynamicSharedArray::<F2, 16>::offset(TILE_OFFSET);
    let b1 = DynamicSharedArray::<F1, 16>::offset(TILE_OFFSET + 16 * m as usize);
    unsafe {
        if DIF {
            dif_stages(a1, a2, t, 0, s0, t.n);
        }
        // No barrier before a DIT copy-in: c is a multiple of blockDim (one tile, or m ≥ 1,024
        // with at most 512 threads), so each thread copies the elements it last wrote.
        let mut c = 0;
        while c < t.n {
            let (g1, g2) = (a1.add(c as usize), a2.add(c as usize));
            let mut j = thread::threadIdx_x();
            while j < m {
                *b1.add(j as usize) = *g1.add(j as usize);
                *b2.add(j as usize) = *g2.add(j as usize);
                j += thread::blockDim_x();
            }
            if DIF { dif_stages(b1, b2, t, s0, t.logn, m) } else { dit_stages(b1, b2, t, t.logn, s0, m) }
            // Each thread copies back the elements it copied in, so the next tile can follow
            // without a barrier.
            let mut j = thread::threadIdx_x();
            while j < m {
                *g1.add(j as usize) = *b1.add(j as usize);
                *g2.add(j as usize) = *b2.add(j as usize);
                j += thread::blockDim_x();
            }
            c += m;
        }
        if DIF {
            thread::sync_threads();
        } else {
            dit_stages(a1, a2, t, s0, 0, t.n);
        }
    }
}

// ---- Wraparound path (spec section 9; `Half` in the host's src/cpu.rs) ----

/// Packs k = 2h limbs as z_j = c_2j + i c_2j+1, j < h, unweighted.
#[inline(always)]
unsafe fn load_packed(lim: *const u32, a1: *mut F1, a2: *mut F2, h: u32) {
    let mut j = thread::threadIdx_x();
    while j < h {
        let i = j as usize;
        unsafe {
            let (re, im) = (*lim.add(2 * i), *lim.add(2 * i + 1));
            *a1.add(i) = F1 { re: re % P1, im: im % P1 };
            *a2.add(i) = F2 { re: re as u64, im: im as u64 };
        }
        j += thread::blockDim_x();
    }
}

/// Position of spectral index (h − rev(p)) mod h, for bit-reversed positions p.
#[inline(always)]
fn partner(p: u32) -> u32 {
    if p == 0 { 0 } else { (3 << (31 - p.leading_zeros())) - 1 - p }
}

/// out ← repack(unpack(z) ⊙ unpack(N)) over h positions (`half_pointwise` on the host, out of
/// place, one output per position) as e · A + d · B with e = z_p + conj z_q, d = z_p − conj z_q
/// (q the partner of p), A at nh[p] and B at nh[h + p] (`half_factors` on the host).
#[inline(always)]
unsafe fn half_pointwise<T: Gf>(z: *const T, out: *mut T, nh: *const T, h: u32) {
    let mut p = thread::threadIdx_x();
    while p < h {
        let pi = p as usize;
        unsafe {
            let (zp, zq) = (*z.add(pi), (*z.add(partner(p) as usize)).conj());
            *out.add(pi) = zp.add(zq).mul(*nh.add(pi)).add(zp.sub(zq).mul(*nh.add(h as usize + pi)));
        }
        p += thread::blockDim_x();
    }
}

// ---- Carry and the fused modular step ----

/// Block-wide OR of `p` that also acts as a barrier (CUDA `__syncthreads_or`).
#[inline(always)]
fn block_or(p: bool) -> bool {
    let r: u32;
    unsafe {
        ptx_asm!(
            "{ .reg .pred %%bp<2>; setp.ne.u32 %%bp0, %1, 0; bar.red.or.pred %%bp1, 0, %%bp0; selp.u32 %0, 1, 0, %%bp1; }",
            out("=r") r,
            in("r") p as u32,
            clobber("memory"),
        );
    }
    r != 0
}

/// The unique C < M31 · M61 with C ≡ c1 (mod M31) and C ≡ c2 (mod M61).
#[inline(always)]
fn crt(c1: u32, c2: u64, inv: u32) -> u128 {
    let c2m = (c2 as u32 & P1) + (c2 >> 31) as u32;
    let c2m = if c2m >= P1 { c2m - P1 } else { c2m };
    let h = mul1(sub1(c1, c2m), inv);
    c2 as u128 + P2 as u128 * h as u128
}

/// Carry into this thread from thread tid − 1: a warp shuffle within the warp, `carries` (one
/// u64 per warp, written by each warp's last lane before the preceding barrier) across warps.
#[inline(always)]
unsafe fn carry_in(c: u64, first: u64, carries: *const u64) -> u64 {
    let tid = thread::threadIdx_x();
    let up = warp::shuffle_up_u64(c, 1);
    if tid == 0 {
        first
    } else if tid % 32 == 0 {
        unsafe { *carries.add((tid / 32 - 1) as usize) }
    } else {
        up
    }
}

/// Adds `cin` to the thread's limbs `l` (those below `count`) and returns the carry out.
#[inline(always)]
fn add_into(l: &mut [u32; CARRY_E as usize], cin: u64, q0: u32, count: u32) -> u64 {
    let mut r = cin;
    let mut e = 0;
    while e < CARRY_E {
        if q0 + e < count {
            let v = l[e as usize] as u64 + r;
            l[e as usize] = v as u32;
            r = v >> 32;
        }
        e += 1;
    }
    r
}

/// Carries into bit i of the sum of generate mask `g` and propagate-or-generate mask `y` with
/// carry `cin` into bit 0, and whether a carry leaves bit 31.
#[inline(always)]
fn lookahead(g: u32, y: u32, cin: u32) -> (u32, bool) {
    let s = g as u64 + y as u64 + cin as u64;
    ((s as u32) ^ g ^ y, s >> 32 != 0)
}

/// Writes the 32-bit limbs of Σ coef(q) 2^(32q) over q < count to `out` and returns the carry
/// out of coefficient count − 1. Each thread converts CARRY_E consecutive coefficients of a
/// chunk of blockDim · CARRY_E and adds its neighbour's carry. The carries left between threads
/// are 0 or 1 and resolve in one carry-lookahead step: generate/propagate ballots per warp, then
/// one scan over the warps, so runs of all-ones limbs cost no extra rounds. `scratch` holds
/// 2 · blockDim / 32 + 2 u64 (per-warp carries, per-warp flags, and the chunk carry in two
/// alternating words, so that each chunk needs three barriers) and aliases neither the source
/// nor `out`.
#[inline(always)]
unsafe fn block_carry<C: Fn(u32) -> u128>(coef: C, count: u32, out: *mut u32, scratch: *mut u64) -> u64 {
    let (tid, nt) = (thread::threadIdx_x(), thread::blockDim_x());
    let nw = (nt / 32) as usize;
    let (carries, flags, chunk) = unsafe { (scratch, scratch.add(nw), scratch.add(2 * nw)) };
    let mut chunk_in = 0u64;
    let mut base = 0;
    let mut parity = 0;
    while base < count {
        let q0 = base + tid * CARRY_E;
        // Thread owning coefficient count − 1 of this chunk; its carry is the chunk's carry out.
        let owner = ((count - 1 - base) / CARRY_E).min(nt - 1);
        let mut l = [0u32; CARRY_E as usize];
        let mut c = 0u64;
        let mut e = 0;
        while e < CARRY_E {
            if q0 + e < count {
                let v = coef(q0 + e) + c as u128;
                l[e as usize] = v as u32;
                c = (v >> 32) as u64;
            }
            e += 1;
        }
        unsafe {
            // Each region was last read before the previous chunk's second or third barrier.
            if tid % 32 == 31 {
                *carries.add((tid / 32) as usize) = c;
            }
            thread::sync_threads();
            let cin = carry_in(c, chunk_in, carries);
            // cin < 2^64 and the thread holds at least 64 bits, so r is 0 or 1.
            let r = add_into(&mut l, cin, q0, count);
            // Carry lookahead over threads below the owner (the owner's carry leaves the chunk).
            let live = tid < owner;
            let mut ones = true;
            let mut e = 0;
            while e < CARRY_E {
                ones &= l[e as usize] == u32::MAX;
                e += 1;
            }
            let (lane, w) = (tid % 32, tid / 32);
            let g = warp::ballot(live && r != 0);
            let y = g | warp::ballot(live && ones);
            let (_, wg) = lookahead(g, y, 0);
            // Per-warp flag: bit 0, the warp generates a carry; bit 1, it only propagates one.
            if lane == 0 {
                *flags.add(w as usize) = wg as u64 | ((y == u32::MAX && !wg) as u64) << 1;
            }
            thread::sync_threads();
            // Every warp scans the per-warp flags itself, which saves a barrier.
            let f = if lane < nt / 32 { *flags.add(lane as usize) } else { 0 };
            let gw = warp::ballot(f & 1 != 0);
            let (cw, _) = lookahead(gw, gw | warp::ballot(f & 2 != 0), 0);
            let (cl, _) = lookahead(g, y, (cw >> w) & 1);
            let r2 = add_into(&mut l, ((cl >> lane) & 1) as u64, q0, count);
            let top = c + r + r2;
            let mut e = 0;
            while e < CARRY_E {
                if q0 + e < count {
                    *out.add((q0 + e) as usize) = l[e as usize];
                }
                e += 1;
            }
            if tid == owner {
                *chunk.add(parity) = top;
            }
            thread::sync_threads();
            chunk_in = *chunk.add(parity);
        }
        parity ^= 1;
        base += nt * CARRY_E;
    }
    chunk_in
}

/// Carry of a product in (a1, a2): coefficient q is the real part of element q for q < n and
/// the imaginary part of element q − n otherwise.
#[inline(always)]
unsafe fn carry_product(a1: *const F1, a2: *const F2, count: u32, out: *mut u32, scratch: *mut u64, t: &Tab) -> u64 {
    let (n, inv) = (t.n, t.crt_inv);
    let coef = move |q: u32| {
        let hi = q >= n;
        let j = (if hi { q - n } else { q }) as usize;
        let (e1, e2) = unsafe { (*a1.add(j), *a2.add(j)) };
        if hi { crt(e1.im, e2.im, inv) } else { crt(e1.re, e2.re, inv) }
    };
    unsafe { block_carry(coef, count, out, scratch) }
}

/// a ← a ⊙ b elementwise, with a in swz order (`at`) when `swizzled` and b in natural order.
#[inline(always)]
unsafe fn pointwise_const(a1: *mut F1, a2: *mut F2, b1: *const F1, b2: *const F2, t: &Tab) {
    let swz = swizzled(t);
    let mut j = thread::threadIdx_x();
    while j < t.n {
        let i = j as usize;
        let ib = if swz { at::<true>(i) } else { i };
        unsafe {
            *a1.add(i) = cmul1(*a1.add(i), *b1.add(ib));
            *a2.add(i) = cmul2(*a2.add(i), *b2.add(ib));
        }
        j += thread::blockDim_x();
    }
}

/// Per-slot batch state in device memory (structure of arrays over slots), passed by value.
/// The host mirrors this layout in `gpu::device::Slots`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Slots {
    /// k limbs per slot.
    pub x: *mut u32,
    /// 2n limbs per slot each, for T and S.
    pub lim_t: *mut u32,
    pub lim_s: *mut u32,
    /// n elements per slot each: F(N), or the wraparound path's factors A | B
    /// (`cpu::half_factors_of`), and F(N′).
    pub nhat1: *const F1,
    pub nhat2: *const F2,
    pub nphat1: *const F1,
    pub nphat2: *const F2,
    /// n elements per slot; global mode only.
    pub work1: *mut F1,
    pub work2: *mut F2,
    /// ew words per slot, most significant bit first.
    pub exp: *const u32,
    pub nbits: *const u32,
    pub pos: *mut u32,
    /// Set to 1 when a carry that must vanish does not.
    pub err: *mut u32,
    pub k: u32,
    pub ew: u32,
    pub base: u32,
}

/// Pointers of slot `slot`'s per-step buffers.
struct Slot {
    x: *mut u32,
    lim_t: *mut u32,
    lim_s: *mut u32,
    nhat1: *const F1,
    nhat2: *const F2,
    nphat1: *const F1,
    nphat2: *const F2,
    err: *mut u32,
}

impl Slots {
    fn slot(&self, slot: usize, n: usize) -> Slot {
        let k = self.k as usize;
        unsafe {
            Slot {
                x: self.x.add(slot * k),
                lim_t: self.lim_t.add(slot * 2 * n),
                lim_s: self.lim_s.add(slot * 2 * n),
                nhat1: self.nhat1.add(slot * n),
                nhat2: self.nhat2.add(slot * n),
                nphat1: self.nphat1.add(slot * n),
                nphat2: self.nphat2.add(slot * n),
                err: self.err.add(slot),
            }
        }
    }
}

/// Steps 2 to 4 of spec section 3: from T in `lim_t`, writes T · R⁻¹ (lazy, in [0, 2N)) to
/// `x`. `x` may be any k-limb buffer; slot `s` provides N̂, N̂′, the limb scratch and `err`.
/// With `WRAP`, steps 3 and 4 are the wraparound path of spec section 9 (k = L/2, tables `ht`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn redc<const WRAP: bool, const TILED: bool>(a1: *mut F1, a2: *mut F2, scratch: *mut u64, x: *mut u32, s: &Slot, k: u32, t: &Tab, ht: &HalfTab) {
    unsafe {
        // m = (T mod R) · N′ mod R
        load_limbs(s.lim_t, k, a1, a2, t);
        dif::<TILED>(a1, a2, t);
        pointwise_const(a1, a2, s.nphat1, s.nphat2, t);
        dit::<TILED>(a1, a2, t);
        carry_product(a1, a2, k, s.lim_s, scratch, t);
        if WRAP {
            // Cyclic m · N mod (R − 1) through length-h transforms of the packed limbs.
            // The packed spectrum goes to elements 0..h, the product to h..2h = n.
            let h = ht.tab.n;
            load_packed(s.lim_s, a1, a2, h);
            dif::<TILED>(a1, a2, &ht.tab);
            let (b1, b2) = (a1.add(h as usize), a2.add(h as usize));
            half_pointwise(a1, b1, s.nhat1, h);
            half_pointwise(a2, b2, s.nhat2, h);
            dit_raw::<TILED>(b1, b2, &ht.tab);
            // G = (cyclic + T mod R) mod (R − 1) = ⌊S / R⌋ + [T mod R ≠ 0], then x = ⌊T / R⌋ + G.
            let (lt, inv) = (s.lim_t, t.crt_inv);
            let coef = move |q: u32| {
                let (e1, e2) = (*b1.add(q as usize / 2), *b2.add(q as usize / 2));
                let c = if q % 2 == 0 { crt(e1.re, e2.re, inv) } else { crt(e1.im, e2.im, inv) };
                c + *lt.add(q as usize) as u128
            };
            let mut c = block_carry(coef, k, x, scratch);
            // End-around carry, 2^(32k) ≡ 1: one fold in practice; a second is reachable in
            // principle (when x + c ≥ R) and is too rare to reach in tests.
            while c != 0 {
                let cin = c as u128;
                c = block_carry(move |q: u32| *x.add(q as usize) as u128 + if q == 0 { cin } else { 0 }, k, x, scratch);
            }
            let top = block_carry(move |q: u32| *x.add(q as usize) as u128 + *lt.add((k + q) as usize) as u128, k, x, scratch);
            if thread::threadIdx_x() == 0 && top != 0 {
                *s.err = 1;
            }
            return;
        }
        let mut nz = false;
        let mut j = thread::threadIdx_x();
        while j < k {
            nz |= *s.lim_t.add(j as usize) != 0;
            j += thread::blockDim_x();
        }
        let flag = block_or(nz) as u32;
        // S = m · N
        load_limbs(s.lim_s, k, a1, a2, t);
        dif::<TILED>(a1, a2, t);
        pointwise_const(a1, a2, s.nhat1, s.nhat2, t);
        dit::<TILED>(a1, a2, t);
        let top_s = carry_product(a1, a2, 2 * k, s.lim_s, scratch, t);
        // t = ⌊T / R⌋ + ⌊S / R⌋ + [T mod R ≠ 0]
        let (lt, ls) = (s.lim_t, s.lim_s);
        let coef = move |q: u32| {
            let i = (k + q) as usize;
            *lt.add(i) as u128 + *ls.add(i) as u128 + (if q == 0 { flag as u128 } else { 0 })
        };
        let top = block_carry(coef, k, x, scratch);
        if thread::threadIdx_x() == 0 && (top_s != 0 || top != 0) {
            *s.err = 1;
        }
    }
}

/// T = carry(F⁻¹(c · F(x) ⊙ F(x))) into `lim_t`; returns the carry out of 2k limbs.
#[inline(always)]
unsafe fn square<const TILED: bool>(a1: *mut F1, a2: *mut F2, scratch: *mut u64, s: &Slot, k: u32, c: u32, t: &Tab) -> u64 {
    unsafe {
        load_limbs(s.x, k, a1, a2, t);
        dif::<TILED>(a1, a2, t);
        let mut j = thread::threadIdx_x();
        while j < t.n {
            let i = j as usize;
            let mut v1 = csqr1(*a1.add(i));
            let mut v2 = csqr2(*a2.add(i));
            if c != 1 {
                v1 = cscale1(v1, c);
                v2 = cscale2(v2, c);
            }
            *a1.add(i) = v1;
            *a2.add(i) = v2;
            j += thread::blockDim_x();
        }
        dit::<TILED>(a1, a2, t);
        carry_product(a1, a2, 2 * k, s.lim_t, scratch, t)
    }
}

/// Up to `bits` steps x ← x² · c^bit · R⁻¹ on slot `first` + blockIdx; `BARE` runs step 1 (the squaring)
/// alone and leaves x unchanged. With `SHARED` the vector lives in shared memory: F2 at offset 0
/// (16n bytes), F1 at 16n, the carry scratch at 24n; otherwise in the slot's global work
/// buffers with the scratch at offset 0 and the transform tiles at `TILE_OFFSET`. Everything
/// below is inlined, so each mode compiles to address-space-specific loads and stores.
#[inline(always)]
unsafe fn steps<const BARE: bool, const SHARED: bool, const WRAP: bool>(t: &Tab, ht: &HalfTab, s: &Slots, first: u32, bits: u32) {
    // Length-n spectra stay in swz order between `dif` and `dit` (the half path's do not).
    let t = &Tab { swz: 1, ..*t };
    let slot = (first + thread::blockIdx_x()) as usize;
    let n = t.n as usize;
    let (a1, a2, scratch): (*mut F1, *mut F2, *mut u64) = if SHARED {
        (DynamicSharedArray::<F1, 16>::offset(16 * n), DynamicSharedArray::<F2, 16>::get(), DynamicSharedArray::<u64, 16>::offset(24 * n))
    } else {
        unsafe { (s.work1.add(slot * n), s.work2.add(slot * n), DynamicSharedArray::<u64, 16>::get()) }
    };
    let sl = s.slot(slot, n);
    unsafe {
        let ex = s.exp.add(slot * s.ew as usize);
        let nb = *s.nbits.add(slot);
        let mut pos = *s.pos.add(slot);
        let mut it = 0;
        while it < bits && pos < nb {
            let bit = (*ex.add((pos >> 5) as usize) >> (31 - (pos & 31))) & 1;
            let c = if bit != 0 { s.base } else { 1 };
            // Global mode transforms through shared-memory tiles.
            let top = if SHARED { square::<false>(a1, a2, scratch, &sl, s.k, c, t) } else { square::<true>(a1, a2, scratch, &sl, s.k, c, t) };
            if thread::threadIdx_x() == 0 && top != 0 {
                *sl.err = 1;
            }
            if !BARE {
                if SHARED {
                    redc::<WRAP, false>(a1, a2, scratch, sl.x, &sl, s.k, t, ht);
                } else {
                    redc::<WRAP, true>(a1, a2, scratch, sl.x, &sl, s.k, t, ht);
                }
            }
            it += 1;
            pos += 1;
        }
        if thread::threadIdx_x() == 0 {
            *s.pos.add(slot) = pos;
        }
    }
}

/// The modular step of spec section 3 in shared mode, one block per slot; the wraparound path
/// when `ht.tab.n` is nonzero. Global mode has its own kernel, `run_steps_global`: in one
/// kernel its register pressure slows the shared path.
#[kernel]
#[launch_bounds(1024)]
pub fn run_steps(t: Tab, ht: HalfTab, s: Slots, first: u32, bits: u32) {
    unsafe {
        if ht.tab.n != 0 { steps::<false, true, true>(&t, &ht, &s, first, bits) } else { steps::<false, true, false>(&t, &ht, &s, first, bits) }
    }
}

/// `run_steps` in global mode, at most 512 threads.
#[kernel]
#[launch_bounds(512)]
pub fn run_steps_global(t: Tab, ht: HalfTab, s: Slots, first: u32, bits: u32) {
    unsafe {
        if ht.tab.n != 0 { steps::<false, false, true>(&t, &ht, &s, first, bits) } else { steps::<false, false, false>(&t, &ht, &s, first, bits) }
    }
}

/// Step 1 of spec section 3 alone, for the overhead measurement of spec section 11.1.
#[kernel]
#[launch_bounds(1024)]
pub fn run_bare(t: Tab, ht: HalfTab, s: Slots, first: u32, bits: u32) {
    unsafe { steps::<true, true, false>(&t, &ht, &s, first, bits) }
}

/// `run_bare` in global mode, at most 512 threads.
#[kernel]
#[launch_bounds(512)]
pub fn run_bare_global(t: Tab, ht: HalfTab, s: Slots, first: u32, bits: u32) {
    unsafe { steps::<true, false, false>(&t, &ht, &s, first, bits) }
}

/// x[out] ← x[a] · x[b] · R⁻¹ with slot `out`'s modulus, in one block. F(x[a]) and F(x[b])
/// go to (a1, a2) and (b1, b2), n elements each; the carry scratch is dynamic shared memory.
#[kernel]
#[launch_bounds(1024)]
pub fn mul_slots(t: Tab, ht: HalfTab, s: Slots, out: u32, ia: u32, ib: u32, a1: *mut F1, a2: *mut F2, b1: *mut F1, b2: *mut F2) {
    let scratch = DynamicSharedArray::<u64, 16>::get();
    let k = s.k as usize;
    let sl = s.slot(out as usize, t.n as usize);
    unsafe {
        load_limbs(s.x.add(ib as usize * k), s.k, b1, b2, &t);
        dif::<false>(b1, b2, &t);
        load_limbs(s.x.add(ia as usize * k), s.k, a1, a2, &t);
        dif::<false>(a1, a2, &t);
        pointwise_const(a1, a2, b1, b2, &t);
        dit::<false>(a1, a2, &t);
        let top = carry_product(a1, a2, 2 * s.k, sl.lim_t, scratch, &t);
        if thread::threadIdx_x() == 0 && top != 0 {
            *sl.err = 1;
        }
        if ht.tab.n != 0 {
            redc::<true, false>(a1, a2, scratch, sl.x, &sl, s.k, &t, &ht);
        } else {
            redc::<false, false>(a1, a2, scratch, sl.x, &sl, s.k, &t, &ht);
        }
    }
}

// ---- Unfused Transform-trait kernels, one block per vector, data in global memory ----
// Block b works on vector `first` + b, so the host can split a batch over several launches.

/// Raw coefficient layout in: element j of vector v holds (c_j, c_{j+n}); weights then DIF.
#[kernel]
pub fn t_forward(t: Tab, d1: *mut F1, d2: *mut F2, first: u32) {
    let off = (first + thread::blockIdx_x()) as usize * t.n as usize;
    unsafe {
        let (a1, a2) = (d1.add(off), d2.add(off));
        let mut j = thread::threadIdx_x();
        while j < t.n {
            let i = j as usize;
            *a1.add(i) = cmul1(*a1.add(i), *t.w1.add(i));
            *a2.add(i) = cmul2(*a2.add(i), *t.w2.add(i));
            j += thread::blockDim_x();
        }
        dif::<false>(a1, a2, &t);
    }
}

#[kernel]
pub fn t_inverse(t: Tab, d1: *mut F1, d2: *mut F2, first: u32) {
    let off = (first + thread::blockIdx_x()) as usize * t.n as usize;
    unsafe { dit::<false>(d1.add(off), d2.add(off), &t) }
}

/// o[v] = scale[v] · a[v] ⊙ b[v]; a null `scale` means 1.
#[kernel]
pub fn t_pointwise(t: Tab, o1: *mut F1, o2: *mut F2, a1: *const F1, a2: *const F2, b1: *const F1, b2: *const F2, scale: *const u32, first: u32) {
    let v = (first + thread::blockIdx_x()) as usize;
    let off = v * t.n as usize;
    let c = if scale.is_null() { 1 } else { unsafe { *scale.add(v) } };
    let mut j = thread::threadIdx_x();
    while j < t.n {
        let i = off + j as usize;
        unsafe {
            let mut v1 = cmul1(*a1.add(i), *b1.add(i));
            let mut v2 = cmul2(*a2.add(i), *b2.add(i));
            if c != 1 {
                v1 = cscale1(v1, c);
                v2 = cscale2(v2, c);
            }
            *o1.add(i) = v1;
            *o2.add(i) = v2;
        }
        j += thread::blockDim_x();
    }
}

/// Forward transforms of per-vector limb strings (k limbs each, stride k) into d.
#[kernel]
pub fn t_load_forward(t: Tab, lim: *const u32, k: u32, d1: *mut F1, d2: *mut F2, first: u32) {
    let v = (first + thread::blockIdx_x()) as usize;
    let off = v * t.n as usize;
    unsafe {
        let (a1, a2) = (d1.add(off), d2.add(off));
        load_limbs(lim.add(v * k as usize), k, a1, a2, &t);
        dif::<false>(a1, a2, &t);
    }
}

//! Backend interfaces.
//!
//! `Transform` is the transform layer of spec section 5. A vector of length `len` (= L)
//! holds L real coefficients; `forward` followed by `pointwise` and `inverse` yields the
//! product of two such vectors modulo X^L + 1, exact while every coefficient stays below
//! `coefficient_bound()`. Internally a vector is L/2 elements of GF(M31²) × GF(M61²) with
//! coefficient j in the real part and coefficient j + L/2 in the imaginary part of element j.
//!
//! `Backend` adds the fused modular step that `ModBatch` drives.

pub trait Limits {
    /// Product of the transform moduli; convolution coefficients must stay below it.
    fn coefficient_bound(&self) -> u128;
    /// Lengths this backend supports, ascending.
    fn supported_lengths(&self) -> &[usize];
    /// True if batches must use k = L/2 limbs (R = 2^(w · L/2)), as the wraparound correction needs.
    fn full_k(&self) -> bool {
        false
    }
}

pub trait Transform: Limits {
    /// Transform-domain data for `count` vectors of one length.
    type Buf;
    /// Per-vector small scalars.
    type Scales;

    /// Uploads `count` coefficient vectors of length `len` (row-major), untransformed.
    fn upload(&self, coeffs: &[u32], len: usize, count: usize) -> Self::Buf;
    /// Downloads untransformed data as exact coefficients in [0, Q) (CRT applied).
    fn download(&self, data: &Self::Buf) -> Vec<u128>;
    fn upload_scales(&self, scales: &[u32]) -> Self::Scales;
    /// In-place, batched over `count` vectors of length `len`.
    fn forward(&self, data: &mut Self::Buf, len: usize, count: usize);
    fn inverse(&self, data: &mut Self::Buf, len: usize, count: usize);
    /// out[i] = scale[i] · a[i] ⊙ b[i].
    fn pointwise(
        &self,
        out: &mut Self::Buf,
        a: &Self::Buf,
        b: &Self::Buf,
        scale: Option<&Self::Scales>,
        len: usize,
        count: usize,
    );
}

/// Static data for one batch, computed on the host.
pub struct BatchSetup<'a> {
    /// Limbs per residue.
    pub k: usize,
    /// Transform length L.
    pub len: usize,
    pub base: u32,
    /// Per slot: N, N′ = −N⁻¹ mod R and 2N, each as k little-endian 32-bit limbs.
    pub n: &'a [Vec<u32>],
    pub nprime: &'a [Vec<u32>],
    pub two_n: &'a [Vec<u32>],
}

pub trait Backend: Limits {
    type State;

    fn create(&self, setup: &BatchSetup) -> anyhow::Result<Self::State>;
    /// Writes k limbs of a Montgomery-form value in [0, 2N) to slot `i`.
    fn load(&self, st: &mut Self::State, i: usize, limbs: &[u32]);
    /// Exponent bits, most significant first: bit p is `(words[p / 32] >> (31 − p % 32)) & 1`.
    fn set_exponent(&self, st: &mut Self::State, i: usize, words: &[u32], nbits: usize);
    /// Runs up to `bits` steps x ← x² · c^bit on every slot with exponent bits left.
    fn run(&self, st: &mut Self::State, bits: usize);
    /// Slot `out` ← a · b · R⁻¹ for slots `a`, `b` that share a modulus.
    fn mul(&self, st: &mut Self::State, out: usize, a: usize, b: usize);
    /// k limbs of slot `i`, a Montgomery-form value in [0, 2N).
    fn read(&self, st: &Self::State, i: usize) -> Vec<u32>;
    /// Exponent bits still to run on slot `i`.
    fn remaining(&self, st: &Self::State, i: usize) -> usize;
}

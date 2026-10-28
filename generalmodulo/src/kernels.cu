// CUDA kernels for the M31 × M61 right-angle transform and the fused Montgomery step.
// Compiled at runtime with NVRTC. One thread block owns one batch slot; the transform
// vector lives in shared memory when it fits and in a per-slot global buffer otherwise.

typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned __int128 u128;

#define P1 0x7fffffffu
#define P2 0x1fffffffffffffffull

#ifndef CARRY_E
#define CARRY_E 16
#endif

// ---- GF(M31) and GF(M61) ----

__device__ __forceinline__ u32 add1(u32 a, u32 b) { u32 s = a + b; return s >= P1 ? s - P1 : s; }
__device__ __forceinline__ u32 sub1(u32 a, u32 b) { return a >= b ? a - b : a + P1 - b; }
__device__ __forceinline__ u32 mul1(u32 a, u32 b) {
    u64 t = (u64)a * b;
    u32 s = (u32)(t & P1) + (u32)(t >> 31);
    return s >= P1 ? s - P1 : s;
}
__device__ __forceinline__ u64 add2(u64 a, u64 b) { u64 s = a + b; return s >= P2 ? s - P2 : s; }
__device__ __forceinline__ u64 sub2(u64 a, u64 b) { return a >= b ? a - b : a + P2 - b; }
__device__ __forceinline__ u64 mul2(u64 a, u64 b) {
    u64 lo = a * b, hi = __umul64hi(a, b);
    u64 s = (lo & P2) + ((hi << 3) | (lo >> 61));
    return s >= P2 ? s - P2 : s;
}

// ---- GF(p²), i² = −1 ----

__device__ __forceinline__ uint2 cadd1(uint2 a, uint2 b) { return make_uint2(add1(a.x, b.x), add1(a.y, b.y)); }
__device__ __forceinline__ uint2 csub1(uint2 a, uint2 b) { return make_uint2(sub1(a.x, b.x), sub1(a.y, b.y)); }
__device__ __forceinline__ uint2 cmul1(uint2 a, uint2 b) {
    return make_uint2(sub1(mul1(a.x, b.x), mul1(a.y, b.y)), add1(mul1(a.x, b.y), mul1(a.y, b.x)));
}
__device__ __forceinline__ uint2 csqr1(uint2 a) {
    u32 m = mul1(a.x, a.y);
    return make_uint2(mul1(add1(a.x, a.y), sub1(a.x, a.y)), add1(m, m));
}
__device__ __forceinline__ uint2 cscale1(uint2 a, u32 c) { return make_uint2(mul1(a.x, c), mul1(a.y, c)); }

__device__ __forceinline__ ulonglong2 cadd2(ulonglong2 a, ulonglong2 b) { return make_ulonglong2(add2(a.x, b.x), add2(a.y, b.y)); }
__device__ __forceinline__ ulonglong2 csub2(ulonglong2 a, ulonglong2 b) { return make_ulonglong2(sub2(a.x, b.x), sub2(a.y, b.y)); }
__device__ __forceinline__ ulonglong2 cmul2(ulonglong2 a, ulonglong2 b) {
    // Karatsuba: three base multiplications.
    u64 ac = mul2(a.x, b.x), bd = mul2(a.y, b.y);
    u64 m = mul2(add2(a.x, a.y), add2(b.x, b.y));
    return make_ulonglong2(sub2(ac, bd), sub2(m, add2(ac, bd)));
}
__device__ __forceinline__ ulonglong2 csqr2(ulonglong2 a) {
    u64 m = mul2(a.x, a.y);
    return make_ulonglong2(mul2(add2(a.x, a.y), sub2(a.x, a.y)), add2(m, m));
}
__device__ __forceinline__ ulonglong2 cscale2(ulonglong2 a, u64 c) { return make_ulonglong2(mul2(a.x, c), mul2(a.y, c)); }

// ---- Per-length tables ----

struct Tab {
    const uint2* tw1; const uint2* itw1; const uint2* w1; const uint2* iw1;
    const ulonglong2* tw2; const ulonglong2* itw2; const ulonglong2* w2; const ulonglong2* iw2;
    int n, logn;
    u32 crt_inv;
};

// ---- Block-level transform pieces (a1, a2 may point to shared or global memory) ----

// Coefficients lim[0..cnt) (cnt ≤ n) as real parts, weighted by ζ^j.
__device__ void load_limbs(const u32* lim, int cnt, uint2* a1, ulonglong2* a2, const Tab& t) {
    for (int j = threadIdx.x; j < t.n; j += blockDim.x) {
        u32 v = j < cnt ? lim[j] : 0;
        u32 v1 = v >= P1 ? v - P1 : v;
        uint2 w1 = t.w1[j];
        ulonglong2 w2 = t.w2[j];
        a1[j] = make_uint2(mul1(w1.x, v1), mul1(w1.y, v1));
        a2[j] = make_ulonglong2(mul2(w2.x, v), mul2(w2.y, v));
    }
}

__device__ void dif(uint2* a1, ulonglong2* a2, const Tab& t) {
    int half = t.n >> 1;
    for (int s = 0; s < t.logn; s++) {
        __syncthreads();
        int lh = t.logn - 1 - s;  // log2 of h
        int h = 1 << lh;
        for (int b = threadIdx.x; b < half; b += blockDim.x) {
            int j = b & (h - 1);
            int i0 = ((b >> lh) << (lh + 1)) + j, i1 = i0 + h;
            int ti = j << s;
            uint2 u1 = a1[i0], v1 = a1[i1];
            a1[i0] = cadd1(u1, v1);
            a1[i1] = cmul1(csub1(u1, v1), t.tw1[ti]);
            ulonglong2 u2 = a2[i0], v2 = a2[i1];
            a2[i0] = cadd2(u2, v2);
            a2[i1] = cmul2(csub2(u2, v2), t.tw2[ti]);
        }
    }
    __syncthreads();
}

// Inverse transform and unweighting by ζ^−j / n.
__device__ void dit(uint2* a1, ulonglong2* a2, const Tab& t) {
    int half = t.n >> 1;
    for (int s = t.logn - 1; s >= 0; s--) {
        __syncthreads();
        int lh = t.logn - 1 - s;
        int h = 1 << lh;
        for (int b = threadIdx.x; b < half; b += blockDim.x) {
            int j = b & (h - 1);
            int i0 = ((b >> lh) << (lh + 1)) + j, i1 = i0 + h;
            int ti = j << s;
            uint2 u1 = a1[i0], v1 = cmul1(a1[i1], t.itw1[ti]);
            a1[i0] = cadd1(u1, v1);
            a1[i1] = csub1(u1, v1);
            ulonglong2 u2 = a2[i0], v2 = cmul2(a2[i1], t.itw2[ti]);
            a2[i0] = cadd2(u2, v2);
            a2[i1] = csub2(u2, v2);
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < t.n; j += blockDim.x) {
        a1[j] = cmul1(a1[j], t.iw1[j]);
        a2[j] = cmul2(a2[j], t.iw2[j]);
    }
    __syncthreads();
}

__device__ __forceinline__ u128 crt(u32 c1, u64 c2, u32 inv) {
    u32 c2m = (u32)(c2 & P1) + (u32)(c2 >> 31);
    c2m = c2m >= P1 ? c2m - P1 : c2m;
    u32 h = mul1(sub1(c1, c2m), inv);
    return (u128)c2 + (u128)P2 * h;
}

// Carry propagation over `count` coefficients coef(q) into 32-bit limbs out[0..count).
// `scratch` holds blockDim.x u64 and may alias the coefficient source only when
// count ≤ blockDim.x · CARRY_E (one chunk, every read precedes the first barrier).
template <typename Coef>
__device__ u64 block_carry(Coef coef, int count, u32* out, u64* scratch) {
    const int T = blockDim.x, tid = threadIdx.x;
    u64 chunk_in = 0;
    for (int base = 0; base < count; base += T * CARRY_E) {
        u32 l[CARRY_E];
        u64 c = 0;
        int q0 = base + tid * CARRY_E;
#pragma unroll
        for (int e = 0; e < CARRY_E; e++) {
            int q = q0 + e;
            u128 v = (q < count ? coef(q) : (u128)0) + c;
            l[e] = (u32)v;
            c = (u64)(v >> 32);
        }
        __syncthreads();
        scratch[tid] = c;
        __syncthreads();
        u64 cin = tid == 0 ? chunk_in : scratch[tid - 1];
        u64 top = c;  // carry out of the last thread, valid in thread T−1
        __syncthreads();
        for (;;) {
            u64 r = cin;
#pragma unroll
            for (int e = 0; e < CARRY_E; e++) {
                u64 v = (u64)l[e] + r;
                l[e] = (u32)v;
                r = v >> 32;
            }
            if (tid == T - 1) top += r;
            scratch[tid] = r;
            if (!__syncthreads_or(tid < T - 1 && r != 0)) break;
            cin = tid == 0 ? 0 : scratch[tid - 1];
            __syncthreads();
        }
#pragma unroll
        for (int e = 0; e < CARRY_E; e++) {
            int q = q0 + e;
            if (q < count) out[q] = l[e];
        }
        __syncthreads();
        if (tid == T - 1) scratch[0] = top;
        __syncthreads();
        chunk_in = scratch[0];
        __syncthreads();
    }
    return chunk_in;
}

// Carry of a product held in (a1, a2): coefficient q is the real part of element q for
// q < n and the imaginary part of element q − n otherwise.
__device__ void carry_product(const uint2* a1, const ulonglong2* a2, int count, u32* out, u64* scratch, const Tab& t) {
    int n = t.n;
    u32 inv = t.crt_inv;
    block_carry([&](int q) -> u128 {
        bool hi = q >= n;
        int j = hi ? q - n : q;
        uint2 e1 = a1[j];
        ulonglong2 e2 = a2[j];
        return crt(hi ? e1.y : e1.x, hi ? e2.y : e2.x, inv);
    }, count, out, scratch);
}

// ---- Fused modular step ----

struct Slots {
    u32* x;              // k limbs per slot (stride k)
    u32* limT;           // 2n limbs per slot
    u32* limS;           // 2n limbs per slot
    const uint2* nhat1; const ulonglong2* nhat2;    // n per slot
    const uint2* nphat1; const ulonglong2* nphat2;  // n per slot
    uint2* work1; ulonglong2* work2;                // n per slot (global mode only)
    const u32* exp;      // ew words per slot
    const u32* nbits;
    u32* pos;
    int k, ew;
    u32 base;
};

__device__ void pointwise_const(uint2* a1, ulonglong2* a2, const uint2* b1, const ulonglong2* b2, const Tab& t) {
    for (int j = threadIdx.x; j < t.n; j += blockDim.x) {
        a1[j] = cmul1(a1[j], b1[j]);
        a2[j] = cmul2(a2[j], b2[j]);
    }
}

// Steps 2–4 of the reduction: from T in limT, writes t = T · R⁻¹ (lazy) to x.
__device__ void redc(uint2* a1, ulonglong2* a2, u64* scratch, u32* x, u32* limT, u32* limS,
                     const uint2* nh1, const ulonglong2* nh2, const uint2* np1, const ulonglong2* np2,
                     int k, const Tab& t) {
    int nz = 0;
    for (int j = threadIdx.x; j < k; j += blockDim.x) nz |= limT[j] != 0;
    int flag = __syncthreads_or(nz);
    // m = (T mod R) · N′ mod R
    load_limbs(limT, k, a1, a2, t);
    dif(a1, a2, t);
    pointwise_const(a1, a2, np1, np2, t);
    dit(a1, a2, t);
    carry_product(a1, a2, k, limS, scratch, t);
    __syncthreads();
    // S = m · N
    load_limbs(limS, k, a1, a2, t);
    dif(a1, a2, t);
    pointwise_const(a1, a2, nh1, nh2, t);
    dit(a1, a2, t);
    carry_product(a1, a2, 2 * k, limS, scratch, t);
    __syncthreads();
    // t = ⌊T / R⌋ + ⌊S / R⌋ + [T mod R ≠ 0]
    block_carry([&](int q) -> u128 {
        return (u128)limT[k + q] + limS[k + q] + (q == 0 ? flag : 0);
    }, k, x, scratch);
    __syncthreads();
}

extern "C" __global__ void run_steps(Tab t, Slots s, int bits, int smem_mode) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int slot = blockIdx.x;
    const int n = t.n, k = s.k;
    uint2* a1;
    ulonglong2* a2;
    u64* scratch;
    if (smem_mode) {
        a2 = (ulonglong2*)smem;
        a1 = (uint2*)(smem + 16 * (size_t)n);
        scratch = (u64*)smem;  // aliases a2; safe because every carry here is one chunk
    } else {
        a1 = s.work1 + (size_t)slot * n;
        a2 = s.work2 + (size_t)slot * n;
        scratch = (u64*)smem;
    }
    u32* x = s.x + (size_t)slot * k;
    u32* limT = s.limT + (size_t)slot * 2 * n;
    u32* limS = s.limS + (size_t)slot * 2 * n;
    const uint2* nh1 = s.nhat1 + (size_t)slot * n;
    const ulonglong2* nh2 = s.nhat2 + (size_t)slot * n;
    const uint2* np1 = s.nphat1 + (size_t)slot * n;
    const ulonglong2* np2 = s.nphat2 + (size_t)slot * n;
    const u32* ex = s.exp + (size_t)slot * s.ew;
    u32 pos = s.pos[slot];
    const u32 nb = s.nbits[slot];

    for (int it = 0; it < bits && pos < nb; it++, pos++) {
        u32 bit = (ex[pos >> 5] >> (31 - (pos & 31))) & 1;
        u32 c = bit ? s.base : 1;
        // T = c · x²
        load_limbs(x, k, a1, a2, t);
        dif(a1, a2, t);
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            uint2 v1 = csqr1(a1[j]);
            ulonglong2 v2 = csqr2(a2[j]);
            if (c != 1) { v1 = cscale1(v1, c); v2 = cscale2(v2, c); }
            a1[j] = v1;
            a2[j] = v2;
        }
        dit(a1, a2, t);
        carry_product(a1, a2, 2 * k, limT, scratch, t);
        __syncthreads();
        redc(a1, a2, scratch, x, limT, limS, nh1, nh2, np1, np2, k, t);
    }
    if (threadIdx.x == 0) s.pos[slot] = pos;
}

// out ← a · b · R⁻¹ for one slot's constants; always uses the global work buffers of
// slot `out` and `fb1`, `fb2` (n elements) for F(b).
extern "C" __global__ void mul_slots(Tab t, Slots s, int out, int ia, int ib, uint2* fb1, ulonglong2* fb2) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int n = t.n, k = s.k;
    u64* scratch = (u64*)smem;
    uint2* a1 = s.work1 + (size_t)out * n;
    ulonglong2* a2 = s.work2 + (size_t)out * n;
    u32* limT = s.limT + (size_t)out * 2 * n;
    u32* limS = s.limS + (size_t)out * 2 * n;
    load_limbs(s.x + (size_t)ib * k, k, fb1, fb2, t);
    dif(fb1, fb2, t);
    load_limbs(s.x + (size_t)ia * k, k, a1, a2, t);
    dif(a1, a2, t);
    pointwise_const(a1, a2, fb1, fb2, t);
    dit(a1, a2, t);
    carry_product(a1, a2, 2 * k, limT, scratch, t);
    __syncthreads();
    redc(a1, a2, scratch, s.x + (size_t)out * k, limT, limS,
         s.nhat1 + (size_t)out * n, s.nhat2 + (size_t)out * n,
         s.nphat1 + (size_t)out * n, s.nphat2 + (size_t)out * n, k, t);
}

// ---- Unfused Transform-trait kernels, one block per vector, data in global memory ----

// Raw coefficient layout in: element j of vector v holds (c_j, c_{j+n}); weights then DIF.
extern "C" __global__ void t_forward(Tab t, uint2* d1, ulonglong2* d2) {
    uint2* a1 = d1 + (size_t)blockIdx.x * t.n;
    ulonglong2* a2 = d2 + (size_t)blockIdx.x * t.n;
    for (int j = threadIdx.x; j < t.n; j += blockDim.x) {
        a1[j] = cmul1(a1[j], t.w1[j]);
        a2[j] = cmul2(a2[j], t.w2[j]);
    }
    dif(a1, a2, t);
}

extern "C" __global__ void t_inverse(Tab t, uint2* d1, ulonglong2* d2) {
    dit(d1 + (size_t)blockIdx.x * t.n, d2 + (size_t)blockIdx.x * t.n, t);
}

extern "C" __global__ void t_pointwise(Tab t, uint2* o1, ulonglong2* o2, const uint2* a1, const ulonglong2* a2,
                                       const uint2* b1, const ulonglong2* b2, const u32* scale) {
    size_t off = (size_t)blockIdx.x * t.n;
    u32 c = scale ? scale[blockIdx.x] : 1;
    for (int j = threadIdx.x; j < t.n; j += blockDim.x) {
        uint2 v1 = cmul1(a1[off + j], b1[off + j]);
        ulonglong2 v2 = cmul2(a2[off + j], b2[off + j]);
        if (c != 1) { v1 = cscale1(v1, c); v2 = cscale2(v2, c); }
        o1[off + j] = v1;
        o2[off + j] = v2;
    }
}

// Forward transforms of per-slot constants (N, N′ limbs, stride k) into dst.
extern "C" __global__ void t_load_forward(Tab t, const u32* lim, int k, uint2* d1, ulonglong2* d2) {
    uint2* a1 = d1 + (size_t)blockIdx.x * t.n;
    ulonglong2* a2 = d2 + (size_t)blockIdx.x * t.n;
    load_limbs(lim + (size_t)blockIdx.x * k, k, a1, a2, t);
    dif(a1, a2, t);
}

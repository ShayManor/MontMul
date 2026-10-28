#![cfg(feature = "gpu")]

mod common;
use common::*;
use generalmodulo::{Backend, Cpu, Gpu, GpuOptions, ModBatch, Multi, Transform};
use rug::Integer;
use rug::rand::RandState;
use std::sync::{RwLock, RwLockReadGuard};

fn gpu() -> Gpu {
    Gpu::new().unwrap()
}

fn gpu_global() -> Gpu {
    Gpu::with_options(GpuOptions { force_global: true, ..Default::default() }).unwrap()
}

/// Tests that measure free device memory hold this exclusively; all others hold it shared.
static DEVICE_MEMORY: RwLock<()> = RwLock::new(());

fn shared() -> RwLockReadGuard<'static, ()> {
    DEVICE_MEMORY.read().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn transform_matches_cpu_bit_for_bit() {
    let _g = shared();
    let (g, c) = (gpu(), Cpu::default());
    let mut rng = rug::rand::RandState::new();
    for e in 3..=20 {
        let len = 1usize << e;
        let count = 3;
        let coeffs: Vec<u32> = (0..len * count).map(|i| if i % len < len / 2 { rng.bits(32) } else { 0 }).collect();
        let (mut gx, mut cx) = (g.upload(&coeffs, len, count), c.upload(&coeffs, len, count));
        g.forward(&mut gx, len, count);
        c.forward(&mut cx, len, count);
        assert_eq!(g.download_raw(&gx), c.download_raw(&cx), "forward, len {len}");
        g.inverse(&mut gx, len, count);
        c.inverse(&mut cx, len, count);
        assert_eq!(g.download(&gx), c.download(&cx), "round trip, len {len}");
    }
}

#[test]
fn exactness_bound() {
    let _g = shared();
    exactness(&gpu(), 1 << 18);
}

#[test]
fn transform_count_and_scales() {
    let _g = shared();
    transform_batched(&gpu(), 1 << 16);
}

#[test]
fn pointwise_scales_above_m31_match_cpu() {
    let _g = shared();
    let (g, c) = (gpu(), Cpu::default());
    let mut rng = RandState::new();
    let (len, count) = (64, 4);
    let scales = [0x7fff_ffff, 0x8000_0004, 0xffff_fffe, u32::MAX];
    let coeffs: Vec<u32> = (0..len * count).map(|_| rng.bits(32)).collect();
    let (mut gx, mut cx) = (g.upload(&coeffs, len, count), c.upload(&coeffs, len, count));
    g.forward(&mut gx, len, count);
    c.forward(&mut cx, len, count);
    let (mut go, mut co) = (g.upload(&vec![0; len * count], len, count), c.upload(&vec![0; len * count], len, count));
    g.pointwise(&mut go, &gx, &gx, Some(&g.upload_scales(&scales)), len, count);
    c.pointwise(&mut co, &cx, &cx, Some(&c.upload_scales(&scales)), len, count);
    assert_eq!(g.download_raw(&go), c.download_raw(&co));
}

#[test]
fn load_forward_matches_upload_forward() {
    let _g = shared();
    let (g, c) = (gpu(), Cpu::default());
    let mut rng = rug::rand::RandState::new();
    for e in [3, 8, 13] {
        let (len, count) = (1usize << e, 5);
        let k = len / 2 - 1;
        let limbs: Vec<u32> = (0..k * count).map(|_| rng.bits(32)).collect();
        let mut padded = vec![0u32; len * count];
        for v in 0..count {
            padded[v * len..v * len + k].copy_from_slice(&limbs[v * k..(v + 1) * k]);
        }
        let mut cx = c.upload(&padded, len, count);
        c.forward(&mut cx, len, count);
        assert_eq!(g.download_raw(&g.load_forward(&limbs, k, len, count)), c.download_raw(&cx), "len {len}");
    }
}

#[test]
fn random_modmul_sizes() {
    let _g = shared();
    for (bits, cases) in [(64, 300), (1000, 200), (10_000, 50), (100_000, 8), (1 << 20, 2), (1 << 22, 1)] {
        for base in [3, 4] {
            random_modmul(gpu, bits, cases, bits, base);
        }
    }
}

#[test]
fn edge_values() {
    let _g = shared();
    for base in [3, 4] {
        edges(gpu, base);
    }
}

#[test]
fn independence() {
    let _g = shared();
    batch_independence(gpu, 3000, &[1, 7, 64, 1024]);
}

#[test]
fn powmod_mul() {
    let _g = shared();
    powmod_and_mul(gpu, 2000, 9);
    // Shared mode at n = 4096: `mul_slots` at 1,024 threads.
    powmod_and_mul(gpu, 87_914, 2);
}

#[test]
fn api() {
    let _g = shared();
    api_contract(gpu);
}

#[test]
fn carries() {
    let _g = shared();
    carry_stress(gpu, 1 << 16);
}

#[test]
fn fermat_tribonacci() {
    let _g = shared();
    end_to_end(gpu, &[6, 7, 11, 87, 98, 215, 802, 1001, 2000, 4000]);
}

#[test]
fn tribonacci_prps() {
    let _g = shared();
    known_prps(gpu, &[4202, 18699], &[4203, 18700]);
}

#[test]
#[ignore = "T(96879), 85k bits"]
fn tribonacci_prp_96879() {
    let _g = shared();
    known_prps(gpu, &[96879], &[]);
}

#[test]
fn global_mode_matches_shared_mode() {
    let _g = shared();
    // 87,914 bits: L = 8192, which is shared mode on the 3060.
    let modes = |b: &mut ModBatch<Gpu>| {
        let (g, st) = b.backend_and_state();
        g.plan_of(st).mode
    };
    let n = vec![rand_odd(&mut RandState::new(), 87_914)];
    assert_eq!(modes(&mut ModBatch::new(gpu(), &n, 3).unwrap()), generalmodulo::gpu::plan::Mode::Shared);
    assert_eq!(modes(&mut ModBatch::new(gpu_global(), &n, 3).unwrap()), generalmodulo::gpu::plan::Mode::Global);
    assert_same_raw(gpu, gpu_global, 87_914, 16, 300);
}

#[test]
fn gpu_matches_cpu_raw() {
    let _g = shared();
    // The default GPU backend runs the wraparound path (k = L/2); `gpu_nowrap` the plain one.
    for bits in [100, 5000, 87_914] {
        assert_same_raw(gpu, || Cpu::with_wraparound(true), bits, 8, 50);
        assert_same_raw(gpu_nowrap, Cpu::default, bits, 8, 50);
    }
}

#[test]
fn long_run_split_into_launches() {
    let _g = shared();
    // 32 slots of 87,914 bits, 200,000 bits of exponent, against pow_mod.
    long_run(gpu, 87_914, 32, 200_000);
}

#[test]
fn oversized_batch_is_error() {
    let _g = shared();
    // L = 2^18 with k just above L / 4 keeps the host cost near 2 MB per slot (N and its limb
    // strings) against at least 88n = 11.5 MB of device memory, global mode; the slot count is
    // sized from free memory, about 5 GB of host RAM on a 32 GB card.
    let n = (Integer::from(1) << 2_097_152u32) + 1u32;
    let slots = gpu().free_memory() / (88 << 17) * 11 / 10 + 1;
    let err = ModBatch::new(gpu(), &vec![n; slots], 3).err().expect("batch must not fit");
    assert!(err.to_string().contains("bytes"), "{err}");
}

/// Creating, running and dropping `count` batches leaves free device memory unchanged.
fn assert_no_leak(make: impl Fn() -> Gpu, count: usize) {
    let _g = DEVICE_MEMORY.write().unwrap_or_else(|e| e.into_inner());
    let mut rng = RandState::new();
    let moduli: Vec<Integer> = (0..64).map(|_| rand_odd(&mut rng, 50_000)).collect();
    let once = || {
        let mut b = ModBatch::new(make(), &moduli, 3).unwrap();
        for i in 0..moduli.len() {
            b.load(i, &Integer::from(1));
            b.set_exponent(i, &Integer::from(12345));
        }
        b.run(5);
        b.mul(0, 0, 0);
    };
    let probe = make();
    once();
    let before = probe.free_memory();
    for _ in 0..count {
        once();
    }
    let after = probe.free_memory();
    assert!(after + (16 << 20) >= before, "free device memory {before} -> {after}");
}

#[test]
fn no_leak_on_drop() {
    assert_no_leak(gpu, 50);
}

/// Two live batches with different L, run alternately, both match pow_mod.
fn two_batches(make: impl Fn() -> Gpu) {
    let mut rng = RandState::new();
    let mut batches = Vec::new();
    for bits in [3000, 50_000] {
        let moduli: Vec<Integer> = (0..4).map(|_| rand_odd(&mut rng, bits)).collect();
        let exps: Vec<Integer> = (0..4).map(|_| Integer::from(Integer::random_bits(300, &mut rng)) | 1u32).collect();
        let mut b = ModBatch::new(make(), &moduli, 3).unwrap();
        for (i, e) in exps.iter().enumerate() {
            b.load(i, &Integer::from(1));
            b.set_exponent(i, e);
        }
        batches.push((b, moduli, exps));
    }
    assert_ne!(batches[0].0.params().len, batches[1].0.params().len);
    while batches.iter().any(|(b, ..)| (0..4).any(|i| b.remaining(i) > 0)) {
        for (b, ..) in batches.iter_mut() {
            b.run(7);
        }
    }
    for (b, moduli, exps) in &batches {
        for i in 0..4 {
            assert_eq!(b.read(i), Integer::from(3).pow_mod(&exps[i], &moduli[i]).unwrap(), "slot {i}");
        }
    }
}

#[test]
fn two_batches_interleaved() {
    let _g = shared();
    two_batches(gpu);
}

#[test]
fn structured_modulus_carries() {
    let _g = shared();
    // N = 2^b − 1 forces long carry ripples (15 to 63 rounds at k = 256 to 1024 in the review's simulation).
    for len in [512usize, 2048, 8192, 65_536] {
        carry_stress(gpu, len);
    }
}

/// x = R − 1 lies outside [0, 2N), so T = 4 · (R − 1)² carries out of 2k limbs; `err` must
/// name slot 1.
fn carry_overflow<B: Backend>(make: impl Fn() -> B, bits: u32) {
    let mut rng = RandState::new();
    let moduli = vec![rand_odd(&mut rng, bits), rand_odd(&mut rng, bits)];
    let mut b = ModBatch::new(make(), &moduli, 4).unwrap();
    let k = b.params().k;
    for i in 0..2 {
        b.load(i, &Integer::from(1));
        b.set_exponent(i, &Integer::from(1));
    }
    let (g, st) = b.backend_and_state();
    g.load(st, 1, &vec![u32::MAX; k]);
    b.run(1);
}

#[test]
#[should_panic(expected = "slot 1")]
fn carry_overflow_panics_naming_the_slot() {
    let _g = shared();
    carry_overflow(gpu, 1000);
}

#[test]
#[should_panic(expected = "slot 1")]
fn carry_overflow_across_chunks() {
    let _g = shared();
    // 87,914 bits: 2k = 8,192 coefficients, four carry chunks at 1,024 threads.
    carry_overflow(gpu, 87_914);
}

#[test]
#[should_panic(expected = "slot 1")]
fn carry_overflow_global_mode() {
    let _g = shared();
    carry_overflow(gpu_global, 87_914);
}

fn gpu_nowrap() -> Gpu {
    Gpu::with_options(GpuOptions { wraparound: false, ..Default::default() }).unwrap()
}

fn gpu_nowrap_global() -> Gpu {
    Gpu::with_options(GpuOptions { wraparound: false, force_global: true, ..Default::default() }).unwrap()
}

#[test]
fn wraparound_matches_cpu_wraparound() {
    let _g = shared();
    assert!(gpu().options().wraparound);
    // Same k = L/2 on both sides, so the raw Montgomery limbs must agree exactly.
    for bits in [64, 200, 400, 1000, 87_914, 100_000] {
        assert_same_raw(gpu, || Cpu::with_wraparound(true), bits, 8, 100);
    }
    assert_same_raw(gpu_global, || Cpu::with_wraparound(true), 87_914, 8, 50);
    assert_same_raw(gpu_global, || Cpu::with_wraparound(true), 200, 4, 50);
    // Global mode with transforms split into global stages and shared-memory tiles: L = 16,384
    // (an odd number of tiled DIT stages) and L = 32,768.
    for bits in [200_000, 400_000] {
        assert_same_raw(gpu_global, || Cpu::with_wraparound(true), bits, 2, 20);
    }
}

#[test]
fn without_wraparound() {
    let _g = shared();
    assert_same_raw(gpu_nowrap, Cpu::default, 100_000, 8, 100);
    assert_same_raw(gpu_nowrap_global, Cpu::default, 87_914, 8, 50);
    // Naturally global and swizzled: n = 16,384, tiles from stage 4.
    assert_same_raw(gpu_nowrap, Cpu::default, 400_000, 2, 20);
    powmod_and_mul(gpu_nowrap, 2000, 9);
    random_modmul(gpu_nowrap, 1000, 100, 1000, 3);
    random_modmul(gpu_nowrap, 100_000, 8, 100_000, 4);
    edges(gpu_nowrap, 3);
    edges(gpu_nowrap, 4);
    carry_stress(gpu_nowrap, 1 << 16);
}

#[test]
fn multi_two_contexts_on_one_device() {
    let _g = shared();
    let two = || Multi::new(vec![gpu(), gpu()]);
    for slots in [1, 5, 16] {
        assert_same_raw(two, || Cpu::with_wraparound(true), 87_914, slots, 50);
    }
    assert_same_raw(two, || Cpu::with_wraparound(true), 1000, 7, 100);
    powmod_and_mul(two, 2000, 9);
    end_to_end(two, &[6, 7, 11, 87, 98, 215, 802, 1001, 2000, 4000]);
}

#[test]
fn multi_all_devices() {
    let _g = shared();
    let all = || Multi::new(Gpu::all().unwrap());
    assert_eq!(all().parts().len(), Gpu::device_count().unwrap());
    assert_same_raw(all, || Cpu::with_wraparound(true), 87_914, 13, 50);
    long_run(all, 87_914, 32, 2_000);
}

#[test]
#[should_panic(expected = "shard 1, slot 1")]
fn multi_carry_overflow_names_the_global_slot() {
    let _g = shared();
    // Slot 1 is slot 0 of shard 1.
    carry_overflow(|| Multi::new(vec![gpu(), gpu()]), 1000);
}

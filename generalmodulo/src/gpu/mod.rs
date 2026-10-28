//! CUDA backend: the M31 × M61 transform and the fused modular step in cuda-oxide kernels
//! (`kernels/`), one thread block per vector or slot. For `Transform`, packing and CRT run on
//! the host with the CPU backend's code, so only the transforms differ, and those match the CPU
//! bit for bit. `Backend` runs whole steps on the device (`run_steps`), split into launches of
//! about `GpuOptions::max_launch_ms`.

pub mod plan;

#[cfg(feature = "gpu")]
mod device;
#[cfg(feature = "gpu")]
pub use device::DevBuf;

#[cfg(feature = "gpu")]
use {
    crate::cpu::{Cpu, CpuBuf, Half, half_factors_of},
    crate::field::{F1, F2, Q, Tables},
    crate::transform::{Backend, BatchSetup, Limits, Transform},
    anyhow::ensure,
    device::{Device, DeviceHalf, DeviceTables, SlotBufs},
    plan::{LaunchPlan, Mode},
    rayon::prelude::*,
    std::collections::HashMap,
    std::sync::{Arc, Mutex},
    std::time::Instant,
};

#[derive(Clone, Copy, Debug)]
pub struct GpuOptions {
    /// CUDA device ordinal.
    pub device: usize,
    /// Keep transform vectors in global memory even when shared memory would fit.
    pub force_global: bool,
    /// Target duration of one kernel launch (the 3060 also drives a display).
    pub max_launch_ms: u32,
    /// Step 3 as a length-L/2 cyclic product with the wraparound correction (spec section 9);
    /// implies k = L/2. Matches `Cpu::with_wraparound(true)` bit for bit. On by default.
    pub wraparound: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        GpuOptions { device: 0, force_global: false, max_launch_ms: 200, wraparound: true }
    }
}

#[cfg(feature = "gpu")]
pub struct Gpu {
    dev: Device,
    opts: GpuOptions,
    max_dyn_smem: u32,
    lengths: Vec<usize>,
    tables: Mutex<HashMap<usize, Arc<DeviceTables>>>,
}

#[cfg(feature = "gpu")]
impl Gpu {
    pub fn new() -> anyhow::Result<Gpu> {
        Self::with_options(GpuOptions::default())
    }

    pub fn with_options(opts: GpuOptions) -> anyhow::Result<Gpu> {
        let dev = Device::new(opts.device)?;
        let max_dyn_smem = dev.max_dyn_smem();
        let lengths = (3..=24).map(|e| 1usize << e).collect();
        Ok(Gpu { dev, opts, max_dyn_smem, lengths, tables: Mutex::new(HashMap::new()) })
    }

    pub fn options(&self) -> GpuOptions {
        self.opts
    }

    pub fn device_name(&self) -> String {
        self.dev.name()
    }

    /// Opt-in maximum of dynamic shared memory per block.
    pub fn max_dyn_smem(&self) -> u32 {
        self.max_dyn_smem
    }

    pub fn sm_count(&self) -> usize {
        self.dev.sm_count()
    }

    /// Free device memory in bytes.
    pub fn free_memory(&self) -> usize {
        self.dev.free_memory()
    }

    /// Launch plan chosen for the batch at `create`.
    pub fn plan_of(&self, st: &GpuState) -> LaunchPlan {
        st.plan
    }

    /// Like `run`, but each step is step 1 of spec section 3 alone (the squaring, spec section
    /// 11.1). Exponent bits are consumed as by `run`; x is left unchanged, so a later `run`
    /// continues from the old x at the advanced position unless the slots are re-armed.
    pub fn run_bare(&self, st: &mut GpuState, bits: usize) {
        self.steps(st, bits, true);
    }

    /// Runs `bits` steps in launches of about `max_launch_ms`. A round of `s` steps covers all
    /// slots in one or more slot ranges; `s` and the range sizes follow from the time per active
    /// slot-step of the latest launch, kept in the state across calls. Without a measurement, the
    /// first launch is one step of at most two slots per SM. One step of one slot is never split,
    /// so above about L = 2^21 a single launch exceeds 200 ms until a multi-block transform
    /// (spec section 7) exists.
    fn steps(&self, st: &mut GpuState, bits: usize, bare: bool) {
        let slots = st.nbits.len();
        let active = |st: &GpuState, vs: std::ops::Range<usize>| vs.filter(|&i| st.pos[i] < st.nbits[i]).count();
        let most = (0..slots).map(|i| (st.nbits[i] - st.pos[i]) as usize).max().unwrap_or(0);
        let mut left = bits.min(most);
        if left == 0 {
            return;
        }
        self.push(st);
        let (p, budget) = (st.plan, self.opts.max_launch_ms as f64);
        while left > 0 {
            let s = match st.ms_per_slot_step[bare as usize] {
                Some(e) => ((budget / (e * active(st, 0..slots).max(1) as f64)) as usize).clamp(1, left.min(u32::MAX as usize)),
                None => 1,
            };
            let mut first = 0;
            while first < slots {
                let r = match st.ms_per_slot_step[bare as usize] {
                    Some(e) => ((budget / (e * s as f64)) as usize).max(1),
                    None => 2 * self.dev.sm_count(),
                };
                let end = (first + r).min(slots);
                let busy = active(st, first..end).max(1);
                let t0 = Instant::now();
                let shared = p.mode == Mode::Shared;
                self.dev.run_steps(&st.tab, st.half.as_ref(), &mut st.bufs, first..end, st.base, s as u32, p.threads, p.smem_bytes, shared, bare);
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                self.check_err(st, "run");
                st.ms_per_slot_step[bare as usize] = Some(ms.max(1e-3) / (busy * s) as f64);
                for i in first..end {
                    st.pos[i] += (s as u32).min(st.nbits[i] - st.pos[i]);
                }
                first = end;
            }
            left -= s;
        }
        if !bare {
            *st.host_x.get_mut().unwrap() = None;
        }
    }

    /// Uploads host-side changes (loaded values, exponents) before a launch.
    fn push(&self, st: &mut GpuState) {
        if st.device_x_stale {
            let x = st.host_x.get_mut().unwrap();
            self.dev.write_x(&mut st.bufs, x.as_ref().unwrap());
            st.device_x_stale = false;
        }
        if st.exp_dirty {
            let ew = st.exps.iter().map(Vec::len).max().unwrap_or(0);
            let mut words = vec![0u32; st.exps.len() * ew];
            for (i, e) in st.exps.iter().enumerate() {
                words[i * ew..i * ew + e.len()].copy_from_slice(e);
            }
            self.dev.set_exponents(&mut st.bufs, &words, ew, &st.nbits, &st.pos);
            st.exp_dirty = false;
        }
    }

    /// Panics naming the first slot whose error word is set, after clearing the error words.
    fn check_err(&self, st: &mut GpuState, what: &str) {
        if let Some(i) = self.dev.read_err(&st.bufs).iter().position(|&e| e != 0) {
            self.dev.clear_err(&mut st.bufs);
            panic!("{what}: carry out of a product that must fit, slot {i}");
        }
    }

    /// Device tables for transform length `len`, uploaded on first use.
    fn tables(&self, len: usize) -> Arc<DeviceTables> {
        let mut cache = self.tables.lock().unwrap();
        cache.entry(len).or_insert_with(|| Arc::new(self.dev.upload_tables(&Tables::new(len / 2)))).clone()
    }

    /// Threads per block of the unfused transform kernels, which work in global memory.
    fn threads(&self, len: usize) -> u32 {
        plan::plan(len / 2, self.max_dyn_smem, true).threads
    }

    /// Calls `launch` on consecutive ranges covering vectors 0..count, sized so that one launch
    /// takes about `max_launch_ms`: the first range is two vectors per SM, each later one as many
    /// as the previous launch's time per vector allows. One vector is never split.
    fn chunked(&self, count: usize, mut launch: impl FnMut(std::ops::Range<usize>)) {
        let (mut first, mut chunk) = (0, 2 * self.dev.sm_count());
        while first < count {
            let end = (first + chunk).min(count);
            let t0 = Instant::now();
            launch(first..end);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            chunk = (((end - first) as f64 * self.opts.max_launch_ms as f64 / ms.max(1e-3)) as usize).max(1);
            first = end;
        }
    }

    /// Transform-domain data without CRT, as `Cpu::download_raw`.
    pub fn download_raw(&self, d: &GpuBuf) -> (Vec<F1>, Vec<F2>) {
        (self.dev.download(&d.f1), self.dev.download(&d.f2))
    }

    /// Weights and forward-transforms `count` strings of `k` limbs each (k ≤ len / 2, row-major).
    pub fn load_forward(&self, limbs: &[u32], k: usize, len: usize, count: usize) -> GpuBuf {
        let n = len / 2;
        let mut d = GpuBuf { len, count, f1: self.dev.zeros(n * count), f2: self.dev.zeros(n * count) };
        let lim = self.dev.upload(&limbs[..k * count]);
        let (t, threads) = (self.tables(len), self.threads(len));
        self.chunked(count, |vs| self.dev.t_load_forward(&t, &lim, k, (&mut d.f1, &mut d.f2), vs, threads));
        d
    }
}

/// `count` transform vectors of length `len` in device memory, layout as `CpuBuf`.
#[cfg(feature = "gpu")]
pub struct GpuBuf {
    pub len: usize,
    pub count: usize,
    f1: DevBuf<F1>,
    f2: DevBuf<F2>,
}

/// One batch on the device, plus host copies of what the host writes or reads.
#[cfg(feature = "gpu")]
pub struct GpuState {
    tab: Arc<DeviceTables>,
    /// Tables of the wraparound path, when on.
    half: Option<DeviceHalf>,
    plan: LaunchPlan,
    k: usize,
    base: u32,
    bufs: SlotBufs,
    /// x of every slot (k limbs, row-major) while the host copy is current.
    host_x: Mutex<Option<Vec<u32>>>,
    /// The host copy of x is newer than the device copy.
    device_x_stale: bool,
    exps: Vec<Vec<u32>>,
    nbits: Vec<u32>,
    /// Next exponent bit per slot; mirrors the device copy, which the kernels advance identically.
    pos: Vec<u32>,
    exp_dirty: bool,
    /// Milliseconds per active slot-step of the latest launch, for `run` and `run_bare`.
    ms_per_slot_step: [Option<f64>; 2],
}

#[cfg(feature = "gpu")]
impl GpuState {
    /// The host copy of x, downloaded if the device copy is newer.
    fn host_x<'a>(&'a self, dev: &Device) -> std::sync::MutexGuard<'a, Option<Vec<u32>>> {
        let mut x = self.host_x.lock().unwrap();
        if x.is_none() {
            *x = Some(dev.read_x(&self.bufs));
        }
        x
    }
}

#[cfg(feature = "gpu")]
impl Limits for Gpu {
    fn coefficient_bound(&self) -> u128 {
        Q
    }
    fn supported_lengths(&self) -> &[usize] {
        &self.lengths
    }
    fn full_k(&self) -> bool {
        self.opts.wraparound
    }
}

#[cfg(feature = "gpu")]
impl Transform for Gpu {
    type Buf = GpuBuf;
    type Scales = DevBuf<u32>;

    fn upload(&self, coeffs: &[u32], len: usize, count: usize) -> GpuBuf {
        let b = Cpu::default().upload(coeffs, len, count);
        GpuBuf { len, count, f1: self.dev.upload(&b.f1), f2: self.dev.upload(&b.f2) }
    }
    fn download(&self, d: &GpuBuf) -> Vec<u128> {
        let (f1, f2) = self.download_raw(d);
        Cpu::default().download(&CpuBuf { len: d.len, count: d.count, f1, f2 })
    }
    fn upload_scales(&self, scales: &[u32]) -> DevBuf<u32> {
        self.dev.upload(scales)
    }
    fn forward(&self, d: &mut GpuBuf, len: usize, count: usize) {
        assert_eq!(d.len, len);
        let (t, threads) = (self.tables(len), self.threads(len));
        self.chunked(count, |vs| self.dev.t_forward(&t, &mut d.f1, &mut d.f2, vs, threads));
    }
    fn inverse(&self, d: &mut GpuBuf, len: usize, count: usize) {
        assert_eq!(d.len, len);
        let (t, threads) = (self.tables(len), self.threads(len));
        self.chunked(count, |vs| self.dev.t_inverse(&t, &mut d.f1, &mut d.f2, vs, threads));
    }
    fn pointwise(&self, out: &mut GpuBuf, a: &GpuBuf, b: &GpuBuf, scale: Option<&DevBuf<u32>>, len: usize, count: usize) {
        assert!(out.len == len && a.len == len && b.len == len);
        let t = self.tables(len);
        let threads = self.threads(len);
        self.chunked(count, |vs| {
            self.dev.t_pointwise(&t, (&mut out.f1, &mut out.f2), (&a.f1, &a.f2), (&b.f1, &b.f2), scale, vs, threads)
        });
    }
}

#[cfg(feature = "gpu")]
impl Backend for Gpu {
    type State = GpuState;

    fn create(&self, st: &BatchSetup) -> anyhow::Result<GpuState> {
        let (k, n, slots) = (st.k, st.len / 2, st.n.len());
        let plan = plan::plan(n, self.max_dyn_smem, self.opts.force_global);
        let global = plan.mode == Mode::Global;
        // Slot state, tables (3n elements per field), the staged N, N′ limbs, exponents of up to
        // k words per slot (N − 1 for a Fermat test), and a margin for allocation granularity.
        let need = SlotBufs::bytes(slots, k, n, global) + 72 * n + 8 * slots * k + 4 * slots * k + (64 << 20);
        let free = self.dev.free_memory();
        ensure!(need <= free, "batch needs {need} bytes, {free} free");
        let tab = self.tables(st.len);
        let mut bufs = self.dev.alloc_slots(slots, k, n, global);
        let (nl, npl) = (self.dev.upload(&st.n.concat()), self.dev.upload(&st.nprime.concat()));
        let threads = self.threads(st.len);
        let with_nhat = !self.opts.wraparound;
        self.chunked(slots, |vs| self.dev.set_constants(&tab, &mut bufs, &nl, &npl, vs, threads, with_nhat));
        let half = self.opts.wraparound.then(|| {
            assert!(k == n, "wraparound needs k = L/2");
            // The half path's pointwise factors come from the CPU code, n elements per slot.
            let hf = Half::new(Arc::new(Tables::new(n / 2)));
            let (mut f1, mut f2) = (vec![F1::default(); slots * n], vec![F2::default(); slots * n]);
            let rows = f1.par_chunks_mut(n).zip(f2.par_chunks_mut(n)).zip(st.n.par_iter());
            rows.for_each(|((r1, r2), limbs)| {
                let (a1, a2) = half_factors_of(limbs, &hf);
                r1.copy_from_slice(&a1);
                r2.copy_from_slice(&a2);
            });
            self.dev.write_half_nhat(&mut bufs, &f1, &f2);
            self.dev.upload_half(&hf)
        });
        Ok(GpuState {
            tab,
            half,
            plan,
            k,
            base: st.base,
            bufs,
            host_x: Mutex::new(Some(vec![0; slots * k])),
            device_x_stale: false,
            exps: vec![vec![]; slots],
            nbits: vec![0; slots],
            pos: vec![0; slots],
            exp_dirty: false,
            ms_per_slot_step: [None; 2],
        })
    }
    fn load(&self, st: &mut GpuState, i: usize, limbs: &[u32]) {
        let k = st.k;
        st.host_x(&self.dev).as_mut().unwrap()[i * k..(i + 1) * k].copy_from_slice(limbs);
        st.device_x_stale = true;
    }
    fn set_exponent(&self, st: &mut GpuState, i: usize, words: &[u32], nbits: usize) {
        assert!(nbits <= 32 * words.len() && nbits <= u32::MAX as usize);
        st.exps[i] = words.to_vec();
        st.nbits[i] = nbits as u32;
        st.pos[i] = 0;
        st.exp_dirty = true;
    }
    fn run(&self, st: &mut GpuState, bits: usize) {
        self.steps(st, bits, false);
    }
    fn mul(&self, st: &mut GpuState, out: usize, a: usize, b: usize) {
        self.push(st);
        self.dev.mul_slots(&st.tab, st.half.as_ref(), &mut st.bufs, out, a, b, st.plan.threads);
        self.check_err(st, "mul");
        *st.host_x.get_mut().unwrap() = None;
    }
    fn read(&self, st: &GpuState, i: usize) -> Vec<u32> {
        st.host_x(&self.dev).as_ref().unwrap()[i * st.k..(i + 1) * st.k].to_vec()
    }
    fn remaining(&self, st: &GpuState, i: usize) -> usize {
        (st.nbits[i] - st.pos[i]) as usize
    }
}

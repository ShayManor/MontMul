//! Thin wrapper over the CUDA driver (cudarc): context, stream, the kernel module, device
//! buffers and one typed launch method per kernel. Every unsafe driver call of the GPU backend
//! lives here. Errors are `Err` while opening the device and panics carrying the driver
//! message afterwards; every launch synchronizes, so a kernel fault panics at its launch.

use crate::cpu::Half;
use crate::field::{Elem, F1, F2, Tables, crt_inv};
use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchArgs, LaunchConfig,
    PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

/// PTX of the `generalmodulo-kernels` crate, compiled by `build.rs`.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/generalmodulo_kernels.ptx"));
const KERNELS: [&str; 9] =
    ["t_forward", "t_inverse", "t_pointwise", "t_load_forward", "run_steps", "run_steps_global", "run_bare", "run_bare_global", "mul_slots"];
/// Dynamic shared memory a launch may use without the per-function opt-in.
const SMEM_DEFAULT: u32 = 48 * 1024;

// SAFETY: plain `repr(C)` integers; all-zero bits are the field element 0.
unsafe impl DeviceRepr for F1 {}
unsafe impl ValidAsZeroBits for F1 {}
unsafe impl DeviceRepr for F2 {}
unsafe impl ValidAsZeroBits for F2 {}

/// Device memory holding `len()` values of `T`.
pub struct DevBuf<T>(CudaSlice<T>);

impl<T> DevBuf<T> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Kernel parameter with the layout of `Tab` in `kernels/src/lib.rs`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Tab {
    tw1: u64,
    itw1: u64,
    w1: u64,
    iw1: u64,
    tw2: u64,
    itw2: u64,
    w2: u64,
    iw2: u64,
    n: u32,
    logn: u32,
    crt_inv: u32,
}

// SAFETY: `repr(C)` integers, passed by value.
unsafe impl DeviceRepr for Tab {}

// The kernel crate's layouts (`.param .align 8 .b8 [80]` for `Tab` in the PTX).
const _: () = assert!(size_of::<F1>() == 8 && align_of::<F1>() == 8);
const _: () = assert!(size_of::<F2>() == 16 && align_of::<F2>() == 16);
const _: () = assert!(size_of::<Tab>() == 80 && align_of::<Tab>() == 8);

/// Kernel parameter with the layout of `HalfTab` in `kernels/src/lib.rs`; all zero when the
/// wraparound path is off.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HalfTab {
    tab: Tab,
}

// SAFETY: `repr(C)` integers, passed by value.
unsafe impl DeviceRepr for HalfTab {}

const _: () = assert!(size_of::<HalfTab>() == 80 && align_of::<HalfTab>() == 8);

/// Tables of the wraparound path (`cpu::Half`) in device memory.
pub struct DeviceHalf {
    arg: HalfTab,
    _tab: DeviceTables,
}

/// `field::Tables` for one length in device memory.
pub struct DeviceTables {
    tab: Tab,
    _f1: [DevBuf<F1>; 4],
    _f2: [DevBuf<F2>; 4],
}

impl DeviceTables {
    /// Complex length n = L / 2.
    pub fn n(&self) -> usize {
        self.tab.n as usize
    }
}

/// Kernel parameter with the layout of `Slots` in `kernels/src/lib.rs`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Slots {
    x: u64,
    lim_t: u64,
    lim_s: u64,
    nhat1: u64,
    nhat2: u64,
    nphat1: u64,
    nphat2: u64,
    work1: u64,
    work2: u64,
    exp: u64,
    nbits: u64,
    pos: u64,
    err: u64,
    k: u32,
    ew: u32,
    base: u32,
}

// SAFETY: `repr(C)` integers, passed by value.
unsafe impl DeviceRepr for Slots {}

const _: () = assert!(size_of::<Slots>() == 120 && align_of::<Slots>() == 8);

/// Per-slot state of one batch (spec section 6), structure of arrays over slots, plus the two
/// vectors `mul_slots` works in. Sizes are fixed at allocation, which keeps the launch methods
/// sound.
pub struct SlotBufs {
    slots: usize,
    k: usize,
    n: usize,
    /// Per-slot work vectors exist (global mode).
    global: bool,
    x: DevBuf<u32>,
    lim_t: DevBuf<u32>,
    lim_s: DevBuf<u32>,
    nhat: (DevBuf<F1>, DevBuf<F2>),
    nphat: (DevBuf<F1>, DevBuf<F2>),
    work: (DevBuf<F1>, DevBuf<F2>),
    mul_a: (DevBuf<F1>, DevBuf<F2>),
    mul_b: (DevBuf<F1>, DevBuf<F2>),
    exp: DevBuf<u32>,
    ew: usize,
    nbits: DevBuf<u32>,
    pos: DevBuf<u32>,
    err: DevBuf<u32>,
}

impl SlotBufs {
    /// Device bytes that `Device::alloc_slots` takes for these sizes, before exponents.
    pub fn bytes(slots: usize, k: usize, n: usize, global: bool) -> usize {
        let per_slot = 4 * k + 2 * 8 * n + 2 * 24 * n + if global { 24 * n } else { 0 } + 3 * 4;
        slots * per_slot + 2 * 24 * n
    }
}

pub struct Device {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    functions: HashMap<&'static str, CudaFunction>,
}

impl Device {
    pub fn new(ordinal: usize) -> anyhow::Result<Device> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.new_stream()?;
        let module = ctx.load_module(Ptx::from_src(PTX))?;
        let mut functions = HashMap::new();
        for name in KERNELS {
            functions.insert(name, module.load_function(name)?);
        }
        Ok(Device { ctx, stream, _module: module, functions })
    }

    pub fn name(&self) -> String {
        self.ctx.name().unwrap_or_else(|e| panic!("device name: {e}"))
    }

    /// Opt-in maximum of dynamic shared memory per block.
    pub fn max_dyn_smem(&self) -> u32 {
        let v = self.ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
        v.unwrap_or_else(|e| panic!("device attribute: {e}")) as u32
    }

    /// Free device memory in bytes.
    pub fn free_memory(&self) -> usize {
        self.ctx.mem_get_info().unwrap_or_else(|e| panic!("memory info: {e}")).0
    }

    pub fn sm_count(&self) -> usize {
        let v = self.ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        v.unwrap_or_else(|e| panic!("device attribute: {e}")) as usize
    }

    pub fn upload<T: DeviceRepr>(&self, host: &[T]) -> DevBuf<T> {
        DevBuf(self.stream.clone_htod(host).unwrap_or_else(|e| panic!("upload: {e}")))
    }

    pub fn zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> DevBuf<T> {
        DevBuf(self.stream.alloc_zeros(len).unwrap_or_else(|e| panic!("alloc {len}: {e}")))
    }

    pub fn download<T: DeviceRepr>(&self, d: &DevBuf<T>) -> Vec<T> {
        self.stream.clone_dtoh(&d.0).unwrap_or_else(|e| panic!("download: {e}"))
    }

    pub fn upload_tables(&self, t: &Tables) -> DeviceTables {
        let f1 = [&radix4_twiddles(&t.t1.tw), &radix4_twiddles(&t.t1.itw), &t.t1.w, &t.t1.iw].map(|v| self.upload(v));
        let f2 = [&radix4_twiddles(&t.t2.tw), &radix4_twiddles(&t.t2.itw), &t.t2.w, &t.t2.iw].map(|v| self.upload(v));
        let tab = Tab {
            tw1: self.ptr(&f1[0]),
            itw1: self.ptr(&f1[1]),
            w1: self.ptr(&f1[2]),
            iw1: self.ptr(&f1[3]),
            tw2: self.ptr(&f2[0]),
            itw2: self.ptr(&f2[1]),
            w2: self.ptr(&f2[2]),
            iw2: self.ptr(&f2[3]),
            n: t.n as u32,
            logn: t.n.trailing_zeros(),
            crt_inv: crt_inv(),
        };
        DeviceTables { tab, _f1: f1, _f2: f2 }
    }

    pub fn upload_half(&self, hf: &Half) -> DeviceHalf {
        let tab = self.upload_tables(&hf.tab);
        DeviceHalf { arg: HalfTab { tab: tab.tab }, _tab: tab }
    }

    fn ptr<T>(&self, d: &DevBuf<T>) -> u64 {
        d.0.device_ptr(&self.stream).0
    }

    /// Copies `host` over all of `d`.
    fn write<T: DeviceRepr>(&self, d: &mut DevBuf<T>, host: &[T]) {
        assert_eq!(d.len(), host.len());
        self.stream.memcpy_htod(host, &mut d.0).unwrap_or_else(|e| panic!("upload: {e}"));
    }

    /// Zeroed state for `slots` slots of k limbs at complex length n (k ≤ n), with per-slot
    /// work vectors when `global`.
    pub fn alloc_slots(&self, slots: usize, k: usize, n: usize, global: bool) -> SlotBufs {
        assert!(slots > 0 && k > 0 && k <= n);
        let pair = |len: usize| (self.zeros(len), self.zeros(len));
        SlotBufs {
            slots,
            k,
            n,
            global,
            x: self.zeros(slots * k),
            lim_t: self.zeros(slots * 2 * n),
            lim_s: self.zeros(slots * 2 * n),
            nhat: pair(slots * n),
            nphat: pair(slots * n),
            work: pair(if global { slots * n } else { 1 }),
            mul_a: pair(n),
            mul_b: pair(n),
            exp: self.zeros(1),
            ew: 0,
            nbits: self.zeros(slots),
            pos: self.zeros(slots),
            err: self.zeros(slots),
        }
    }

    /// N̂ = F(N) and N̂′ = F(N′) for slots `vs`, from k limbs per slot of all slots (row-major).
    /// Without `with_nhat` only N̂′ is set (the wraparound path writes its own N̂).
    #[allow(clippy::too_many_arguments)]
    pub fn set_constants(
        &self,
        t: &DeviceTables,
        b: &mut SlotBufs,
        n: &DevBuf<u32>,
        nprime: &DevBuf<u32>,
        vs: Range<usize>,
        threads: u32,
        with_nhat: bool,
    ) {
        assert!(t.n() == b.n && vs.end <= b.slots);
        if with_nhat {
            self.t_load_forward(t, n, b.k, (&mut b.nhat.0, &mut b.nhat.1), vs.clone(), threads);
        }
        self.t_load_forward(t, nprime, b.k, (&mut b.nphat.0, &mut b.nphat.1), vs, threads);
    }

    /// Overwrites N̂ with the wraparound path's factors, 2h = n elements per slot (`half_factors_of`).
    pub fn write_half_nhat(&self, b: &mut SlotBufs, f1: &[F1], f2: &[F2]) {
        self.write(&mut b.nhat.0, f1);
        self.write(&mut b.nhat.1, f2);
    }

    /// Writes k limbs per slot.
    pub fn write_x(&self, b: &mut SlotBufs, x: &[u32]) {
        self.write(&mut b.x, x);
    }

    pub fn read_x(&self, b: &SlotBufs) -> Vec<u32> {
        self.download(&b.x)
    }

    /// Exponents of `ew` words per slot (row-major), their bit counts and the next bit to run.
    pub fn set_exponents(&self, b: &mut SlotBufs, words: &[u32], ew: usize, nbits: &[u32], pos: &[u32]) {
        assert_eq!(words.len(), b.slots * ew);
        assert!(nbits.iter().all(|&nb| nb as usize <= 32 * ew), "exponent longer than {ew} words");
        b.exp = if ew == 0 { self.zeros(1) } else { self.upload(words) };
        b.ew = ew;
        self.write(&mut b.nbits, nbits);
        self.write(&mut b.pos, pos);
    }

    /// Per-slot error words, 1 where a carry that must vanish did not.
    pub fn read_err(&self, b: &SlotBufs) -> Vec<u32> {
        self.download(&b.err)
    }

    pub fn clear_err(&self, b: &mut SlotBufs) {
        self.write(&mut b.err, &vec![0; b.slots]);
    }

    fn slots_arg(&self, b: &SlotBufs, base: u32) -> Slots {
        Slots {
            x: self.ptr(&b.x),
            lim_t: self.ptr(&b.lim_t),
            lim_s: self.ptr(&b.lim_s),
            nhat1: self.ptr(&b.nhat.0),
            nhat2: self.ptr(&b.nhat.1),
            nphat1: self.ptr(&b.nphat.0),
            nphat2: self.ptr(&b.nphat.1),
            work1: self.ptr(&b.work.0),
            work2: self.ptr(&b.work.1),
            exp: self.ptr(&b.exp),
            nbits: self.ptr(&b.nbits),
            pos: self.ptr(&b.pos),
            err: self.ptr(&b.err),
            k: b.k as u32,
            ew: b.ew as u32,
            base,
        }
    }

    /// Up to `bits` steps on slots `vs` (kernel `run_steps`, or `run_bare` for step 1 alone;
    /// their `_global` variants in global mode),
    /// one block of `threads` per slot. In shared mode the vector lives in `smem` bytes of
    /// dynamic shared memory; otherwise in the per-slot work vectors. `half` selects the
    /// wraparound path.
    #[allow(clippy::too_many_arguments)]
    pub fn run_steps(
        &self,
        t: &DeviceTables,
        half: Option<&DeviceHalf>,
        b: &mut SlotBufs,
        vs: Range<usize>,
        base: u32,
        bits: u32,
        threads: u32,
        smem: u32,
        shared: bool,
        bare: bool,
    ) {
        assert!(t.n() == b.n && vs.end <= b.slots);
        check_threads(threads);
        let scratch = super::plan::scratch_bytes(threads) as usize;
        let need = if shared { 24 * b.n + scratch } else { super::plan::global_smem_bytes(b.n) as usize };
        assert!(smem as usize >= need && (shared || b.global), "{smem} B smem, need {need} (shared {shared})");
        // The global-mode kernels are compiled for at most 512 threads.
        assert!(shared || threads <= 512, "{threads} threads in global mode");
        let name = match (bare, shared) {
            (false, true) => "run_steps",
            (false, false) => "run_steps_global",
            (true, true) => "run_bare",
            (true, false) => "run_bare_global",
        };
        let s = self.slots_arg(b, base);
        let ht = half.map_or(HalfTab::default(), |h| h.arg);
        let first = vs.start as u32;
        let mut args = self.stream.launch_builder(&self.functions[name]);
        args.arg(&t.tab).arg(&ht).arg(&s).arg(&first).arg(&bits);
        self.launch(name, args, vs.len(), threads, smem);
    }

    /// x[out] ← x[a] · x[b] · R⁻¹ with slot `out`'s modulus, one block of `threads`.
    #[allow(clippy::too_many_arguments)]
    pub fn mul_slots(&self, t: &DeviceTables, half: Option<&DeviceHalf>, b: &mut SlotBufs, out: usize, a: usize, bb: usize, threads: u32) {
        assert!(t.n() == b.n && out < b.slots && a < b.slots && bb < b.slots);
        check_threads(threads);
        let s = self.slots_arg(b, 1);
        let ht = half.map_or(HalfTab::default(), |h| h.arg);
        let idx = [out as u32, a as u32, bb as u32];
        let mut args = self.stream.launch_builder(&self.functions["mul_slots"]);
        args.arg(&t.tab).arg(&ht).arg(&s).arg(&idx[0]).arg(&idx[1]).arg(&idx[2]);
        args.arg(&mut b.mul_a.0.0).arg(&mut b.mul_a.1.0).arg(&mut b.mul_b.0.0).arg(&mut b.mul_b.1.0);
        self.launch("mul_slots", args, 1, threads, super::plan::scratch_bytes(threads));
    }

    /// Weights and forward-transforms vectors `vs` of (f1, f2).
    pub fn t_forward(&self, t: &DeviceTables, f1: &mut DevBuf<F1>, f2: &mut DevBuf<F2>, vs: Range<usize>, threads: u32) {
        check_vectors(t, vs.end, &[f1.len(), f2.len()]);
        let first = vs.start as u32;
        let mut args = self.stream.launch_builder(&self.functions["t_forward"]);
        args.arg(&t.tab).arg(&mut f1.0).arg(&mut f2.0).arg(&first);
        self.launch("t_forward", args, vs.len(), threads, 0);
    }

    /// Inverse-transforms and unweights vectors `vs` of (f1, f2).
    pub fn t_inverse(&self, t: &DeviceTables, f1: &mut DevBuf<F1>, f2: &mut DevBuf<F2>, vs: Range<usize>, threads: u32) {
        check_vectors(t, vs.end, &[f1.len(), f2.len()]);
        let first = vs.start as u32;
        let mut args = self.stream.launch_builder(&self.functions["t_inverse"]);
        args.arg(&t.tab).arg(&mut f1.0).arg(&mut f2.0).arg(&first);
        self.launch("t_inverse", args, vs.len(), threads, 0);
    }

    /// out[v] = scale[v] · a[v] ⊙ b[v] for v in `vs`; `None` scales by 1.
    #[allow(clippy::too_many_arguments)]
    pub fn t_pointwise(
        &self,
        t: &DeviceTables,
        out: (&mut DevBuf<F1>, &mut DevBuf<F2>),
        a: (&DevBuf<F1>, &DevBuf<F2>),
        b: (&DevBuf<F1>, &DevBuf<F2>),
        scale: Option<&DevBuf<u32>>,
        vs: Range<usize>,
        threads: u32,
    ) {
        let lens = [out.0.len(), out.1.len(), a.0.len(), a.1.len(), b.0.len(), b.1.len()];
        check_vectors(t, vs.end, &lens);
        assert!(scale.is_none_or(|s| s.len() >= vs.end), "vectors {vs:?}, too few scales");
        let (null, first) = (0u64, vs.start as u32);
        let mut args = self.stream.launch_builder(&self.functions["t_pointwise"]);
        args.arg(&t.tab).arg(&mut out.0.0).arg(&mut out.1.0).arg(&a.0.0).arg(&a.1.0).arg(&b.0.0).arg(&b.1.0);
        match scale {
            Some(s) => args.arg(&s.0),
            None => args.arg(&null),
        };
        args.arg(&first);
        self.launch("t_pointwise", args, vs.len(), threads, 0);
    }

    /// Weights and forward-transforms limb strings `vs` of `k` limbs each (stride k, k ≤ n).
    pub fn t_load_forward(
        &self,
        t: &DeviceTables,
        limbs: &DevBuf<u32>,
        k: usize,
        out: (&mut DevBuf<F1>, &mut DevBuf<F2>),
        vs: Range<usize>,
        threads: u32,
    ) {
        check_vectors(t, vs.end, &[out.0.len(), out.1.len()]);
        assert!(k <= t.n() && limbs.len() >= k * vs.end, "limb strings {vs:?} of {k}, buffer {}", limbs.len());
        let (k32, first) = (k as u32, vs.start as u32);
        let mut args = self.stream.launch_builder(&self.functions["t_load_forward"]);
        args.arg(&t.tab).arg(&limbs.0).arg(&k32).arg(&mut out.0.0).arg(&mut out.1.0).arg(&first);
        self.launch("t_load_forward", args, vs.len(), threads, 0);
    }

    /// Launches `blocks` blocks of `threads` and waits for completion; panics on any error.
    fn launch(&self, name: &str, mut args: LaunchArgs, blocks: usize, threads: u32, smem: u32) {
        if blocks == 0 {
            return;
        }
        if smem > SMEM_DEFAULT {
            let attr = CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
            self.functions[name].set_attribute(attr, smem as i32).unwrap_or_else(|e| panic!("{name}: {smem} B smem: {e}"));
        }
        let cfg = LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem };
        // SAFETY: each caller pushes exactly the parameter list of kernel `name` in
        // `kernels/src/lib.rs` and has checked that every buffer covers the blocks launched.
        let r = unsafe { args.launch(cfg) }.and_then(|_| self.stream.synchronize());
        r.unwrap_or_else(|e| panic!("kernel {name} ({blocks} × {threads}): {e}"));
    }
}

/// The fused kernels need whole warps and at most 1,024 threads (see `plan`).
/// Twiddles r^t for t < n/2 (r^(n/2) = −1) as the kernels read them: the table itself, then
/// for each radix-4 pass with lower stage σ (q = n/2^(σ+2), from offset 2n − 6q) the runs R^j,
/// R^2j and R^3j over j < q, R = r^(2^σ), so that each pass reads them contiguously. The
/// butterflies take r^(n/4) = ±i (i for tw, −i for itw).
fn radix4_twiddles<F: Elem>(v: &[F]) -> Vec<F> {
    let n = 2 * v.len();
    assert!(n < 4 || v[n / 4] == F::i() || v[n / 4] == F::default().sub(F::i()));
    let at = |t: usize| if t < n / 2 { v[t] } else { F::default().sub(v[t - n / 2]) };
    let mut out = v.to_vec();
    let (mut q, mut sigma) = (n / 4, 0);
    while q >= 1 {
        for k in 1..=3 {
            out.extend((0..q).map(|j| at((k * j) << sigma)));
        }
        q /= 2;
        sigma += 1;
    }
    out
}

fn check_threads(threads: u32) {
    assert!(threads.is_multiple_of(32) && threads <= 1024, "{threads} threads per block");
}

/// Every buffer must hold `count` vectors of n elements.
fn check_vectors(t: &DeviceTables, count: usize, lens: &[usize]) {
    let need = count * t.n();
    assert!(lens.iter().all(|&l| l >= need), "{count} vectors of {} need {need} elements, have {lens:?}", t.n());
}

//! Batched modular squaring for generic odd moduli: Montgomery reduction over an
//! M31 × M61 number-theoretic transform, with CPU and CUDA backends.

pub mod batch;
pub mod cpu;
pub mod field;
pub mod gmp;
pub mod gpu;
pub mod multi;
pub mod score;
pub mod seq;
pub mod transform;

pub use batch::{ModBatch, Params, fermat};
pub use cpu::Cpu;
pub use gmp::Gmp;
pub use multi::Multi;
#[cfg(feature = "gpu")]
pub use gpu::{Gpu, GpuOptions};
pub use transform::{Backend, Limits, Transform};

//! Batched modular squaring for generic odd moduli: Montgomery reduction over an
//! M31 × M61 number-theoretic transform, with CPU and CUDA backends.

pub mod batch;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod field;
pub mod transform;

pub use batch::{ModBatch, Params, fermat};
pub use cpu::Cpu;
pub use transform::{Backend, Transform};

//! Several backends as one, e.g. one `Gpu` per device: the slots of a batch are split into
//! contiguous shards, one per part, and every shard runs on its own host thread. Slots are
//! independent, so only `mul` across shards needs the host: it reads both operands, multiplies
//! with `rug` and loads the result.

use crate::batch::{W, from_limbs, to_limbs};
use crate::transform::{Backend, BatchSetup, Limits};
use rug::Integer;
use std::ops::Range;

pub struct Multi<B> {
    parts: Vec<B>,
}

impl<B: Limits> Multi<B> {
    /// Panics on an empty vector or on parts whose `Limits` differ.
    pub fn new(parts: Vec<B>) -> Multi<B> {
        assert!(!parts.is_empty(), "Multi::new: no parts");
        for (i, p) in parts.iter().enumerate().skip(1) {
            let q = &parts[0];
            assert!(
                p.coefficient_bound() == q.coefficient_bound()
                    && p.supported_lengths() == q.supported_lengths()
                    && p.full_k() == q.full_k(),
                "Multi::new: part {i} has other limits than part 0"
            );
        }
        Multi { parts }
    }
}

impl<B> Multi<B> {
    pub fn parts(&self) -> &[B] {
        &self.parts
    }
}

/// Slot ranges of the shards: contiguous, sizes differ by at most one, none empty.
fn split(slots: usize, parts: usize) -> Vec<Range<usize>> {
    let shards = parts.min(slots);
    (0..shards).map(|s| s * slots / shards..(s + 1) * slots / shards).collect()
}

/// Calls `f(shard, item)` for every item, each on its own scoped thread (directly for one item).
fn each<I: Send, T: Send>(items: Vec<I>, f: impl Fn(usize, I) -> T + Sync) -> Vec<T> {
    if items.len() == 1 {
        return items.into_iter().map(|it| f(0, it)).collect();
    }
    std::thread::scope(|s| {
        let f = &f;
        let hs: Vec<_> = items.into_iter().enumerate().map(|(i, it)| s.spawn(move || f(i, it))).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
    })
}

/// Runs `f` on the shard `s` whose first slot is `start`, re-raising a panic with the shard and,
/// for a message ending in "slot i" (a carry-out), the global slot start + i.
fn on_shard<T>(s: usize, start: usize, f: impl FnOnce() -> T) -> T {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|e| {
        let msg = e.downcast_ref::<&str>().map(|m| m.to_string()).or_else(|| e.downcast_ref::<String>().cloned());
        let Some(msg) = msg else { std::panic::resume_unwind(e) };
        match msg.rsplit_once("slot ").and_then(|(head, i)| Some((head, i.parse::<usize>().ok()?))) {
            Some((head, i)) => panic!("{head}shard {s}, slot {}", start + i),
            None => panic!("shard {s} (first slot {start}): {msg}"),
        }
    })
}

pub struct MultiState<S> {
    shards: Vec<S>,
    /// First slot of each shard.
    starts: Vec<usize>,
    k: usize,
    /// N per slot, for `mul` across shards.
    n: Vec<Integer>,
}

impl<S> MultiState<S> {
    /// (shard, index within the shard) of slot `i`.
    fn locate(&self, i: usize) -> (usize, usize) {
        assert!(i < self.n.len(), "slot {i} out of range");
        let s = self.starts.partition_point(|&f| f <= i) - 1;
        (s, i - self.starts[s])
    }
}

impl<B: Limits> Limits for Multi<B> {
    fn coefficient_bound(&self) -> u128 {
        self.parts[0].coefficient_bound()
    }
    fn supported_lengths(&self) -> &[usize] {
        self.parts[0].supported_lengths()
    }
    fn full_k(&self) -> bool {
        self.parts[0].full_k()
    }
}

impl<B> Backend for Multi<B>
where
    B: Backend + Sync,
    B::State: Send,
{
    type State = MultiState<B::State>;

    fn create(&self, st: &BatchSetup) -> anyhow::Result<Self::State> {
        let ranges = split(st.n.len(), self.parts.len());
        let starts = ranges.iter().map(|r| r.start).collect();
        let shards = each(ranges, |p, r| {
            let setup = BatchSetup {
                k: st.k,
                len: st.len,
                base: st.base,
                n: &st.n[r.clone()],
                nprime: &st.nprime[r.clone()],
                two_n: &st.two_n[r],
            };
            self.parts[p].create(&setup)
        });
        let shards = shards.into_iter().collect::<anyhow::Result<_>>()?;
        Ok(MultiState { shards, starts, k: st.k, n: st.n.iter().map(|l| from_limbs(l)).collect() })
    }
    fn load(&self, st: &mut Self::State, i: usize, limbs: &[u32]) {
        let (s, j) = st.locate(i);
        self.parts[s].load(&mut st.shards[s], j, limbs);
    }
    fn set_exponent(&self, st: &mut Self::State, i: usize, words: &[u32], nbits: usize) {
        let (s, j) = st.locate(i);
        self.parts[s].set_exponent(&mut st.shards[s], j, words, nbits);
    }
    fn run(&self, st: &mut Self::State, bits: usize) {
        let starts = &st.starts;
        each(st.shards.iter_mut().collect(), |s, sh| on_shard(s, starts[s], || self.parts[s].run(sh, bits)));
    }
    /// Within one shard the part computes it; across shards the host does, with the result in
    /// [0, N).
    fn mul(&self, st: &mut Self::State, out: usize, a: usize, b: usize) {
        let ((so, o), (sa, ja), (sb, jb)) = (st.locate(out), st.locate(a), st.locate(b));
        if so == sa && sa == sb {
            let shard = &mut st.shards[so];
            return on_shard(so, st.starts[so], || self.parts[so].mul(shard, o, ja, jb));
        }
        let n = &st.n[out];
        let rinv = (Integer::from(1) << (st.k as u32 * W)).invert(n).unwrap();
        let x = (from_limbs(&self.read(st, a)) * from_limbs(&self.read(st, b)) * rinv).modulo(n);
        self.load(st, out, &to_limbs(&x, st.k));
    }
    fn read(&self, st: &Self::State, i: usize) -> Vec<u32> {
        let (s, j) = st.locate(i);
        self.parts[s].read(&st.shards[s], j)
    }
    fn remaining(&self, st: &Self::State, i: usize) -> usize {
        let (s, j) = st.locate(i);
        self.parts[s].remaining(&st.shards[s], j)
    }
}

#[cfg(test)]
mod tests {
    use super::on_shard;

    #[test]
    #[should_panic(expected = "run: carry out, shard 1, slot 4")]
    fn shard_panic_names_global_slot() {
        on_shard(1, 3, || panic!("run: carry out, slot {}", 1));
    }

    #[test]
    #[should_panic(expected = "shard 2 (first slot 5): lazy range violated")]
    fn shard_panic_without_slot() {
        on_shard(2, 5, || panic!("lazy range violated"));
    }
}

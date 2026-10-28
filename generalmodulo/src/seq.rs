//! Integer sequences used as test moduli.

use rug::Integer;

/// Tribonacci T(n) with T(0) = T(1) = 0, T(2) = 1 (OEIS A000073).
pub fn tribonacci(n: usize) -> Integer {
    let (mut a, mut b, mut c) = (Integer::from(0), Integer::from(0), Integer::from(1));
    for _ in 0..n {
        let d = Integer::from(&a + &b) + &c;
        a = b;
        b = c;
        c = d;
    }
    a
}

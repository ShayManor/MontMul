# GeneralModulo
Generalized modulo reduction for non-Mersenne numbers

GPU accelerated, efficient, batched, calculations of 3^(N-1) mod N for N > 2^100,000
1) Square N with forward NTT, multiplication, inverse NTT O(nlogn)
2) Reduce mod N with Montgomery reduction
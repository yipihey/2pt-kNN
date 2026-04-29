//! Reproducible deterministic subsampling helpers.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Return `min(k, n)` distinct indices in `0..n`, drawn deterministically
/// using `seed`. Output is sorted ascending.
pub fn subsample_indices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let kk = k.min(n);
    if kk == 0 {
        return Vec::new();
    }
    if kk == n {
        return (0..n).collect();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..n).collect();
    // Partial Fisher-Yates: choose kk distinct indices.
    for i in 0..kk {
        let j = rng.gen_range(i..n);
        idx.swap(i, j);
    }
    idx.truncate(kk);
    idx.sort_unstable();
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn deterministic_under_same_seed() {
        let a = subsample_indices(1000, 50, 42);
        let b = subsample_indices(1000, 50, 42);
        assert_eq!(a, b);
        assert_eq!(a.len(), 50);
        assert!(a.iter().all(|&x| x < 1000));
        let s: HashSet<_> = a.iter().collect();
        assert_eq!(s.len(), 50);
    }

    #[test]
    fn different_seeds_differ() {
        let a = subsample_indices(1000, 50, 1);
        let b = subsample_indices(1000, 50, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn k_larger_than_n_returns_all() {
        let v = subsample_indices(5, 100, 1);
        assert_eq!(v, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn k_zero_returns_empty() {
        let v = subsample_indices(10, 0, 1);
        assert!(v.is_empty());
    }
}

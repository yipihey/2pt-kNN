//! Local 2D weighted exact-cumulative-distribution (ECDF) engine.
//!
//! Given local points `(x_i, y_i, w_i)` (already extracted from the catalog
//! by the angular-search front-end) and tabulated thresholds `u_a` and `z_b`,
//! we want
//!
//! ```text
//! K(u_a, z_b) = sum_i w_i * 1[x_i <= u_a] * 1[y_i <= z_b]
//! ```
//!
//! for every (a, b). We use a sweep over `u_a` (ascending) combined with a
//! Fenwick (binary indexed) tree on the y axis. The y axis is compressed
//! against the user-supplied threshold array `z[]`: each point gets a bucket
//! `partition_point(z, |t| *t < y)`. Points whose y exceeds `z.last()` are
//! discarded for ECDF purposes.
//!
//! Cost per query: O(m log n_z + n_u * n_z) where m is the local count.
//! In typical kNN aperture regimes m is small (< few hundred) and the second
//! term dominates; the algorithm is also embarrassingly parallel over queries.
//!
//! A naive `histogram` backend is also provided for cross-validation.

/// Backend selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcdfBackend {
    /// Sweep-line + Fenwick. Exact and recommended.
    Sweep,
    /// 2D histogram with prefix sums. Exact and used for validation.
    Histogram,
}

/// Locate the bucket of `y` in `z_edges`. Returns `n_z` if `y > z_edges.last()`.
#[inline]
pub fn z_bucket(z_edges: &[f64], y: f64) -> usize {
    // partition_point finds the first index where the predicate is false.
    z_edges.partition_point(|t| *t < y)
}

/// Sweep-line + Fenwick implementation.
///
/// Inputs:
/// - `xs`, `ys`, `ws`: local points (length m).
/// - `u_edges`: sorted ascending angular thresholds (length n_u).
/// - `z_edges`: sorted ascending redshift thresholds (length n_z).
/// - `out`: output buffer of length n_u * n_z, row-major: out[a * n_z + b].
pub fn ecdf2d_sweep(
    xs: &[f64],
    ys: &[f64],
    ws: &[f64],
    u_edges: &[f64],
    z_edges: &[f64],
    out: &mut [f64],
) {
    let n_u = u_edges.len();
    let n_z = z_edges.len();
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert_eq!(xs.len(), ws.len());
    debug_assert_eq!(out.len(), n_u * n_z);

    // Sort points by x using an index permutation to avoid mutating callers.
    let m = xs.len();
    let mut order: Vec<u32> = (0..m as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        xs[a as usize]
            .partial_cmp(&xs[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Fenwick tree over n_z buckets (1-indexed).
    let mut bit: Vec<f64> = vec![0.0; n_z + 1];

    let mut ptr = 0usize;
    for a in 0..n_u {
        let u_a = u_edges[a];
        while ptr < m {
            let i = order[ptr] as usize;
            if xs[i] > u_a {
                break;
            }
            let bucket = z_bucket(z_edges, ys[i]);
            if bucket < n_z {
                fenwick_add(&mut bit, bucket + 1, ws[i]);
            }
            ptr += 1;
        }
        // Prefix sums up to bucket b => K(u_a, z_b).
        // Each prefix is O(log n_z); the whole row is O(n_z log n_z).
        let row = a * n_z;
        for b in 0..n_z {
            out[row + b] = fenwick_prefix(&bit, b + 1);
        }
    }
}

#[inline]
fn fenwick_add(bit: &mut [f64], mut i: usize, w: f64) {
    let n = bit.len() - 1;
    while i <= n {
        bit[i] += w;
        i += i & i.wrapping_neg();
    }
}

#[inline]
fn fenwick_prefix(bit: &[f64], mut i: usize) -> f64 {
    let mut s = 0.0;
    while i > 0 {
        s += bit[i];
        i -= i & i.wrapping_neg();
    }
    s
}

/// Histogram backend: bin into a 2D array sized (n_u, n_z) using
/// `searchsorted` on each axis, then take 2D prefix sums.
///
/// Note: this counts a point if x_i <= u_edges.last() and y_i <= z_edges.last();
/// points outside both ranges are discarded just as in the sweep backend.
pub fn ecdf2d_histogram(
    xs: &[f64],
    ys: &[f64],
    ws: &[f64],
    u_edges: &[f64],
    z_edges: &[f64],
    out: &mut [f64],
) {
    let n_u = u_edges.len();
    let n_z = z_edges.len();
    debug_assert_eq!(out.len(), n_u * n_z);
    out.fill(0.0);
    if n_u == 0 || n_z == 0 {
        return;
    }
    // Bin each point into (a, b).
    for i in 0..xs.len() {
        let a = u_edges.partition_point(|t| *t < xs[i]);
        let b = z_edges.partition_point(|t| *t < ys[i]);
        if a < n_u && b < n_z {
            out[a * n_z + b] += ws[i];
        }
    }
    // Cumulative along z axis (rows).
    for a in 0..n_u {
        let row = a * n_z;
        for b in 1..n_z {
            out[row + b] += out[row + b - 1];
        }
    }
    // Cumulative along u axis (cols).
    for b in 0..n_z {
        for a in 1..n_u {
            out[a * n_z + b] += out[(a - 1) * n_z + b];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn brute_force(
        xs: &[f64],
        ys: &[f64],
        ws: &[f64],
        u_edges: &[f64],
        z_edges: &[f64],
    ) -> Vec<f64> {
        let n_u = u_edges.len();
        let n_z = z_edges.len();
        let mut out = vec![0.0; n_u * n_z];
        for a in 0..n_u {
            for b in 0..n_z {
                let mut s = 0.0;
                for i in 0..xs.len() {
                    if xs[i] <= u_edges[a] && ys[i] <= z_edges[b] {
                        s += ws[i];
                    }
                }
                out[a * n_z + b] = s;
            }
        }
        out
    }

    #[test]
    fn sweep_matches_brute_random() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for trial in 0..20 {
            let m = 50 + trial * 10;
            let xs: Vec<f64> = (0..m).map(|_| rng.gen_range(0.0..1.0)).collect();
            let ys: Vec<f64> = (0..m).map(|_| rng.gen_range(0.0..1.0)).collect();
            let ws: Vec<f64> = (0..m).map(|_| rng.gen_range(0.5..2.0)).collect();
            let mut u_edges: Vec<f64> = (0..7).map(|i| 0.1 + 0.12 * i as f64).collect();
            u_edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut z_edges: Vec<f64> = (0..9).map(|i| 0.05 + 0.10 * i as f64).collect();
            z_edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let want = brute_force(&xs, &ys, &ws, &u_edges, &z_edges);
            let mut got = vec![0.0; u_edges.len() * z_edges.len()];
            ecdf2d_sweep(&xs, &ys, &ws, &u_edges, &z_edges, &mut got);
            for k in 0..got.len() {
                assert_abs_diff_eq!(got[k], want[k], epsilon = 1e-10);
            }
            let mut got_h = vec![0.0; u_edges.len() * z_edges.len()];
            ecdf2d_histogram(&xs, &ys, &ws, &u_edges, &z_edges, &mut got_h);
            for k in 0..got_h.len() {
                assert_abs_diff_eq!(got_h[k], want[k], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn empty_inputs_are_safe() {
        let xs: Vec<f64> = vec![];
        let ys: Vec<f64> = vec![];
        let ws: Vec<f64> = vec![];
        let u = [0.1, 0.2, 0.3];
        let z = [0.5, 1.0];
        let mut out = vec![1.0; u.len() * z.len()];
        ecdf2d_sweep(&xs, &ys, &ws, &u, &z, &mut out);
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }
}

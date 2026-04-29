//! Shell counts: K_q(u; z_minus, z_plus) = K_q(u, z_plus) - K_q(u, z_minus).

/// Compute shell counts for a single query from its 2D ECDF table `k_2d`
/// of shape `(n_u, n_z)` (row-major). `intervals` is a list of `(l, r)` index
/// pairs into `z_edges`, with `l < r`. The result `out` is row-major shape
/// `(n_u, n_intervals)`.
///
/// Convention: `K(u, z_l)` is the cumulative count up to z_edges[l] (inclusive).
/// The shell `(l, r)` therefore counts points with `z_edges[l] < z_i <= z_edges[r]`.
/// To count "below the lowest edge" use `l = -1` semantics by representing it
/// as `(usize::MAX, r)` (treated as zero left-cumulative).
pub fn shell_counts(
    k_2d: &[f64],
    n_u: usize,
    n_z: usize,
    intervals: &[(usize, usize)],
    out: &mut [f64],
) {
    let n_intervals = intervals.len();
    debug_assert_eq!(k_2d.len(), n_u * n_z);
    debug_assert_eq!(out.len(), n_u * n_intervals);
    for a in 0..n_u {
        let row_in = a * n_z;
        let row_out = a * n_intervals;
        for (ell, &(l, r)) in intervals.iter().enumerate() {
            let kr = if r == usize::MAX || r >= n_z {
                if r == usize::MAX {
                    0.0
                } else {
                    k_2d[row_in + n_z - 1]
                }
            } else {
                k_2d[row_in + r]
            };
            let kl = if l == usize::MAX {
                0.0
            } else {
                k_2d[row_in + l]
            };
            out[row_out + ell] = kr - kl;
        }
    }
}

/// Build adjacent-shell intervals: `[(0,1), (1,2), ..., (n_z-2, n_z-1)]`.
pub fn adjacent_intervals(n_z: usize) -> Vec<(usize, usize)> {
    if n_z < 2 {
        return Vec::new();
    }
    (0..(n_z - 1)).map(|i| (i, i + 1)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn shell_difference_is_correct() {
        // n_u=2, n_z=3
        let k = vec![
            // row a=0
            1.0, 3.0, 6.0,
            // row a=1
            2.0, 4.0, 9.0,
        ];
        let intervals = vec![(0, 1), (1, 2), (0, 2)];
        let mut out = vec![0.0; 2 * 3];
        shell_counts(&k, 2, 3, &intervals, &mut out);
        // a=0:
        assert_abs_diff_eq!(out[0], 3.0 - 1.0);
        assert_abs_diff_eq!(out[1], 6.0 - 3.0);
        assert_abs_diff_eq!(out[2], 6.0 - 1.0);
        // a=1:
        assert_abs_diff_eq!(out[3], 4.0 - 2.0);
        assert_abs_diff_eq!(out[4], 9.0 - 4.0);
        assert_abs_diff_eq!(out[5], 9.0 - 2.0);
    }

    #[test]
    fn adjacent_pairs_helper() {
        assert_eq!(adjacent_intervals(0), vec![]);
        assert_eq!(adjacent_intervals(1), vec![]);
        assert_eq!(adjacent_intervals(3), vec![(0, 1), (1, 2)]);
    }
}

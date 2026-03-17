//! Geometric pair-distance kernel for the LCA estimator.
//!
//! Given two bounding boxes (left and right child of an LCA node), compute the
//! fraction of uniformly distributed point pairs whose separation falls into
//! each radial bin.
//!
//! Current implementation: Monte Carlo sampling (MC reference).
//! Future: analytic 1D marginals + 3D CDF (piecewise polynomial).

use super::moments::BBox3;

// ---------------------------------------------------------------------------
// MC kernel
// ---------------------------------------------------------------------------

/// Compute bin fractions via Monte Carlo sampling.
///
/// For each bin j, returns the fraction of uniform point pairs (one from each box)
/// whose separation r satisfies `bin_edges[j] <= r < bin_edges[j+1]`.
///
/// Fractions sum to ≤ 1.0 (pairs outside all bins are not counted).
///
/// # Arguments
/// - `bbox_left`, `bbox_right`: bounding boxes of the two child partitions.
/// - `bin_edges`: radial bin edges (physical, not squared). Length = n_bins + 1.
/// - `n_samples`: number of uniform samples per box (total pairs = n_samples²).
/// - `box_size`: if `Some`, apply periodic minimum-image convention.
/// - `seed`: RNG seed for reproducibility.
pub fn bin_fractions_mc(
    bbox_left: &BBox3,
    bbox_right: &BBox3,
    bin_edges: &[f64],
    n_samples: usize,
    box_size: Option<f64>,
    seed: u64,
) -> Vec<f64> {
    let n_bins = bin_edges.len() - 1;
    let mut counts = vec![0u64; n_bins];

    // Quick reject: if minimum box separation > r_max, all fractions are 0.
    let r_max = bin_edges[n_bins];
    let min_d2 = bbox_left.min_dist_sq(bbox_right);
    if min_d2 > r_max * r_max {
        return vec![0.0; n_bins];
    }

    // Quick reject: if maximum box separation < r_min, all fractions are 0.
    let r_min = bin_edges[0];
    let max_d2 = bbox_left.max_dist_sq(bbox_right);
    if max_d2 < r_min * r_min {
        return vec![0.0; n_bins];
    }

    // Precompute squared bin edges.
    let r2_edges: Vec<f64> = bin_edges.iter().map(|&r| r * r).collect();

    // Simple LCG for speed (we need many samples, quality is secondary).
    let mut rng_state = seed ^ 0x517cc1b727220a95;

    let left_side = bbox_left.side_lengths();
    let right_side = bbox_right.side_lengths();

    let total_pairs = (n_samples as u64) * (n_samples as u64);

    // Generate samples from left box.
    let left_samples: Vec<[f64; 3]> = (0..n_samples)
        .map(|_| {
            [
                bbox_left.lo[0] + lcg_f64(&mut rng_state) * left_side[0],
                bbox_left.lo[1] + lcg_f64(&mut rng_state) * left_side[1],
                bbox_left.lo[2] + lcg_f64(&mut rng_state) * left_side[2],
            ]
        })
        .collect();

    // Generate samples from right box.
    let right_samples: Vec<[f64; 3]> = (0..n_samples)
        .map(|_| {
            [
                bbox_right.lo[0] + lcg_f64(&mut rng_state) * right_side[0],
                bbox_right.lo[1] + lcg_f64(&mut rng_state) * right_side[1],
                bbox_right.lo[2] + lcg_f64(&mut rng_state) * right_side[2],
            ]
        })
        .collect();

    // Count pairs in each bin.
    for left_pt in &left_samples {
        for right_pt in &right_samples {
            let r2 = dist_sq(left_pt, right_pt, box_size);

            if r2 < r2_edges[0] || r2 >= r2_edges[n_bins] {
                continue;
            }

            // Binary search for bin.
            let bin = match r2_edges.binary_search_by(|e| e.partial_cmp(&r2).unwrap()) {
                Ok(i) => i.min(n_bins - 1),
                Err(i) => {
                    if i == 0 { continue; }
                    (i - 1).min(n_bins - 1)
                }
            };

            counts[bin] += 1;
        }
    }

    // Normalize to fractions.
    counts
        .iter()
        .map(|&c| c as f64 / total_pairs as f64)
        .collect()
}

// ---------------------------------------------------------------------------
// Distance computation
// ---------------------------------------------------------------------------

#[inline]
fn dist_sq(a: &[f64; 3], b: &[f64; 3], box_size: Option<f64>) -> f64 {
    let mut r2 = 0.0;
    for k in 0..3 {
        let mut d = a[k] - b[k];
        if let Some(bs) = box_size {
            if d > 0.5 * bs {
                d -= bs;
            } else if d < -0.5 * bs {
                d += bs;
            }
        }
        r2 += d * d;
    }
    r2
}

// ---------------------------------------------------------------------------
// Simple LCG RNG (fast, sufficient for MC sampling)
// ---------------------------------------------------------------------------

#[inline]
fn lcg_f64(state: &mut u64) -> f64 {
    // LCG constants from Knuth MMIX.
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    // Map upper 53 bits to [0, 1).
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separated_cubes_single_bin() {
        // Two unit cubes separated by 10 units along x.
        // All pair distances are in [9, ~12.3].
        let left = BBox3 { lo: [0.0, 0.0, 0.0], hi: [1.0, 1.0, 1.0] };
        let right = BBox3 { lo: [10.0, 0.0, 0.0], hi: [11.0, 1.0, 1.0] };

        // Single bin covering the entire range.
        let bin_edges = vec![8.0, 13.0];
        let fracs = bin_fractions_mc(&left, &right, &bin_edges, 500, None, 42);

        // All pairs should fall in the one bin.
        assert!(
            (fracs[0] - 1.0).abs() < 0.05,
            "expected ~1.0, got {}",
            fracs[0]
        );
    }

    #[test]
    fn far_apart_cubes_zero_fraction() {
        // Two cubes very far apart, bin range too small.
        let left = BBox3 { lo: [0.0, 0.0, 0.0], hi: [1.0, 1.0, 1.0] };
        let right = BBox3 { lo: [100.0, 0.0, 0.0], hi: [101.0, 1.0, 1.0] };

        let bin_edges = vec![1.0, 5.0, 10.0];
        let fracs = bin_fractions_mc(&left, &right, &bin_edges, 100, None, 42);

        // Should be all zeros (quick reject by min_dist_sq).
        for f in &fracs {
            assert_eq!(*f, 0.0);
        }
    }

    #[test]
    fn fractions_sum_leq_one() {
        // Two overlapping cubes.
        let left = BBox3 { lo: [0.0, 0.0, 0.0], hi: [5.0, 5.0, 5.0] };
        let right = BBox3 { lo: [2.0, 2.0, 2.0], hi: [7.0, 7.0, 7.0] };

        let bin_edges = vec![0.1, 1.0, 3.0, 6.0, 10.0, 15.0];
        let fracs = bin_fractions_mc(&left, &right, &bin_edges, 500, None, 123);

        let total: f64 = fracs.iter().sum();
        assert!(
            total <= 1.0 + 1e-10,
            "fractions sum to {} > 1.0",
            total
        );
        // Should have some non-zero bins.
        assert!(total > 0.1, "fractions sum to {} is too small", total);
    }
}

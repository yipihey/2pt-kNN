//! Dilution ladder for multi-scale ξ(r) estimation.
//!
//! The dilution ladder extends kNN measurements to large scales by creating
//! a geometric sequence of random, volume-filling subsamples. At each level ℓ,
//! the data are randomly partitioned into R_ℓ = 8^ℓ disjoint subsets.
//!
//! The overlap between adjacent levels provides consistency checks and
//! the scatter across subsamples gives empirical variance estimates.

use crate::estimator::{linear_bins, KnnCdfs};
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Result from one dilution level within one mock.
pub struct LevelResult {
    pub level: usize,
    pub dilution_factor: usize,
    pub r_centers: Vec<f64>,
    pub xi_mean: Vec<f64>,
    pub xi_std: Vec<f64>,
    pub n_subsamples: usize,
    pub cdf_dd: Option<KnnCdfSummary>,
    pub cdf_dr: Option<KnnCdfSummary>,
    pub cdf_rr: Option<KnnCdfSummary>,
}

/// kNN-CDF summary averaged across subsamples at one dilution level.
#[derive(Debug, Clone)]
pub struct KnnCdfSummary {
    pub r_values: Vec<f64>,
    pub k_values: Vec<usize>,
    /// cdf_mean[k_idx][r_idx]
    pub cdf_mean: Vec<Vec<f64>>,
    /// cdf_std[k_idx][r_idx]
    pub cdf_std: Vec<Vec<f64>>,
    pub n_subsamples: usize,
}

/// Composite ξ(r) stitched from multiple levels.
pub struct CompositeXi {
    pub r: Vec<f64>,
    pub xi: Vec<f64>,
    pub xi_std: Vec<f64>,
    pub level_tag: Vec<usize>,
}

/// CDFs from all dilution levels, tagged by level.
pub struct CompositeCdfs {
    pub levels: Vec<LevelCdfs>,
}

/// CDFs for a single dilution level.
pub struct LevelCdfs {
    pub level: usize,
    pub cdf_dd: Option<KnnCdfSummary>,
    pub cdf_dr: Option<KnnCdfSummary>,
    pub cdf_rr: Option<KnnCdfSummary>,
}

/// A single dilution level.
#[derive(Debug)]
pub struct DilutionLevel {
    /// Dilution factor R_ℓ = 8^ℓ
    pub dilution_factor: usize,
    /// Level index ℓ
    pub level: usize,
    /// Indices into the original data array for each subsample
    pub subsamples: Vec<Vec<usize>>,
}

/// The full dilution ladder.
pub struct DilutionLadder {
    pub levels: Vec<DilutionLevel>,
}

impl DilutionLadder {
    /// Build a dilution ladder from n_data points up to the given max level.
    ///
    /// Level 0: full data (1 subset)
    /// Level ℓ: 8^ℓ disjoint, volume-filling subsets of n_data / 8^ℓ points each
    pub fn build(n_data: usize, max_level: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..n_data).collect();
        indices.shuffle(&mut rng);

        let mut levels = Vec::with_capacity(max_level + 1);

        for level in 0..=max_level {
            let r_ell = 8usize.pow(level as u32);
            if r_ell > n_data {
                break;
            }
            let subset_size = n_data / r_ell;
            if subset_size == 0 {
                break;
            }

            let subsamples: Vec<Vec<usize>> = (0..r_ell)
                .map(|m| {
                    let start = m * subset_size;
                    let end = start + subset_size;
                    indices[start..end].to_vec()
                })
                .collect();

            levels.push(DilutionLevel {
                dilution_factor: r_ell,
                level,
                subsamples,
            });
        }

        DilutionLadder { levels }
    }

    /// Characteristic scale probed at level ℓ with k-th neighbor.
    /// r_char(k, R_ℓ) ~ (k · R_ℓ / n̄)^{1/3}
    pub fn r_char(k: usize, dilution_factor: usize, nbar: f64) -> f64 {
        (k as f64 * dilution_factor as f64 / nbar).cbrt()
    }

    /// Characteristic kNN reach: r_char(k, R) = (k·R / (n̄·4π/3))^{1/3}.
    ///
    /// Uses the same convention as CoxMockParams::r_char_k, extended
    /// with a dilution factor R.
    pub fn r_char_knn(k: usize, dilution_factor: usize, nbar: f64) -> f64 {
        let four_pi_3 = 4.0 / 3.0 * std::f64::consts::PI;
        (k as f64 * dilution_factor as f64 / (nbar * four_pi_3)).cbrt()
    }

    /// Effective k_max for a given dilution level.
    /// Safety cap: don't query more than n_sub/4 neighbors.
    pub fn effective_k_max(k_max: usize, n_data: usize, dilution_factor: usize) -> usize {
        let n_sub = n_data / dilution_factor;
        k_max.min(n_sub / 4)
    }

    /// Build a reverse lookup: for each original data point, which subsample
    /// does it belong to at each dilution level?
    ///
    /// Returns `Vec<Vec<Option<usize>>>` indexed as `[level_idx][point_idx]`,
    /// giving `Some(subsample_idx)` or `None` if the point was dropped due to
    /// integer truncation at that level.
    pub fn inverse_map(&self, n_data: usize) -> Vec<Vec<Option<usize>>> {
        self.levels
            .iter()
            .map(|level| {
                let mut map = vec![None; n_data];
                for (sub_idx, sub) in level.subsamples.iter().enumerate() {
                    for &pt_idx in sub {
                        map[pt_idx] = Some(sub_idx);
                    }
                }
                map
            })
            .collect()
    }

    /// Compute bin edges for a specific dilution level.
    ///
    /// Level 0: `[r_min, r_char_knn(k_max, 1, nbar)]`
    /// Level ℓ>0: `[r_char_knn(k_max, R_{ℓ-1}, nbar), r_char_knn(k_eff, R_ℓ, nbar)]`
    ///
    /// Upper bound is capped at `box_size / 2`.
    pub fn bin_edges_for_level(
        level: usize,
        k_max: usize,
        nbar: f64,
        n_bins: usize,
        r_min: f64,
        box_size: f64,
        n_data: usize,
    ) -> Vec<f64> {
        let r_lo = if level == 0 {
            r_min
        } else {
            let r_prev = 8usize.pow((level - 1) as u32);
            Self::r_char_knn(k_max, r_prev, nbar)
        };

        let r_ell = 8usize.pow(level as u32);
        let k_eff = if level == 0 {
            k_max
        } else {
            Self::effective_k_max(k_max, n_data, r_ell)
        };

        let r_hi = Self::r_char_knn(k_eff, r_ell, nbar).min(box_size / 2.0);

        linear_bins(r_lo, r_hi, n_bins)
    }
}

/// Average KnnCdfs across subsamples → KnnCdfSummary (mean + std).
pub fn average_cdfs(cdf_subs: &[KnnCdfs]) -> KnnCdfSummary {
    assert!(!cdf_subs.is_empty());
    let first = &cdf_subs[0];
    let n_k = first.k_values.len();
    let n_r = first.r_values.len();
    let n_s = cdf_subs.len();
    let n_sf = n_s as f64;

    let mut cdf_mean = vec![vec![0.0; n_r]; n_k];
    let mut cdf_std = vec![vec![0.0; n_r]; n_k];

    for ki in 0..n_k {
        for ri in 0..n_r {
            let sum: f64 = cdf_subs.iter().map(|c| c.cdf_values[ki][ri]).sum();
            cdf_mean[ki][ri] = sum / n_sf;
        }
        if n_s > 1 {
            for ri in 0..n_r {
                let m = cdf_mean[ki][ri];
                let var: f64 = cdf_subs
                    .iter()
                    .map(|c| (c.cdf_values[ki][ri] - m).powi(2))
                    .sum::<f64>()
                    / (n_sf - 1.0);
                cdf_std[ki][ri] = var.sqrt();
            }
        }
    }

    KnnCdfSummary {
        r_values: first.r_values.clone(),
        k_values: first.k_values.clone(),
        cdf_mean,
        cdf_std,
        n_subsamples: n_s,
    }
}

/// Concatenate disjoint level results into a composite ξ(r) and composite CDFs.
pub fn stitch_levels(results: &[LevelResult]) -> (CompositeXi, CompositeCdfs) {
    let mut r = Vec::new();
    let mut xi = Vec::new();
    let mut xi_std = Vec::new();
    let mut level_tag = Vec::new();

    let mut cdf_levels = Vec::new();

    for res in results {
        for i in 0..res.r_centers.len() {
            r.push(res.r_centers[i]);
            xi.push(res.xi_mean[i]);
            xi_std.push(res.xi_std[i]);
            level_tag.push(res.level);
        }
        cdf_levels.push(LevelCdfs {
            level: res.level,
            cdf_dd: res.cdf_dd.clone(),
            cdf_dr: res.cdf_dr.clone(),
            cdf_rr: res.cdf_rr.clone(),
        });
    }

    (
        CompositeXi {
            r,
            xi,
            xi_std,
            level_tag,
        },
        CompositeCdfs {
            levels: cdf_levels,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimator::KnnCdfs;
    use std::collections::HashSet;

    #[test]
    fn test_subsamples_are_disjoint() {
        let ladder = DilutionLadder::build(1000, 2, 42);
        for level in &ladder.levels {
            let mut seen = HashSet::new();
            for sub in &level.subsamples {
                for &idx in sub {
                    assert!(seen.insert(idx), "Duplicate index {} at level {}", idx, level.level);
                }
            }
        }
    }

    #[test]
    fn test_subsamples_cover_all_indices() {
        let n = 1000;
        let ladder = DilutionLadder::build(n, 2, 42);
        for level in &ladder.levels {
            let r_ell = level.dilution_factor;
            let subset_size = n / r_ell;
            let expected_total = subset_size * r_ell;
            let all: HashSet<usize> = level.subsamples.iter().flatten().copied().collect();
            assert_eq!(
                all.len(),
                expected_total,
                "Level {} union size {} != expected {}",
                level.level,
                all.len(),
                expected_total,
            );
        }
    }

    #[test]
    fn test_subsample_counts() {
        let n = 1000;
        let ladder = DilutionLadder::build(n, 2, 42);
        for level in &ladder.levels {
            let r_ell = level.dilution_factor;
            let subset_size = n / r_ell;
            assert_eq!(level.subsamples.len(), r_ell);
            for sub in &level.subsamples {
                assert_eq!(sub.len(), subset_size);
            }
        }
    }

    #[test]
    fn test_different_seeds() {
        let a = DilutionLadder::build(1000, 1, 42);
        let b = DilutionLadder::build(1000, 1, 99);
        // Level 0 is the full data (both should be identical sets), but
        // level 1 partitions should differ with different seeds.
        assert!(a.levels.len() > 1 && b.levels.len() > 1);
        let a_sub0: HashSet<usize> = a.levels[1].subsamples[0].iter().copied().collect();
        let b_sub0: HashSet<usize> = b.levels[1].subsamples[0].iter().copied().collect();
        assert_ne!(a_sub0, b_sub0, "Different seeds should produce different partitions");
    }

    #[test]
    fn test_level_hierarchy() {
        // Level 2 (64 subs) should be a refinement of level 1 (8 subs).
        // Each level-1 sub corresponds to 8 level-2 subs with the same indices.
        let n = 512;
        let ladder = DilutionLadder::build(n, 2, 42);
        assert!(ladder.levels.len() >= 3);
        let l1_all: HashSet<usize> = ladder.levels[1]
            .subsamples
            .iter()
            .flatten()
            .copied()
            .collect();
        let l2_all: HashSet<usize> = ladder.levels[2]
            .subsamples
            .iter()
            .flatten()
            .copied()
            .collect();
        // Level 2 uses a subset of (or same) indices as level 1
        // Actually both come from the same shuffled array, so l2_all ⊆ l1_all
        // (l2 may use fewer indices due to integer truncation)
        assert!(l2_all.is_subset(&l1_all));
    }

    #[test]
    fn test_effective_k_max() {
        assert_eq!(DilutionLadder::effective_k_max(128, 1000, 1), 128);
        assert_eq!(DilutionLadder::effective_k_max(128, 1000, 8), 31);
        assert_eq!(DilutionLadder::effective_k_max(128, 1000, 64), 3);
    }

    #[test]
    fn test_bin_edges_contiguous() {
        let nbar = 1e-3;
        let n = 10000;
        let k_max = 64;
        let e0 = DilutionLadder::bin_edges_for_level(0, k_max, nbar, 20, 1.0, 500.0, n);
        let e1 = DilutionLadder::bin_edges_for_level(1, k_max, nbar, 20, 1.0, 500.0, n);
        // Level 1's lower bound should match level 0's upper bound
        let diff = (e1[0] - e0[e0.len() - 1]).abs();
        assert!(
            diff < 1e-10,
            "Level boundaries don't match: e0_hi={}, e1_lo={}",
            e0[e0.len() - 1],
            e1[0],
        );
    }

    #[test]
    fn test_stitch_levels() {
        let results = vec![
            LevelResult {
                level: 0,
                dilution_factor: 1,
                r_centers: vec![1.0, 2.0],
                xi_mean: vec![0.1, 0.2],
                xi_std: vec![0.01, 0.02],
                n_subsamples: 1,
                cdf_dd: None,
                cdf_dr: None,
                cdf_rr: None,
            },
            LevelResult {
                level: 1,
                dilution_factor: 8,
                r_centers: vec![3.0, 4.0],
                xi_mean: vec![0.3, 0.4],
                xi_std: vec![0.03, 0.04],
                n_subsamples: 8,
                cdf_dd: None,
                cdf_dr: None,
                cdf_rr: None,
            },
        ];
        let (composite, cdfs) = stitch_levels(&results);
        assert_eq!(composite.r, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(composite.xi, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(composite.xi_std, vec![0.01, 0.02, 0.03, 0.04]);
        assert_eq!(composite.level_tag, vec![0, 0, 1, 1]);
        assert_eq!(cdfs.levels.len(), 2);
    }

    #[test]
    fn test_average_cdfs() {
        // Two synthetic CDF measurements
        let cdf1 = KnnCdfs {
            r_values: vec![1.0, 2.0, 3.0],
            k_values: vec![1, 2],
            cdf_values: vec![
                vec![0.1, 0.4, 0.8],
                vec![0.0, 0.2, 0.6],
            ],
            n_queries: 100,
        };
        let cdf2 = KnnCdfs {
            r_values: vec![1.0, 2.0, 3.0],
            k_values: vec![1, 2],
            cdf_values: vec![
                vec![0.2, 0.5, 0.9],
                vec![0.1, 0.3, 0.7],
            ],
            n_queries: 100,
        };

        let summary = average_cdfs(&[cdf1, cdf2]);
        assert_eq!(summary.n_subsamples, 2);
        assert_eq!(summary.k_values, vec![1, 2]);
        assert_eq!(summary.r_values, vec![1.0, 2.0, 3.0]);

        // Mean of [0.1, 0.2] = 0.15
        assert!((summary.cdf_mean[0][0] - 0.15).abs() < 1e-12);
        // Mean of [0.4, 0.5] = 0.45
        assert!((summary.cdf_mean[0][1] - 0.45).abs() < 1e-12);
        // Std of [0.1, 0.2] = sqrt((0.05^2+0.05^2)/1) = 0.0707...
        assert!((summary.cdf_std[0][0] - (0.005f64 / 1.0).sqrt()).abs() < 1e-10);
    }
}

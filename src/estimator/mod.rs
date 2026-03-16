//! Landy–Szalay estimator via kNN pair-count densities.
//!
//! The primary estimator is Landy–Szalay in the Hamilton form:
//!
//!   1 + ξ̂(r) = D^DD(r) · D^RR(r) / [D^DR(r)]²
//!
//! where D^{XY}(r) = Σ_k pdf^{XY}_k(r) are the summed kNN pair-count
//! densities reconstructed from the distance distributions.
//!
//! For uniform periodic boxes (no weights), D^RR is the Erlang distribution
//! (analytic), and the estimator simplifies to DD/DR (Davis–Peebles).

#[cfg(not(target_arch = "wasm32"))]
use crate::tree::PointTree;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// kNN query result for a single query point: distances to k nearest neighbors.
/// Sorted by increasing distance. Distances are Euclidean (not squared).
#[derive(Debug, Clone)]
pub struct KnnDistances {
    /// Euclidean distances to the k nearest neighbors, sorted ascending
    pub distances: Vec<f64>,
}

/// Raw kNN distance distributions from a set of queries against a tree.
#[derive(Debug)]
pub struct KnnDistributions {
    /// For each query point, the distances to its k nearest neighbors
    pub per_query: Vec<KnnDistances>,
    /// k_max used
    pub k_max: usize,
}

/// Pair-count density evaluated at a set of radii.
/// D(r) = d<N>/dr evaluated by kernel density estimation on the kNN distances.
#[derive(Debug, Clone)]
pub struct PairCountDensity {
    /// Radii at which the density is evaluated
    pub r: Vec<f64>,
    /// Pair-count density values at each radius
    pub density: Vec<f64>,
}

/// The kNN Landy–Szalay estimator.
pub struct LandySzalayKnn {
    /// Maximum number of neighbors per query
    pub k_max: usize,
    /// Bandwidth for kernel density estimation of the CDF derivative
    pub bandwidth: f64,
}

impl LandySzalayKnn {
    pub fn new(k_max: usize) -> Self {
        Self {
            k_max,
            bandwidth: 0.0, // auto-select
        }
    }

    /// Set the KDE bandwidth explicitly. If 0, it will be auto-selected.
    pub fn with_bandwidth(mut self, bw: f64) -> Self {
        self.bandwidth = bw;
        self
    }

    /// Run kNN queries from `queries` against the tree built on `data`.
    ///
    /// Returns the distance distributions: for each query point, the
    /// Euclidean distances to its k_max nearest data neighbors.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn query_distances(
        &self,
        data_tree: &PointTree,
        queries: &[[f64; 3]],
    ) -> KnnDistributions {
        let k_max = self.k_max;

        #[cfg(feature = "parallel")]
        let iter = queries.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = queries.iter();

        let per_query: Vec<KnnDistances> = iter
            .map(|q| {
                let neighbors = data_tree.nearest_k(q, k_max);
                let distances: Vec<f64> = neighbors.iter().map(|n| n.dist).collect();
                KnnDistances { distances }
            })
            .collect();

        KnnDistributions { per_query, k_max }
    }

    /// Run kNN queries with periodic boundary conditions.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn query_distances_periodic(
        &self,
        data_tree: &PointTree,
        queries: &[[f64; 3]],
        box_size: f64,
    ) -> KnnDistributions {
        let k_max = self.k_max;

        #[cfg(feature = "parallel")]
        let iter = queries.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = queries.iter();

        let per_query: Vec<KnnDistances> = iter
            .map(|q| {
                let neighbors = data_tree.nearest_k_periodic(q, k_max, box_size);
                let distances: Vec<f64> = neighbors.iter().map(|n| n.dist).collect();
                KnnDistances { distances }
            })
            .collect();

        KnnDistributions { per_query, k_max }
    }

    /// Compute the empirical CDF of the k-th nearest neighbor distance.
    ///
    /// CDF_k(r) = fraction of query points whose k-th NN distance ≤ r.
    pub fn empirical_cdf(
        dists: &KnnDistributions,
        k: usize,
        r_values: &[f64],
    ) -> Vec<f64> {
        assert!(k >= 1 && k <= dists.k_max);
        let n = dists.per_query.len() as f64;

        r_values
            .iter()
            .map(|&r| {
                let count = dists
                    .per_query
                    .iter()
                    .filter(|qd| qd.distances.len() >= k && qd.distances[k - 1] <= r)
                    .count();
                count as f64 / n
            })
            .collect()
    }

    /// Compute empirical CDFs for multiple k values simultaneously.
    ///
    /// For each k, extracts k-th NN distances, sorts, then binary-searches r-grid.
    /// O(n log n + n_r log n) per k — much faster than calling empirical_cdf in a loop.
    pub fn empirical_cdfs(
        dists: &KnnDistributions,
        k_values: &[usize],
        r_values: &[f64],
    ) -> KnnCdfs {
        let n = dists.per_query.len();
        let n_f = n as f64;

        let cdf_values: Vec<Vec<f64>> = k_values
            .iter()
            .map(|&k| {
                assert!(k >= 1 && k <= dists.k_max);
                // Extract k-th NN distances and sort
                let mut dk: Vec<f64> = dists
                    .per_query
                    .iter()
                    .filter_map(|qd| {
                        if qd.distances.len() >= k {
                            Some(qd.distances[k - 1])
                        } else {
                            None
                        }
                    })
                    .collect();
                dk.sort_by(|a, b| a.partial_cmp(b).unwrap());

                // Binary search each r value
                r_values
                    .iter()
                    .map(|&r| {
                        let count = dk.partition_point(|&d| d <= r);
                        count as f64 / n_f
                    })
                    .collect()
            })
            .collect();

        KnnCdfs {
            r_values: r_values.to_vec(),
            k_values: k_values.to_vec(),
            cdf_values,
            n_queries: n,
        }
    }

    /// Compute the summed pair-count density D(r) = Σ_k pdf_k(r).
    ///
    /// This is the derivative of the mean neighbor count <N>(r) = Σ_k CDF_k(r).
    /// We estimate it by histogramming all kNN distances into radial bins
    /// and normalizing.
    pub fn pair_count_density(
        dists: &KnnDistributions,
        r_edges: &[f64],
    ) -> PairCountDensity {
        let n_bins = r_edges.len() - 1;
        let n_queries = dists.per_query.len() as f64;
        let mut counts = vec![0.0f64; n_bins];

        // Each neighbor distance contributes to the histogram.
        // The sum over all k gives the total pair-count density.
        for qd in &dists.per_query {
            for &d in &qd.distances {
                // Find bin
                if d < r_edges[0] || d >= r_edges[n_bins] {
                    continue;
                }
                // Binary search for bin
                let bin = match r_edges.binary_search_by(|edge| {
                    edge.partial_cmp(&d).unwrap()
                }) {
                    Ok(i) => i.min(n_bins - 1),
                    Err(i) => (i - 1).min(n_bins - 1),
                };
                counts[bin] += 1.0;
            }
        }

        // Normalize: divide by n_queries and by shell volume (4πr²Δr)
        let mut r_centers = Vec::with_capacity(n_bins);
        let mut density = Vec::with_capacity(n_bins);
        for i in 0..n_bins {
            let r_lo = r_edges[i];
            let r_hi = r_edges[i + 1];
            let r_mid = 0.5 * (r_lo + r_hi);
            let dr = r_hi - r_lo;
            let shell_vol = 4.0 * std::f64::consts::PI * r_mid * r_mid * dr;

            r_centers.push(r_mid);
            density.push(counts[i] / (n_queries * shell_vol));
        }

        PairCountDensity {
            r: r_centers,
            density,
        }
    }

    /// Estimate ξ(r) via the Landy–Szalay estimator in Hamilton form.
    ///
    /// 1 + ξ̂(r) = D^DD(r) · D^RR(r) / [D^DR(r)]²
    ///
    /// For a uniform periodic box without weights, D^RR is the uniform
    /// expectation (n̄ per unit volume), and the estimator simplifies to:
    ///
    /// 1 + ξ̂(r) = D^DD(r) / D^DR(r)   (Davis–Peebles)
    ///
    /// We provide both forms.
    pub fn estimate_xi_dp(
        dd: &PairCountDensity,
        dr: &PairCountDensity,
    ) -> XiEstimate {
        assert_eq!(dd.r.len(), dr.r.len());
        let r = dd.r.clone();
        let xi: Vec<f64> = dd
            .density
            .iter()
            .zip(dr.density.iter())
            .map(|(&dd_val, &dr_val)| {
                if dr_val > 0.0 {
                    dd_val / dr_val - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        XiEstimate { r, xi }
    }

    /// Full Landy–Szalay estimator using DD, DR, and RR pair-count densities.
    ///
    /// The Hamilton form DD·RR/DR² requires all densities on the same
    /// footing. Since `pair_count_density` normalizes by n_queries, and
    /// DD queries N_d points while DR/RR query N_r points, we must scale
    /// RR by (N_d / N_r) so that it measures the same effective density
    /// as DD and DR.
    ///
    /// `n_data` and `n_random` are the catalog sizes used for DD and
    /// DR/RR queries respectively.
    pub fn estimate_xi_ls(
        dd: &PairCountDensity,
        dr: &PairCountDensity,
        rr: &PairCountDensity,
        n_data: usize,
        n_random: usize,
    ) -> XiEstimate {
        assert_eq!(dd.r.len(), dr.r.len());
        assert_eq!(dd.r.len(), rr.r.len());
        let r = dd.r.clone();
        // Scale RR density to match DD normalization:
        // DD is count / (N_d × V_shell) ≈ n̄_data
        // RR is count / (N_r × V_shell) ≈ n̄_random = (N_r/N_d) × n̄_data
        // Multiply RR by (N_d / N_r) to get n̄_data scale.
        let rr_scale = n_data as f64 / n_random as f64;
        let xi: Vec<f64> = dd
            .density
            .iter()
            .zip(dr.density.iter())
            .zip(rr.density.iter())
            .map(|((&dd_val, &dr_val), &rr_val)| {
                let rr_scaled = rr_val * rr_scale;
                if rr_scaled > 0.0 && dr_val > 0.0 {
                    // Hamilton form: DD·RR_scaled/DR² − 1
                    (dd_val * rr_scaled) / (dr_val * dr_val) - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        XiEstimate { r, xi }
    }
}

/// CDF values for selected k-values evaluated at a common r-grid.
/// Indexed as cdf_values[k_idx][r_idx].
#[derive(Debug, Clone)]
pub struct KnnCdfs {
    pub r_values: Vec<f64>,
    pub k_values: Vec<usize>,
    /// cdf_values[k_idx][r_idx] = CDF_{k_values[k_idx]}(r_values[r_idx])
    pub cdf_values: Vec<Vec<f64>>,
    pub n_queries: usize,
}

/// Result of a correlation function estimation.
#[derive(Debug, Clone)]
pub struct XiEstimate {
    /// Separation values (bin centers)
    pub r: Vec<f64>,
    /// Estimated ξ(r) values
    pub xi: Vec<f64>,
}

impl XiEstimate {
    /// Multiply ξ by r² for plotting (common convention for power-law signals).
    pub fn r2_xi(&self) -> Vec<f64> {
        self.r
            .iter()
            .zip(self.xi.iter())
            .map(|(&r, &xi)| r * r * xi)
            .collect()
    }
}

/// Helper: generate linearly spaced bin edges.
pub fn linear_bins(r_min: f64, r_max: f64, n_bins: usize) -> Vec<f64> {
    let dr = (r_max - r_min) / n_bins as f64;
    (0..=n_bins).map(|i| r_min + i as f64 * dr).collect()
}

/// Helper: generate logarithmically spaced bin edges.
pub fn log_bins(r_min: f64, r_max: f64, n_bins: usize) -> Vec<f64> {
    let log_min = r_min.ln();
    let log_max = r_max.ln();
    let d_log = (log_max - log_min) / n_bins as f64;
    (0..=n_bins)
        .map(|i| (log_min + i as f64 * d_log).exp())
        .collect()
}

/// Powers of 2 up to k_eff: [1, 2, 4, 8, ..., k_eff].
pub fn cdf_k_values(k_eff: usize) -> Vec<usize> {
    let mut ks = Vec::new();
    let mut k = 1;
    while k <= k_eff {
        ks.push(k);
        k *= 2;
    }
    ks
}

/// Dense linearly-spaced r-grid for CDF evaluation.
pub fn cdf_r_grid(r_lo: f64, r_hi: f64, n_points: usize) -> Vec<f64> {
    assert!(n_points >= 2);
    let dr = (r_hi - r_lo) / (n_points - 1) as f64;
    (0..n_points).map(|i| r_lo + i as f64 * dr).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdf_k_values() {
        assert_eq!(cdf_k_values(1), vec![1]);
        assert_eq!(cdf_k_values(2), vec![1, 2]);
        assert_eq!(cdf_k_values(3), vec![1, 2]);
        assert_eq!(cdf_k_values(4), vec![1, 2, 4]);
        assert_eq!(cdf_k_values(8), vec![1, 2, 4, 8]);
        assert_eq!(cdf_k_values(64), vec![1, 2, 4, 8, 16, 32, 64]);
        assert_eq!(cdf_k_values(100), vec![1, 2, 4, 8, 16, 32, 64]);
    }

    #[test]
    fn test_cdf_r_grid() {
        let grid = cdf_r_grid(0.0, 10.0, 11);
        assert_eq!(grid.len(), 11);
        assert!((grid[0] - 0.0).abs() < 1e-12);
        assert!((grid[10] - 10.0).abs() < 1e-12);
        assert!((grid[5] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_empirical_cdfs_matches_single() {
        // Build synthetic KnnDistributions
        let per_query: Vec<KnnDistances> = (0..100)
            .map(|i| {
                let base = (i as f64 + 1.0) * 0.1;
                KnnDistances {
                    distances: vec![base, base * 2.0, base * 3.0, base * 4.0],
                }
            })
            .collect();
        let dists = KnnDistributions {
            per_query,
            k_max: 4,
        };

        let r_values: Vec<f64> = (1..=20).map(|i| i as f64 * 0.5).collect();
        let k_values = vec![1, 2, 4];

        let batch = LandySzalayKnn::empirical_cdfs(&dists, &k_values, &r_values);
        assert_eq!(batch.cdf_values.len(), 3);
        assert_eq!(batch.k_values, k_values);
        assert_eq!(batch.r_values, r_values);

        // Compare with individual calls
        for (ki, &k) in k_values.iter().enumerate() {
            let single = LandySzalayKnn::empirical_cdf(&dists, k, &r_values);
            for (ri, &val) in single.iter().enumerate() {
                assert!(
                    (batch.cdf_values[ki][ri] - val).abs() < 1e-12,
                    "Mismatch at k={}, r_idx={}: batch={}, single={}",
                    k,
                    ri,
                    batch.cdf_values[ki][ri],
                    val,
                );
            }
        }
    }
}

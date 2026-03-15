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

use crate::tree::PointTree;
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
    pub fn query_distances(
        &self,
        data_tree: &PointTree,
        queries: &[[f64; 3]],
    ) -> KnnDistributions {
        let k_max = self.k_max;

        let per_query: Vec<KnnDistances> = queries
            .par_iter()
            .map(|q| {
                let neighbors = data_tree.nearest_k(q, k_max);
                let distances: Vec<f64> = neighbors.iter().map(|n| n.dist_sq.sqrt()).collect();
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
    pub fn estimate_xi_ls(
        dd: &PairCountDensity,
        dr: &PairCountDensity,
        rr: &PairCountDensity,
    ) -> XiEstimate {
        assert_eq!(dd.r.len(), dr.r.len());
        assert_eq!(dd.r.len(), rr.r.len());
        let r = dd.r.clone();
        let xi: Vec<f64> = dd
            .density
            .iter()
            .zip(dr.density.iter())
            .zip(rr.density.iter())
            .map(|((&dd_val, &dr_val), &rr_val)| {
                if rr_val > 0.0 && dr_val > 0.0 {
                    // Hamilton form: DD·RR/DR² − 1
                    (dd_val * rr_val) / (dr_val * dr_val) - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        XiEstimate { r, xi }
    }
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

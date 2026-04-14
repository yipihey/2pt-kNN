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

pub mod cumulants;
pub mod profile;
pub mod savgol;

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

// ============================================================================
// Per-point density profiles
// ============================================================================

/// Volume of a sphere: V = 4π/3 · r³.
#[inline]
pub fn volume(r: f64) -> f64 {
    (4.0 / 3.0) * std::f64::consts::PI * r * r * r
}

/// Per-point cumulative neighbor count from kNN distances, merged across
/// dilution levels. Each (radius, count) pair says "N(<r) = count".
#[derive(Debug, Clone)]
pub struct CumulativeProfile {
    /// Sorted ascending by radius.
    pub radii: Vec<f64>,
    /// N(<r) at each radius.
    pub counts: Vec<f64>,
}

impl CumulativeProfile {
    /// Build from raw kNN distances at a single dilution level.
    ///
    /// `distances` are the k nearest-neighbor Euclidean distances (sorted ascending).
    /// `dilution_factor` scales the raw rank to the full-density count:
    /// count_k = k * dilution_factor.
    pub fn from_knn(distances: &[f64], dilution_factor: usize) -> Self {
        let df = dilution_factor as f64;
        let radii: Vec<f64> = distances.to_vec();
        let counts: Vec<f64> = (1..=distances.len()).map(|k| k as f64 * df).collect();
        CumulativeProfile { radii, counts }
    }

    /// Merge profiles from multiple dilution levels into one sorted profile.
    pub fn merge(levels: &[Self]) -> Self {
        let total: usize = levels.iter().map(|l| l.radii.len()).sum();
        let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(total);
        for level in levels {
            for (r, c) in level.radii.iter().zip(level.counts.iter()) {
                pairs.push((*r, *c));
            }
        }
        // Sort by radius (stable: preserve level ordering at ties)
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let radii = pairs.iter().map(|p| p.0).collect();
        let counts = pairs.iter().map(|p| p.1).collect();
        CumulativeProfile { radii, counts }
    }

    /// Evaluate N(<r) at a single radius via linear interpolation in volume space.
    ///
    /// Below the smallest sample: extrapolate from (0, 0) to (V_1, count_1).
    /// Above the largest sample: clamp to the last count.
    pub fn eval_at_radius(&self, r: f64) -> f64 {
        if self.radii.is_empty() {
            return 0.0;
        }
        let v = volume(r);
        let v0 = volume(self.radii[0]);
        if v <= v0 {
            // Extrapolate from origin (0, 0) to (V_1, count_1)
            if v0 > 0.0 {
                self.counts[0] * v / v0
            } else {
                self.counts[0]
            }
        } else {
            let vn = volume(*self.radii.last().unwrap());
            if v >= vn {
                return *self.counts.last().unwrap();
            }
            // Find bracketing interval in volume space
            // We need the index i such that V(radii[i]) <= v < V(radii[i+1])
            let idx = self
                .radii
                .partition_point(|&ri| volume(ri) <= v)
                .saturating_sub(1);
            let vi = volume(self.radii[idx]);
            let vi1 = volume(self.radii[idx + 1]);
            let ci = self.counts[idx];
            let ci1 = self.counts[idx + 1];
            if (vi1 - vi).abs() < 1e-30 {
                ci
            } else {
                ci + (ci1 - ci) * (v - vi) / (vi1 - vi)
            }
        }
    }

    /// Evaluate N(<r) at multiple radii.
    pub fn eval_at_radii(&self, r: &[f64]) -> Vec<f64> {
        r.iter().map(|&ri| self.eval_at_radius(ri)).collect()
    }

    /// Volume-space density dN/dV at radius r.
    ///
    /// Returns the slope of the piecewise-linear N(V) interpolant.
    /// For a Poisson field this is constant (= n̄).
    /// Beyond the last sample radius, returns 0 (clamped region).
    pub fn volume_density(&self, r: f64) -> f64 {
        if self.radii.is_empty() {
            return 0.0;
        }
        let v = volume(r);
        let v0 = volume(self.radii[0]);
        if v <= v0 {
            // Extrapolation region: slope from (0,0) to (V_1, count_1)
            if v0 > 0.0 {
                self.counts[0] / v0
            } else {
                0.0
            }
        } else {
            let vn = volume(*self.radii.last().unwrap());
            if v >= vn {
                return 0.0;
            }
            let idx = self
                .radii
                .partition_point(|&ri| volume(ri) <= v)
                .saturating_sub(1);
            let vi = volume(self.radii[idx]);
            let vi1 = volume(self.radii[idx + 1]);
            if (vi1 - vi).abs() < 1e-30 {
                0.0
            } else {
                (self.counts[idx + 1] - self.counts[idx]) / (vi1 - vi)
            }
        }
    }

    /// Volume-space density dN/dV at multiple radii.
    pub fn volume_density_at_radii(&self, r: &[f64]) -> Vec<f64> {
        r.iter().map(|&ri| self.volume_density(ri)).collect()
    }
}

/// Per-point DD and DR profiles for one data point.
#[derive(Debug, Clone)]
pub struct PointProfile {
    pub point_index: usize,
    pub dd: CumulativeProfile,
    pub dr: CumulativeProfile,
}

/// How to compute the RR reference.
pub enum RrMode {
    /// n̄_R · V(r) — exact for Poisson randoms, noise-free.
    Analytic,
    /// Empirically measured RR cumulative profile.
    Empirical(CumulativeProfile),
}

/// Per-point xi evaluated on a common grid.
#[derive(Debug, Clone)]
pub struct PointXi {
    pub point_index: usize,
    pub r: Vec<f64>,
    pub xi: Vec<f64>,
}

/// Collection of all per-point profiles + catalog metadata.
pub struct PerPointProfiles {
    pub profiles: Vec<PointProfile>,
    pub nbar_data: f64,
    pub nbar_random: f64,
    pub box_size: f64,
}

impl PerPointProfiles {
    /// Evaluate per-point **volume-averaged** ξ̄_i(<r) on a common radial grid.
    ///
    /// With `RrMode::Analytic` (periodic box, known geometry): uses the direct
    /// estimator ξ̄_i(<r) = DD_i(<r) / (n̄_D · V(r)) − 1, which is unbiased
    /// because the denominator is exact (no stochastic DR in the denominator).
    ///
    /// With `RrMode::Empirical`: uses the Hamilton form
    ///   1 + ξ̄_i(<r) = DD_i(<r) · RR(<r) / DR_i(<r)² × (n̄_R / n̄_D)
    /// which corrects for non-trivial survey geometry but has ratio bias
    /// when applied per-point.
    ///
    /// To recover the differential (shell) ξ(r), use [`evaluate_xi_differential`].
    pub fn evaluate_xi(&self, r_grid: &[f64], rr_mode: &RrMode) -> Vec<PointXi> {
        self.profiles
            .iter()
            .map(|prof| {
                let dd_vals = prof.dd.eval_at_radii(r_grid);

                let xi: Vec<f64> = match rr_mode {
                    RrMode::Analytic => {
                        // Direct estimator: DD_i / (n̄_D · V) − 1
                        // Unbiased: denominator is exact, no DR noise.
                        r_grid
                            .iter()
                            .enumerate()
                            .map(|(i, &r)| {
                                let expected = self.nbar_data * volume(r);
                                if expected > 0.0 {
                                    dd_vals[i] / expected - 1.0
                                } else {
                                    0.0
                                }
                            })
                            .collect()
                    }
                    RrMode::Empirical(ref rr_prof) => {
                        // Hamilton form: DD · RR / DR² × (n̄_R/n̄_D) − 1
                        let dr_vals = prof.dr.eval_at_radii(r_grid);
                        let nbar_ratio = self.nbar_random / self.nbar_data;
                        r_grid
                            .iter()
                            .enumerate()
                            .map(|(i, &r)| {
                                let dd_r = dd_vals[i];
                                let dr_r = dr_vals[i];
                                let rr_r = rr_prof.eval_at_radius(r);
                                if dr_r > 0.0 && rr_r > 0.0 {
                                    (dd_r * rr_r) / (dr_r * dr_r) * nbar_ratio - 1.0
                                } else {
                                    0.0
                                }
                            })
                            .collect()
                    }
                };

                PointXi {
                    point_index: prof.point_index,
                    r: r_grid.to_vec(),
                    xi,
                }
            })
            .collect()
    }

    /// Compute the mean volume-averaged ξ̄(<r) across all points.
    pub fn mean_xi(&self, r_grid: &[f64], rr_mode: &RrMode) -> XiEstimate {
        let all = self.evaluate_xi(r_grid, rr_mode);
        let n = all.len() as f64;
        let xi: Vec<f64> = (0..r_grid.len())
            .map(|i| all.iter().map(|px| px.xi[i]).sum::<f64>() / n)
            .collect();
        XiEstimate {
            r: r_grid.to_vec(),
            xi,
        }
    }

    /// Evaluate per-point **differential** ξ_i(r) via Savitzky-Golay differentiation
    /// of the cumulative estimator.
    ///
    /// The cumulative Hamilton estimator gives ξ̄_i(<r) (smooth, bin-free).
    /// The differential ξ is recovered by the exact inversion:
    ///
    ///   ξ(r) = ξ̄(<r) + (r/3) dξ̄/dr
    ///
    /// The Savitzky-Golay filter simultaneously smooths ξ̄ and computes its
    /// analytic derivative, so we never form a noisy finite difference.
    ///
    /// **Requires a uniformly-spaced `r_grid`** (e.g. from [`cdf_r_grid`]).
    ///
    /// # Parameters
    /// - `sg_half_window`: half-width of the SG window (full width = 2m+1).
    ///   Larger = smoother but loses small-scale features.
    /// - `sg_poly_order`: polynomial degree for the local fit (typically 2 or 3).
    pub fn evaluate_xi_differential(
        &self,
        r_grid: &[f64],
        rr_mode: &RrMode,
        sg_half_window: usize,
        sg_poly_order: usize,
    ) -> Vec<PointXi> {
        let xi_bar_all = self.evaluate_xi(r_grid, rr_mode);
        let h = if r_grid.len() >= 2 {
            r_grid[1] - r_grid[0]
        } else {
            1.0
        };

        xi_bar_all
            .into_iter()
            .map(|px| {
                let (xi_bar_smooth, dxi_bar_dr) =
                    savgol::sg_smooth_diff(&px.xi, h, sg_half_window, sg_poly_order);

                // ξ(r) = ξ̄(<r) + (r/3) dξ̄/dr
                let xi: Vec<f64> = px
                    .r
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| xi_bar_smooth[i] + (r / 3.0) * dxi_bar_dr[i])
                    .collect();

                PointXi {
                    point_index: px.point_index,
                    r: px.r,
                    xi,
                }
            })
            .collect()
    }

    /// Compute the mean differential ξ(r) across all points.
    pub fn mean_xi_differential(
        &self,
        r_grid: &[f64],
        rr_mode: &RrMode,
        sg_half_window: usize,
        sg_poly_order: usize,
    ) -> XiEstimate {
        let all = self.evaluate_xi_differential(r_grid, rr_mode, sg_half_window, sg_poly_order);
        let n = all.len() as f64;
        let xi: Vec<f64> = (0..r_grid.len())
            .map(|i| all.iter().map(|px| px.xi[i]).sum::<f64>() / n)
            .collect();
        XiEstimate {
            r: r_grid.to_vec(),
            xi,
        }
    }
}

/// Remove self-pairs from DD query results.
///
/// When querying data against its own tree, the first neighbor is the
/// point itself (distance ≈ 0). We query k_max+1 neighbors and drop
/// the self-pair, keeping exactly k_max real neighbors.
pub fn exclude_self_pairs(dists: KnnDistributions, k_max: usize) -> KnnDistributions {
    let filtered: Vec<KnnDistances> = dists
        .per_query
        .into_iter()
        .map(|qd| {
            let real: Vec<f64> = qd
                .distances
                .into_iter()
                .filter(|&d| d > 1e-10)
                .take(k_max)
                .collect();
            KnnDistances { distances: real }
        })
        .collect();

    KnnDistributions {
        per_query: filtered,
        k_max,
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
    fn test_cumulative_profile_from_knn() {
        let dists = vec![1.0, 2.0, 3.0];
        let cp = CumulativeProfile::from_knn(&dists, 1);
        assert_eq!(cp.radii, vec![1.0, 2.0, 3.0]);
        assert_eq!(cp.counts, vec![1.0, 2.0, 3.0]);

        let cp8 = CumulativeProfile::from_knn(&dists, 8);
        assert_eq!(cp8.counts, vec![8.0, 16.0, 24.0]);
    }

    #[test]
    fn test_cumulative_profile_eval_at_sample_points() {
        // At sample radii, eval should recover exact counts
        let dists = vec![10.0, 20.0, 30.0];
        let cp = CumulativeProfile::from_knn(&dists, 1);
        for (i, &r) in dists.iter().enumerate() {
            let val = cp.eval_at_radius(r);
            assert!(
                (val - (i + 1) as f64).abs() < 1e-10,
                "At r={}, expected {}, got {}",
                r,
                i + 1,
                val,
            );
        }
    }

    #[test]
    fn test_cumulative_profile_interpolation_volume_space() {
        // Two points: r=10 → count=1, r=20 → count=2
        // Interpolation is linear in V = 4π/3·r³
        let cp = CumulativeProfile {
            radii: vec![10.0, 20.0],
            counts: vec![1.0, 2.0],
        };
        let v10 = volume(10.0);
        let v20 = volume(20.0);
        // Midpoint in volume space
        let v_mid = 0.5 * (v10 + v20);
        let r_mid = (v_mid / (4.0 / 3.0 * std::f64::consts::PI)).cbrt();
        let val = cp.eval_at_radius(r_mid);
        assert!(
            (val - 1.5).abs() < 1e-10,
            "Volume-space midpoint should give count 1.5, got {}",
            val,
        );
    }

    #[test]
    fn test_cumulative_profile_extrapolation() {
        let cp = CumulativeProfile {
            radii: vec![10.0, 20.0],
            counts: vec![1.0, 2.0],
        };
        // Below minimum: extrapolate from origin
        let val_small = cp.eval_at_radius(5.0);
        // v(5)/v(10) = (5/10)^3 = 0.125, so count = 1.0 * 0.125
        assert!((val_small - 0.125).abs() < 1e-10);

        // Above maximum: clamp
        let val_big = cp.eval_at_radius(100.0);
        assert!((val_big - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_profile_merge() {
        let cp1 = CumulativeProfile::from_knn(&[1.0, 2.0], 1);
        let cp2 = CumulativeProfile::from_knn(&[1.5, 3.0], 8);
        let merged = CumulativeProfile::merge(&[cp1, cp2]);
        assert_eq!(merged.radii, vec![1.0, 1.5, 2.0, 3.0]);
        assert_eq!(merged.counts, vec![1.0, 8.0, 2.0, 16.0]);
    }

    #[test]
    fn test_volume_density_uniform() {
        // For a uniform field with n̄ = 0.001, N(<r) = n̄ · V(r).
        // dN/dV should be n̄ everywhere.
        let nbar = 0.001;
        let radii: Vec<f64> = (1..=20).map(|i| i as f64 * 5.0).collect();
        let counts: Vec<f64> = radii.iter().map(|&r| nbar * volume(r)).collect();
        let cp = CumulativeProfile { radii, counts };

        for r in [10.0, 30.0, 50.0, 80.0] {
            let rho = cp.volume_density(r);
            assert!(
                (rho - nbar).abs() < 1e-10,
                "At r={}, volume_density={}, expected nbar={}",
                r,
                rho,
                nbar,
            );
        }
    }

    #[test]
    fn test_xi_bar_to_xi_power_law() {
        // For ξ(r) = A/r², ξ̄(<r) = 3A/r².
        // The inversion ξ = ξ̄ + (r/3)dξ̄/dr should recover A/r².
        //
        // We test via SG differentiation on a synthetic ξ̄ grid.
        let a = 1000.0;
        let n = 201;
        let r_lo = 5.0;
        let r_hi = 100.0;
        let h = (r_hi - r_lo) / (n - 1) as f64;
        let r_grid: Vec<f64> = (0..n).map(|i| r_lo + i as f64 * h).collect();

        // Synthetic ξ̄(<r) = 3A/r²
        let xi_bar: Vec<f64> = r_grid.iter().map(|&r| 3.0 * a / (r * r)).collect();

        let (xi_bar_smooth, dxi_bar_dr) = savgol::sg_smooth_diff(&xi_bar, h, 5, 3);

        // ξ(r) = ξ̄ + (r/3) dξ̄/dr
        let xi_recovered: Vec<f64> = r_grid
            .iter()
            .enumerate()
            .map(|(i, &r)| xi_bar_smooth[i] + (r / 3.0) * dxi_bar_dr[i])
            .collect();

        // Deep interior: very tight tolerance (SG is exact for polynomials
        // up to its order; the 1/r² curve is nearly polynomial locally)
        for i in 20..(n - 20) {
            let r = r_grid[i];
            let expected = a / (r * r);
            let rel_err = (xi_recovered[i] - expected).abs() / expected;
            assert!(
                rel_err < 1e-4,
                "At r={:.1}, xi_recovered={:.6}, expected={:.6}, rel_err={:.2e}",
                r,
                xi_recovered[i],
                expected,
                rel_err,
            );
        }
        // Near-boundary region (i=10..20): looser tolerance, still good
        for i in 10..20 {
            let r = r_grid[i];
            let expected = a / (r * r);
            let rel_err = (xi_recovered[i] - expected).abs() / expected;
            assert!(
                rel_err < 5e-3,
                "Near-boundary r={:.1}: rel_err={:.2e}",
                r,
                rel_err,
            );
        }
    }

    #[test]
    fn test_xi_bar_to_xi_coxmock_form() {
        // CoxMock: ξ(r) = C/r²·(1 − r/ℓ), ξ̄(<r) = 3C/r²·(1 − r/(2ℓ))
        let c = 500.0;
        let ell = 200.0;
        let n = 301;
        let r_lo = 3.0;
        let r_hi = 150.0;
        let h = (r_hi - r_lo) / (n - 1) as f64;
        let r_grid: Vec<f64> = (0..n).map(|i| r_lo + i as f64 * h).collect();

        let xi_bar: Vec<f64> = r_grid
            .iter()
            .map(|&r| 3.0 * c / (r * r) * (1.0 - r / (2.0 * ell)))
            .collect();

        let (xi_bar_smooth, dxi_bar_dr) = savgol::sg_smooth_diff(&xi_bar, h, 5, 3);

        let xi_recovered: Vec<f64> = r_grid
            .iter()
            .enumerate()
            .map(|(i, &r)| xi_bar_smooth[i] + (r / 3.0) * dxi_bar_dr[i])
            .collect();

        // Interior: the C/r²·(1−r/ℓ) form has steeper curvature than
        // a pure power law near small r, so SG accuracy is ~2e-4 there.
        for i in 20..(n - 20) {
            let r = r_grid[i];
            let expected = c / (r * r) * (1.0 - r / ell);
            let rel_err = if expected.abs() > 1e-6 {
                (xi_recovered[i] - expected).abs() / expected.abs()
            } else {
                (xi_recovered[i] - expected).abs()
            };
            assert!(
                rel_err < 5e-4,
                "At r={:.1}, xi_recovered={:.6}, expected={:.6}, rel_err={:.2e}",
                r,
                xi_recovered[i],
                expected,
                rel_err,
            );
        }
        // Near-boundary: looser tolerance
        for i in 10..20 {
            let r = r_grid[i];
            let expected = c / (r * r) * (1.0 - r / ell);
            let rel_err = if expected.abs() > 1e-6 {
                (xi_recovered[i] - expected).abs() / expected.abs()
            } else {
                (xi_recovered[i] - expected).abs()
            };
            assert!(
                rel_err < 5e-3,
                "Near-boundary r={:.1}: rel_err={:.2e}",
                r,
                rel_err,
            );
        }
    }

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

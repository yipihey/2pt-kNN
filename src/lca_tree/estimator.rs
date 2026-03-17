//! Single-traversal ξ estimator using the LCA decomposition.
//!
//! For each internal node v:
//!   α = W_D_total / W_R_total
//!   δ_L = W_D_L - α × W_R_L
//!   δ_R = W_D_R - α × W_R_R
//!   n_v = W_R_L × W_R_R
//!   f_j = bin_fractions(bbox_L, bbox_R, ...)
//!
//!   numerator[j]   += δ_L × δ_R × f_j
//!   denominator[j] += n_v × f_j
//!
//! Result: ξ[j] = numerator[j] / denominator[j]
//!
//! Uses Kahan compensated summation for numerical stability.

use super::LcaTree;
use super::kernel;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the LCA ξ estimator.
#[derive(Debug, Clone)]
pub struct LcaEstimatorConfig {
    /// Radial bin edges (physical units, not squared). Length = n_bins + 1.
    pub bin_edges: Vec<f64>,
    /// Number of MC samples per box for the geometric kernel.
    pub mc_samples: usize,
    /// Box size for periodic boundary conditions (None = non-periodic).
    pub box_size: Option<f64>,
    /// RNG seed for MC kernel.
    pub seed: u64,
}

impl LcaEstimatorConfig {
    pub fn new(bin_edges: Vec<f64>, box_size: Option<f64>) -> Self {
        Self {
            bin_edges,
            mc_samples: 512,
            box_size,
            seed: 12345,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of the LCA ξ estimation.
#[derive(Debug, Clone)]
pub struct LcaXiEstimate {
    /// Bin centers (geometric mean of edges).
    pub r: Vec<f64>,
    /// Bin edges.
    pub bin_edges: Vec<f64>,
    /// Landy-Szalay ξ estimate per bin.
    pub xi: Vec<f64>,
    /// Raw numerator per bin (for diagnostics).
    pub numerator: Vec<f64>,
    /// Raw denominator per bin (for diagnostics).
    pub denominator: Vec<f64>,
    /// Number of internal nodes processed.
    pub n_nodes_processed: usize,
    /// Number of nodes skipped (all fractions zero).
    pub n_nodes_skipped: usize,
}

// ---------------------------------------------------------------------------
// Kahan accumulator
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct KahanAcc {
    sum: f64,
    comp: f64,
}

impl KahanAcc {
    fn new() -> Self { Self { sum: 0.0, comp: 0.0 } }

    #[inline]
    fn add(&mut self, val: f64) {
        let y = val - self.comp;
        let t = self.sum + y;
        self.comp = (t - self.sum) - y;
        self.sum = t;
    }

    fn value(&self) -> f64 { self.sum }
}

// ---------------------------------------------------------------------------
// Estimator
// ---------------------------------------------------------------------------

/// Run the single-traversal LCA ξ estimator.
///
/// The tree must have been built with `LcaTree::build` and randoms inserted
/// via `randoms::insert_randoms`.
pub fn estimate_xi(tree: &LcaTree, config: &LcaEstimatorConfig) -> LcaXiEstimate {
    let n_bins = config.bin_edges.len() - 1;
    let mut num_acc: Vec<KahanAcc> = (0..n_bins).map(|_| KahanAcc::new()).collect();
    let mut den_acc: Vec<KahanAcc> = (0..n_bins).map(|_| KahanAcc::new()).collect();

    // Global ratio α = W_D / W_R.
    let w_d_total = tree.total_data_weight();
    let w_r_total = tree.total_rand_weight();

    if w_r_total == 0.0 || w_d_total == 0.0 {
        return empty_result(&config.bin_edges);
    }

    let alpha = w_d_total / w_r_total;

    let mut n_processed = 0usize;
    let mut n_skipped = 0usize;

    // Traverse all internal nodes (skip sentinel at index 0).
    for (node_idx, node) in tree.nodes.iter().enumerate().skip(1) {
        // Overdensity signals.
        let delta_l = node.data_mono_left.w - alpha * node.rand_mono_left.w;
        let delta_r = node.data_mono_right.w - alpha * node.rand_mono_right.w;

        // Pair normalization from randoms.
        let n_v = node.rand_mono_left.w * node.rand_mono_right.w;

        // Skip if no random pairs at this node.
        if n_v == 0.0 {
            n_skipped += 1;
            continue;
        }

        // Compute geometric bin fractions via MC.
        let per_node_seed = config.seed ^ (node_idx as u64);
        let fracs = kernel::bin_fractions_mc(
            &node.bbox_left,
            &node.bbox_right,
            &config.bin_edges,
            config.mc_samples,
            config.box_size,
            per_node_seed,
        );

        // Check if all fractions are zero.
        let all_zero = fracs.iter().all(|&f| f == 0.0);
        if all_zero {
            n_skipped += 1;
            continue;
        }

        // Accumulate into bins.
        for j in 0..n_bins {
            if fracs[j] > 0.0 {
                num_acc[j].add(delta_l * delta_r * fracs[j]);
                den_acc[j].add(n_v * fracs[j]);
            }
        }

        n_processed += 1;
    }

    // Compute ξ = numerator / denominator.
    let numerator: Vec<f64> = num_acc.iter().map(|a| a.value()).collect();
    let denominator: Vec<f64> = den_acc.iter().map(|a| a.value()).collect();

    let xi: Vec<f64> = numerator
        .iter()
        .zip(denominator.iter())
        .map(|(&n, &d)| if d.abs() > 0.0 { n / d } else { 0.0 })
        .collect();

    // Bin centers (geometric mean).
    let r: Vec<f64> = (0..n_bins)
        .map(|i| (config.bin_edges[i] * config.bin_edges[i + 1]).sqrt())
        .collect();

    LcaXiEstimate {
        r,
        bin_edges: config.bin_edges.clone(),
        xi,
        numerator,
        denominator,
        n_nodes_processed: n_processed,
        n_nodes_skipped: n_skipped,
    }
}

fn empty_result(bin_edges: &[f64]) -> LcaXiEstimate {
    let n_bins = bin_edges.len() - 1;
    let r: Vec<f64> = (0..n_bins)
        .map(|i| (bin_edges[i] * bin_edges[i + 1]).sqrt())
        .collect();
    LcaXiEstimate {
        r,
        bin_edges: bin_edges.to_vec(),
        xi: vec![0.0; n_bins],
        numerator: vec![0.0; n_bins],
        denominator: vec![0.0; n_bins],
        n_nodes_processed: 0,
        n_nodes_skipped: 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lca_tree::{LcaTree, randoms};
    use rand::SeedableRng;
    use rand::Rng;

    fn uniform_random(n: usize, box_size: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                [
                    rng.gen::<f64>() * box_size,
                    rng.gen::<f64>() * box_size,
                    rng.gen::<f64>() * box_size,
                ]
            })
            .collect()
    }

    #[test]
    fn poisson_null_xi_near_zero() {
        let box_size = 100.0;
        let n_data = 2_000;
        let n_rand = 4_000;

        let data = uniform_random(n_data, box_size, 42);
        let rand_pos = uniform_random(n_rand, box_size, 43);
        let data_weights = vec![1.0; n_data];
        let rand_weights = vec![1.0; n_rand];

        let mut tree = LcaTree::build(&data, &data_weights, Some(16), Some(box_size));
        randoms::insert_randoms(&mut tree, &rand_pos, &rand_weights);

        let config = LcaEstimatorConfig {
            bin_edges: crate::xi_morton::cell_pairs::log_bin_edges(5.0, 40.0, 8),
            mc_samples: 256,
            box_size: Some(box_size),
            seed: 99,
        };

        let result = estimate_xi(&tree, &config);

        // Poisson → ξ ≈ 0.
        for (r, xi) in result.r.iter().zip(result.xi.iter()) {
            assert!(
                xi.abs() < 0.5,
                "ξ({:.1}) = {:.4}, expected ~0 for Poisson (N_D={}, N_R={})",
                r, xi, n_data, n_rand
            );
        }

        assert!(result.n_nodes_processed > 0, "no nodes were processed");
    }
}

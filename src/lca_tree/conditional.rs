//! Phase 4: Conditional statistics — tidal-environment–dependent clustering.
//!
//! Bins galaxies by their tidal eigenvalues at a coarse scale, then measures
//! counts-in-cells and the two-point function ξ(r) separately in each
//! tidal environment class.
//!
//! ## Conditional counts-in-cells (§4.1)
//!
//! At each fine scale L_f < L_c, measure the counts distribution separately
//! in each tidal bin (Void, Pancake, Filament, Halo).
//!
//! ## Conditional two-point function (§4.2)
//!
//! Using the tree-moment estimator, measure ξ̂_ℓ(s) separately for galaxy
//! pairs whose LCA node falls in each tidal environment class.
//!
//! Key identity: the sum over all environment pairs reproduces the
//! unconditional ξ̂_ℓ(s) exactly (checksum).

use std::collections::HashMap;
use super::LcaTree;
use super::tidal::WebType;
use super::estimator::LcaEstimatorConfig;
use super::kernel;

// ---------------------------------------------------------------------------
// Tidal binning
// ---------------------------------------------------------------------------

/// Tidal environment binning configuration.
#[derive(Debug, Clone)]
pub struct TidalBinning {
    /// Growth factor D₊ for classification.
    pub d_plus: f64,
    /// Whether to use 2LPT classification (vs 1LPT).
    pub use_2lpt: bool,
}

impl Default for TidalBinning {
    fn default() -> Self {
        Self { d_plus: 1.0, use_2lpt: false }
    }
}

impl TidalBinning {
    /// Classify eigenvalues into a WebType.
    pub fn classify(&self, eigenvalues: &[f64; 3]) -> WebType {
        if self.use_2lpt {
            WebType::classify_2lpt(eigenvalues, self.d_plus)
        } else {
            WebType::classify_1lpt(eigenvalues, self.d_plus)
        }
    }
}

// ---------------------------------------------------------------------------
// Conditional counts-in-cells
// ---------------------------------------------------------------------------

/// Counts distribution in a single tidal environment bin.
#[derive(Debug, Clone)]
pub struct CountsDistribution {
    /// Number of cells in this environment.
    pub n_cells: usize,
    /// Total number of galaxies in these cells.
    pub n_galaxies: usize,
    /// Mean count per cell.
    pub mean: f64,
    /// Variance of counts.
    pub variance: f64,
    /// Histogram of counts: counts_hist[k] = number of cells with k galaxies.
    pub counts_hist: Vec<usize>,
}

/// Compute conditional counts-in-cells from per-particle tidal classifications.
///
/// Given a tree (which partitions particles into leaves), count galaxies per leaf
/// and group by the tidal environment of each leaf (determined by the mean
/// tidal eigenvalues of particles in that leaf).
///
/// # Arguments
/// - `tree`: built KD-tree
/// - `particle_eigenvalues`: per-particle tidal eigenvalues [λ₁, λ₂, λ₃]
/// - `binning`: tidal binning configuration
pub fn conditional_counts_in_cells(
    tree: &LcaTree,
    particle_eigenvalues: &[[f64; 3]],
    binning: &TidalBinning,
) -> HashMap<WebType, CountsDistribution> {
    // Collect leaf spans.
    let leaves = collect_leaf_spans(tree);
    let mut env_counts: HashMap<WebType, Vec<usize>> = HashMap::new();

    for &(start, end) in &leaves {
        let n_particles = end - start;
        if n_particles == 0 { continue; }

        // Mean eigenvalues in this leaf.
        let mut mean_eig = [0.0f64; 3];
        for i in start..end {
            for k in 0..3 {
                mean_eig[k] += particle_eigenvalues[i][k];
            }
        }
        for k in 0..3 {
            mean_eig[k] /= n_particles as f64;
        }

        let web_type = binning.classify(&mean_eig);
        env_counts.entry(web_type).or_default().push(n_particles);
    }

    let mut result = HashMap::new();
    for (wt, counts) in env_counts {
        let n_cells = counts.len();
        let n_galaxies: usize = counts.iter().sum();
        let mean = n_galaxies as f64 / n_cells as f64;
        let variance = if n_cells > 1 {
            counts.iter().map(|&c| (c as f64 - mean).powi(2)).sum::<f64>() / (n_cells - 1) as f64
        } else {
            0.0
        };

        let max_count = counts.iter().copied().max().unwrap_or(0);
        let mut hist = vec![0usize; max_count + 1];
        for &c in &counts {
            hist[c] += 1;
        }

        result.insert(wt, CountsDistribution {
            n_cells,
            n_galaxies,
            mean,
            variance,
            counts_hist: hist,
        });
    }

    result
}

// ---------------------------------------------------------------------------
// Conditional two-point function
// ---------------------------------------------------------------------------

/// Environment pair key for conditional ξ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnvPair(pub WebType, pub WebType);

impl EnvPair {
    /// Canonical ordering: (min, max) by discriminant.
    pub fn canonical(a: WebType, b: WebType) -> Self {
        let da = a as u8;
        let db = b as u8;
        if da <= db { EnvPair(a, b) } else { EnvPair(b, a) }
    }
}

/// Conditional ξ measurement in one environment pair.
#[derive(Debug, Clone)]
pub struct ConditionalXi {
    /// Bin centers.
    pub r: Vec<f64>,
    /// ξ estimate per bin.
    pub xi: Vec<f64>,
    /// Numerator per bin.
    pub numerator: Vec<f64>,
    /// Denominator per bin.
    pub denominator: Vec<f64>,
    /// Number of nodes contributing to this pair.
    pub n_nodes: usize,
}

/// Estimate ξ separately for each environment pair.
///
/// At each internal node v, determine the tidal environment of the left
/// and right subtrees (from the mean eigenvalues of particles), compute
/// the environment pair, and accumulate into the corresponding ξ estimator.
///
/// The sum over all environment pairs reproduces the unconditional ξ exactly.
///
/// # Arguments
/// - `tree`: built KD-tree with randoms inserted
/// - `node_web_types`: pre-computed web type for each internal node's left and right children
/// - `config`: estimator configuration (bin edges, MC samples, etc.)
pub fn conditional_xi(
    tree: &LcaTree,
    per_node_env: &[(WebType, WebType)],  // indexed by node index, (left_env, right_env)
    config: &LcaEstimatorConfig,
) -> HashMap<EnvPair, ConditionalXi> {
    let n_bins = config.bin_edges.len() - 1;
    let w_d_total = tree.total_data_weight();
    let w_r_total = tree.total_rand_weight();

    if w_r_total == 0.0 || w_d_total == 0.0 {
        return HashMap::new();
    }
    let alpha = w_d_total / w_r_total;

    // Accumulators per environment pair.
    let mut accumulators: HashMap<EnvPair, (Vec<f64>, Vec<f64>, usize)> = HashMap::new();

    for (node_idx, node) in tree.nodes.iter().enumerate().skip(1) {
        if node_idx >= per_node_env.len() { break; }

        let delta_l = node.data_mono_left.w - alpha * node.rand_mono_left.w;
        let delta_r = node.data_mono_right.w - alpha * node.rand_mono_right.w;
        let n_v = node.rand_mono_left.w * node.rand_mono_right.w;
        if n_v == 0.0 { continue; }

        let per_node_seed = config.seed ^ (node_idx as u64);
        let fracs = kernel::bin_fractions_mc(
            &node.bbox_left,
            &node.bbox_right,
            &config.bin_edges,
            config.mc_samples,
            config.box_size,
            per_node_seed,
        );

        if fracs.iter().all(|&f| f == 0.0) { continue; }

        let (env_l, env_r) = per_node_env[node_idx];
        let pair = EnvPair::canonical(env_l, env_r);

        let (num, den, count) = accumulators.entry(pair).or_insert_with(|| {
            (vec![0.0; n_bins], vec![0.0; n_bins], 0)
        });

        for j in 0..n_bins {
            if fracs[j] > 0.0 {
                num[j] += delta_l * delta_r * fracs[j];
                den[j] += n_v * fracs[j];
            }
        }
        *count += 1;
    }

    // Convert to ConditionalXi.
    let r: Vec<f64> = (0..n_bins)
        .map(|i| (config.bin_edges[i] * config.bin_edges[i + 1]).sqrt())
        .collect();

    accumulators.into_iter().map(|(pair, (num, den, count))| {
        let xi: Vec<f64> = num.iter().zip(den.iter())
            .map(|(&n, &d)| if d.abs() > 0.0 { n / d } else { 0.0 })
            .collect();
        (pair, ConditionalXi {
            r: r.clone(),
            xi,
            numerator: num,
            denominator: den,
            n_nodes: count,
        })
    }).collect()
}

/// Classify each internal node's left/right subtrees by mean tidal eigenvalues.
///
/// Returns a parallel array indexed by node index: `(left_web_type, right_web_type)`.
/// Index 0 (sentinel) gets (Void, Void).
pub fn classify_nodes(
    tree: &LcaTree,
    particle_eigenvalues: &[[f64; 3]],
    binning: &TidalBinning,
) -> Vec<(WebType, WebType)> {
    let n_nodes = tree.nodes.len();
    let mut result = vec![(WebType::Void, WebType::Void); n_nodes];

    for (idx, node) in tree.nodes.iter().enumerate().skip(1) {
        let start = node.particle_start as usize;
        let end = node.particle_end as usize;
        let mid = start + (end - start) / 2;

        let left_env = mean_web_type(particle_eigenvalues, start, mid, binning);
        let right_env = mean_web_type(particle_eigenvalues, mid, end, binning);
        result[idx] = (left_env, right_env);
    }

    result
}

/// Mean eigenvalues → web type for a particle range.
fn mean_web_type(
    eigenvalues: &[[f64; 3]],
    start: usize,
    end: usize,
    binning: &TidalBinning,
) -> WebType {
    if start >= end { return WebType::Void; }
    let n = (end - start) as f64;
    let mut mean = [0.0f64; 3];
    for i in start..end {
        for k in 0..3 {
            mean[k] += eigenvalues[i][k];
        }
    }
    for k in 0..3 { mean[k] /= n; }
    binning.classify(&mean)
}

/// Verify that conditional ξ sums to unconditional ξ (checksum).
///
/// Returns the maximum relative error across all bins.
pub fn conditional_xi_checksum(
    conditional: &HashMap<EnvPair, ConditionalXi>,
    unconditional_num: &[f64],
    unconditional_den: &[f64],
) -> f64 {
    let n_bins = unconditional_num.len();
    let mut sum_num = vec![0.0f64; n_bins];
    let mut sum_den = vec![0.0f64; n_bins];

    for cxi in conditional.values() {
        for j in 0..n_bins {
            sum_num[j] += cxi.numerator[j];
            sum_den[j] += cxi.denominator[j];
        }
    }

    let mut max_err = 0.0f64;
    for j in 0..n_bins {
        if unconditional_den[j].abs() > 1e-30 {
            let err_num = (sum_num[j] - unconditional_num[j]).abs()
                / unconditional_num[j].abs().max(1e-30);
            let err_den = (sum_den[j] - unconditional_den[j]).abs()
                / unconditional_den[j].abs().max(1e-30);
            max_err = max_err.max(err_num).max(err_den);
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn collect_leaf_spans(tree: &LcaTree) -> Vec<(usize, usize)> {
    let mut leaves = Vec::new();
    if tree.nodes.len() <= 1 {
        if !tree.weights.is_empty() {
            leaves.push((0, tree.weights.len()));
        }
        return leaves;
    }
    for node in tree.nodes.iter().skip(1) {
        let start = node.particle_start as usize;
        let end = node.particle_end as usize;
        let mid = start + (end - start) / 2;
        if node.left == 0 && mid > start { leaves.push((start, mid)); }
        if node.right == 0 && end > mid { leaves.push((mid, end)); }
    }
    leaves
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
        (0..n).map(|_| {
            [rng.gen::<f64>() * box_size, rng.gen::<f64>() * box_size, rng.gen::<f64>() * box_size]
        }).collect()
    }

    #[test]
    fn conditional_cic_basic() {
        let positions = uniform_random(200, 100.0, 42);
        let weights = vec![1.0; 200];
        let tree = LcaTree::build(&positions, &weights, Some(8), None);

        // Assign random eigenvalues (all small → all Void for d_plus=1).
        let eigenvalues: Vec<[f64; 3]> = vec![[0.1, 0.05, 0.01]; 200];
        let binning = TidalBinning::default();
        let result = conditional_counts_in_cells(&tree, &eigenvalues, &binning);

        // All should be Void.
        assert!(result.contains_key(&WebType::Void));
        let void_counts = &result[&WebType::Void];
        assert!(void_counts.n_cells > 0);
        assert_eq!(void_counts.n_galaxies, 200);
    }

    #[test]
    fn classify_nodes_basic() {
        let positions = uniform_random(100, 100.0, 42);
        let weights = vec![1.0; 100];
        let tree = LcaTree::build(&positions, &weights, Some(8), None);

        let eigenvalues: Vec<[f64; 3]> = vec![[0.1, 0.05, 0.01]; 100];
        let binning = TidalBinning::default();
        let node_envs = classify_nodes(&tree, &eigenvalues, &binning);

        assert_eq!(node_envs.len(), tree.nodes.len());
        // All should be Void for these small eigenvalues.
        for &(l, r) in node_envs.iter().skip(1) {
            assert_eq!(l, WebType::Void);
            assert_eq!(r, WebType::Void);
        }
    }

    #[test]
    fn conditional_xi_sums_to_unconditional() {
        let box_size = 100.0;
        let n_data = 500;
        let n_rand = 1000;

        let data = uniform_random(n_data, box_size, 42);
        let rand_pos = uniform_random(n_rand, box_size, 43);
        let data_weights = vec![1.0; n_data];
        let rand_weights = vec![1.0; n_rand];

        let mut tree = LcaTree::build(&data, &data_weights, Some(16), Some(box_size));
        randoms::insert_randoms(&mut tree, &rand_pos, &rand_weights);

        let config = LcaEstimatorConfig {
            bin_edges: crate::xi_morton::cell_pairs::log_bin_edges(5.0, 40.0, 5),
            mc_samples: 128,
            box_size: Some(box_size),
            seed: 99,
        };

        // Unconditional ξ.
        let uncond = super::super::estimator::estimate_xi(&tree, &config);

        // Assign mixed eigenvalues to get multiple web types.
        let eigenvalues: Vec<[f64; 3]> = (0..n_data)
            .map(|i| {
                if i % 3 == 0 { [2.0, 1.5, 0.1] }  // Filament
                else if i % 3 == 1 { [0.1, 0.05, 0.01] }  // Void
                else { [3.0, 2.0, 1.5] }  // Halo
            })
            .collect();

        let binning = TidalBinning::default();
        let node_envs = classify_nodes(&tree, &eigenvalues, &binning);
        let cond = conditional_xi(&tree, &node_envs, &config);

        let checksum_err = conditional_xi_checksum(
            &cond, &uncond.numerator, &uncond.denominator,
        );

        assert!(
            checksum_err < 1e-10,
            "conditional ξ checksum error: {:.2e}",
            checksum_err
        );
    }
}

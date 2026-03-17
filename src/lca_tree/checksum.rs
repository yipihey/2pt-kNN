//! Pair-weight verification (Eq. 2 of Abel et al. 2026).
//!
//! After building the tree, verify that the sum of left × right weight products
//! over all internal nodes, plus intra-leaf pair weights, equals the expected
//! total pair weight:
//!
//! ```text
//! Σ_v W_L(v) × W_R(v) + Σ_leaf Σ_{i<j} w_i·w_j = (W_total² − Σ w_i²) / 2
//! ```
//!
//! With leaf_size > 1, pairs within a leaf have no LCA node, so the identity
//! must include the leaf contribution.
//!
//! Uses Kahan compensated summation for machine-precision accuracy.

use super::LcaTree;

// ---------------------------------------------------------------------------
// Kahan accumulator (duplicated from xi_morton for independence)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct KahanAccumulator {
    sum: f64,
    comp: f64,
}

impl KahanAccumulator {
    fn new() -> Self {
        Self { sum: 0.0, comp: 0.0 }
    }

    #[inline]
    fn add(&mut self, val: f64) {
        let y = val - self.comp;
        let t = self.sum + y;
        self.comp = (t - self.sum) - y;
        self.sum = t;
    }

    fn value(&self) -> f64 {
        self.sum
    }
}

// ---------------------------------------------------------------------------
// Checksum
// ---------------------------------------------------------------------------

/// Result of the pair-weight checksum.
#[derive(Debug)]
pub struct ChecksumResult {
    /// Sum of W_L × W_R over internal nodes + intra-leaf pairs.
    pub tree_sum: f64,
    /// Expected value: (W_total² − Σ w_i²) / 2.
    pub expected: f64,
    /// Relative error: |tree_sum - expected| / expected.
    pub relative_error: f64,
    /// Whether the checksum passed (relative error < threshold).
    pub passed: bool,
}

/// Compute the sum of w_i × w_j for all i < j within a particle span.
fn intra_leaf_pair_weight(weights: &[f64], start: usize, end: usize) -> f64 {
    let mut acc = KahanAccumulator::new();
    for i in start..end {
        for j in (i + 1)..end {
            acc.add(weights[i] * weights[j]);
        }
    }
    acc.value()
}

/// Collect leaf spans from the tree.
///
/// A leaf is a partition [start..end] that is not further split.
/// For each internal node, if left=0 then [start..mid] is a leaf,
/// and if right=0 then [mid..end] is a leaf.
/// Also, if the root doesn't exist (all particles in one leaf), the
/// entire array is a leaf.
fn collect_leaf_spans(tree: &LcaTree) -> Vec<(usize, usize)> {
    let mut leaves = Vec::new();

    if tree.nodes.len() <= 1 {
        // No internal nodes — entire array is one leaf.
        if !tree.weights.is_empty() {
            leaves.push((0, tree.weights.len()));
        }
        return leaves;
    }

    // For each internal node, check if children are leaves.
    for node in tree.nodes.iter().skip(1) {
        let start = node.particle_start as usize;
        let end = node.particle_end as usize;
        let mid = start + (end - start) / 2;

        if node.left == 0 {
            // Left child is a leaf: [start..mid]
            if mid > start {
                leaves.push((start, mid));
            }
        }
        if node.right == 0 {
            // Right child is a leaf: [mid..end]
            if end > mid {
                leaves.push((mid, end));
            }
        }
    }

    leaves
}

/// Verify the pair-weight identity on the data monopoles.
///
/// Checks: Σ_v W_D_L × W_D_R + Σ_leaf pairs = (W² − Σ w_i²) / 2
///
/// `threshold`: maximum allowed relative error (typically 1e-12).
pub fn verify_data_checksum(tree: &LcaTree, threshold: f64) -> ChecksumResult {
    // Internal node contribution: Σ W_D_L × W_D_R.
    let mut tree_acc = KahanAccumulator::new();
    for node in tree.nodes.iter().skip(1) {
        tree_acc.add(node.data_mono_left.w * node.data_mono_right.w);
    }

    // Leaf contribution: intra-leaf pair weights.
    let leaves = collect_leaf_spans(tree);
    for &(start, end) in &leaves {
        tree_acc.add(intra_leaf_pair_weight(&tree.weights, start, end));
    }

    let tree_sum = tree_acc.value();

    // Expected: (W_total² − Σ w_i²) / 2.
    let mut w_total_acc = KahanAccumulator::new();
    let mut w_sq_acc = KahanAccumulator::new();
    for &w in &tree.weights {
        w_total_acc.add(w);
        w_sq_acc.add(w * w);
    }
    let w_total = w_total_acc.value();
    let w_sq_sum = w_sq_acc.value();
    let expected = (w_total * w_total - w_sq_sum) / 2.0;

    let relative_error = if expected.abs() > 0.0 {
        (tree_sum - expected).abs() / expected.abs()
    } else {
        tree_sum.abs()
    };

    ChecksumResult {
        tree_sum,
        expected,
        relative_error,
        passed: relative_error < threshold,
    }
}

/// Verify the pair-weight identity on the random monopoles.
///
/// Should be called after `randoms::insert_randoms`.
/// Note: for randoms, we don't have per-leaf pair weights readily available
/// since randoms aren't stored in the tree. This verifies only the internal
/// node contribution, which is exact when leaf_size = 1.
pub fn verify_random_checksum(tree: &LcaTree, rand_weights: &[f64], threshold: f64) -> ChecksumResult {
    // For randoms, the tree structure doesn't perfectly partition them into
    // leaves the same way, so we check a looser bound.
    // The internal node sum should be close to the total pair weight.
    let mut tree_acc = KahanAccumulator::new();
    for node in tree.nodes.iter().skip(1) {
        tree_acc.add(node.rand_mono_left.w * node.rand_mono_right.w);
    }
    let tree_sum = tree_acc.value();

    let mut w_total_acc = KahanAccumulator::new();
    let mut w_sq_acc = KahanAccumulator::new();
    for &w in rand_weights {
        w_total_acc.add(w);
        w_sq_acc.add(w * w);
    }
    let w_total = w_total_acc.value();
    let w_sq_sum = w_sq_acc.value();
    let expected = (w_total * w_total - w_sq_sum) / 2.0;

    let relative_error = if expected.abs() > 0.0 {
        (tree_sum - expected).abs() / expected.abs()
    } else {
        tree_sum.abs()
    };

    ChecksumResult {
        tree_sum,
        expected,
        relative_error,
        passed: relative_error < threshold,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lca_tree::LcaTree;

    #[test]
    fn checksum_uniform_weights() {
        // 100 particles with weight 1.0 each.
        // Expected: (100² - 100×1²) / 2 = (10000 - 100) / 2 = 4950.
        let positions: Vec<[f64; 3]> = (0..100)
            .map(|i| {
                let f = i as f64;
                [f, f * 0.5, f * 0.3]
            })
            .collect();
        let weights = vec![1.0; 100];

        let tree = LcaTree::build(&positions, &weights, Some(4), None);
        let result = verify_data_checksum(&tree, 1e-12);

        assert!(
            result.passed,
            "checksum failed: tree_sum={}, expected={}, rel_err={}",
            result.tree_sum, result.expected, result.relative_error
        );
        assert!((result.expected - 4950.0).abs() < 1e-10);
    }

    #[test]
    fn checksum_varied_weights() {
        let positions: Vec<[f64; 3]> = (0..50)
            .map(|i| {
                let f = i as f64;
                [f * 2.0, f * 3.0, f * 1.5]
            })
            .collect();
        let weights: Vec<f64> = (0..50).map(|i| (i + 1) as f64).collect();

        let tree = LcaTree::build(&positions, &weights, Some(4), None);
        let result = verify_data_checksum(&tree, 1e-12);

        let w_total: f64 = weights.iter().sum();
        let w_sq: f64 = weights.iter().map(|w| w * w).sum();
        let expected = (w_total * w_total - w_sq) / 2.0;

        assert!(
            result.passed,
            "checksum failed: tree_sum={}, expected={}, rel_err={}",
            result.tree_sum, result.expected, result.relative_error
        );
        assert!((result.expected - expected).abs() < 1e-6);
    }

    #[test]
    fn checksum_leaf_size_1() {
        // With leaf_size=1, the tree captures all pairs.
        let positions: Vec<[f64; 3]> = (0..20)
            .map(|i| {
                let f = i as f64;
                [f, f * 0.7, f * 0.3]
            })
            .collect();
        let weights = vec![1.0; 20];

        let tree = LcaTree::build(&positions, &weights, Some(1), None);
        let result = verify_data_checksum(&tree, 1e-12);

        assert!(
            result.passed,
            "checksum failed with leaf_size=1: tree_sum={}, expected={}, rel_err={}",
            result.tree_sum, result.expected, result.relative_error
        );
    }
}

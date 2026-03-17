//! Random particle insertion into the LCA tree.
//!
//! For each random particle, descend from the root comparing
//! `pos[split_axis]` vs `split_val` to accumulate weight into
//! `rand_mono_left` or `rand_mono_right` at each visited node.
//!
//! Cost: O(N_R × log(N_D / leaf_size)).

use super::LcaTree;

/// Insert random particles into the tree, populating `rand_mono_left` and
/// `rand_mono_right` on every internal node.
///
/// Must be called after `LcaTree::build` and before `estimator::estimate_xi`.
pub fn insert_randoms(
    tree: &mut LcaTree,
    positions: &[[f64; 3]],
    weights: &[f64],
) {
    assert_eq!(positions.len(), weights.len());

    // Reset all random monopoles.
    for node in tree.nodes.iter_mut().skip(1) {
        node.rand_mono_left.w = 0.0;
        node.rand_mono_right.w = 0.0;
    }

    let root_idx = match tree.root() {
        Some(r) => r,
        None => return, // no internal nodes
    };

    for (i, pos) in positions.iter().enumerate() {
        let w = weights[i];
        insert_one(tree, root_idx, pos, w);
    }
}

/// Descend from `node_idx`, at each internal node deciding left or right
/// based on the split, and accumulating the random weight.
fn insert_one(tree: &mut LcaTree, node_idx: usize, pos: &[f64; 3], weight: f64) {
    let mut idx = node_idx;

    loop {
        let node = &tree.nodes[idx];
        let axis = node.split_axis as usize;
        let split = node.split_val;

        let go_left = pos[axis] < split;

        // Accumulate weight into the appropriate side.
        if go_left {
            tree.nodes[idx].rand_mono_left.w += weight;
        } else {
            tree.nodes[idx].rand_mono_right.w += weight;
        }

        // Descend to child.
        let child = if go_left {
            tree.nodes[idx].left
        } else {
            tree.nodes[idx].right
        };

        if child == 0 {
            break; // reached a leaf
        }

        idx = child as usize;
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
    fn insert_randoms_weight_conservation() {
        // Build tree from data.
        let positions: Vec<[f64; 3]> = (0..64)
            .map(|i| {
                let f = i as f64;
                [f, f * 0.5, f * 0.3]
            })
            .collect();
        let weights = vec![1.0; 64];
        let mut tree = LcaTree::build(&positions, &weights, Some(4), None);

        // Insert randoms (same positions, different weights).
        let rand_positions: Vec<[f64; 3]> = (0..100)
            .map(|i| {
                let f = i as f64 * 0.63;
                [f, f * 0.4, f * 0.25]
            })
            .collect();
        let rand_weights = vec![2.0; 100];

        insert_randoms(&mut tree, &rand_positions, &rand_weights);

        // At the root, total random weight should equal Σ rand_weights = 200.
        let root = &tree.nodes[tree.root().unwrap()];
        let total_rand = root.rand_mono_left.w + root.rand_mono_right.w;
        assert!(
            (total_rand - 200.0).abs() < 1e-10,
            "total random weight = {total_rand}, expected 200.0"
        );
    }

    #[test]
    fn insert_randoms_all_left_or_right() {
        // All data on the right, all randoms on the left.
        let data_pos: Vec<[f64; 3]> = (0..40)
            .map(|i| [50.0 + i as f64, 0.0, 0.0])
            .collect();
        let data_wts = vec![1.0; 40];
        let mut tree = LcaTree::build(&data_pos, &data_wts, Some(4), None);

        // Randoms all at x < split_val of root.
        let root = &tree.nodes[tree.root().unwrap()];
        let split = root.split_val;

        let rand_pos: Vec<[f64; 3]> = (0..20)
            .map(|_| [split - 10.0, 0.0, 0.0])
            .collect();
        let rand_wts = vec![1.0; 20];

        insert_randoms(&mut tree, &rand_pos, &rand_wts);

        let root = &tree.nodes[tree.root().unwrap()];
        // All randoms should go left at the root.
        assert!(
            (root.rand_mono_left.w - 20.0).abs() < 1e-10,
            "rand_mono_left = {}, expected 20.0",
            root.rand_mono_left.w
        );
    }
}

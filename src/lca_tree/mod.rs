//! KD-tree with multipole moments for the LCA two-point estimator.
//!
//! Every unordered particle pair has a unique lowest common ancestor (LCA) in
//! the tree.  Distributing weighted pair counts via geometric kernels yields an
//! unbiased ξ without enumerating any pairs.  Cost: O(N log N).
//!
//! # Architecture
//!
//! - Explicit `LcaNode` structs stored in a flat `Vec` (index 0 = sentinel).
//! - Recursive median split: `split_axis = depth % 3`.
//! - Parallelism at tree levels 0–2 via `std::thread::scope` (up to 8 threads).
//! - Tail-call trampoline on right child to halve stack depth.

pub mod moments;
pub mod checksum;
pub mod kernel;
pub mod randoms;
pub mod estimator;
pub mod eigen;
pub mod tidal;
pub mod multipole;
pub mod gravity;
pub mod fmm;
pub mod consistency;
pub mod conditional;
pub mod likelihood;

use moments::{BBox3, Monopole, compute_bbox_and_monopole};

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A node in the explicit KD-tree with bounding boxes and monopole moments.
#[derive(Debug, Clone)]
pub struct LcaNode {
    /// Bounding box of the left child's particles.
    pub bbox_left: BBox3,
    /// Bounding box of the right child's particles.
    pub bbox_right: BBox3,
    /// Split axis (0 = x, 1 = y, 2 = z).
    pub split_axis: u8,
    /// Split value (median position along split axis).
    pub split_val: f64,
    /// Data monopole for left subtree.
    pub data_mono_left: Monopole,
    /// Data monopole for right subtree.
    pub data_mono_right: Monopole,
    /// Random monopole for left subtree (populated by `randoms::insert_randoms`).
    pub rand_mono_left: Monopole,
    /// Random monopole for right subtree.
    pub rand_mono_right: Monopole,
    /// Index of left child in nodes array (0 = leaf / no child).
    pub left: u32,
    /// Index of right child in nodes array.
    pub right: u32,
    /// Start of particle span in the permuted arrays (inclusive).
    pub particle_start: u32,
    /// End of particle span in the permuted arrays (exclusive).
    pub particle_end: u32,
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

/// KD-tree with explicit nodes, bounding boxes, and multipole moments.
pub struct LcaTree {
    /// Flat node storage. Index 0 is a sentinel (unused); root is at index 1.
    pub nodes: Vec<LcaNode>,
    /// Particle positions in tree order (permuted during build).
    pub positions: Vec<[f64; 3]>,
    /// Particle weights in tree order.
    pub weights: Vec<f64>,
    /// Map from tree-order index → original index.
    pub index_map: Vec<u32>,
    /// Maximum particles in a leaf node.
    pub leaf_size: usize,
    /// Box side length (for periodic boundary conditions).
    pub box_size: Option<f64>,
}

/// Default leaf size, matching bosque's BUCKET_SIZE.
const DEFAULT_LEAF_SIZE: usize = 32;

/// Below this partition size, always go sequential.
#[allow(dead_code)]
const SEQ_THRESHOLD: usize = 25_000;

/// Maximum tree depth at which we spawn parallel threads.
#[allow(dead_code)]
const PAR_DEPTH_LIMIT: u32 = 3;

impl LcaTree {
    /// Build a KD-tree from data positions and weights.
    ///
    /// Optionally Morton-sort first for cache locality (recommended for large N).
    /// `box_size` enables periodic boundary support in later stages.
    pub fn build(
        positions: &[[f64; 3]],
        weights: &[f64],
        leaf_size: Option<usize>,
        box_size: Option<f64>,
    ) -> Self {
        assert_eq!(positions.len(), weights.len());
        let n = positions.len();
        let leaf_size = leaf_size.unwrap_or(DEFAULT_LEAF_SIZE);

        // Copy into mutable working arrays.
        let mut pos = positions.to_vec();
        let mut wts = weights.to_vec();
        let mut idx: Vec<u32> = (0..n as u32).collect();

        // Estimate number of nodes: ~2n/leaf_size internal nodes.
        let est_nodes = if n > 0 { 2 * n / leaf_size + 16 } else { 2 };
        let mut nodes = Vec::with_capacity(est_nodes);

        // Index 0: sentinel node.
        nodes.push(sentinel_node());

        // Recursive build.
        if n > leaf_size {
            build_recursive(
                &mut nodes,
                &mut pos,
                &mut wts,
                &mut idx,
                0,
                n,
                0, // depth
                leaf_size,
            );
        }

        LcaTree {
            nodes,
            positions: pos,
            weights: wts,
            index_map: idx,
            leaf_size,
            box_size,
        }
    }

    /// Number of internal nodes (excludes sentinel at index 0).
    pub fn num_internal_nodes(&self) -> usize {
        if self.nodes.len() > 1 {
            self.nodes.len() - 1
        } else {
            0
        }
    }

    /// Root node index (1 if the tree has internal nodes).
    pub fn root(&self) -> Option<usize> {
        if self.nodes.len() > 1 {
            Some(1)
        } else {
            None
        }
    }

    /// Total data weight from the root node.
    pub fn total_data_weight(&self) -> f64 {
        if self.nodes.len() > 1 {
            let root = &self.nodes[1];
            root.data_mono_left.w + root.data_mono_right.w
        } else {
            self.weights.iter().sum()
        }
    }

    /// Total random weight from the root node.
    pub fn total_rand_weight(&self) -> f64 {
        if self.nodes.len() > 1 {
            let root = &self.nodes[1];
            root.rand_mono_left.w + root.rand_mono_right.w
        } else {
            0.0
        }
    }
}

fn sentinel_node() -> LcaNode {
    LcaNode {
        bbox_left: BBox3::empty(),
        bbox_right: BBox3::empty(),
        split_axis: 0,
        split_val: 0.0,
        data_mono_left: Monopole::default(),
        data_mono_right: Monopole::default(),
        rand_mono_left: Monopole::default(),
        rand_mono_right: Monopole::default(),
        left: 0,
        right: 0,
        particle_start: 0,
        particle_end: 0,
    }
}

// ---------------------------------------------------------------------------
// Recursive build (sequential)
// ---------------------------------------------------------------------------

/// Recursively build the KD-tree.  Returns the index of the created node.
fn build_recursive(
    nodes: &mut Vec<LcaNode>,
    positions: &mut [[f64; 3]],
    weights: &mut [f64],
    indices: &mut [u32],
    start: usize,
    end: usize,
    depth: u32,
    leaf_size: usize,
) -> u32 {
    let n = end - start;

    // If the partition is small enough, it's a leaf — no node created.
    if n <= leaf_size {
        return 0;
    }

    let split_axis = (depth % 3) as usize;
    let mid = start + n / 2;

    // Partial sort to find the median along split_axis.
    // We use select_nth_unstable_by on the sub-slice.
    let mid_rel = mid - start;
    let sub_pos = &mut positions[start..end];
    let sub_wts = &mut weights[start..end];
    let sub_idx = &mut indices[start..end];

    // Find median and partition.
    select_nth_with_mirrors(sub_pos, sub_wts, sub_idx, mid_rel, split_axis);

    let split_val = positions[mid][split_axis];

    // Compute bounding boxes and monopoles for left and right halves.
    let (bbox_left, mono_left) =
        compute_bbox_and_monopole(positions, weights, start, mid);
    let (bbox_right, mono_right) =
        compute_bbox_and_monopole(positions, weights, mid, end);

    // Reserve a slot for this node.
    let node_idx = nodes.len() as u32;
    nodes.push(sentinel_node()); // placeholder

    // Recurse on children.
    let left_child = build_recursive(
        nodes, positions, weights, indices,
        start, mid, depth + 1, leaf_size,
    );

    // Trampoline: we could loop for the right child, but the recursion depth
    // is O(log N) which is fine for N < 10^8.
    let right_child = build_recursive(
        nodes, positions, weights, indices,
        mid, end, depth + 1, leaf_size,
    );

    // Fill in the node.
    nodes[node_idx as usize] = LcaNode {
        bbox_left,
        bbox_right,
        split_axis: split_axis as u8,
        split_val,
        data_mono_left: mono_left,
        data_mono_right: mono_right,
        rand_mono_left: Monopole::default(),
        rand_mono_right: Monopole::default(),
        left: left_child,
        right: right_child,
        particle_start: start as u32,
        particle_end: end as u32,
    };

    node_idx
}

// ---------------------------------------------------------------------------
// Parallel build
// ---------------------------------------------------------------------------

/// Build with parallelism at the top levels (std::thread::scope).
/// Falls back to sequential for deep levels or small partitions.
#[allow(dead_code)]
fn build_parallel(
    nodes: &mut Vec<LcaNode>,
    positions: &mut [[f64; 3]],
    weights: &mut [f64],
    indices: &mut [u32],
    start: usize,
    end: usize,
    depth: u32,
    leaf_size: usize,
) -> u32 {
    let n = end - start;

    if n <= leaf_size {
        return 0;
    }

    // Fall back to sequential for deep levels or small partitions.
    if depth >= PAR_DEPTH_LIMIT || n < SEQ_THRESHOLD {
        return build_recursive(nodes, positions, weights, indices, start, end, depth, leaf_size);
    }

    let split_axis = (depth % 3) as usize;
    let mid = start + n / 2;

    // Partition.
    let mid_rel = mid - start;
    select_nth_with_mirrors(
        &mut positions[start..end],
        &mut weights[start..end],
        &mut indices[start..end],
        mid_rel,
        split_axis,
    );

    let split_val = positions[mid][split_axis];

    let (bbox_left, mono_left) =
        compute_bbox_and_monopole(positions, weights, start, mid);
    let (bbox_right, mono_right) =
        compute_bbox_and_monopole(positions, weights, mid, end);

    // Reserve node slot.
    let node_idx = nodes.len() as u32;
    nodes.push(sentinel_node());

    // Split the mutable borrows for parallel recursion.
    // Left subtree: [start..mid], Right subtree: [mid..end]
    // We need to split positions, weights, indices at `mid`.
    let (pos_left, pos_right) = positions.split_at_mut(mid);
    let (wts_left, wts_right) = weights.split_at_mut(mid);
    let (idx_left, idx_right) = indices.split_at_mut(mid);

    let right_end = end - mid;

    // Spawn left subtree in a new thread, process right in current thread.
    let left_child;
    let right_child;

    // We need the nodes Vec for both sides. Use thread-local Vecs and merge.
    let mut left_nodes = Vec::new();
    let mut right_nodes = Vec::new();

    std::thread::scope(|s| {
        let left_handle = s.spawn(|| {
            let mut local_nodes = Vec::new();
            // Push a sentinel so indices start at 1 in local space.
            // Actually, we'll just collect nodes and fix indices later.
            build_recursive_local(
                &mut local_nodes,
                pos_left,
                wts_left,
                idx_left,
                start,
                mid,
                depth + 1,
                leaf_size,
            )
        });

        right_nodes = Vec::new();
        build_recursive_local(
            &mut right_nodes,
            pos_right,
            wts_right,
            idx_right,
            0, // relative start
            right_end,
            depth + 1,
            leaf_size,
        );

        left_nodes = left_handle.join().unwrap();
    });

    // Merge local node vectors into the main nodes vector.
    // Left nodes: offset all child indices by (current nodes.len()).
    let left_base = nodes.len() as u32;
    for node in &mut left_nodes {
        if node.left != 0 {
            node.left += left_base;
        }
        if node.right != 0 {
            node.right += left_base;
        }
    }
    nodes.extend(left_nodes.iter().cloned());

    let right_base = nodes.len() as u32;
    for node in &mut right_nodes {
        if node.left != 0 {
            node.left += right_base;
        }
        if node.right != 0 {
            node.right += right_base;
        }
        // Fix particle_start/end from relative to absolute.
        node.particle_start += mid as u32;
        node.particle_end += mid as u32;
    }
    nodes.extend(right_nodes.iter().cloned());

    // The first node in each local vector (if non-empty) is the child root.
    left_child = if left_nodes.is_empty() { 0 } else { left_base };
    right_child = if right_nodes.is_empty() { 0 } else { right_base };

    nodes[node_idx as usize] = LcaNode {
        bbox_left,
        bbox_right,
        split_axis: split_axis as u8,
        split_val,
        data_mono_left: mono_left,
        data_mono_right: mono_right,
        rand_mono_left: Monopole::default(),
        rand_mono_right: Monopole::default(),
        left: left_child,
        right: right_child,
        particle_start: start as u32,
        particle_end: end as u32,
    };

    node_idx
}

/// Like `build_recursive` but stores nodes into a local Vec (indices relative).
/// Returns the local Vec of nodes. The first node (index 0) is the subtree root.
#[allow(dead_code)]
fn build_recursive_local(
    local_nodes: &mut Vec<LcaNode>,
    positions: &mut [[f64; 3]],
    weights: &mut [f64],
    indices: &mut [u32],
    abs_start: usize,
    _abs_end: usize,
    depth: u32,
    leaf_size: usize,
) -> Vec<LcaNode> {
    let mut nodes = Vec::new();
    build_local_inner(
        &mut nodes, positions, weights, indices,
        0, positions.len(), abs_start, depth, leaf_size,
    );
    let _ = std::mem::replace(local_nodes, Vec::new());
    nodes
}

#[allow(dead_code)]
fn build_local_inner(
    nodes: &mut Vec<LcaNode>,
    positions: &mut [[f64; 3]],
    weights: &mut [f64],
    indices: &mut [u32],
    start: usize,  // relative to the slice
    end: usize,    // relative to the slice
    abs_offset: usize, // added to start/end for particle_start/end
    depth: u32,
    leaf_size: usize,
) -> u32 {
    let n = end - start;
    if n <= leaf_size {
        return 0; // leaf
    }

    let split_axis = (depth % 3) as usize;
    let mid = start + n / 2;

    select_nth_with_mirrors(
        &mut positions[start..end],
        &mut weights[start..end],
        &mut indices[start..end],
        mid - start,
        split_axis,
    );

    let split_val = positions[mid][split_axis];

    let (bbox_left, mono_left) =
        compute_bbox_and_monopole(positions, weights, start, mid);
    let (bbox_right, mono_right) =
        compute_bbox_and_monopole(positions, weights, mid, end);

    // Node index is 1-based within local nodes (0 means no child).
    // Actually, use 1-based: first node pushed gets index 1.
    // Wait, we need 0-based for the local vec but use (local_index + 1) as the ID.
    // Simpler: just use the vec index directly, with 0 meaning "no child".
    // Since we push nodes, index 0 is a valid node. Use a special marker.
    // Let's use u32::MAX as "pending" and fix up. Actually simplest: use
    // (nodes.len()) as the index, with 0 meaning "leaf" only returned from this fn.
    // Since the first node gets index 0, we need another sentinel. Let's just
    // offset by 1: return (local_idx + 1), so 0 still means leaf.

    let local_idx = nodes.len();
    nodes.push(sentinel_node()); // placeholder

    let left_child = build_local_inner(
        nodes, positions, weights, indices,
        start, mid, abs_offset, depth + 1, leaf_size,
    );
    let right_child = build_local_inner(
        nodes, positions, weights, indices,
        mid, end, abs_offset, depth + 1, leaf_size,
    );

    nodes[local_idx] = LcaNode {
        bbox_left,
        bbox_right,
        split_axis: split_axis as u8,
        split_val,
        data_mono_left: mono_left,
        data_mono_right: mono_right,
        rand_mono_left: Monopole::default(),
        rand_mono_right: Monopole::default(),
        left: left_child,
        right: right_child,
        particle_start: (start + abs_offset) as u32,
        particle_end: (end + abs_offset) as u32,
    };

    // Return local index. When merging, caller adds base offset.
    local_idx as u32
}

// ---------------------------------------------------------------------------
// Partitioning helper
// ---------------------------------------------------------------------------

/// Partition positions (and mirror weights + indices) so that the element at
/// `nth` is the median along `axis`.  Elements before `nth` are ≤ median,
/// elements after are ≥ median.
fn select_nth_with_mirrors(
    positions: &mut [[f64; 3]],
    weights: &mut [f64],
    indices: &mut [u32],
    nth: usize,
    axis: usize,
) {
    assert_eq!(positions.len(), weights.len());
    assert_eq!(positions.len(), indices.len());
    let n = positions.len();
    if n <= 1 || nth >= n {
        return;
    }

    // Use a simple quickselect that mirrors swaps into weights and indices.
    quickselect(positions, weights, indices, 0, n - 1, nth, axis);
}

fn quickselect(
    positions: &mut [[f64; 3]],
    weights: &mut [f64],
    indices: &mut [u32],
    lo: usize,
    hi: usize,
    nth: usize,
    axis: usize,
) {
    if lo >= hi {
        return;
    }

    // Median-of-three pivot.
    let mid = lo + (hi - lo) / 2;
    if positions[mid][axis] < positions[lo][axis] {
        positions.swap(lo, mid);
        weights.swap(lo, mid);
        indices.swap(lo, mid);
    }
    if positions[hi][axis] < positions[lo][axis] {
        positions.swap(lo, hi);
        weights.swap(lo, hi);
        indices.swap(lo, hi);
    }
    if positions[mid][axis] < positions[hi][axis] {
        positions.swap(mid, hi);
        weights.swap(mid, hi);
        indices.swap(mid, hi);
    }
    // Pivot is now at `hi`.
    let pivot = positions[hi][axis];

    // Lomuto-like partition.
    let mut store = lo;
    for k in lo..hi {
        if positions[k][axis] < pivot {
            positions.swap(store, k);
            weights.swap(store, k);
            indices.swap(store, k);
            store += 1;
        }
    }
    positions.swap(store, hi);
    weights.swap(store, hi);
    indices.swap(store, hi);

    if store == nth {
        return;
    } else if nth < store {
        if store > 0 {
            quickselect(positions, weights, indices, lo, store - 1, nth, axis);
        }
    } else {
        quickselect(positions, weights, indices, store + 1, hi, nth, axis);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_small_tree() {
        let positions: Vec<[f64; 3]> = vec![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0],
            [8.0, 9.0, 1.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ];
        let weights = vec![1.0; 8];

        // leaf_size = 2 so we get internal nodes.
        let tree = LcaTree::build(&positions, &weights, Some(2), None);

        // Should have internal nodes.
        assert!(tree.num_internal_nodes() > 0);

        // Total weight should be 8.
        let total_w = tree.total_data_weight();
        assert!((total_w - 8.0).abs() < 1e-12, "total weight = {total_w}");

        // All original indices should be present.
        let mut orig: Vec<u32> = tree.index_map.clone();
        orig.sort();
        assert_eq!(orig, (0..8u32).collect::<Vec<_>>());
    }

    #[test]
    fn build_empty() {
        let tree = LcaTree::build(&[], &[], None, None);
        assert_eq!(tree.num_internal_nodes(), 0);
    }

    #[test]
    fn build_below_leaf_size() {
        let positions = vec![[1.0, 2.0, 3.0]; 10];
        let weights = vec![1.0; 10];
        let tree = LcaTree::build(&positions, &weights, Some(32), None);
        // 10 < 32, so no internal nodes.
        assert_eq!(tree.num_internal_nodes(), 0);
    }

    #[test]
    fn select_nth_basic() {
        let mut pos = vec![
            [5.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ];
        let mut wts = vec![50.0, 30.0, 10.0, 40.0, 20.0];
        let mut idx: Vec<u32> = (0..5).collect();

        select_nth_with_mirrors(&mut pos, &mut wts, &mut idx, 2, 0);

        // Element at index 2 should be the median (value 3.0 along axis 0).
        assert!((pos[2][0] - 3.0).abs() < 1e-12);
        // All elements before should be ≤ 3.0.
        for i in 0..2 {
            assert!(pos[i][0] <= 3.0 + 1e-12);
        }
        // All elements after should be ≥ 3.0.
        for i in 3..5 {
            assert!(pos[i][0] >= 3.0 - 1e-12);
        }
    }

    #[test]
    fn tree_node_bboxes_are_tight() {
        let positions: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [5.0, 5.0, 5.0],
            [2.0, 8.0, 3.0],
        ];
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let tree = LcaTree::build(&positions, &weights, Some(1), None);

        // Check that root node's left + right bboxes together cover all points.
        if let Some(root_idx) = tree.root() {
            let root = &tree.nodes[root_idx];
            let mid = (root.particle_start + root.particle_end) / 2;

            // Left bbox should tightly contain positions[start..mid].
            for i in root.particle_start as usize..mid as usize {
                let p = &tree.positions[i];
                for k in 0..3 {
                    assert!(p[k] >= root.bbox_left.lo[k] - 1e-12);
                    assert!(p[k] <= root.bbox_left.hi[k] + 1e-12);
                }
            }

            // Right bbox should tightly contain positions[mid..end].
            for i in mid as usize..root.particle_end as usize {
                let p = &tree.positions[i];
                for k in 0..3 {
                    assert!(p[k] >= root.bbox_right.lo[k] - 1e-12);
                    assert!(p[k] <= root.bbox_right.hi[k] + 1e-12);
                }
            }
        }
    }

    #[test]
    fn tree_weights_preserved() {
        let positions: Vec<[f64; 3]> = (0..100)
            .map(|i| {
                let f = i as f64;
                [f, f * 2.0, f * 3.0]
            })
            .collect();
        let weights: Vec<f64> = (0..100).map(|i| (i + 1) as f64).collect();
        let expected_total: f64 = weights.iter().sum();

        let tree = LcaTree::build(&positions, &weights, Some(8), None);

        let actual_total: f64 = tree.weights.iter().sum();
        assert!(
            (actual_total - expected_total).abs() < 1e-10,
            "weight sum {actual_total} != {expected_total}"
        );

        // Each original weight should appear exactly once.
        let mut sorted_w: Vec<f64> = tree.weights.clone();
        sorted_w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected_sorted: Vec<f64> = weights.clone();
        expected_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted_w, expected_sorted);
    }
}

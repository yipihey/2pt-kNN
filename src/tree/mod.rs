//! KD-tree abstraction over bosque.
//!
//! We wrap bosque's in-place KD-tree to provide kNN queries returning
//! (distance, index) pairs sorted by distance. All heavy lifting is
//! delegated to bosque's tree-accelerated queries.
//!
//! Bosque (with default `sqrt-dist` feature) returns Euclidean distances,
//! not squared distances. Our `Neighbor::dist` field stores the actual
//! Euclidean distance.

use std::fmt;

/// Result of a kNN query: Euclidean distance and original index.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    pub dist: f64,
    pub index: usize,
}

/// A 3D point cloud with an in-place KD-tree built on it.
///
/// After construction, the positions array is permuted (bosque builds in-place).
/// We keep an index map so that `Neighbor::index` refers to the original
/// insertion order, not the permuted position in the array.
pub struct PointTree {
    /// Permuted positions (bosque tree structure is implicit in this ordering)
    positions: Vec<[f64; 3]>,
    /// Maps permuted index → original index
    index_map: Vec<usize>,
    /// Number of points
    len: usize,
}

impl fmt::Debug for PointTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PointTree")
            .field("len", &self.len)
            .finish()
    }
}

impl PointTree {
    /// Build a KD-tree from a set of 3D positions.
    ///
    /// The input is consumed: bosque permutes the array in-place.
    /// We use `build_tree_with_indices` to track the permutation so
    /// query results can be mapped back to original insertion order.
    pub fn build(positions: Vec<[f64; 3]>) -> Self {
        let len = positions.len();
        let mut pos = positions;
        let mut idxs: Vec<u32> = (0..len as u32).collect();

        bosque::tree::build_tree_with_indices(&mut pos, &mut idxs);

        // idxs[i] now holds the original index of the point at permuted position i
        let index_map: Vec<usize> = idxs.into_iter().map(|i| i as usize).collect();

        Self {
            positions: pos,
            index_map,
            len,
        }
    }

    /// Number of points in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Find the single nearest neighbor of `query`.
    pub fn nearest_one(&self, query: &[f64; 3]) -> Neighbor {
        let (dist, perm_idx) = bosque::tree::nearest_one(&self.positions, query);
        Neighbor {
            dist,
            index: self.index_map[perm_idx],
        }
    }

    /// Find the k nearest neighbors of `query`, sorted by distance.
    ///
    /// Returns up to `k` neighbors. If the tree has fewer than `k` points,
    /// returns all points.
    pub fn nearest_k(&self, query: &[f64; 3], k: usize) -> Vec<Neighbor> {
        let k = k.min(self.len);
        if k == 0 {
            return Vec::new();
        }

        let mut results = bosque::tree::nearest_k(&self.positions, query, k);
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        results
            .into_iter()
            .map(|(dist, perm_idx)| Neighbor {
                dist,
                index: self.index_map[perm_idx],
            })
            .collect()
    }

    /// Find the k nearest neighbors with periodic boundary conditions.
    ///
    /// Distances wrap around a cubic box of side `box_size`.
    pub fn nearest_k_periodic(&self, query: &[f64; 3], k: usize, box_size: f64) -> Vec<Neighbor> {
        let k = k.min(self.len);
        if k == 0 {
            return Vec::new();
        }

        let mut results =
            bosque::tree::nearest_k_periodic(&self.positions, query, k, 0.0, box_size);
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        results
            .into_iter()
            .map(|(dist, perm_idx)| Neighbor {
                dist,
                index: self.index_map[perm_idx],
            })
            .collect()
    }

    /// Access the (permuted) positions slice.
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_one_identity() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let tree = PointTree::build(pts);
        let n = tree.nearest_one(&[0.01, 0.0, 0.0]);
        assert_eq!(n.index, 0);
        assert!(n.dist < 0.02);
    }

    #[test]
    fn test_nearest_k() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        let tree = PointTree::build(pts);
        let neighbors = tree.nearest_k(&[0.5, 0.0, 0.0], 2);
        assert_eq!(neighbors.len(), 2);
        let indices: Vec<usize> = neighbors.iter().map(|n| n.index).collect();
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        // Check distances are Euclidean (not squared)
        assert!((neighbors[0].dist - 0.5).abs() < 1e-10);
        assert!((neighbors[1].dist - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_k_periodic() {
        // Point near boundary should find periodic neighbor
        let pts = vec![
            [1.0, 5.0, 5.0],
            [9.0, 5.0, 5.0], // periodic dist to pt0 = 2.0
            [3.0, 5.0, 5.0], // dist to pt0 = 2.0
        ];
        let tree = PointTree::build(pts);
        let nn = tree.nearest_k_periodic(&[1.0, 5.0, 5.0], 3, 10.0);
        assert_eq!(nn.len(), 3);
        assert!(nn[0].dist < 1e-10); // self
        // Both neighbors at distance 2.0
        assert!((nn[1].dist - 2.0).abs() < 1e-10);
        assert!((nn[2].dist - 2.0).abs() < 1e-10);
    }
}

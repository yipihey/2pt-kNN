//! KD-tree abstraction over bosque.
//!
//! We wrap bosque's in-place KD-tree to provide kNN queries returning
//! (distance, index) pairs sorted by distance. The trait `KnnTree` allows
//! swapping backends if needed.

use std::fmt;

/// Result of a kNN query: squared distance and original index.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    pub dist_sq: f64,
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
    /// We track the original indices so query results can be mapped back.
    pub fn build(positions: Vec<[f64; 3]>) -> Self {
        let len = positions.len();

        // Tag each point with its original index, then separate
        let tagged: Vec<([f64; 3], usize)> = positions
            .into_iter()
            .enumerate()
            .map(|(i, p)| (p, i))
            .collect();

        // Extract just the positions for bosque
        let mut pos: Vec<[f64; 3]> = tagged.iter().map(|(p, _)| *p).collect();

        // Build tree in-place (permutes pos)
        bosque::tree::build_tree(&mut pos);

        // Reconstruct index map: after build_tree, pos is permuted.
        // We need to figure out which original point ended up where.
        // Since bosque permutes in-place, we need to track this.
        //
        // Strategy: build a second tagged array, sort it the same way.
        // Actually, bosque doesn't expose the permutation. We'll use
        // a workaround: build from (pos, original_idx) pairs.
        //
        // For now, we use a simpler approach: store (pos, orig_idx) pairs
        // and do the sort ourselves. This is a TODO to optimize with
        // bosque internals if it exposes the permutation.

        // Re-tag: find each permuted position's original index by matching.
        // This is O(n²) and only acceptable for initial development.
        // TODO: Use bosque's internal permutation tracking or build
        // a custom in-place sort that tracks indices.
        let mut index_map = vec![0usize; len];
        let mut used = vec![false; len];
        for (new_idx, new_pos) in pos.iter().enumerate() {
            for (orig_idx, (orig_pos, _)) in tagged.iter().enumerate() {
                if !used[orig_idx]
                    && new_pos[0] == orig_pos[0]
                    && new_pos[1] == orig_pos[1]
                    && new_pos[2] == orig_pos[2]
                {
                    index_map[new_idx] = orig_idx;
                    used[orig_idx] = true;
                    break;
                }
            }
        }

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
        let (dist_sq, perm_idx) = bosque::tree::nearest_one(&self.positions, query);
        Neighbor {
            dist_sq,
            index: self.index_map[perm_idx],
        }
    }

    /// Find the k nearest neighbors of `query`, sorted by distance.
    ///
    /// Returns up to `k` neighbors. If the tree has fewer than `k` points,
    /// returns all points.
    ///
    /// TODO: This uses repeated single-neighbor queries with exclusion,
    /// which is O(k · log n) but with high constant factors. Replace with
    /// a proper k-nearest traversal in bosque when available.
    pub fn nearest_k(&self, query: &[f64; 3], k: usize) -> Vec<Neighbor> {
        // For the initial implementation, we do a brute-force scan.
        // This is correct but slow for large datasets. The first optimization
        // target is to add a proper nearest_k to bosque or use its tree
        // structure directly.
        //
        // For CoxMock validation at N ~ 10^4–10^5, this is acceptable.
        let k = k.min(self.len);
        let mut dists: Vec<(f64, usize)> = self
            .positions
            .iter()
            .enumerate()
            .map(|(perm_idx, pos)| {
                let dx = pos[0] - query[0];
                let dy = pos[1] - query[1];
                let dz = pos[2] - query[2];
                (dx * dx + dy * dy + dz * dz, perm_idx)
            })
            .collect();

        // Partial sort to get k smallest
        dists.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.truncate(k);
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        dists
            .into_iter()
            .map(|(dist_sq, perm_idx)| Neighbor {
                dist_sq,
                index: self.index_map[perm_idx],
            })
            .collect()
    }

    /// Find the k nearest neighbors with periodic boundary conditions.
    ///
    /// Distances wrap around a cubic box of side `box_size`.
    pub fn nearest_k_periodic(&self, query: &[f64; 3], k: usize, box_size: f64) -> Vec<Neighbor> {
        let k = k.min(self.len);
        let half = box_size * 0.5;
        let mut dists: Vec<(f64, usize)> = self
            .positions
            .iter()
            .enumerate()
            .map(|(perm_idx, pos)| {
                let mut dist_sq = 0.0;
                for dim in 0..3 {
                    let mut d = (pos[dim] - query[dim]).abs();
                    if d > half {
                        d = box_size - d;
                    }
                    dist_sq += d * d;
                }
                (dist_sq, perm_idx)
            })
            .collect();

        dists.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.truncate(k);
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        dists
            .into_iter()
            .map(|(dist_sq, perm_idx)| Neighbor {
                dist_sq,
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
        // Nearest should be point 0 or 1
        let indices: Vec<usize> = neighbors.iter().map(|n| n.index).collect();
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }
}

//! Angular radius search via a small KD-tree on 3D unit vectors.
//!
//! For each query point we want all catalog points within an angular distance
//! `theta_max`. We work on unit vectors with chord (Euclidean) distance:
//!
//! ```text
//! chord(theta) = 2 sin(theta / 2)
//! ```
//!
//! A KD-tree radius query in chord space recovers exactly the angular ball.
//! RA periodicity is handled implicitly by the unit-vector mapping.

use crate::haversine::{radec_to_unit, theta_to_chord};

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct Node {
    /// Index in the original input ordering.
    idx: u32,
    /// Splitting axis 0,1,2 (kept for debugging/visualization).
    axis: u8,
    /// Left subtree root (in `nodes`), or `u32::MAX` for none.
    left: u32,
    right: u32,
    /// Bounding box (axis-aligned) of all points beneath this node.
    bb_min: [f64; 3],
    bb_max: [f64; 3],
}

const NIL: u32 = u32::MAX;

/// Compact KD-tree on 3D unit vectors with squared chord radius queries.
pub struct AngularTree {
    points: Vec<[f64; 3]>,
    nodes: Vec<Node>,
    root: u32,
}

impl AngularTree {
    /// Build the tree from RA/Dec arrays in radians. Lengths must match.
    pub fn new(ra: &[f64], dec: &[f64]) -> Self {
        assert_eq!(ra.len(), dec.len(), "ra and dec must have equal length");
        let mut points: Vec<[f64; 3]> = ra
            .iter()
            .zip(dec.iter())
            .map(|(&r, &d)| radec_to_unit(r, d))
            .collect();
        let n = points.len();
        let mut indices: Vec<u32> = (0..n as u32).collect();
        let mut nodes: Vec<Node> = Vec::with_capacity(n);

        // Build recursively over `indices` slice; mutates `points` order indirectly via indices.
        let root = if n == 0 {
            NIL
        } else {
            build(&mut points, &mut indices[..], &mut nodes)
        };

        AngularTree {
            points,
            nodes,
            root,
        }
    }

    /// Number of points indexed.
    pub fn len(&self) -> usize {
        self.points.len()
    }
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Return the unit vector for the point with original index `i`.
    pub fn unit(&self, i: usize) -> &[f64; 3] {
        &self.points[i]
    }

    /// Append all original indices `i` whose angular distance to the query
    /// `(ra_q, dec_q)` is `<= theta_max` to `out` (cleared first).
    pub fn query_radius(
        &self,
        ra_q: f64,
        dec_q: f64,
        theta_max: f64,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        if self.root == NIL || theta_max <= 0.0 {
            return;
        }
        let q = radec_to_unit(ra_q, dec_q);
        let chord = theta_to_chord(theta_max);
        let r2 = chord * chord;
        self.query_recursive(self.root, &q, r2, out);
    }

    /// Convenience: allocate and return.
    pub fn query_radius_vec(&self, ra_q: f64, dec_q: f64, theta_max: f64) -> Vec<usize> {
        let mut v = Vec::new();
        self.query_radius(ra_q, dec_q, theta_max, &mut v);
        v
    }

    fn query_recursive(
        &self,
        node_idx: u32,
        q: &[f64; 3],
        r2: f64,
        out: &mut Vec<usize>,
    ) {
        if node_idx == NIL {
            return;
        }
        let node = &self.nodes[node_idx as usize];

        // Prune by bounding box squared distance.
        if bb_dist2(&node.bb_min, &node.bb_max, q) > r2 {
            return;
        }

        // Test this point.
        let p = &self.points[node.idx as usize];
        let dx = p[0] - q[0];
        let dy = p[1] - q[1];
        let dz = p[2] - q[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 <= r2 {
            out.push(node.idx as usize);
        }

        // Descend.
        self.query_recursive(node.left, q, r2, out);
        self.query_recursive(node.right, q, r2, out);
    }
}

fn bb_dist2(bb_min: &[f64; 3], bb_max: &[f64; 3], q: &[f64; 3]) -> f64 {
    let mut d2 = 0.0;
    for k in 0..3 {
        let v = if q[k] < bb_min[k] {
            bb_min[k] - q[k]
        } else if q[k] > bb_max[k] {
            q[k] - bb_max[k]
        } else {
            0.0
        };
        d2 += v * v;
    }
    d2
}

/// Build a KD-tree over `indices`, returning the index in `nodes` of the root
/// of this subtree. Points are not reordered; only indices are partitioned.
fn build(points: &mut [[f64; 3]], indices: &mut [u32], nodes: &mut Vec<Node>) -> u32 {
    if indices.is_empty() {
        return NIL;
    }

    // Compute bounding box.
    let first = points[indices[0] as usize];
    let mut bb_min = first;
    let mut bb_max = first;
    for &i in indices.iter().skip(1) {
        let p = points[i as usize];
        for k in 0..3 {
            if p[k] < bb_min[k] {
                bb_min[k] = p[k];
            }
            if p[k] > bb_max[k] {
                bb_max[k] = p[k];
            }
        }
    }

    // Choose the axis with the largest extent.
    let mut axis = 0usize;
    let mut span = bb_max[0] - bb_min[0];
    for k in 1..3 {
        let s = bb_max[k] - bb_min[k];
        if s > span {
            span = s;
            axis = k;
        }
    }

    // Partition by median on chosen axis using nth-element.
    let mid = indices.len() / 2;
    indices.select_nth_unstable_by(mid, |&a, &b| {
        points[a as usize][axis]
            .partial_cmp(&points[b as usize][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Reserve a slot for this node so children get later indices (bottom-up build).
    let pivot = indices[mid];
    let placeholder_idx = nodes.len() as u32;
    nodes.push(Node {
        idx: pivot,
        axis: axis as u8,
        left: NIL,
        right: NIL,
        bb_min,
        bb_max,
    });

    let (left_part, right_with_pivot) = indices.split_at_mut(mid);
    // right_with_pivot[0] is the pivot; recurse on left_part and right_with_pivot[1..]
    let left_root = build(points, left_part, nodes);
    let right_root = build(points, &mut right_with_pivot[1..], nodes);
    nodes[placeholder_idx as usize].left = left_root;
    nodes[placeholder_idx as usize].right = right_root;
    placeholder_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::haversine::haversine;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::f64::consts::PI;

    fn brute_force(
        ra: &[f64],
        dec: &[f64],
        ra_q: f64,
        dec_q: f64,
        theta_max: f64,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        for i in 0..ra.len() {
            if haversine(ra[i], dec[i], ra_q, dec_q) <= theta_max {
                out.push(i);
            }
        }
        out.sort();
        out
    }

    #[test]
    fn query_matches_brute_force_random() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 5000;
        let ra: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..(2.0 * PI))).collect();
        // sample dec uniformly on the sphere
        let dec: Vec<f64> = (0..n).map(|_| (1.0 - 2.0 * rng.gen::<f64>()).asin()).collect();
        let tree = AngularTree::new(&ra, &dec);
        for _ in 0..50 {
            let raq = rng.gen_range(0.0..(2.0 * PI));
            let decq = (1.0 - 2.0 * rng.gen::<f64>()).asin();
            for &theta_deg in &[0.5_f64, 1.0, 5.0, 30.0] {
                let theta_max = theta_deg.to_radians();
                let mut got = tree.query_radius_vec(raq, decq, theta_max);
                got.sort();
                let want = brute_force(&ra, &dec, raq, decq, theta_max);
                assert_eq!(got, want, "theta={}deg", theta_deg);
            }
        }
    }

    #[test]
    fn handles_ra_wrap() {
        // Place a point near RA=0 and query near RA=2pi.
        let ra = vec![0.01_f64, 6.27_f64];
        let dec = vec![0.0_f64, 0.0_f64];
        let tree = AngularTree::new(&ra, &dec);
        let theta_max = 0.05_f64;
        let mut got = tree.query_radius_vec(2.0 * PI, 0.0, theta_max);
        got.sort();
        // Both points are within ~0.014 rad of RA=2pi, so both should be returned.
        assert_eq!(got, vec![0, 1]);
    }

    #[test]
    fn empty_tree_is_safe() {
        let tree = AngularTree::new(&[], &[]);
        let got = tree.query_radius_vec(0.0, 0.0, 0.5);
        assert!(got.is_empty());
        assert!(tree.is_empty());
    }
}

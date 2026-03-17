//! FMM-style tree walk for tidal tensor evaluation.
//!
//! Implements a Barnes-Hut style tree walk (simplified FMM) that descends the
//! KD-tree and accumulates the tidal tensor T_{ab}(x_i) at every particle.
//!
//! At each level of descent, the algorithm:
//! 1. For the current node, consider all sibling/cousin nodes.
//! 2. If a node passes the opening-angle criterion (h/r < θ), accept its
//!    multipole contribution.
//! 3. Otherwise, recurse into its children.
//! 4. At leaf level, handle near-field interactions via direct summation.
//!
//! # Scale-dependent output
//!
//! During the walk, the cumulative tidal tensor is recorded at each tree depth,
//! giving T_{ab}(x_i; L_ℓ) — the tidal field smoothed at scale L_ℓ.

use super::LcaTree;
use super::eigen::Sym3x3;
use super::multipole::{NodeMultipole, MultipoleOrder, compute_multipole, merge_multipoles};
use super::gravity::{tidal_from_node, tidal_direct, should_open};
use super::tidal::{TidalAccum, ScaleTidal};

/// Configuration for the FMM tidal evaluation.
#[derive(Debug, Clone)]
pub struct FmmConfig {
    /// Opening angle θ for the Barnes-Hut criterion.
    /// Smaller = more accurate but slower. Typical: 0.5–0.7.
    pub theta: f64,
    /// Multipole expansion order.
    pub order: MultipoleOrder,
    /// Gravitational softening length squared (to avoid singularity).
    pub softening_sq: f64,
    /// Whether to store scale-dependent tidal tensors.
    pub store_scale_tidal: bool,
    /// Maximum number of tree levels for scale-dependent storage.
    /// If None, uses the actual tree depth.
    pub max_levels: Option<usize>,
}

impl Default for FmmConfig {
    fn default() -> Self {
        Self {
            theta: 0.5,
            order: MultipoleOrder::Quadrupole,
            softening_sq: 0.0,
            store_scale_tidal: false,
            max_levels: None,
        }
    }
}

/// Result of the FMM tidal evaluation.
pub struct FmmResult {
    /// Per-particle tidal tensor (accumulated from all sources).
    pub tidal: Vec<TidalAccum>,
    /// Per-particle scale-dependent tidal tensors (if `store_scale_tidal`).
    /// Indexed as scale_tidal[particle_idx].
    pub scale_tidal: Option<Vec<ScaleTidal>>,
    /// Number of node-particle multipole interactions.
    pub n_multipole_interactions: u64,
    /// Number of direct particle-particle interactions.
    pub n_direct_interactions: u64,
    /// Maximum tree depth encountered.
    pub max_depth: usize,
    /// Characteristic scale at each depth level.
    pub depth_scales: Vec<f64>,
}

/// Compute multipole moments for all nodes in the tree (bottom-up).
///
/// Returns a parallel array indexed by node index in `tree.nodes`.
pub fn compute_tree_multipoles(tree: &LcaTree) -> Vec<NodeMultipole> {
    let n_nodes = tree.nodes.len();
    let mut multipoles = vec![NodeMultipole::default(); n_nodes];

    if n_nodes <= 1 {
        return multipoles;
    }

    // Bottom-up pass: process nodes in reverse order (children before parents).
    // Since nodes are appended during build (parent index < child index),
    // reverse iteration gives us bottom-up order.
    for i in (1..n_nodes).rev() {
        let node = &tree.nodes[i];
        let start = node.particle_start as usize;
        let end = node.particle_end as usize;
        let mid = start + (end - start) / 2;

        let has_left = node.left != 0;
        let has_right = node.right != 0;

        if !has_left && !has_right {
            // Both children are leaves — compute directly from particles.
            multipoles[i] = compute_multipole(
                &tree.positions, &tree.weights, start, end,
            );
        } else if has_left && has_right {
            // Both children are internal nodes — merge their multipoles.
            let left_mp = &multipoles[node.left as usize];
            let right_mp = &multipoles[node.right as usize];

            // Use weighted center of mass as parent center.
            let total_mass = left_mp.monopole + right_mp.monopole;
            let parent_center = if total_mass > 0.0 {
                [
                    (left_mp.center[0] * left_mp.monopole + right_mp.center[0] * right_mp.monopole) / total_mass,
                    (left_mp.center[1] * left_mp.monopole + right_mp.center[1] * right_mp.monopole) / total_mass,
                    (left_mp.center[2] * left_mp.monopole + right_mp.center[2] * right_mp.monopole) / total_mass,
                ]
            } else {
                let cl = &left_mp.center;
                let cr = &right_mp.center;
                [(cl[0] + cr[0]) / 2.0, (cl[1] + cr[1]) / 2.0, (cl[2] + cr[2]) / 2.0]
            };

            multipoles[i] = merge_multipoles(left_mp, right_mp, &parent_center);
        } else if has_left {
            // Left is internal, right is leaf.
            let left_mp = multipoles[node.left as usize].clone();
            let right_mp = compute_multipole(&tree.positions, &tree.weights, mid, end);
            let total_mass = left_mp.monopole + right_mp.monopole;
            let parent_center = if total_mass > 0.0 {
                [
                    (left_mp.center[0] * left_mp.monopole + right_mp.center[0] * right_mp.monopole) / total_mass,
                    (left_mp.center[1] * left_mp.monopole + right_mp.center[1] * right_mp.monopole) / total_mass,
                    (left_mp.center[2] * left_mp.monopole + right_mp.center[2] * right_mp.monopole) / total_mass,
                ]
            } else {
                let cl = &left_mp.center;
                let cr = &right_mp.center;
                [(cl[0] + cr[0]) / 2.0, (cl[1] + cr[1]) / 2.0, (cl[2] + cr[2]) / 2.0]
            };
            multipoles[i] = merge_multipoles(&left_mp, &right_mp, &parent_center);
        } else {
            // Right is internal, left is leaf.
            let left_mp = compute_multipole(&tree.positions, &tree.weights, start, mid);
            let right_mp = multipoles[node.right as usize].clone();
            let total_mass = left_mp.monopole + right_mp.monopole;
            let parent_center = if total_mass > 0.0 {
                [
                    (left_mp.center[0] * left_mp.monopole + right_mp.center[0] * right_mp.monopole) / total_mass,
                    (left_mp.center[1] * left_mp.monopole + right_mp.center[1] * right_mp.monopole) / total_mass,
                    (left_mp.center[2] * left_mp.monopole + right_mp.center[2] * right_mp.monopole) / total_mass,
                ]
            } else {
                let cl = &left_mp.center;
                let cr = &right_mp.center;
                [(cl[0] + cr[0]) / 2.0, (cl[1] + cr[1]) / 2.0, (cl[2] + cr[2]) / 2.0]
            };
            multipoles[i] = merge_multipoles(&left_mp, &right_mp, &parent_center);
        }
    }

    multipoles
}

/// Compute the depth of the tree.
fn tree_depth(tree: &LcaTree) -> usize {
    if tree.nodes.len() <= 1 { return 0; }
    fn depth_recursive(tree: &LcaTree, node_idx: u32) -> usize {
        if node_idx == 0 { return 0; }
        let node = &tree.nodes[node_idx as usize];
        let ld = depth_recursive(tree, node.left);
        let rd = depth_recursive(tree, node.right);
        1 + ld.max(rd)
    }
    depth_recursive(tree, 1)
}

/// Compute characteristic scale at each tree depth.
fn depth_scales(tree: &LcaTree, max_depth: usize) -> Vec<f64> {
    let box_size = tree.box_size.unwrap_or_else(|| {
        // Estimate from root node bounding box.
        if tree.nodes.len() > 1 {
            let root = &tree.nodes[1];
            let side_l = root.bbox_left.side_lengths();
            let side_r = root.bbox_right.side_lengths();
            let max_l = side_l[0].max(side_l[1]).max(side_l[2]);
            let max_r = side_r[0].max(side_r[1]).max(side_r[2]);
            max_l.max(max_r) * 2.0
        } else {
            1.0
        }
    });

    (0..=max_depth)
        .map(|d| box_size / (1 << d) as f64)
        .collect()
}

/// Run the FMM tidal evaluation on all particles in the tree.
///
/// Uses a Barnes-Hut tree walk: for each particle, descend the tree and
/// evaluate the tidal tensor from far-field nodes via multipole expansion,
/// falling back to direct summation for near-field interactions.
pub fn evaluate_tidal_field(
    tree: &LcaTree,
    multipoles: &[NodeMultipole],
    config: &FmmConfig,
) -> FmmResult {
    let n_particles = tree.positions.len();
    let max_d = tree_depth(tree);
    let n_levels = config.max_levels.unwrap_or(max_d + 1).min(max_d + 1);
    let scales = depth_scales(tree, max_d);

    let mut tidal: Vec<TidalAccum> = (0..n_particles).map(|_| TidalAccum::new()).collect();
    let mut scale_tidal: Option<Vec<ScaleTidal>> = if config.store_scale_tidal {
        Some((0..n_particles).map(|_| ScaleTidal::new(n_levels)).collect())
    } else {
        None
    };

    let mut n_multipole = 0u64;
    let mut n_direct = 0u64;

    if tree.nodes.len() <= 1 {
        // No internal nodes — do direct summation over all pairs.
        for i in 0..n_particles {
            let fp = &tree.positions[i];
            let t = tidal_direct(fp, &tree.positions, &tree.weights, config.softening_sq);
            tidal[i].add_contribution(&t);
            n_direct += (n_particles - 1) as u64;
        }
        return FmmResult {
            tidal,
            scale_tidal,
            n_multipole_interactions: n_multipole,
            n_direct_interactions: n_direct,
            max_depth: 0,
            depth_scales: scales,
        };
    }

    // For each particle, do a tree walk starting from root.
    for i in 0..n_particles {
        let fp = tree.positions[i];
        let mut particle_tidal = TidalAccum::new();

        walk_tree_for_particle(
            tree,
            multipoles,
            config,
            1, // root node
            &fp,
            i,
            0, // depth
            &mut particle_tidal,
            &mut scale_tidal,
            &mut n_multipole,
            &mut n_direct,
        );

        tidal[i] = particle_tidal;
    }

    FmmResult {
        tidal,
        scale_tidal,
        n_multipole_interactions: n_multipole,
        n_direct_interactions: n_direct,
        max_depth: max_d,
        depth_scales: scales,
    }
}

/// Recursive tree walk for a single particle.
fn walk_tree_for_particle(
    tree: &LcaTree,
    multipoles: &[NodeMultipole],
    config: &FmmConfig,
    node_idx: u32,
    field_point: &[f64; 3],
    particle_idx: usize,
    depth: usize,
    accum: &mut TidalAccum,
    scale_tidal: &mut Option<Vec<ScaleTidal>>,
    n_multipole: &mut u64,
    n_direct: &mut u64,
) {
    if node_idx == 0 {
        return;
    }

    let node = &tree.nodes[node_idx as usize];
    let mp = &multipoles[node_idx as usize];
    let start = node.particle_start as usize;
    let end = node.particle_end as usize;

    // Check if the particle is inside this node's range.
    let particle_in_node = particle_idx >= start && particle_idx < end;

    // If the particle is NOT in this node and the node passes the opening criterion,
    // accept the multipole.
    if !particle_in_node && !should_open(mp, field_point, config.theta) {
        let t = tidal_from_node(mp, field_point, config.order);
        accum.add_contribution(&t);
        *n_multipole += 1;

        // Record at this depth for scale-dependent output.
        if let Some(ref mut st) = scale_tidal {
            let n_levels = st[particle_idx].n_levels();
            if depth < n_levels {
                super::eigen::sym3_add(&mut st[particle_idx].t_ij[depth], &t);
            }
        }
        return;
    }

    // If both children are leaves, do direct summation.
    let has_left_child = node.left != 0;
    let has_right_child = node.right != 0;
    let mid = start + (end - start) / 2;

    if !has_left_child && !has_right_child {
        // Leaf node — direct summation over particles, excluding self.
        for j in start..end {
            if j == particle_idx { continue; }
            let rp = &tree.positions[j];
            let rx = field_point[0] - rp[0];
            let ry = field_point[1] - rp[1];
            let rz = field_point[2] - rp[2];
            let r2 = rx * rx + ry * ry + rz * rz + config.softening_sq;
            if r2 < 1e-30 { continue; }

            let r = r2.sqrt();
            let r3_inv = 1.0 / (r * r2);
            let r5_inv = r3_inv / r2;
            let rv = [rx, ry, rz];
            let w = tree.weights[j];

            let t: Sym3x3 = [
                w * (3.0 * rv[0] * rv[0] * r5_inv - r3_inv),
                w * (3.0 * rv[1] * rv[1] * r5_inv - r3_inv),
                w * (3.0 * rv[2] * rv[2] * r5_inv - r3_inv),
                w * 3.0 * rv[0] * rv[1] * r5_inv,
                w * 3.0 * rv[0] * rv[2] * r5_inv,
                w * 3.0 * rv[1] * rv[2] * r5_inv,
            ];
            accum.add_contribution(&t);

            if let Some(ref mut st) = scale_tidal {
                let n_levels = st[particle_idx].n_levels();
                if depth < n_levels {
                    super::eigen::sym3_add(&mut st[particle_idx].t_ij[depth], &t);
                }
            }
            *n_direct += 1;
        }
        return;
    }

    // Recurse into children.
    // Handle the case where one child is a leaf and the other is internal.

    if has_left_child {
        walk_tree_for_particle(
            tree, multipoles, config,
            node.left, field_point, particle_idx,
            depth + 1, accum, scale_tidal, n_multipole, n_direct,
        );
    } else {
        // Left child is a leaf [start..mid] — direct summation.
        for j in start..mid {
            if j == particle_idx { continue; }
            let t = single_particle_tidal(field_point, &tree.positions[j], tree.weights[j], config.softening_sq);
            accum.add_contribution(&t);
            if let Some(ref mut st) = scale_tidal {
                let n_levels = st[particle_idx].n_levels();
                if depth + 1 < n_levels {
                    super::eigen::sym3_add(&mut st[particle_idx].t_ij[depth + 1], &t);
                }
            }
            *n_direct += 1;
        }
    }

    if has_right_child {
        walk_tree_for_particle(
            tree, multipoles, config,
            node.right, field_point, particle_idx,
            depth + 1, accum, scale_tidal, n_multipole, n_direct,
        );
    } else {
        // Right child is a leaf [mid..end] — direct summation.
        for j in mid..end {
            if j == particle_idx { continue; }
            let t = single_particle_tidal(field_point, &tree.positions[j], tree.weights[j], config.softening_sq);
            accum.add_contribution(&t);
            if let Some(ref mut st) = scale_tidal {
                let n_levels = st[particle_idx].n_levels();
                if depth + 1 < n_levels {
                    super::eigen::sym3_add(&mut st[particle_idx].t_ij[depth + 1], &t);
                }
            }
            *n_direct += 1;
        }
    }
}

/// Tidal tensor from a single source particle (used in leaf-level direct sum).
#[inline]
fn single_particle_tidal(
    field_point: &[f64; 3],
    source: &[f64; 3],
    weight: f64,
    softening_sq: f64,
) -> Sym3x3 {
    let rx = field_point[0] - source[0];
    let ry = field_point[1] - source[1];
    let rz = field_point[2] - source[2];
    let r2 = rx * rx + ry * ry + rz * rz + softening_sq;

    if r2 < 1e-30 {
        return [0.0; 6];
    }

    let r = r2.sqrt();
    let r3_inv = 1.0 / (r * r2);
    let r5_inv = r3_inv / r2;
    let rv = [rx, ry, rz];
    let w = weight;

    [
        w * (3.0 * rv[0] * rv[0] * r5_inv - r3_inv),
        w * (3.0 * rv[1] * rv[1] * r5_inv - r3_inv),
        w * (3.0 * rv[2] * rv[2] * r5_inv - r3_inv),
        w * 3.0 * rv[0] * rv[1] * r5_inv,
        w * 3.0 * rv[0] * rv[2] * r5_inv,
        w * 3.0 * rv[1] * rv[2] * r5_inv,
    ]
}

/// Cumulate scale tidal tensors: convert per-depth contributions to
/// running cumulative sums. After this, `scale_tidal[i].t_ij[ℓ]` contains
/// the tidal tensor from all sources at depths 0..=ℓ.
pub fn cumulate_scale_tidal(scale_tidal: &mut [ScaleTidal]) {
    for st in scale_tidal.iter_mut() {
        let n = st.t_ij.len();
        for l in 1..n {
            for k in 0..6 {
                st.t_ij[l][k] += st.t_ij[l - 1][k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::LcaTree;
    use super::super::eigen::sym3_frob_sq;

    fn make_test_particles(n: usize, seed: u64) -> (Vec<[f64; 3]>, Vec<f64>) {
        let mut state = seed ^ 0x517cc1b727220a95;
        let lcg = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*s >> 11) as f64 / (1u64 << 53) as f64
        };
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|_| [lcg(&mut state) * 100.0, lcg(&mut state) * 100.0, lcg(&mut state) * 100.0])
            .collect();
        let weights = vec![1.0; n];
        (positions, weights)
    }

    #[test]
    fn compute_tree_multipoles_mass_conserved() {
        let (positions, weights) = make_test_particles(200, 42);
        let tree = LcaTree::build(&positions, &weights, Some(8), None);
        let multipoles = compute_tree_multipoles(&tree);

        if let Some(root) = tree.root() {
            let root_mp = &multipoles[root];
            let total_w: f64 = tree.weights.iter().sum();
            assert!(
                (root_mp.monopole - total_w).abs() < 1e-10,
                "root monopole {} != total weight {}",
                root_mp.monopole, total_w
            );
        }
    }

    #[test]
    fn fmm_vs_direct_small() {
        // Small problem: compare FMM tidal against direct summation.
        let (positions, weights) = make_test_particles(100, 123);
        let tree = LcaTree::build(&positions, &weights, Some(4), None);
        let multipoles = compute_tree_multipoles(&tree);

        let config = FmmConfig {
            theta: 0.3, // tight opening angle for accuracy
            order: MultipoleOrder::Quadrupole,
            softening_sq: 1e-4, // small softening for stability
            store_scale_tidal: false,
            max_levels: None,
        };

        let result = evaluate_tidal_field(&tree, &multipoles, &config);

        // Compare against direct summation for a few particles.
        // We must exclude self-interaction to match the FMM.
        let n_check = 10.min(tree.positions.len());
        for i in 0..n_check {
            let fp = &tree.positions[i];
            // Direct sum excluding self.
            let mut t_direct = [0.0f64; 6];
            for j in 0..tree.positions.len() {
                if j == i { continue; }
                let t = single_particle_tidal(fp, &tree.positions[j], tree.weights[j], config.softening_sq);
                for k in 0..6 { t_direct[k] += t[k]; }
            }
            let t_fmm = &result.tidal[i].t_ij;

            let err = sym3_frob_err(&t_direct, t_fmm);
            assert!(
                err < 0.05,
                "particle {}: FMM vs direct relative error = {:.4}",
                i, err
            );
        }
    }

    #[test]
    fn fmm_tidal_traceless() {
        // Tidal tensor should be traceless at every particle.
        let (positions, weights) = make_test_particles(50, 77);
        let tree = LcaTree::build(&positions, &weights, Some(4), None);
        let multipoles = compute_tree_multipoles(&tree);

        let config = FmmConfig {
            theta: 0.5,
            softening_sq: 1e-4,
            ..FmmConfig::default()
        };

        let result = evaluate_tidal_field(&tree, &multipoles, &config);

        for (i, ta) in result.tidal.iter().enumerate() {
            let trace = ta.t_ij[0] + ta.t_ij[1] + ta.t_ij[2];
            let frob = super::super::eigen::sym3_frob_sq(&ta.t_ij).sqrt();
            let rel_trace = if frob > 1e-30 { trace.abs() / frob } else { trace.abs() };
            assert!(
                rel_trace < 1e-4,
                "particle {}: trace = {:.2e}, frob = {:.2e}, relative = {:.2e}",
                i, trace, frob, rel_trace
            );
        }
    }

    #[test]
    fn scale_tidal_cumulation() {
        let mut st = vec![ScaleTidal::new(3)];
        st[0].t_ij[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        st[0].t_ij[1] = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        st[0].t_ij[2] = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        cumulate_scale_tidal(&mut st);

        assert!((st[0].t_ij[0][0] - 1.0).abs() < 1e-14);
        assert!((st[0].t_ij[1][0] - 3.0).abs() < 1e-14); // 1 + 2
        assert!((st[0].t_ij[2][0] - 6.0).abs() < 1e-14); // 1 + 2 + 3
    }

    fn sym3_frob_err(a: &[f64; 6], b: &[f64; 6]) -> f64 {
        let diff = [
            a[0] - b[0], a[1] - b[1], a[2] - b[2],
            a[3] - b[3], a[4] - b[4], a[5] - b[5],
        ];
        let norm_a = sym3_frob_sq(a).sqrt().max(1e-30);
        sym3_frob_sq(&diff).sqrt() / norm_a
    }
}

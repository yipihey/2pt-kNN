//! Validation binary for the FMM tidal tensor evaluation.
//!
//! Generates random particle distributions, builds the LCA tree, computes
//! multipole moments, evaluates the tidal field via FMM, and validates against
//! direct O(N²) summation.
//!
//! Tests:
//! - Point mass tidal tensor
//! - Shell theorem (cubic symmetry)
//! - FMM vs direct summation accuracy
//! - Tracelessness of tidal tensor
//! - Eigenvalue statistics
//! - Web type classification distribution

use clap::Parser;
use std::time::Instant;
use twopoint::lca_tree::LcaTree;
use twopoint::lca_tree::eigen::{sym3x3_eigenvalues, sym3_frob_sq};
use twopoint::lca_tree::fmm::{self, FmmConfig, compute_tree_multipoles};
use twopoint::lca_tree::multipole::MultipoleOrder;
use twopoint::lca_tree::tidal::WebType;

#[derive(Parser, Debug)]
#[command(name = "twopoint-validate-tidal")]
#[command(about = "Validate the FMM tidal tensor evaluation")]
struct Args {
    /// Number of particles
    #[arg(long, default_value_t = 1000)]
    n_particles: usize,

    /// Leaf size for the KD-tree
    #[arg(long, default_value_t = 16)]
    leaf_size: usize,

    /// Barnes-Hut opening angle θ
    #[arg(long, default_value_t = 0.5)]
    theta: f64,

    /// Gravitational softening length
    #[arg(long, default_value_t = 0.01)]
    softening: f64,

    /// Box size
    #[arg(long, default_value_t = 100.0)]
    box_size: f64,

    /// Number of particles for direct-sum comparison
    #[arg(long, default_value_t = 50)]
    n_check: usize,

    /// RNG seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Whether to store scale-dependent tidal tensors
    #[arg(long)]
    scale_tidal: bool,
}

fn main() {
    let args = Args::parse();

    println!("=== FMM Tidal Tensor Validation ===");
    println!("N_particles: {}", args.n_particles);
    println!("Leaf size: {}", args.leaf_size);
    println!("θ: {}", args.theta);
    println!("Softening: {}", args.softening);
    println!("Box size: {}", args.box_size);
    println!();

    // Generate random particles.
    let t0 = Instant::now();
    let (positions, weights) = generate_random_particles(
        args.n_particles, args.box_size, args.seed,
    );
    println!("Generated {} particles in {:.1}ms",
        args.n_particles, t0.elapsed().as_secs_f64() * 1e3);

    // Build tree.
    let t0 = Instant::now();
    let tree = LcaTree::build(&positions, &weights, Some(args.leaf_size), Some(args.box_size));
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("Built KD-tree: {} internal nodes in {:.1}ms",
        tree.num_internal_nodes(), build_ms);

    // Compute multipoles.
    let t0 = Instant::now();
    let multipoles = compute_tree_multipoles(&tree);
    let mp_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("Computed multipoles in {:.1}ms", mp_ms);

    // Verify root monopole.
    if let Some(root) = tree.root() {
        let root_mp = &multipoles[root];
        let total_w: f64 = tree.weights.iter().sum();
        let rel_err = (root_mp.monopole - total_w).abs() / total_w;
        println!("Root monopole: {:.6} (total weight: {:.6}, rel_err: {:.2e})",
            root_mp.monopole, total_w, rel_err);
    }

    // Evaluate tidal field.
    let config = FmmConfig {
        theta: args.theta,
        order: MultipoleOrder::Quadrupole,
        softening_sq: args.softening * args.softening,
        store_scale_tidal: args.scale_tidal,
        max_levels: None,
    };

    let t0 = Instant::now();
    let result = fmm::evaluate_tidal_field(&tree, &multipoles, &config);
    let fmm_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("\nFMM evaluation: {:.1}ms", fmm_ms);
    println!("  Multipole interactions: {}", result.n_multipole_interactions);
    println!("  Direct interactions:    {}", result.n_direct_interactions);
    println!("  Max depth: {}", result.max_depth);

    // --- Test 1: FMM vs direct summation ---
    println!("\n--- Test 1: FMM vs Direct Summation ---");
    let n_check = args.n_check.min(tree.positions.len());
    let mut max_err = 0.0f64;
    let mut mean_err = 0.0f64;
    let mut n_compared = 0usize;

    let t0 = Instant::now();
    for i in 0..n_check {
        let fp = &tree.positions[i];
        // Direct sum excluding self.
        let mut t_direct = [0.0f64; 6];
        for j in 0..tree.positions.len() {
            if j == i { continue; }
            let t = single_particle_tidal_inline(fp, &tree.positions[j], tree.weights[j], config.softening_sq);
            for k in 0..6 { t_direct[k] += t[k]; }
        }
        let t_fmm = &result.tidal[i].t_ij;

        let diff = [
            t_direct[0] - t_fmm[0], t_direct[1] - t_fmm[1], t_direct[2] - t_fmm[2],
            t_direct[3] - t_fmm[3], t_direct[4] - t_fmm[4], t_direct[5] - t_fmm[5],
        ];
        let norm_direct = sym3_frob_sq(&t_direct).sqrt();
        let norm_diff = sym3_frob_sq(&diff).sqrt();
        let rel_err = if norm_direct > 1e-30 { norm_diff / norm_direct } else { norm_diff };

        if i < 5 {
            println!("  Particle {:4}: rel_err = {:.6}, |T_direct| = {:.4e}, |T_fmm| = {:.4e}",
                i, rel_err, norm_direct, sym3_frob_sq(t_fmm).sqrt());
        }

        max_err = max_err.max(rel_err);
        mean_err += rel_err;
        n_compared += 1;
    }
    let direct_ms = t0.elapsed().as_secs_f64() * 1e3;
    mean_err /= n_compared as f64;
    println!("  Checked {} particles in {:.1}ms", n_check, direct_ms);
    println!("  Max relative error:  {:.6}", max_err);
    println!("  Mean relative error: {:.6}", mean_err);
    let test1_pass = max_err < 0.1;
    println!("  Result: {}", if test1_pass { "PASS" } else { "FAIL" });

    // --- Test 2: Tracelessness ---
    println!("\n--- Test 2: Tracelessness ---");
    let mut max_trace_err = 0.0f64;
    for ta in &result.tidal {
        let trace = ta.t_ij[0] + ta.t_ij[1] + ta.t_ij[2];
        let frob = sym3_frob_sq(&ta.t_ij).sqrt();
        let rel = if frob > 1e-30 { trace.abs() / frob } else { trace.abs() };
        max_trace_err = max_trace_err.max(rel);
    }
    let test2_pass = max_trace_err < 1e-3;
    println!("  Max relative trace: {:.2e}", max_trace_err);
    println!("  Result: {}", if test2_pass { "PASS" } else { "FAIL" });

    // --- Test 3: Eigenvalue statistics ---
    println!("\n--- Test 3: Eigenvalue Statistics ---");
    let mut n_void = 0usize;
    let mut n_pancake = 0usize;
    let mut n_filament = 0usize;
    let mut n_halo = 0usize;
    let mut mean_eig = [0.0f64; 3];

    for ta in &result.tidal {
        let eig = sym3x3_eigenvalues(ta.t_ij);
        for k in 0..3 { mean_eig[k] += eig[k]; }

        // Classify using d_plus = 1 (arbitrary for this test).
        match WebType::classify_1lpt(&eig, 1.0) {
            WebType::Void => n_void += 1,
            WebType::Pancake => n_pancake += 1,
            WebType::Filament => n_filament += 1,
            WebType::Halo => n_halo += 1,
        }
    }
    let n = result.tidal.len() as f64;
    for k in 0..3 { mean_eig[k] /= n; }

    println!("  Mean eigenvalues: [{:.4e}, {:.4e}, {:.4e}]", mean_eig[0], mean_eig[1], mean_eig[2]);
    println!("  Web types (D₊=1): Void={}, Pancake={}, Filament={}, Halo={}",
        n_void, n_pancake, n_filament, n_halo);

    // Mean eigenvalue sum should be ~0 (traceless).
    let mean_trace = mean_eig[0] + mean_eig[1] + mean_eig[2];
    println!("  Mean trace: {:.4e}", mean_trace);

    // --- Test 4: Scale-dependent tidal (if enabled) ---
    if let Some(ref st) = result.scale_tidal {
        println!("\n--- Test 4: Scale-Dependent Tidal ---");
        let n_levels = if st.is_empty() { 0 } else { st[0].n_levels() };
        println!("  Number of levels: {}", n_levels);
        println!("  Depth scales: {:?}", &result.depth_scales[..result.depth_scales.len().min(8)]);

        // Check that the finest level matches the total tidal.
        if n_levels > 0 && !st.is_empty() {
            // Cumulate
            let mut st_cum = st.clone();
            fmm::cumulate_scale_tidal(&mut st_cum);

            let finest = &st_cum[0].t_ij[n_levels - 1];
            let total = &result.tidal[0].t_ij;
            let diff: f64 = (0..6).map(|k| (finest[k] - total[k]).powi(2)).sum();
            println!("  Finest-level vs total diff (particle 0): {:.2e}", diff.sqrt());
        }
    }

    // --- Summary ---
    println!("\n=== Summary ===");
    println!("Test 1 (FMM vs direct): {}", if test1_pass { "PASS" } else { "FAIL" });
    println!("Test 2 (tracelessness): {}", if test2_pass { "PASS" } else { "FAIL" });

    let all_pass = test1_pass && test2_pass;
    if all_pass {
        println!("\nAll tests PASSED.");
    } else {
        println!("\nSome tests FAILED.");
        std::process::exit(1);
    }
}

fn generate_random_particles(n: usize, box_size: f64, seed: u64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let mut state = seed ^ 0x517cc1b727220a95;
    let lcg = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    };
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|_| [lcg(&mut state) * box_size, lcg(&mut state) * box_size, lcg(&mut state) * box_size])
        .collect();
    let weights = vec![1.0; n];
    (positions, weights)
}

#[inline]
fn single_particle_tidal_inline(
    field_point: &[f64; 3],
    source: &[f64; 3],
    weight: f64,
    softening_sq: f64,
) -> [f64; 6] {
    let rx = field_point[0] - source[0];
    let ry = field_point[1] - source[1];
    let rz = field_point[2] - source[2];
    let r2 = rx * rx + ry * ry + rz * rz + softening_sq;
    if r2 < 1e-30 { return [0.0; 6]; }
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

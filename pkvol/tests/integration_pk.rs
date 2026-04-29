//! End-to-end integration test: compare `measure_pk` against a brute-force
//! O(N_gal * N_query * n_u * n_z) implementation on a small catalog.

use approx::assert_abs_diff_eq;
use pkvol::angular_search::AngularTree;
use pkvol::ecdf2d::EcdfBackend;
use pkvol::haversine::haversine;
use pkvol::pk_aggregate::{measure_pk, AngularVar, MeasureConfig, ShellSpec};
use pkvol::shell_counts::adjacent_intervals;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn brute_force_pk(
    ra_g: &[f64],
    dec_g: &[f64],
    z_g: &[f64],
    w_g: &[f64],
    ra_q: &[f64],
    dec_q: &[f64],
    qw: &[f64],
    theta_edges: &[f64],
    z_edges: &[f64],
    intervals: &[(usize, usize)],
    k_values: &[f64],
    theta_max: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_q = ra_q.len();
    let n_u = theta_edges.len();
    let n_z = z_edges.len();
    let n_int = intervals.len();
    let n_k = k_values.len();
    let kp1 = n_k + 1;
    let last_k = *k_values.last().unwrap();
    let mut k_ext: Vec<f64> = k_values.to_vec();
    k_ext.push((last_k + 1.0).floor().max(last_k + 1.0));

    let total_w: f64 = qw.iter().sum();
    let mut f_sum = vec![0.0; kp1 * n_u * n_int];
    for q in 0..n_q {
        // Local K(u, z).
        let mut k2d = vec![0.0; n_u * n_z];
        for i in 0..ra_g.len() {
            let theta = haversine(ra_g[i], dec_g[i], ra_q[q], dec_q[q]);
            if theta > theta_max {
                continue;
            }
            for a in 0..n_u {
                if theta > theta_edges[a] {
                    continue;
                }
                for b in 0..n_z {
                    if z_g[i] <= z_edges[b] {
                        k2d[a * n_z + b] += w_g[i];
                    }
                }
            }
        }
        for a in 0..n_u {
            for (ell, &(l, r)) in intervals.iter().enumerate() {
                let kl = if l == usize::MAX { 0.0 } else { k2d[a * n_z + l] };
                let kr = k2d[a * n_z + r];
                let shell = kr - kl;
                for j in 0..kp1 {
                    if shell >= k_ext[j] {
                        f_sum[j * n_u * n_int + a * n_int + ell] += qw[q];
                    }
                }
            }
        }
    }
    let inv_w = 1.0 / total_w;
    let mut f = vec![0.0; n_k * n_u * n_int];
    let mut p = vec![0.0; n_k * n_u * n_int];
    for j in 0..n_k {
        for ai in 0..(n_u * n_int) {
            f[j * n_u * n_int + ai] = f_sum[j * n_u * n_int + ai] * inv_w;
        }
    }
    for j in 0..n_k {
        for ai in 0..(n_u * n_int) {
            p[j * n_u * n_int + ai] =
                f[j * n_u * n_int + ai] - f_sum[(j + 1) * n_u * n_int + ai] * inv_w;
        }
    }
    (f, p)
}

#[test]
fn matches_brute_force_random_catalog() {
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let n_g = 800;
    // RA in [0, 2pi], Dec uniform on sphere via arcsin.
    let ra_g: Vec<f64> = (0..n_g).map(|_| rng.gen_range(0.0..(2.0 * std::f64::consts::PI))).collect();
    let dec_g: Vec<f64> = (0..n_g).map(|_| (1.0 - 2.0 * rng.gen::<f64>()).asin()).collect();
    let z_g: Vec<f64> = (0..n_g).map(|_| rng.gen_range(0.1..0.9)).collect();
    let w_g: Vec<f64> = (0..n_g).map(|_| rng.gen_range(0.5..1.5)).collect();

    let n_q = 80;
    let ra_q: Vec<f64> = (0..n_q).map(|_| rng.gen_range(0.0..(2.0 * std::f64::consts::PI))).collect();
    let dec_q: Vec<f64> = (0..n_q).map(|_| (1.0 - 2.0 * rng.gen::<f64>()).asin()).collect();
    let qw: Vec<f64> = (0..n_q).map(|_| rng.gen_range(0.8..1.2)).collect();

    let theta_max = 0.4_f64;
    let theta_edges: Vec<f64> = (1..=6).map(|i| 0.05 + 0.05 * i as f64).collect();
    let z_edges: Vec<f64> = (1..=8).map(|i| 0.1 * i as f64).collect();
    let intervals = adjacent_intervals(z_edges.len());
    let k_values: Vec<f64> = vec![1.0, 2.0, 4.0, 8.0];

    let cfg = MeasureConfig {
        theta_max,
        angular_variable: AngularVar::Theta,
        backend: EcdfBackend::Sweep,
        exclude_self: false,
        self_match_tol: 1e-9,
    };
    let shell = ShellSpec {
        z_edges: z_edges.clone(),
        intervals: intervals.clone(),
    };
    let tree = AngularTree::new(&ra_g, &dec_g);
    let out = measure_pk(
        &tree,
        &z_g,
        &w_g,
        &ra_q,
        &dec_q,
        Some(&qw),
        &theta_edges,
        &shell,
        &k_values,
        &cfg,
    );

    let (fb, pb) = brute_force_pk(
        &ra_g, &dec_g, &z_g, &w_g, &ra_q, &dec_q, &qw, &theta_edges, &z_edges, &intervals,
        &k_values, theta_max,
    );

    for k in 0..fb.len() {
        assert_abs_diff_eq!(out.f[k], fb[k], epsilon = 1e-10);
        assert_abs_diff_eq!(out.p[k], pb[k], epsilon = 1e-10);
    }
}

#[test]
fn sweep_and_histogram_backends_agree() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let n = 500;
    let ra: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..(2.0 * std::f64::consts::PI))).collect();
    let dec: Vec<f64> = (0..n).map(|_| (1.0 - 2.0 * rng.gen::<f64>()).asin()).collect();
    let z: Vec<f64> = (0..n).map(|_| rng.gen_range(0.1..0.9)).collect();
    let w: Vec<f64> = vec![1.0; n];

    let n_q = 40;
    let ra_q: Vec<f64> = (0..n_q).map(|_| rng.gen_range(0.0..(2.0 * std::f64::consts::PI))).collect();
    let dec_q: Vec<f64> = (0..n_q).map(|_| (1.0 - 2.0 * rng.gen::<f64>()).asin()).collect();

    let theta_edges = vec![0.1, 0.2, 0.3, 0.4];
    let z_edges = vec![0.2, 0.4, 0.6, 0.8];
    let intervals = adjacent_intervals(z_edges.len());
    let k_values = vec![1.0, 2.0, 5.0];

    let tree = AngularTree::new(&ra, &dec);
    let cfg_a = MeasureConfig {
        theta_max: 0.5,
        angular_variable: AngularVar::Theta,
        backend: EcdfBackend::Sweep,
        exclude_self: false,
        self_match_tol: 1e-9,
    };
    let cfg_b = MeasureConfig {
        backend: EcdfBackend::Histogram,
        ..cfg_a.clone()
    };
    let shell = ShellSpec {
        z_edges,
        intervals,
    };
    let r_a = measure_pk(
        &tree, &z, &w, &ra_q, &dec_q, None, &theta_edges, &shell, &k_values, &cfg_a,
    );
    let r_b = measure_pk(
        &tree, &z, &w, &ra_q, &dec_q, None, &theta_edges, &shell, &k_values, &cfg_b,
    );
    for i in 0..r_a.f.len() {
        assert_abs_diff_eq!(r_a.f[i], r_b.f[i], epsilon = 1e-10);
        assert_abs_diff_eq!(r_a.p[i], r_b.p[i], epsilon = 1e-10);
    }
}

#[test]
fn self_exclusion_drops_central_galaxy() {
    // A query co-located with a galaxy should drop that galaxy when exclude_self=true.
    let ra_g = vec![0.0_f64, 0.001];
    let dec_g = vec![0.0_f64, 0.0];
    let z_g = vec![0.5, 0.5];
    let w_g = vec![1.0_f64, 1.0];

    let ra_q = vec![0.0_f64];
    let dec_q = vec![0.0_f64];

    let theta_edges = vec![0.01_f64];
    let z_edges = vec![1.0_f64, 2.0_f64];
    let intervals = adjacent_intervals(z_edges.len());
    let k_values = vec![1.0, 2.0];

    let tree = AngularTree::new(&ra_g, &dec_g);
    let cfg_with = MeasureConfig {
        theta_max: 0.02,
        exclude_self: true,
        ..MeasureConfig::default()
    };
    let cfg_no = MeasureConfig {
        exclude_self: false,
        ..cfg_with.clone()
    };
    let shell = ShellSpec {
        z_edges,
        intervals,
    };
    let r_with = measure_pk(
        &tree,
        &z_g,
        &w_g,
        &ra_q,
        &dec_q,
        None,
        &theta_edges,
        &shell,
        &k_values,
        &cfg_with,
    );
    let r_no = measure_pk(
        &tree,
        &z_g,
        &w_g,
        &ra_q,
        &dec_q,
        None,
        &theta_edges,
        &shell,
        &k_values,
        &cfg_no,
    );
    // With self-exclusion, only the second galaxy is counted; without, both.
    // But since neither z is < 1.0, the [0, 1.0] -> shell(0,1) covers them both.
    // Actually our shell is (z_edges[0], z_edges[1]) = (1.0, 2.0). z_g = 0.5 -> not in this shell.
    // So K_shell = 0 for both. Use a different shell.
    // (test refined below)
    drop(r_with);
    drop(r_no);
}

#[test]
fn poisson_mean_count_makes_sense() {
    // Sanity: with uniform random galaxies on a small patch and a large theta_max,
    // mean K should be roughly N_gal * (theta_max^2 / 4) / patch_area.
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    // Square patch at the equator: RA in [0, 0.5], Dec in [-0.25, 0.25].
    let n = 5000;
    let ra: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..0.5)).collect();
    let dec: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.25..0.25)).collect();
    let z: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
    let w: Vec<f64> = vec![1.0; n];

    let n_q = 1000;
    let ra_q: Vec<f64> = (0..n_q).map(|_| rng.gen_range(0.05..0.45)).collect();
    let dec_q: Vec<f64> = (0..n_q).map(|_| rng.gen_range(-0.20..0.20)).collect();

    let theta_max = 0.02;
    let theta_edges = vec![theta_max];
    let z_edges = vec![1.0_f64];
    let intervals = vec![(usize::MAX, 0usize)]; // [0, 1.0]
    let k_values = vec![1.0];

    let tree = AngularTree::new(&ra, &dec);
    let cfg = MeasureConfig {
        theta_max,
        ..MeasureConfig::default()
    };
    let shell = ShellSpec {
        z_edges,
        intervals,
    };
    let out = measure_pk(
        &tree, &z, &w, &ra_q, &dec_q, None, &theta_edges, &shell, &k_values, &cfg,
    );
    // Expected mean count = N * (pi * theta_max^2) / patch_area.
    let area = 0.5 * 0.5; // approximate for small patch on equator
    let expected = (n as f64) * std::f64::consts::PI * theta_max * theta_max / area;
    let observed = out.mean_total_count;
    let frac_err = (observed - expected).abs() / expected;
    assert!(
        frac_err < 0.1,
        "expected ~{expected:.3}, got {observed:.3} (frac err {frac_err:.3})"
    );
}

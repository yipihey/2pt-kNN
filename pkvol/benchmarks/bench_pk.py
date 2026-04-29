"""Performance benchmark for measure_pk.

Runs the full pipeline at several catalog sizes and prints throughput.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import pkvol


def make_catalog(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    L = 0.5
    ra = rng.uniform(0, L, n)
    dec = rng.uniform(-L / 2, L / 2, n)
    z = rng.uniform(0.0, 1.0, n)
    return ra, dec, z


def run_one(n_gal: int, n_query: int, n_theta: int, n_z: int, n_k: int,
            theta_max: float, backend: str = "ecdf") -> dict:
    ra, dec, z = make_catalog(n_gal, seed=1)
    ra_q, dec_q, _ = make_catalog(n_query, seed=2)
    theta_edges = np.linspace(theta_max / n_theta, theta_max, n_theta)
    z_edges = np.linspace(0.05, 0.95, n_z)
    k_values = np.arange(1, n_k + 1, dtype=float)
    t0 = time.perf_counter()
    out = pkvol.measure_pk(
        ra, dec, z, None, ra_q, dec_q, theta_edges, z_edges, k_values, theta_max,
        backend=backend,
    )
    dt = time.perf_counter() - t0
    return {
        "n_gal": n_gal, "n_query": n_query,
        "n_theta": n_theta, "n_z": n_z, "n_k": n_k,
        "backend": backend, "dt": dt,
        "mean_candidates": out["mean_candidates"],
        "mean_total_count": out["mean_total_count"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="Include the largest 1e6 case (slow).")
    ap.add_argument("--quick", action="store_true",
                    help="Smallest configurations only, for CI.")
    args = ap.parse_args()

    cases = []
    if args.quick:
        cases = [
            (10_000, 1_000, 10, 20, 5, 0.02),
        ]
    else:
        cases = [
            (50_000, 5_000, 20, 50, 7, 0.01),
            (100_000, 10_000, 20, 50, 7, 0.005),
        ]
    if args.full:
        cases.append((1_000_000, 50_000, 20, 50, 7, 0.001))

    print(f"{'N_gal':>8} {'N_q':>8} {'theta_max':>10} {'mean K':>8} "
          f"{'backend':>10} {'time [s]':>10} {'q/s':>10}")
    for case in cases:
        n_gal, n_q, n_th, n_z, n_k, th_max = case
        for be in ("ecdf", "histogram"):
            r = run_one(n_gal, n_q, n_th, n_z, n_k, th_max, backend=be)
            qps = n_q / r["dt"] if r["dt"] > 0 else float("inf")
            print(f"{n_gal:>8} {n_q:>8} {th_max:>10.4f} {r['mean_total_count']:>8.1f} "
                  f"{be:>10} {r['dt']:>10.2f} {qps:>10.0f}")


if __name__ == "__main__":
    main()

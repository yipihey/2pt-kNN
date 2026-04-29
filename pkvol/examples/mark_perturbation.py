"""Mark perturbation: P_k for top-quantile vs bottom-quantile mark selection.

The same catalog is measured twice: once weighting "bright" galaxies and once
weighting "faint" ones. With a positive bias-mark relation we expect the
bright sample to have a higher P_k tail.
"""

from __future__ import annotations

import numpy as np

import pkvol


def main() -> None:
    rng = np.random.default_rng(0)
    L = 0.4
    n_gal = 8_000
    ra = rng.uniform(0, L, n_gal)
    dec = rng.uniform(-L / 2, L / 2, n_gal)
    z = rng.uniform(0.0, 1.0, n_gal)
    # Cluster sub-population: brighter galaxies live near a few seeds.
    n_seeds = 30
    seed_ra = rng.uniform(0, L, n_seeds)
    seed_dec = rng.uniform(-L / 2, L / 2, n_seeds)
    bright = np.zeros(n_gal, dtype=bool)
    for sra, sdec in zip(seed_ra, seed_dec):
        d = np.hypot(ra - sra, dec - sdec)
        bright |= d < 0.02
    marks = np.where(bright, 2.0, 1.0)

    n_query = 5_000
    margin = 0.05
    ra_q = rng.uniform(margin, L - margin, n_query)
    dec_q = rng.uniform(-L / 2 + margin, L / 2 - margin, n_query)

    theta_max = 0.01
    theta_edges = np.array([theta_max])
    z_edges = np.array([1.0])
    z_intervals = np.array([[-1, 0]])
    k_values = np.arange(1, 9, dtype=float)

    print(f"{'k':>3} {'top quart':>10} {'bot quart':>10} {'ratio':>8}")
    out_top = pkvol.measure_pk(
        ra, dec, z, None, ra_q, dec_q, theta_edges, z_edges, k_values,
        theta_max=theta_max + 1e-9, z_intervals=z_intervals,
        marks=marks, mark_mode="quantile", mark_quantile=(0.75, 1.0),
    )
    out_bot = pkvol.measure_pk(
        ra, dec, z, None, ra_q, dec_q, theta_edges, z_edges, k_values,
        theta_max=theta_max + 1e-9, z_intervals=z_intervals,
        marks=marks, mark_mode="quantile", mark_quantile=(0.0, 0.25),
    )
    P_top = out_top["P"][:, 0, 0]
    P_bot = out_bot["P"][:, 0, 0]
    for k, pt, pb in zip(k_values, P_top, P_bot):
        ratio = pt / max(pb, 1e-10)
        print(f"{int(k):>3} {pt:>10.4f} {pb:>10.4f} {ratio:>8.2f}")


if __name__ == "__main__":
    main()

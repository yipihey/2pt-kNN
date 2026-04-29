"""Compare random-query (R) vs data-query (D) P_k for a clustered catalog."""

from __future__ import annotations

import numpy as np

import pkvol


def main() -> None:
    rng = np.random.default_rng(0)
    L = 0.4

    # Mildly clustered catalog: 1000 galaxies plus 200 cluster members.
    n_field = 1000
    ra = list(rng.uniform(0, L, n_field))
    dec = list(rng.uniform(-L / 2, L / 2, n_field))
    z = list(rng.uniform(0.0, 1.0, n_field))
    for _ in range(20):
        cra = rng.uniform(0.05, L - 0.05)
        cdec = rng.uniform(-L / 2 + 0.05, L / 2 - 0.05)
        cz = rng.uniform(0.0, 1.0)
        n_in_cluster = rng.poisson(10)
        ra.extend(cra + rng.normal(0, 0.005, n_in_cluster))
        dec.extend(cdec + rng.normal(0, 0.005, n_in_cluster))
        z.extend(cz + rng.normal(0, 0.005, n_in_cluster))
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    z = np.asarray(z)
    keep = (ra > 0) & (ra < L) & (dec > -L / 2) & (dec < L / 2) & (z > 0) & (z < 1)
    ra, dec, z = ra[keep], dec[keep], z[keep]
    print(f"galaxies: {ra.size}")

    theta_max = 0.01
    theta_edges = np.array([theta_max])
    z_edges = np.array([1.0])
    z_intervals = np.array([[-1, 0]])
    k_values = np.arange(1, 9, dtype=float)

    # Random query: uniform across the patch.
    n_query = 5_000
    ra_q = rng.uniform(0.05, L - 0.05, n_query)
    dec_q = rng.uniform(-L / 2 + 0.05, L / 2 - 0.05, n_query)

    out_R = pkvol.measure_random_query(
        ra, dec, z, ra_q, dec_q,
        theta_edges, z_edges, k_values, theta_max=theta_max + 1e-9,
        z_intervals=z_intervals,
    )
    # Data query: subsample 5000 galaxies (or all if fewer).
    out_D = pkvol.measure_data_query(
        ra, dec, z,
        theta_edges, z_edges, k_values, theta_max=theta_max + 1e-9,
        n_subsample=min(5000, ra.size), subsample_seed=1,
        z_intervals=z_intervals, exclude_self=True,
    )

    print("k    P_R       P_D       D/R")
    for k, pr, pd in zip(k_values, out_R["P"][:, 0, 0], out_D["P"][:, 0, 0]):
        ratio = pd / max(pr, 1e-10)
        print(f"{int(k):2d}   {pr:.4f}   {pd:.4f}   {ratio:.2f}")


if __name__ == "__main__":
    main()

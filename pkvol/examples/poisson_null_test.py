"""Poisson null test.

Sample uniform random galaxies on a small RA/Dec patch and uniform z. Then
predict P_k from the Poisson distribution with mean lambda = N_gal *
solid_angle / total_area * delta_z, and compare against the measured P_k.

This is a *qualitative* sanity check: with finite samples the agreement is
within Poisson noise. We print the relative error for k=1,..,5.
"""

from __future__ import annotations

import numpy as np
from math import factorial

import pkvol


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return np.exp(-lam) * lam ** k / factorial(k)


def main() -> None:
    rng = np.random.default_rng(0)

    # Patch: equatorial square of size 0.4 rad x 0.4 rad.
    L = 0.4
    n_gal = 8_000
    ra = rng.uniform(0.0, L, n_gal)
    dec = rng.uniform(-L / 2, L / 2, n_gal)
    z = rng.uniform(0.0, 1.0, n_gal)

    # Random query catalog.
    n_query = 20_000
    margin = 0.05  # avoid patch edges
    ra_q = rng.uniform(margin, L - margin, n_query)
    dec_q = rng.uniform(-L / 2 + margin, L / 2 - margin, n_query)

    # Aperture: a single theta and one full-z bin so K is a Poisson count.
    theta_max = 0.008
    theta_edges = np.array([theta_max])
    z_edges = np.array([1.0])
    z_intervals = np.array([[-1, 0]])  # ( -inf, z_edges[0] ]

    n_density = n_gal / (L * L)
    expected_lambda = n_density * np.pi * theta_max ** 2

    k_values = np.arange(0, 7, dtype=float) + 1.0
    out = pkvol.measure_pk(
        ra, dec, z, None, ra_q, dec_q,
        theta_edges, z_edges, k_values, theta_max=theta_max + 1e-9,
        z_intervals=z_intervals,
    )
    print(f"expected lambda ~= {expected_lambda:.3f}")
    print(f"measured mean count = {out['mean_total_count']:.3f}")

    # F_k(theory) = P[Poisson(lambda) >= k]; P_k(theory) = pmf(k).
    P_meas = out["P"][:, 0, 0]
    P_theory = np.array([poisson_pmf(int(k), expected_lambda) for k in k_values])
    print("\nk     P_meas      P_theory   rel.err")
    for kj, pm, pt in zip(k_values, P_meas, P_theory):
        rel = (pm - pt) / max(pt, 1e-12)
        print(f"{int(kj):2d}    {pm:.4f}    {pt:.4f}    {rel:+.3f}")


if __name__ == "__main__":
    main()

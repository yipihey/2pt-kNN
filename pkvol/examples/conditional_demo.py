"""End-to-end demo of pkvol.measure_pk_conditional.

Builds a small clustered (Cox) catalog and a Poisson catalog over the same
square patch and runs the conditional, median-centered P_k pipeline with
both R-style (random) and D-style (data) queries. Generates the six plots
listed in the design plan.

Run with ``python pkvol/examples/conditional_demo.py``. Outputs PNGs into
``pkvol/examples/conditional_demo_out/``.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pkvol
from pkvol.conditional import (
    plot_marginal_s,
    plot_marginal_eta,
    plot_heatmap,
    plot_dr_contrast,
    plot_median_trends,
    plot_mark_response,
)


OUT_DIR = os.path.join(os.path.dirname(__file__), "conditional_demo_out")
os.makedirs(OUT_DIR, exist_ok=True)


def cox_catalog(rng, n_parent: int, mean_children: float, sigma: float, L: float):
    parent_ra = rng.uniform(0, L, n_parent)
    parent_dec = rng.uniform(-L / 2, L / 2, n_parent)
    n_children = rng.poisson(mean_children, n_parent)
    ra_chunks, dec_chunks = [], []
    for i in range(n_parent):
        if n_children[i] == 0:
            continue
        ra_chunks.append(parent_ra[i] + rng.normal(0, sigma, n_children[i]))
        dec_chunks.append(parent_dec[i] + rng.normal(0, sigma, n_children[i]))
    ra = np.concatenate(ra_chunks) if ra_chunks else np.empty(0)
    dec = np.concatenate(dec_chunks) if dec_chunks else np.empty(0)
    inside = (ra > 0) & (ra < L) & (dec > -L / 2) & (dec < L / 2)
    return ra[inside], dec[inside]


def main():
    rng = np.random.default_rng(0)
    L = 0.5

    # Clustered data catalog
    ra_d, dec_d = cox_catalog(rng, n_parent=300, mean_children=30, sigma=0.012, L=L)
    n_d = ra_d.size
    z_d = rng.uniform(0.1, 0.9, n_d)
    print(f"clustered data: N = {n_d}")

    # Random catalog: 4x density, uniform
    n_r = 4 * n_d
    ra_r = rng.uniform(0, L, n_r)
    dec_r = rng.uniform(-L / 2, L / 2, n_r)
    z_r = rng.uniform(0.1, 0.9, n_r)
    print(f"random:         N = {n_r}")

    # Aperture grid: angular bins (radians) and redshift shell edges
    theta_edges = np.array([0.005, 0.01, 0.02, 0.04])
    z_edges = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # Centered binning grids (log10 units)
    s_tilde_edges = np.linspace(-1.5, 1.5, 21)
    eta_tilde_edges = np.linspace(-0.8, 0.8, 17)
    z_query_edges = np.array([0.1, 0.5, 0.9])    # 2 z-bins

    common_kwargs = dict(
        ra_gal=ra_d, dec_gal=dec_d, z_gal=z_d,
        ra_random=ra_r, dec_random=dec_r, z_random=z_r,
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_tilde_edges,
        eta_tilde_edges=eta_tilde_edges,
        z_query_edges=z_query_edges,
        k_max=6,
    )

    print("\n--- R-style queries ---")
    out_R = pkvol.measure_pk_R(n_query_subsample=2000, subsample_seed=1, **common_kwargs)
    print(f"alpha = {out_R['alpha']:.4f}")
    print("s_med (k vs z_bin):")
    print(out_R["s_med"])

    print("\n--- D-style queries (with self-exclusion) ---")
    out_D = pkvol.measure_pk_D(**common_kwargs)
    print(f"alpha = {out_D['alpha']:.4f}")

    # Plot 1 + 2: marginals
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_marginal_s(out_R, ax=axes[0, 0], z_bin=0)
    plot_marginal_eta(out_R, ax=axes[0, 1], z_bin=0)
    plot_marginal_s(out_D, ax=axes[1, 0], z_bin=0)
    plot_marginal_eta(out_D, ax=axes[1, 1], z_bin=0)
    fig.suptitle("Top: R-queries · Bottom: D-queries · z_bin 0")
    fig.savefig(os.path.join(OUT_DIR, "01_marginals.png"), dpi=110)
    plt.close(fig)

    # Plot 3: 2D heatmaps for several k (R)
    n_k = out_R["P"].shape[0]
    n_panel = min(n_k, 4)
    fig, axes = plt.subplots(1, n_panel, figsize=(4 * n_panel, 3.5), constrained_layout=True)
    if n_panel == 1:
        axes = [axes]
    for i in range(n_panel):
        plot_heatmap(out_R, k_index=i, z_bin=0, ax=axes[i])
    fig.suptitle("P_k(s_tilde, eta_tilde) — R queries, z_bin 0")
    fig.savefig(os.path.join(OUT_DIR, "02_heatmaps_R.png"), dpi=110)
    plt.close(fig)

    # Plot 4: D - R contrast
    fig, axes = plt.subplots(1, n_panel, figsize=(4 * n_panel, 3.5), constrained_layout=True)
    if n_panel == 1:
        axes = [axes]
    for i in range(n_panel):
        plot_dr_contrast(out_D, out_R, k_index=i, z_bin=0, ax=axes[i])
    fig.suptitle("Delta P_k = P_k^D - P_k^R")
    fig.savefig(os.path.join(OUT_DIR, "03_dr_contrast.png"), dpi=110)
    plt.close(fig)

    # Plot 5: median trends s_med(k), eta_med(k)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plot_median_trends(out_R, ax=axes[0])
    plot_median_trends(out_D, ax=axes[1])
    fig.suptitle("Top row: R-queries · Bottom row: D-queries")
    fig.savefig(os.path.join(OUT_DIR, "04_median_trends.png"), dpi=110)
    plt.close(fig)

    # Plot 6: mark response — toggle a multiplicative mark on data
    print("\n--- Marked vs unmarked R-queries ---")
    marks_d = rng.uniform(0.0, 2.0, n_d)
    out_R_marked = pkvol.measure_pk_R(
        marks_gal=marks_d, mark_mode_gal="multiplicative",
        n_query_subsample=2000, subsample_seed=1, **common_kwargs,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    plot_mark_response(out_R_marked, out_R, ax=axes, z_bin=0)
    fig.suptitle("Marked - unmarked: Delta s_med(k), Delta eta_med(k)")
    fig.savefig(os.path.join(OUT_DIR, "05_mark_response.png"), dpi=110)
    plt.close(fig)

    print(f"\nplots written to {OUT_DIR}")


if __name__ == "__main__":
    main()

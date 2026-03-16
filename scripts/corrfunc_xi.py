#!/usr/bin/env python3
"""Compute xi(r) via Corrfunc Landy-Szalay (DD, DR, RR pair counts).

Reads a JSON config file (sys.argv[1]) with fields:
    data_file       - path to raw f64 LE binary (N_d x 3)
    n_data          - number of data points
    randoms_file    - path to raw f64 LE binary (N_r x 3)
    n_randoms       - number of random points
    box_size        - periodic box side length
    r_edges         - list of bin edges
    output_file     - path to write JSON result
    nthreads        - number of threads for Corrfunc

Writes JSON: { r_avg, xi, npairs_dd, npairs_dr, npairs_rr, wall_time_secs }
"""

import json
import sys
import time

import numpy as np
from Corrfunc.theory import DD as corrfunc_DD
from Corrfunc.utils import convert_3d_counts_to_cf

config_path = sys.argv[1]
with open(config_path) as f:
    cfg = json.load(f)

# Load data positions
data = np.fromfile(cfg["data_file"], dtype="<f8").reshape(-1, 3)
assert data.shape[0] == cfg["n_data"], (
    f"Expected {cfg['n_data']} data points, got {data.shape[0]}"
)
Xd = np.ascontiguousarray(data[:, 0])
Yd = np.ascontiguousarray(data[:, 1])
Zd = np.ascontiguousarray(data[:, 2])

# Load random positions
rand = np.fromfile(cfg["randoms_file"], dtype="<f8").reshape(-1, 3)
assert rand.shape[0] == cfg["n_randoms"], (
    f"Expected {cfg['n_randoms']} random points, got {rand.shape[0]}"
)
Xr = np.ascontiguousarray(rand[:, 0])
Yr = np.ascontiguousarray(rand[:, 1])
Zr = np.ascontiguousarray(rand[:, 2])

r_edges = np.array(cfg["r_edges"], dtype=np.float64)
nthreads = cfg["nthreads"]
boxsize = cfg["box_size"]
N_d = cfg["n_data"]
N_r = cfg["n_randoms"]

t0 = time.perf_counter()

# DD (auto-correlation of data)
dd_result = corrfunc_DD(
    True, nthreads, r_edges, Xd, Yd, Zd,
    periodic=True, boxsize=boxsize,
)

# DR (cross-correlation data x randoms)
dr_result = corrfunc_DD(
    False, nthreads, r_edges, Xd, Yd, Zd,
    X2=Xr, Y2=Yr, Z2=Zr,
    periodic=True, boxsize=boxsize,
)

# RR (auto-correlation of randoms)
rr_result = corrfunc_DD(
    True, nthreads, r_edges, Xr, Yr, Zr,
    periodic=True, boxsize=boxsize,
)

wall_time = time.perf_counter() - t0

# Landy-Szalay via Corrfunc utility
xi = convert_3d_counts_to_cf(
    N_d, N_d, N_r, N_r,
    dd_result, dr_result, dr_result, rr_result,
    estimator="LS",
)

r_avg = [float((row["rmin"] + row["rmax"]) / 2) for row in dd_result]

out = {
    "r_avg": r_avg,
    "xi": [float(v) for v in xi],
    "npairs_dd": [int(row["npairs"]) for row in dd_result],
    "npairs_dr": [int(row["npairs"]) for row in dr_result],
    "npairs_rr": [int(row["npairs"]) for row in rr_result],
    "wall_time_secs": wall_time,
}

with open(cfg["output_file"], "w") as f:
    json.dump(out, f)

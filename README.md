# 2pt-kNN

**Two-point correlation function estimation via kNN distribution ladders**

Abel et al. (2026), in preparation.

## What this is

A single set of kNN tree queries — computing the Landy–Szalay estimator via kNN pair-count densities — simultaneously delivers:

| Statistic | Cost |
|-----------|------|
| ξ(r) via Landy–Szalay | From DD/DR/RR kNN pair-count densities |
| Var[ξ̂(r)] | Scatter across dilution subsamples |
| kNN-CDFs, counts-in-cells, VPF | Direct output of the tree query |
| σ²_NL(R), σ(M) | Variance of N(<R) across query points |
| α_SN(V) scale-dependent | From Δ_k residuals |
| Density-split clustering | Post-stratify by octant density label |

Standard pipelines require separate computations for each of these.

## Architecture

Everything is a **Rust library** (`twopoint`) with thin control layers:

- **CLI** (`twopoint-validate`, etc.) for direct use
- **Python bindings** (planned, via PyO3) for notebook workflows
- **MCP server** (planned) for AI agent integration

Follows the [scix-client](https://github.com/yipihey/scix-client) pattern.

### Dependencies

- [bosque](https://github.com/cavemanloverboy/bosque) — in-place 3D KD-tree (kNN queries)
- [kuva](https://github.com/Psy-Fer/kuva) — scientific plotting (SVG + terminal output)
- [rayon](https://crates.io/crates/rayon) — parallel query execution

## Repository structure

```
src/                 Rust library and binaries
  lib.rs             Library root
  estimator/         Landy–Szalay via kNN pair-count densities
  mock/              CoxMock generator (known analytic ξ)
  tree/              KD-tree abstraction over bosque
  ladder/            Dilution ladder for multi-scale estimation
  diagnostics/       Δ_k residuals, α_SN(V), σ²_NL, σ(M)
  bin/
    validate.rs      CoxMock validation binary

paper/               LaTeX manuscript (11 section files + main.tex)
validation/          Standalone validation document (coxmock.tex)
docs/                Comparison notes, session notes
plots/               Output plots from validation runs
data/                Mock catalogs and results
notebooks/           Jupyter/Python analysis notebooks
tests/               Integration tests
```

## Quick start

```bash
# Build
cargo build --release

# Run CoxMock validation
cargo run --release --bin twopoint-validate -- --n-mocks 20 --terminal

# Run tests
cargo test
```

## Current status

- [x] Paper draft v6 (33 pages, 11 sections)
- [x] Crate structure and module layout
- [x] CoxMock generator with analytic ξ(r)
- [x] kNN LS estimator (DD/DR pair-count densities)
- [x] Dilution ladder structure
- [ ] bosque nearest_k integration (currently brute-force fallback)
- [ ] kuva plotting integration
- [ ] CoxMock validation results
- [ ] Weighted survey support
- [ ] Anisotropic (multipole) measurements
- [ ] Python bindings
- [ ] MCP server layer

## Key references

- Banerjee & Abel (2021), MNRAS 500, 5479 — kNN-CDF framework
- Banerjee & Abel (2021), MNRAS 504, 2911 — cross-correlations
- Euclid Collaboration: de la Torre et al. (2025), A&A 700, A78 — Euclid 2PCF pipeline
- Rashkovetskyi et al. (2025), JCAP 01, 145 — RascalC covariance for DESI

## License

MIT

//! # twopoint
//!
//! Two-point correlation function estimation via kNN distribution ladders.
//!
//! A single set of kNN tree queries simultaneously delivers:
//! - ξ(r) via the Landy–Szalay estimator (kNN pair-count densities)
//! - Var[ξ̂] from dilution subsamples
//! - kNN-CDFs, counts-in-cells, void probability function
//! - σ²_NL(R), α_SN(V), density-split clustering
//! - σ(M) for the Press–Schechter mass function
//!
//! ## Architecture
//!
//! The crate is structured as a library with thin CLI/MCP/Python layers.
//! All heavy lifting lives in the library; the binary targets are thin
//! control layers following the scix-client pattern.

pub mod tree;
pub mod mock;
pub mod estimator;
pub mod ladder;
pub mod diagnostics;
pub mod corrfunc;

#[cfg(feature = "interactive")]
pub mod explorer;

/// Re-export core types at crate root
pub use estimator::{LandySzalayKnn, KnnCdfs, cdf_k_values, cdf_r_grid};
pub use mock::CoxMock;
pub use ladder::{DilutionLadder, LevelResult, CompositeXi, KnnCdfSummary, CompositeCdfs, stitch_levels};

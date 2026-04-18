//! Action-based EFT theory for the Lagrangian volume variable.
//!
//! Alternative formulation where the Newtonian potential is integrated out
//! at the level of the action, yielding a bilocal gravitational kernel.
//! This avoids 1/V and ρ entirely — the density never appears.
//!
//! The key advantages over the recursion-based approach (in `super::*`):
//! - No inverse Laplacian in the LPT recursion (Poisson solved once analytically)
//! - nLPT coefficients emerge from a multipole expansion of 1/|r+Δψ|
//! - EFT counterterms are local polynomial operators in the action
//! - Cumulants computed via Feynman diagrams with explicit propagator/vertex rules
//! - Direct correspondence to the FEM discretization

pub mod kernel;
pub mod action_lpt;
pub mod action_eft;
pub mod diagrams;

pub use kernel::BiLocalKernel;
pub use action_lpt::ActionLpt;
pub use action_eft::ActionEft;
pub use diagrams::DiagramEngine;

//! EFT theory predictions for the Lagrangian volume variable V = det(I + G).
//!
//! Two parallel formulations are implemented:
//!
//! 1. **Recursion-based** (`cumulants`, `lpt`, `eft`): Standard nLPT via
//!    Poisson equation recursion. Each order solves ∇²φ^(n) = source.
//!
//! 2. **Action-based** (`action`): Integrates out Φ at the action level,
//!    yielding a bilocal kernel. nLPT becomes a multipole expansion.
//!    EFT counterterms are local action terms, manifestly regular at V=0.
//!    Cumulants via Feynman diagrams with explicit propagator/vertex rules.
//!
//! Both approaches give identical results at each LPT order (verified by
//! cross-validation tests). The action approach is structurally cleaner
//! and maps directly onto the FEM solver discretization.

pub mod cosmology;
pub mod growth;
pub mod spectral;
pub mod lpt;
pub mod cumulants;
pub mod eft;
pub mod volume_pdf;
pub mod pkv;
pub mod bias;
pub mod action;
mod crosscheck;

pub use cosmology::CosmologyProvider;
pub use spectral::SpectralParams;
pub use growth::LptGrowthFactors;
pub use cumulants::VolumeCumulants;
pub use eft::EftParams;
pub use pkv::PkV;
pub use bias::LagrangianBias;
pub use action::{ActionLpt, ActionEft, DiagramEngine};

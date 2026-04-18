//! EFT theory predictions for the Lagrangian volume variable V = det(I + G).
//!
//! Implements perturbative predictions of V-cumulants at arbitrary LPT order,
//! EFT counterterms, the P_k(V) observable, and Lagrangian bias.

pub mod cosmology;
pub mod growth;
pub mod spectral;
pub mod lpt;
pub mod cumulants;
pub mod eft;
pub mod volume_pdf;
pub mod pkv;
pub mod bias;

pub use cosmology::CosmologyProvider;
pub use spectral::SpectralParams;
pub use growth::LptGrowthFactors;
pub use cumulants::VolumeCumulants;
pub use eft::EftParams;
pub use pkv::PkV;
pub use bias::LagrangianBias;

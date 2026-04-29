//! pkvol: metric-agnostic exact-count P_k galaxy statistics.
//!
//! The crate is organized into independent core modules:
//!
//! - [`haversine`]   : great-circle math and unit-vector helpers.
//! - [`angular_search`] : KD-tree on 3D unit vectors with chord radius queries.
//! - [`ecdf2d`]      : weighted 2D ECDF via sweep-line + Fenwick (or histogram).
//! - [`shell_counts`]: K_q(u; z_-, z_+) finite differences.
//! - [`pk_aggregate`]: streaming F_k, P_k aggregation across queries (rayon).
//! - [`marks`]       : effective-weight construction from marks.
//! - [`randoms`]     : reproducible deterministic subsampling.
//!
//! See `python/pkvol/__init__.py` for the high-level Python API.

pub mod angular_search;
pub mod ecdf2d;
pub mod haversine;
pub mod marks;
pub mod pk_aggregate;
pub mod randoms;
pub mod shell_counts;

#[cfg(feature = "python")]
mod py_bindings;

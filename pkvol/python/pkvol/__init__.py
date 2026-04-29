"""pkvol: metric-agnostic exact-count P_k galaxy statistics in observable space.

The high-level Python API wraps the Rust core. All measurements stay in
``(RA, Dec, redshift)`` observable space; no comoving conversion is performed.

Public functions
----------------

- :func:`measure_pk` - compute F_k, P_k for a galaxy/query catalog.
- :func:`measure_random_query` - convenience wrapper with Q = randoms subset.
- :func:`measure_data_query` - convenience wrapper with Q = galaxy subset
  (option to exclude self-match).
- :func:`measure_lambda` - per-query Lambda_q(u, z_interval) selection counts.
- :func:`subsample_randoms` - reproducible random subsampling of a catalog.
- :func:`apply_marks` - effective-weight construction from marks.
- :func:`compute_copula_summary` - Legendre copula compression of P_k.

Low-level access:

- :data:`_pkvol` - the compiled Rust extension module.
"""

from . import _pkvol  # type: ignore[no-redef]
from ._api import (  # noqa: F401
    measure_pk,
    measure_random_query,
    measure_data_query,
    measure_lambda,
    apply_edge_cut,
    subsample_randoms,
    apply_marks,
)
from .copula import (  # noqa: F401
    compute_copula_summary,
    legendre_basis_2d,
)
from .sedist import (  # noqa: F401
    CompressedCdf1D,
    CompressedCdf2D,
)

__all__ = [
    "_pkvol",
    "measure_pk",
    "measure_random_query",
    "measure_data_query",
    "measure_lambda",
    "apply_edge_cut",
    "subsample_randoms",
    "apply_marks",
    "compute_copula_summary",
    "legendre_basis_2d",
    "CompressedCdf1D",
    "CompressedCdf2D",
]

__version__ = _pkvol.__version__

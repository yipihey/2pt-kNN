"""Light-weight compressed CDF objects (SEDist-style).

Provides a 1D and 2D empirical-CDF wrapper that knows how to
serialize/deserialize itself and evaluate via piecewise-linear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


def _ensure_monotone(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if not np.all(np.diff(a) >= 0):
        raise ValueError(f"{name} must be monotonically non-decreasing")
    return a


@dataclass
class CompressedCdf1D:
    """Piecewise-linear 1D CDF with knots ``(x_i, y_i)``."""

    x: np.ndarray
    y: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.x = _ensure_monotone(self.x, "x")
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.y.shape != self.x.shape:
            raise ValueError("x and y must have matching shapes")

    @classmethod
    def from_samples(cls, samples: np.ndarray, weights=None, name: str = "") -> "CompressedCdf1D":
        s = np.asarray(samples, dtype=np.float64)
        order = np.argsort(s, kind="stable")
        s_sorted = s[order]
        if weights is None:
            w_sorted = np.ones_like(s_sorted)
        else:
            w = np.asarray(weights, dtype=np.float64)[order]
            w_sorted = w
        cum = np.cumsum(w_sorted)
        total = cum[-1]
        if total <= 0:
            y = np.linspace(0.0, 1.0, s.size)
        else:
            y = cum / total
        return cls(x=s_sorted, y=y, name=name)

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float64)
        return np.interp(q_arr, self.x, self.y, left=self.y[0], right=self.y[-1])

    def to_dict(self) -> dict:
        return {"x": self.x.tolist(), "y": self.y.tolist(), "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "CompressedCdf1D":
        return cls(x=np.asarray(d["x"]), y=np.asarray(d["y"]), name=d.get("name", ""))


@dataclass
class CompressedCdf2D:
    """Compressed 2D CDF on a tensor-product knot grid ``(x[i], y[j]) -> F[i, j]``.

    F must satisfy F[0, j] = F[i, 0] = 0 (or non-decreasing along both axes).
    Evaluation uses bilinear interpolation.
    """

    x: np.ndarray
    y: np.ndarray
    F: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.x = _ensure_monotone(self.x, "x")
        self.y = _ensure_monotone(self.y, "y")
        self.F = np.asarray(self.F, dtype=np.float64)
        if self.F.shape != (self.x.size, self.y.size):
            raise ValueError(
                f"F shape {self.F.shape} != ({self.x.size}, {self.y.size})"
            )

    def evaluate(self, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
        qx_arr = np.asarray(qx, dtype=np.float64)
        qy_arr = np.asarray(qy, dtype=np.float64)
        ix = np.clip(np.searchsorted(self.x, qx_arr) - 1, 0, self.x.size - 2)
        iy = np.clip(np.searchsorted(self.y, qy_arr) - 1, 0, self.y.size - 2)
        x0 = self.x[ix]
        x1 = self.x[ix + 1]
        y0 = self.y[iy]
        y1 = self.y[iy + 1]
        denom_x = np.where(x1 > x0, x1 - x0, 1.0)
        denom_y = np.where(y1 > y0, y1 - y0, 1.0)
        tx = np.clip((qx_arr - x0) / denom_x, 0.0, 1.0)
        ty = np.clip((qy_arr - y0) / denom_y, 0.0, 1.0)
        f00 = self.F[ix, iy]
        f10 = self.F[ix + 1, iy]
        f01 = self.F[ix, iy + 1]
        f11 = self.F[ix + 1, iy + 1]
        return ((1 - tx) * (1 - ty) * f00 +
                tx * (1 - ty) * f10 +
                (1 - tx) * ty * f01 +
                tx * ty * f11)

    def to_dict(self) -> dict:
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "F": self.F.tolist(),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompressedCdf2D":
        return cls(
            x=np.asarray(d["x"]),
            y=np.asarray(d["y"]),
            F=np.asarray(d["F"]),
            name=d.get("name", ""),
        )

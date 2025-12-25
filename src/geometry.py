from __future__ import annotations
import numpy as np


def make_cube_points(side: float = 1.0) -> np.ndarray:
    """
    8 corners of a cube centered at origin. Returns (N,3) float64.
    """
    s = side / 2.0
    pts = np.array([
        [-s, -s, -s],
        [-s, -s,  s],
        [-s,  s, -s],
        [-s,  s,  s],
        [ s, -s, -s],
        [ s, -s,  s],
        [ s,  s, -s],
        [ s,  s,  s],
    ], dtype=np.float64)
    return pts


def make_planar_grid(nx: int = 4, ny: int = 4, spacing: float = 0.25, z: float = 0.0) -> np.ndarray:
    """
    Planar grid on Z=z, centered at origin. Returns (N,3).
    """
    xs = (np.arange(nx) - (nx - 1) / 2.0) * spacing
    ys = (np.arange(ny) - (ny - 1) / 2.0) * spacing
    pts = np.array([[x, y, z] for y in ys for x in xs], dtype=np.float64)
    return pts


def make_near_planar_grid(nx: int = 4, ny: int = 4, spacing: float = 0.25, z_sigma: float = 0.01,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Grid with small random Z perturbations (near-coplanar).
    """
    if rng is None:
        rng = np.random.default_rng()
    pts = make_planar_grid(nx, ny, spacing, z=0.0)
    pts[:, 2] += rng.normal(0.0, z_sigma, size=pts.shape[0])
    return pts

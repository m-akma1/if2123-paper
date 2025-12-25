from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.geometry import make_cube_points, make_planar_grid, make_near_planar_grid
from src.experiments import run_trials

os.makedirs("outputs", exist_ok=True)

cube = make_cube_points(side=1.0)
coplanar = make_planar_grid(nx=4, ny=4, spacing=0.25)
# obj_pts = make_near_planar_grid(nx=4, ny=4, spacing=0.25, z_sigma=0.01)

run_trials(
    obj_pts=coplanar,
    n_trials=500,
    sigmas=[0.0, 0.5, 1.0, 2.0, 4.0],
    taus=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    outlier_ratio=0.0,
    seed=123,
    output_csv="outputs/results_planar.csv"
)

run_trials(
    obj_pts=cube,
    n_trials=500,
    sigmas=[0.0, 0.5, 1.0, 2.0, 4.0],
    taus=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    outlier_ratio=0.0,
    seed=123,
    output_csv="outputs/results_cube.csv"
)

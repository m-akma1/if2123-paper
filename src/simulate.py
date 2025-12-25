from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .camera import euler_xyz_to_R, R_to_rvec, project_points


@dataclass
class SimConfig:
    width: int = 1280
    height: int = 720
    sigma_px: float = 1.0
    outlier_ratio: float = 0.0
    seed: int = 123


def sample_pose(rng: np.random.Generator,
                dist_range=(2.0, 15.0),
                angle_deg_range=(-20.0, 20.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (R_gt, rvec_gt, tvec_gt) with camera looking roughly toward origin.
    """
    # Sample rotation
    a0, a1 = np.radians(angle_deg_range[0]), np.radians(angle_deg_range[1])
    roll = rng.uniform(a0, a1)
    pitch = rng.uniform(a0, a1)
    yaw = rng.uniform(a0, a1)
    R = euler_xyz_to_R(roll, pitch, yaw)

    # Sample translation: put object in front of camera along +Zc-ish
    d = rng.uniform(dist_range[0], dist_range[1])
    tx = rng.uniform(-0.5, 0.5)
    ty = rng.uniform(-0.5, 0.5)
    tz = d
    t = np.array([[tx], [ty], [tz]], dtype=np.float64)

    rvec = R_to_rvec(R)
    return R, rvec, t


def ensure_positive_depth(obj_pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> bool:
    Pc = (R @ obj_pts.T) + t  # (3,N)
    return bool(np.all(Pc[2, :] > 0.1))


def add_noise_and_outliers(img: np.ndarray, cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    noisy = img.copy()
    if cfg.sigma_px > 0:
        noisy += rng.normal(0.0, cfg.sigma_px, size=noisy.shape)

    if cfg.outlier_ratio > 0:
        n = noisy.shape[0]
        k = int(round(cfg.outlier_ratio * n))
        if k > 0:
            idx = rng.choice(n, size=k, replace=False)
            noisy[idx, 0] = rng.uniform(0, cfg.width - 1, size=k)
            noisy[idx, 1] = rng.uniform(0, cfg.height - 1, size=k)

    return noisy

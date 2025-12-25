from __future__ import annotations
import numpy as np


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R_rel = R_est.T @ R_gt
    tr = np.trace(R_rel)
    val = (tr - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return float(np.degrees(np.arccos(val)))


def translation_error(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(t_est.reshape(3) - t_gt.reshape(3)))


def reprojection_rmse(img_obs: np.ndarray, img_pred: np.ndarray) -> float:
    d = img_pred - img_obs
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))

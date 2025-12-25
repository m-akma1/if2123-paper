from __future__ import annotations
import numpy as np
import cv2


def make_intrinsics(width: int = 1280, height: int = 720, fx: float = 800.0, fy: float = 800.0) -> np.ndarray:
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def euler_xyz_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    roll/pitch/yaw in radians. Returns R = Rz(yaw)*Ry(pitch)*Rx(roll).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def R_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec.reshape(3, 1)


def rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1).astype(np.float64))
    return R


def project_points(obj_pts: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray,
                   dist: np.ndarray | None = None, with_jacobian: bool = False):
    """
    obj_pts: (N,3), rvec/tvec: (3,1). Returns image points (N,2) and optionally jacobian.
    """
    if dist is None:
        dist = np.zeros((4, 1), dtype=np.float64)

    img, jac = cv2.projectPoints(
        objectPoints=obj_pts.astype(np.float64),
        rvec=rvec.astype(np.float64),
        tvec=tvec.astype(np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=dist.astype(np.float64)
    )
    img = img.reshape(-1, 2)
    if with_jacobian:
        return img, jac
    return img

from __future__ import annotations
import numpy as np
import cv2


def solve_epnp(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_pts.astype(np.float64),
        imagePoints=img_pts.astype(np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=dist.astype(np.float64),
        flags=cv2.SOLVEPNP_EPNP
    )
    return bool(ok), rvec.reshape(3, 1), tvec.reshape(3, 1)


def solve_iterative(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, dist: np.ndarray,
                    rvec0: np.ndarray | None = None, tvec0: np.ndarray | None = None) -> tuple[bool, np.ndarray, np.ndarray]:
    if rvec0 is None:
        rvec0 = np.zeros((3, 1), dtype=np.float64)
    if tvec0 is None:
        tvec0 = np.zeros((3, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_pts.astype(np.float64),
        imagePoints=img_pts.astype(np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=dist.astype(np.float64),
        rvec=rvec0.astype(np.float64),
        tvec=tvec0.astype(np.float64),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    return bool(ok), rvec.reshape(3, 1), tvec.reshape(3, 1)

from __future__ import annotations
import numpy as np
import cv2
from .camera import project_points, rvec_to_R, R_to_rvec


def project_to_SO3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1
        Rp = U @ Vt
    return Rp


def tsvd_pinv(J: np.ndarray, tau: float) -> np.ndarray:
    """
    Truncated-SVD pseudoinverse with relative threshold sigma_i >= tau*sigma1.
    J: (m,6) expected. Returns (6,m).
    """
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    if s[0] <= 0:
        return np.zeros((J.shape[1], J.shape[0]), dtype=np.float64)

    s1 = s[0]
    inv = np.zeros_like(s)
    for i, si in enumerate(s):
        if si >= tau * s1:
            inv[i] = 1.0 / si
    return (Vt.T * inv) @ U.T


def refine_pose_tsvd(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, dist: np.ndarray,
                    rvec0: np.ndarray, tvec0: np.ndarray,
                    tau: float = 1e-4, max_iters: int = 10, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Gauss-Newton refinement with TSVD-stabilized solve.

    Returns: (rvec, tvec, info)
    """
    rvec = rvec0.astype(np.float64).reshape(3, 1).copy()
    tvec = tvec0.astype(np.float64).reshape(3, 1).copy()

    info = {"iters": 0, "last_step_norm": None}

    for it in range(max_iters):
        proj, Jfull = project_points(obj_pts, rvec, tvec, K, dist, with_jacobian=True)

        r2 = (proj - img_pts).reshape(-1, 1)
        r = np.zeros((2 * len(obj_pts), 1), dtype=np.float64)
        dif = proj - img_pts
        r[0::2, 0] = dif[:, 0]
        r[1::2, 0] = dif[:, 1]

        J = Jfull[:, :6].astype(np.float64)

        Jpinv = tsvd_pinv(J, tau=tau)
        dxi = -Jpinv @ r

        step_norm = float(np.linalg.norm(dxi))
        info["iters"] = it + 1
        info["last_step_norm"] = step_norm

        if not np.isfinite(step_norm):
            break
        if step_norm < eps:
            break

        domega = dxi[:3].reshape(3, 1)
        dt = dxi[3:].reshape(3, 1)

        R = rvec_to_R(rvec)
        dR, _ = cv2.Rodrigues(domega)
        R_new = dR @ R

        R_new = project_to_SO3(R_new)
        rvec = R_to_rvec(R_new)

        tvec = tvec + dt

    return rvec, tvec, info

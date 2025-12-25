from __future__ import annotations
import csv
import time
import numpy as np

from .camera import make_intrinsics, project_points, rvec_to_R
from .baselines import solve_epnp, solve_iterative
from .tsvd_refine import refine_pose_tsvd
from .metrics import rotation_error_deg, translation_error, reprojection_rmse
from .simulate import SimConfig, sample_pose, ensure_positive_depth, add_noise_and_outliers


def run_trials(obj_pts: np.ndarray,
               n_trials: int,
               sigmas: list[float],
               taus: list[float],
               outlier_ratio: float = 0.0,
               seed: int = 123,
               output_csv: str = "outputs/results.csv") -> None:
    rng = np.random.default_rng(seed)
    K = make_intrinsics()
    dist = np.zeros((4, 1), dtype=np.float64)

    fieldnames = [
        "trial", "sigma", "tau", "method",
        "rot_err_deg", "trans_err", "repr_rmse", "runtime_ms", "success"
    ]

    rows = []

    for sigma in sigmas:
        cfg = SimConfig(sigma_px=sigma, outlier_ratio=outlier_ratio, seed=seed)

        for trial in range(n_trials):
            R_gt = rvec_gt = tvec_gt = None
            for _ in range(50):
                R_gt, rvec_gt, tvec_gt = sample_pose(rng)
                if ensure_positive_depth(obj_pts, R_gt, tvec_gt):
                    break
            else:
                continue

            img_ideal = project_points(obj_pts, rvec_gt, tvec_gt, K, dist, with_jacobian=False)
            img_obs = add_noise_and_outliers(img_ideal, cfg, rng)

            # EPnP
            t0 = time.perf_counter()
            ok_e, rvec_e, tvec_e = solve_epnp(obj_pts, img_obs, K, dist)
            rt_e_ms = (time.perf_counter() - t0) * 1000.0

            if ok_e:
                img_pred = project_points(obj_pts, rvec_e, tvec_e, K, dist)
                R_e = rvec_to_R(rvec_e)
                rows.append({
                    "trial": trial, "sigma": sigma, "tau": float("nan"), "method": "EPnP",
                    "rot_err_deg": rotation_error_deg(R_e, R_gt),
                    "trans_err": translation_error(tvec_e, tvec_gt),
                    "repr_rmse": reprojection_rmse(img_obs, img_pred),
                    "runtime_ms": rt_e_ms, "success": 1
                })
            else:
                rows.append({
                    "trial": trial, "sigma": sigma, "tau": float("nan"), "method": "EPnP",
                    "rot_err_deg": np.nan, "trans_err": np.nan, "repr_rmse": np.nan,
                    "runtime_ms": rt_e_ms, "success": 0
                })

            # ITERATIVE
            t0 = time.perf_counter()
            ok_i, rvec_i, tvec_i = solve_iterative(
                obj_pts, img_obs, K, dist,
                rvec0=rvec_e if ok_e else None,
                tvec0=tvec_e if ok_e else None
            )
            rt_i_ms = (time.perf_counter() - t0) * 1000.0

            if ok_i:
                img_pred = project_points(obj_pts, rvec_i, tvec_i, K, dist)
                R_i = rvec_to_R(rvec_i)
                rows.append({
                    "trial": trial, "sigma": sigma, "tau": float("nan"), "method": "ITERATIVE",
                    "rot_err_deg": rotation_error_deg(R_i, R_gt),
                    "trans_err": translation_error(tvec_i, tvec_gt),
                    "repr_rmse": reprojection_rmse(img_obs, img_pred),
                    "runtime_ms": rt_i_ms, "success": 1
                })
            else:
                rows.append({
                    "trial": trial, "sigma": sigma, "tau": float("nan"), "method": "ITERATIVE",
                    "rot_err_deg": np.nan, "trans_err": np.nan, "repr_rmse": np.nan,
                    "runtime_ms": rt_i_ms, "success": 0
                })

            # EPnP + TSVD
            for tau in taus:
                t0 = time.perf_counter()
                if ok_e:
                    rvec_p, tvec_p, info = refine_pose_tsvd(
                        obj_pts, img_obs, K, dist, rvec_e, tvec_e,
                        tau=tau, max_iters=10
                    )
                    ok_p = np.all(np.isfinite(rvec_p)) and np.all(np.isfinite(tvec_p))
                else:
                    ok_p = False
                    rvec_p = np.full((3, 1), np.nan)
                    tvec_p = np.full((3, 1), np.nan)

                rt_p_ms = (time.perf_counter() - t0) * 1000.0

                if ok_p:
                    img_pred = project_points(obj_pts, rvec_p, tvec_p, K, dist)
                    R_p = rvec_to_R(rvec_p)
                    rows.append({
                        "trial": trial, "sigma": sigma, "tau": tau, "method": "EPnP+TSVD",
                        "rot_err_deg": rotation_error_deg(R_p, R_gt),
                        "trans_err": translation_error(tvec_p, tvec_gt),
                        "repr_rmse": reprojection_rmse(img_obs, img_pred),
                        "runtime_ms": rt_p_ms, "success": 1
                    })
                else:
                    rows.append({
                        "trial": trial, "sigma": sigma, "tau": tau, "method": "EPnP+TSVD",
                        "rot_err_deg": np.nan, "trans_err": np.nan, "repr_rmse": np.nan,
                        "runtime_ms": rt_p_ms, "success": 0
                    })

    # CSV Output
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")

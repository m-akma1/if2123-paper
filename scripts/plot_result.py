from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_csv(path: str) -> List[dict]:
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def closest_value(values: List[float], target: float) -> float:
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return float("nan")
    return min(values, key=lambda v: abs(v - target))


def group_stats(values: List[float]) -> Tuple[float, float, float]:
    """
    Returns (median, p25, p75) ignoring NaNs.
    """
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(arr)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to results CSV (e.g., outputs/results_planar.csv)")
    ap.add_argument("--out", default="outputs/figs", help="Output folder for figures")
    ap.add_argument("--tau-ref", type=float, default=1e-4, help="tau value used for noise/failure plots")
    ap.add_argument("--sigma-ref", type=float, default=1.0, help="sigma value used for tau ablation plot")
    ap.add_argument("--show", action="store_true", help="Also show plots interactively")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rows = load_csv(args.csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    parsed = []
    for r in rows:
        parsed.append({
            "trial": int(r["trial"]),
            "sigma": _to_float(r["sigma"]),
            "tau": _to_float(r["tau"]),
            "method": r["method"],
            "rot_err_deg": _to_float(r["rot_err_deg"]),
            "trans_err": _to_float(r["trans_err"]),
            "repr_rmse": _to_float(r["repr_rmse"]),
            "runtime_ms": _to_float(r["runtime_ms"]),
            "success": int(float(r["success"])),
        })

    methods = sorted(set(r["method"] for r in parsed))
    sigmas = sorted(set(r["sigma"] for r in parsed if np.isfinite(r["sigma"])))
    taus = sorted(set(r["tau"] for r in parsed if np.isfinite(r["tau"])))

    tau_used = closest_value(taus, args.tau_ref) if taus else float("nan")
    sigma_used = closest_value(sigmas, args.sigma_ref) if sigmas else float("nan")

    print(f"Detected methods: {methods}")
    print(f"Detected sigmas: {sigmas}")
    print(f"Detected taus:   {taus}")
    print(f"Using tau_ref   ~ {tau_used}")
    print(f"Using sigma_ref ~ {sigma_used}")

    # Helper: filter
    def filt(method: str, sigma: float | None = None, tau: float | None = None, success_only: bool = True):
        out = []
        for r in parsed:
            if r["method"] != method:
                continue
            if sigma is not None and not math.isclose(r["sigma"], sigma, rel_tol=0, abs_tol=1e-12):
                continue

            # Only filter by tau for EPnP+TSVD
            if method == "EPnP+TSVD" and tau is not None:
                if not math.isclose(r["tau"], tau, rel_tol=0, abs_tol=1e-12):
                    continue

            if success_only and r["success"] != 1:
                continue
            out.append(r)
        return out

    # Plot 1: Noise sweep (median rotation + IQR)
    if np.isfinite(tau_used):
        plt.figure()
        for m in methods:
            meds, p25s, p75s = [], [], []
            for s in sigmas:
                rs = filt(m, sigma=s, tau=tau_used, success_only=True)
                med, p25, p75 = group_stats([x["rot_err_deg"] for x in rs])
                meds.append(med); p25s.append(p25); p75s.append(p75)
            x = np.array(sigmas, dtype=np.float64)
            y = np.array(meds, dtype=np.float64)
            y1 = np.array(p25s, dtype=np.float64)
            y2 = np.array(p75s, dtype=np.float64)
            plt.plot(x, y, marker="o", label=m)
            plt.fill_between(x, y1, y2, alpha=0.2)

        plt.xlabel("Pixel noise σ (px)")
        plt.ylabel("Rotation error (deg), median (IQR band)")
        plt.title(f"Noise sweep (tau≈{tau_used:g})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        outpath = os.path.join(args.out, "noise_sweep_rot.png")
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Wrote {outpath}")
        if args.show:
            plt.show()
        plt.close()

        # Translation plot
        plt.figure()
        for m in methods:
            meds, p25s, p75s = [], [], []
            for s in sigmas:
                rs = filt(m, sigma=s, tau=tau_used, success_only=True)
                med, p25, p75 = group_stats([x["trans_err"] for x in rs])
                meds.append(med); p25s.append(p25); p75s.append(p75)
            x = np.array(sigmas, dtype=np.float64)
            y = np.array(meds, dtype=np.float64)
            y1 = np.array(p25s, dtype=np.float64)
            y2 = np.array(p75s, dtype=np.float64)
            plt.plot(x, y, marker="o", label=m)
            plt.fill_between(x, y1, y2, alpha=0.2)

        plt.xlabel("Pixel noise σ (px)")
        plt.ylabel("Translation error ||t - t_gt|| (units), median (IQR band)")
        plt.title(f"Noise sweep (tau≈{tau_used:g})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        outpath = os.path.join(args.out, "noise_sweep_trans.png")
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Wrote {outpath}")
        if args.show:
            plt.show()
        plt.close()

    # Plot 2: Failure rate vs sigma
    if np.isfinite(tau_used):
        plt.figure()
        for m in methods:
            fail_rates = []
            for s in sigmas:
                rs = filt(m, sigma=s, tau=tau_used, success_only=False)
                if not rs:
                    fail_rates.append(float("nan"))
                    continue
                succ = np.mean([x["success"] for x in rs])
                fail_rates.append(1.0 - succ)
            plt.plot(sigmas, fail_rates, marker="o", label=m)

        plt.xlabel("Pixel noise σ (px)")
        plt.ylabel("Failure rate")
        plt.ylim(-0.05, 1.05)
        plt.title(f"Failure rate vs noise (tau≈{tau_used:g})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        outpath = os.path.join(args.out, "failure_rate_vs_sigma.png")
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Wrote {outpath}")
        if args.show:
            plt.show()
        plt.close()

    # Plot 3: Tau ablation at fixed sigma (EPnP+TSVD + baseline line)
    if np.isfinite(sigma_used) and taus:
        plt.figure()

        # EPnP+TSVD curve
        if "EPnP+TSVD" in methods:
            meds = []
            for t in taus:
                rs = filt("EPnP+TSVD", sigma=sigma_used, tau=t, success_only=True)
                med, _, _ = group_stats([x["rot_err_deg"] for x in rs])
                meds.append(med)
            plt.plot(taus, meds, marker="o", label="EPnP+TSVD")

        # Baseline ITERATIVE
        if "ITERATIVE" in methods and np.isfinite(tau_used):
            rs = filt("ITERATIVE", sigma=sigma_used, tau=tau_used, success_only=True)
            med, _, _ = group_stats([x["rot_err_deg"] for x in rs])
            if np.isfinite(med):
                plt.axhline(med, linestyle="--", label=f"ITERATIVE (tau≈{tau_used:g})")

        plt.xscale("log")
        plt.xlabel("TSVD threshold τ (relative)")
        plt.ylabel("Rotation error (deg), median")
        plt.title(f"Tau ablation at σ≈{sigma_used:g} px")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        outpath = os.path.join(args.out, "tau_ablation_rot.png")
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Wrote {outpath}")
        if args.show:
            plt.show()
        plt.close()


if __name__ == "__main__":
    main()

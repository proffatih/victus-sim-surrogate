"""
generate_dataset.py
===================
Generate a reproducible Latin-hypercube design-of-experiments (DoE) dataset
mapping design parameters theta = (C, cx, Q, uD) to the heat-solver QoIs
(Tavg, Tmax).  Also records validation data (MMS convergence, QoI mesh
convergence) and the measured single-solve wall-clock time of the high-fidelity
FE solver, which is the reference cost for the speed-up study.

Outputs (written to ../results and ../data):
  results/mms_convergence.csv
  results/qoi_mesh_convergence.csv
  results/solver_timing.json
  data/doe_dataset.csv           (the LHS training/test pool)

All randomness seeded.  Refinement level for the dataset is REFINE_DATA.
"""

import os
import json
import time
import numpy as np
from scipy.stats import qmc

from solver import (solve_heat, PARAM_BOUNDS, PARAM_NAMES,
                    manufactured_solution_test, mesh_convergence_qoi)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

REFINE_DATA = 6      # 4225 dofs -> mesh-converged QoI (see convergence study)
N_SAMPLES = 600      # size of the LHS pool
SEED = 20260617


def write_validation():
    # MMS convergence
    hs, errs, rate = manufactured_solution_test(refine_levels=(2, 3, 4, 5, 6))
    with open(os.path.join(RESULTS, "mms_convergence.csv"), "w") as fh:
        fh.write("h,l2_error\n")
        for h, e in zip(hs, errs):
            fh.write(f"{h:.8f},{e:.8e}\n")
    # QoI mesh convergence for a representative parameter set
    theta_ref = [3.0, 0.5, 12.0, 1.0]
    rows = mesh_convergence_qoi(theta_ref, refine_levels=(2, 3, 4, 5, 6, 7))
    with open(os.path.join(RESULTS, "qoi_mesh_convergence.csv"), "w") as fh:
        fh.write("refine,ndofs,Tavg,Tmax\n")
        for r, nd, ta, tm in rows:
            fh.write(f"{r},{nd},{ta:.8f},{tm:.8f}\n")
    print(f"[validation] MMS convergence rate = {rate:.3f} (P1 theory = 2.0)")
    return rate


def time_single_solve(theta, n_rep=20):
    """Median wall-clock of one high-fidelity solve at REFINE_DATA."""
    # warm up (JIT/cache)
    solve_heat(theta, refine=REFINE_DATA)
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        solve_heat(theta, refine=REFINE_DATA)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.std(ts))


def main():
    rate = write_validation()

    # LHS design
    sampler = qmc.LatinHypercube(d=4, seed=SEED)
    unit = sampler.random(N_SAMPLES)
    X = qmc.scale(unit, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])

    print(f"[dataset] solving {N_SAMPLES} cases at refine={REFINE_DATA} ...")
    t0 = time.perf_counter()
    Y = np.empty((N_SAMPLES, 2))
    for i, th in enumerate(X):
        r = solve_heat(th, refine=REFINE_DATA)
        Y[i] = (r["Tavg"], r["Tmax"])
        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{N_SAMPLES}")
    total = time.perf_counter() - t0

    header = ",".join(PARAM_NAMES + ["Tavg", "Tmax"])
    np.savetxt(os.path.join(DATA, "doe_dataset.csv"),
               np.hstack([X, Y]), delimiter=",", header=header, comments="")

    # Timing: median per-solve cost
    med, sd = time_single_solve([3.0, 0.5, 12.0, 1.0])
    timing = {
        "refine_level": REFINE_DATA,
        "ndofs": int(solve_heat([3.0, 0.5, 12.0, 1.0], refine=REFINE_DATA)["ndofs"]),
        "median_solve_seconds": med,
        "std_solve_seconds": sd,
        "n_samples": N_SAMPLES,
        "total_dataset_build_seconds": total,
        "mms_rate": rate,
        "seed": SEED,
    }
    with open(os.path.join(RESULTS, "solver_timing.json"), "w") as fh:
        json.dump(timing, fh, indent=2)
    print(f"[timing] median single solve = {med*1e3:.2f} ms ({timing['ndofs']} dofs)")
    print(f"[dataset] wrote {N_SAMPLES} rows; total build {total:.1f} s")


if __name__ == "__main__":
    main()

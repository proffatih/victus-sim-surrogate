"""
active_learning.py
==================
Compare uncertainty-based ACTIVE LEARNING against random sampling for building
a GP surrogate of the heat-solver QoI Tavg.

Protocol (repeated over several random seeds and averaged):
  * Fixed independent TEST set of N_TEST points (LHS, separate seed).
  * Start from a small initial design (N_INIT LHS points).
  * Candidate pool of N_POOL LHS points (the "unlabelled" set).
  * At each iteration, evaluate the FE solver on BATCH new points chosen either:
        - active : the pool points with the largest GP predictive std,
        - random : uniformly at random from the remaining pool.
  * Refit the GP, record test RMSE / R2 vs cumulative number of FE solves.

The FE solver is the *only* expensive oracle; the metric of interest is the
number of FE calls needed to reach a target accuracy.

Outputs:
  results/active_learning_curves.csv   (method, n_train, rmse, r2, seed)
  results/active_learning_summary.json (FE calls to reach target RMSE)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from solver import solve_heat, PARAM_BOUNDS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

N_TEST = 300
N_POOL = 500
N_INIT = 12
BATCH = 4
N_ITER = 22          # final design size = N_INIT + N_ITER*BATCH
SEEDS = [0, 1, 2, 3, 4]
TARGET_RMSE = 0.005  # accuracy target for the "FE calls to target" metric
# Note: post-hoc the summary JSON also reports savings at 0.01 / 0.005 / 0.003
# plus the fraction of budgets where active beats random (see recompute step).


def make_gp():
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=[1.0] * 4, length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(1e-4, (1e-8, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=2, random_state=0)


def lhs(n, seed):
    u = qmc.LatinHypercube(d=4, seed=seed).random(n)
    return qmc.scale(u, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])


def evaluate(X):
    return np.array([solve_heat(t, refine=6)["Tavg"] for t in X])


def run_one(method, seed, Xtest, ytest):
    pool = lhs(N_POOL, seed=1000 + seed)
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(N_POOL, N_INIT, replace=False)
    mask = np.zeros(N_POOL, dtype=bool)
    mask[init_idx] = True

    Xtr = pool[mask]
    ytr = evaluate(Xtr)

    scaler = StandardScaler().fit(pool)
    curve = []
    for it in range(N_ITER + 1):
        gp = make_gp()
        gp.fit(scaler.transform(Xtr), ytr)
        mu = gp.predict(scaler.transform(Xtest))
        rmse = float(np.sqrt(mean_squared_error(ytest, mu)))
        r2 = float(r2_score(ytest, mu))
        curve.append((method, len(ytr), rmse, r2, seed))
        if it == N_ITER:
            break
        # choose next batch from the remaining pool
        rem = np.where(~mask)[0]
        if method == "active":
            _, sd = gp.predict(scaler.transform(pool[rem]), return_std=True)
            pick = rem[np.argsort(sd)[-BATCH:]]
        else:  # random
            pick = rng.choice(rem, BATCH, replace=False)
        mask[pick] = True
        Xnew = pool[pick]
        ynew = evaluate(Xnew)
        Xtr = np.vstack([Xtr, Xnew])
        ytr = np.concatenate([ytr, ynew])
    return curve


def main():
    # fixed independent test set
    Xtest = lhs(N_TEST, seed=999)
    ytest = evaluate(Xtest)

    rows = []
    for seed in SEEDS:
        for method in ["active", "random"]:
            rows += run_one(method, seed, Xtest, ytest)
            print(f"  done method={method} seed={seed}")
    df = pd.DataFrame(rows, columns=["method", "n_train", "rmse", "r2", "seed"])
    df.to_csv(os.path.join(RESULTS, "active_learning_curves.csv"), index=False)

    # FE calls to reach target RMSE (mean over seeds)
    summary = {"target_rmse": TARGET_RMSE}
    for method in ["active", "random"]:
        calls = []
        for seed in SEEDS:
            sub = df[(df.method == method) & (df.seed == seed)].sort_values("n_train")
            hit = sub[sub.rmse <= TARGET_RMSE]
            calls.append(int(hit.n_train.iloc[0]) if len(hit) else np.nan)
        calls = np.array(calls, dtype=float)
        summary[method] = {
            "fe_calls_to_target_mean": float(np.nanmean(calls)),
            "fe_calls_to_target_std": float(np.nanstd(calls)),
            "reached_in_all_seeds": bool(np.all(~np.isnan(calls))),
        }
    if summary["random"]["fe_calls_to_target_mean"] > 0:
        summary["sample_saving_fraction"] = (
            1.0 - summary["active"]["fe_calls_to_target_mean"]
            / summary["random"]["fe_calls_to_target_mean"])
    with open(os.path.join(RESULTS, "active_learning_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[active learning] summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

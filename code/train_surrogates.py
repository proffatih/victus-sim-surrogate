"""
train_surrogates.py
===================
Train and compare three surrogate models for the heat-solver QoI Tavg:
  * Gaussian Process (GP) regression with an anisotropic RBF + white kernel
  * Gradient Boosting Regressor (GBR)
  * Multi-layer Perceptron (MLP)

Outputs:
  results/surrogate_metrics.csv     test + 5-fold CV R2 and RMSE per model
  results/gp_calibration.csv        observed coverage vs nominal for GP intervals
  results/parity_<model>.csv        y_true vs y_pred on the held-out test set
  results/speedup.json              measured surrogate prediction vs FE solve cost

Primary QoI for the comparison is Tavg.  All seeds fixed.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(RESULTS, exist_ok=True)

SEED = 20260617
QOI = "Tavg"


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def build_gp():
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=[1.0] * 4, length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(1e-3, (1e-8, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=4, random_state=SEED)


def build_gbr():
    return GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                     learning_rate=0.05, subsample=0.9,
                                     random_state=SEED)


def build_mlp():
    return Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                             alpha=1e-4, max_iter=4000, random_state=SEED)),
    ])


def main():
    df = pd.read_csv(os.path.join(DATA, "doe_dataset.csv"))
    X = df[["C", "cx", "Q", "uD"]].values
    y = df[QOI].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=SEED)
    # GP works on standardized inputs
    gp_scaler = StandardScaler().fit(Xtr)

    models = {
        "GP": build_gp(),
        "GBR": build_gbr(),
        "MLP": build_mlp(),
    }

    rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, model in models.items():
        if name == "GP":
            Xtr_m, Xte_m = gp_scaler.transform(Xtr), gp_scaler.transform(Xte)
            Xcv = gp_scaler.transform(X)
        else:
            Xtr_m, Xte_m, Xcv = Xtr, Xte, X

        t0 = time.perf_counter()
        model.fit(Xtr_m, ytr)
        fit_t = time.perf_counter() - t0

        yp = model.predict(Xte_m)
        r2 = r2_score(yte, yp)
        rm = rmse(yte, yp)

        cv_r2 = cross_val_score(model, Xcv, y, cv=kf, scoring="r2")
        cv_rmse = -cross_val_score(model, Xcv, y, cv=kf,
                                   scoring="neg_root_mean_squared_error")

        rows.append({
            "model": name,
            "test_R2": r2, "test_RMSE": rm,
            "cv_R2_mean": cv_r2.mean(), "cv_R2_std": cv_r2.std(),
            "cv_RMSE_mean": cv_rmse.mean(), "cv_RMSE_std": cv_rmse.std(),
            "fit_seconds": fit_t,
        })
        # parity data
        pd.DataFrame({"y_true": yte, "y_pred": yp}).to_csv(
            os.path.join(RESULTS, f"parity_{name}.csv"), index=False)
        print(f"[{name}] test R2={r2:.4f} RMSE={rm:.4e} | "
              f"CV R2={cv_r2.mean():.4f}+-{cv_r2.std():.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "surrogate_metrics.csv"),
                              index=False)

    # ---- GP calibration: coverage of predictive intervals ----
    gp = build_gp()
    gp.fit(gp_scaler.transform(Xtr), ytr)
    mu, sd = gp.predict(gp_scaler.transform(Xte), return_std=True)
    z = (yte - mu) / sd
    nominal = np.linspace(0.05, 0.99, 30)
    cov = []
    for p in nominal:
        zc = norm.ppf(0.5 + p / 2.0)   # two-sided
        cov.append(float(np.mean(np.abs(z) <= zc)))
    pd.DataFrame({"nominal": nominal, "observed": cov}).to_csv(
        os.path.join(RESULTS, "gp_calibration.csv"), index=False)
    # also save standardized residuals for a calibration histogram
    pd.DataFrame({"z": z}).to_csv(os.path.join(RESULTS, "gp_residuals.csv"),
                                  index=False)
    print(f"[GP calibration] mean |z|={np.mean(np.abs(z)):.3f} "
          f"(ideal ~0.80 for N(0,1)); 95% coverage="
          f"{np.mean(np.abs(z) <= 1.96):.3f}")

    # ---- Speedup: surrogate predict vs FE solve ----
    from solver import solve_heat
    theta = [3.0, 0.5, 12.0, 1.0]
    solve_heat(theta, refine=6)  # warm
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        solve_heat(theta, refine=6)
    fe_t = (time.perf_counter() - t0) / n

    Xq = gp_scaler.transform(np.tile(theta, (1, 1)))
    gp.predict(Xq)  # warm
    npred = 2000
    Xbatch = gp_scaler.transform(np.tile(theta, (npred, 1)))
    t0 = time.perf_counter()
    gp.predict(Xbatch)
    gp_t = (time.perf_counter() - t0) / npred

    # GBR single-prediction cost
    gbr = build_gbr(); gbr.fit(Xtr, ytr)
    gbr.predict(np.tile(theta, (1, 1)))
    t0 = time.perf_counter()
    gbr.predict(np.tile(theta, (npred, 1)))
    gbr_t = (time.perf_counter() - t0) / npred

    speed = {
        "fe_solve_seconds": fe_t,
        "gp_predict_seconds": gp_t,
        "gbr_predict_seconds": gbr_t,
        "speedup_gp": fe_t / gp_t,
        "speedup_gbr": fe_t / gbr_t,
    }
    with open(os.path.join(RESULTS, "speedup.json"), "w") as fh:
        json.dump(speed, fh, indent=2)
    print(f"[speedup] FE={fe_t*1e3:.2f} ms  GP={gp_t*1e6:.2f} us  "
          f"GBR={gbr_t*1e6:.2f} us  -> GP {speed['speedup_gp']:.0f}x, "
          f"GBR {speed['speedup_gbr']:.0f}x")


if __name__ == "__main__":
    main()

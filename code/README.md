# Reproducible code — Fast uncertainty-aware ML surrogate for parametric heat conduction

Author: Prof. Dr. Fatih Gül (Recep Tayyip Erdoğan University, EEE / AI&IoT Lab).

All results in the manuscript regenerate from fixed random seeds.

## Requirements
```
python >= 3.10
numpy scipy scikit-learn matplotlib scikit-fem
```
Install: `pip install numpy scipy scikit-learn matplotlib scikit-fem`

## Pipeline (run in order, from this `code/` directory)
1. `python solver.py`            — verify solver (MMS rate ~1.97, QoI mesh convergence).
2. `python generate_dataset.py`  — write `../results/{mms_convergence,qoi_mesh_convergence}.csv`,
                                    `../results/solver_timing.json`, and the 600-point LHS dataset
                                    `../data/doe_dataset.csv`.
3. `python train_surrogates.py`  — GP/GBR/MLP metrics, GP calibration, measured speed-up
                                    → `../results/{surrogate_metrics.csv, gp_calibration.csv,
                                    gp_residuals.csv, parity_*.csv, speedup.json}`.
4. `python active_learning.py`   — active vs random learning curves and summary
                                    → `../results/{active_learning_curves.csv, active_learning_summary.json}`.
   (Note: the FE-calls-to-target summary is recomputed for targets 0.01/0.005/0.003.)
5. `python make_figures.py`      — all six figures (PDF + 300-dpi PNG) → `../figures/`.

## Seeds
Dataset seed `20260617`; surrogate split/CV seed `20260617`; active-learning seeds `0..4`,
test-set seed `999`, pool seeds `1000+seed`.

## Model
2-D steady-state heat conduction on the unit square, P1 triangular FEM (scikit-fem),
mixed Dirichlet/Neumann/Robin BCs, Gaussian conductivity inclusion + Gaussian source.
Parameters θ = (C, c_x, Q, u_D); QoI = (T_avg, T_max). See `solver.py` docstring.

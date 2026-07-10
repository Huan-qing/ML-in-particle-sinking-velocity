"""
prep.py  —  Shared data preparation for the settling-velocity model comparison.
 
ONE place for cleaning + stratified split + feature selection + target transform,
so every model trains/tests on the IDENTICAL rows and the leaderboard is fair.
 
Typical use in any model notebook/script:
 
    import prep
    d = prep.get_data()                 # uses prep.CONFIG
    # 80% training pool (train+val) and the common 20% test set:
    Xtr, Xte = d.X_trainval, d.X_test
    ytr, yte = d.y_trainval, d.y_test   # model-space target (log10 if CONFIG.log_target)
    ...
    res = prep.evaluate("MyModel", d.y_test_orig, pred_test)   # metrics on ORIGINAL units
 
Models that need a validation set (TabNet, GNN, early-stopping LightGBM) use
d.X_train / d.X_val for tuning, then refit on d.X_trainval for the final number.
 
Keep CONFIG identical across every model. The split is saved to split_indices.npz
so the first run creates it and all later runs reuse the exact same rows.
"""

from types import SimpleNamespace
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------------
# CONFIG  — the single source of truth. Change here, not in individual models.
# ----------------------------------------------------------------------------
CONFIG = SimpleNamespace(
    # file_path="data_each_particle_aspectrat_circ.csv",
    # file_path = "predictions_all_particles.csv",
    file_path = "data_each_particle_od_sinking.csv" ,
    feature_cols=[0, 5, 9],   # same predictors for every model
    target_col=-1,            # last column = SettlingVelocity
    abs_col=9,                # Orientation: take magnitude
    # predict log10(SV)? set once, applies to all models
    log_target=False,
    test_size=0.20,           # frozen common test fraction
    # fraction of the (train+val) remainder -> 16% of total
    val_size=0.20,
    # quantile bins of log(target) used for stratification
    n_bins=10,
    seed=42,
    split_path="split_indices.npz",
)


# ----------------------------------------------------------------------------
# Cleaning  — must be byte-identical across all models, so it lives here.
# ----------------------------------------------------------------------------
def load_clean(cfg=CONFIG):
    """Load CSV, keep positive-target rows, abs() the orientation column, drop NaNs."""
    df = pd.read_csv(cfg.file_path)
    df = df[df.iloc[:, cfg.target_col] > 0].copy()      # drop non-positive SV
    df.iloc[:, cfg.abs_col] = df.iloc[:, cfg.abs_col].abs()
    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Stratified split  — created once, then reused from disk.
# ----------------------------------------------------------------------------
def make_or_load_split(target, cfg=CONFIG):
    """Return (train_idx, val_idx, test_idx). Stratified on quantile bins of log(target).

    val_size is a fraction of the (train+val) remainder, so the defaults give 64/16/20.
    """
    if os.path.exists(cfg.split_path):
        z = np.load(cfg.split_path)
        return z["train"], z["val"], z["test"]
    idx = np.arange(len(target))
    bins = pd.qcut(np.log10(target), q=cfg.n_bins,
                   labels=False, duplicates="drop")
    tr_val, test = train_test_split(
        idx, test_size=cfg.test_size, random_state=cfg.seed, stratify=bins)
    tr, val = train_test_split(
        tr_val, test_size=cfg.val_size, random_state=cfg.seed, stratify=bins[tr_val])
    np.savez(cfg.split_path, train=tr, val=val, test=test)
    return tr, val, test


# ----------------------------------------------------------------------------
# Assemble everything a model needs.
# ----------------------------------------------------------------------------
def get_data(cfg=CONFIG):
    df = load_clean(cfg)
    tr, va, te = make_or_load_split(df.iloc[:, cfg.target_col].values, cfg)
    trva = np.concatenate([tr, va])

    X = df.iloc[:, cfg.feature_cols].values.astype(np.float64)
    y_orig = df.iloc[:, cfg.target_col].values.astype(np.float64)
    y = np.log10(y_orig) if cfg.log_target else y_orig   # model-space target

    split_label = np.empty(len(df), dtype=object)
    split_label[tr] = "train"
    split_label[va] = "val"
    split_label[te] = "test"

    return SimpleNamespace(
        df=df,
        feature_names=df.columns[cfg.feature_cols].tolist(),
        target_name=df.columns[cfg.target_col],
        # indices
        train_idx=tr, val_idx=va, test_idx=te, trainval_idx=trva,
        split_label=split_label,
        # model-space target slices
        X_train=X[tr],     y_train=y[tr],
        X_val=X[va],       y_val=y[va],
        X_test=X[te],      y_test=y[te],
        X_trainval=X[trva], y_trainval=y[trva],
        X_all=X,           y_all=y,
        # original-scale target slices (always use these for metrics)
        y_train_orig=y_orig[tr], y_val_orig=y_orig[va],
        y_test_orig=y_orig[te],  y_trainval_orig=y_orig[trva],
        y_all_orig=y_orig,
        cfg=cfg,
    )


# ----------------------------------------------------------------------------
# Helpers shared by every model.
# ----------------------------------------------------------------------------
def scale(ref, *arrays):
    """Fit StandardScaler on `ref` (e.g. X_trainval) and transform every array.

    Returns (fitted_scaler, transformed_arrays...). Prevents train/test leakage.
    """
    sc = StandardScaler().fit(ref)
    return (sc, *[sc.transform(a) for a in arrays])


def inverse(y_modelspace, cfg=CONFIG):
    """Map a model-space prediction back to original SV units."""
    return np.power(10.0, y_modelspace) if cfg.log_target else y_modelspace


def evaluate(name, y_true_orig, y_pred_modelspace, cfg=CONFIG, note=""):
    """Compute MAE / RMSE / R2 on ORIGINAL units (inverting log if needed)."""
    pred = np.asarray(inverse(y_pred_modelspace, cfg)).ravel()
    true = np.asarray(y_true_orig).ravel()
    mse = mean_squared_error(true, pred)
    return {
        "model": name,
        "MAE": mean_absolute_error(true, pred),
        "RMSE": float(np.sqrt(mse)),
        "MSE": mse,
        "R2": r2_score(true, pred),
        "note": note,
    }

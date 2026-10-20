"""
Evaluation metrics (manuscript Section 2.7).

Study I is scored with the metrics of the previous study, MAPE and the outlier
fraction %Nout (residual > 3 sigma); Studies II and III use RMSE and
Spearman's rho as principal metrics.  ``compute_all_metrics`` returns all of
them at once so that every run writes the same columns.  All metrics are
computed on the original log x scale (the target standardisation of the
neural networks is inverted before scoring).
"""

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#: metric columns written by every run, in a fixed order
METRIC_KEYS = ["RMSE", "MAE", "R2", "MAPE", "Nout_pct",
               "Pearson_r", "Spearman_rho", "Kendall_tau"]


def safe_pearsonr(yt, yp):
    if len(yt) < 2 or np.std(yt) == 0 or np.std(yp) == 0: return np.nan
    return float(pearsonr(yt, yp)[0])


def safe_spearmanr(yt, yp):
    if len(yt) < 2: return np.nan
    try:
        rho, _ = spearmanr(yt, yp)
        return float(rho) if np.isfinite(rho) else np.nan
    except Exception:
        return np.nan


def safe_kendalltau(yt, yp):
    if len(yt) < 2: return np.nan
    try:
        tau, _ = kendalltau(yt, yp)
        return float(tau) if np.isfinite(tau) else np.nan
    except Exception:
        return np.nan


def compute_mape(y_true, y_pred):
    """Mean absolute percentage error on the log x scale [%]."""
    yt = np.asarray(y_true, dtype=float); yp = np.asarray(y_pred, dtype=float)
    return float(100.0 * np.mean(np.abs(yt - yp) / (np.abs(yt) + 1e-10)))


def compute_nout_percent(y_true, y_pred):
    """Fraction of points whose residual exceeds three standard deviations [%]."""
    yt = np.asarray(y_true, dtype=float); yp = np.asarray(y_pred, dtype=float)
    if len(yt) < 2: return np.nan
    residuals = yp - yt
    sigma = float(np.std(residuals))
    if sigma == 0: return 0.0
    return float(100.0 * np.sum(np.abs(residuals) > 3.0 * sigma) / len(yt))


def compute_all_metrics(y_true, y_pred):
    """RMSE, MAE, R2, MAPE, %Nout, Pearson r, Spearman rho, Kendall tau and bias."""
    yt = np.asarray(y_true, dtype=float); yp = np.asarray(y_pred, dtype=float)
    n = len(yt)
    if n == 0:
        return {k: np.nan for k in
                ["n", "RMSE", "MAE", "R2", "MAPE", "Nout_pct",
                 "Pearson_r", "Spearman_rho", "Kendall_tau", "Bias"]}
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if n >= 2 and np.var(yt) > 0 else np.nan
    return {
        "n": int(n), "RMSE": rmse, "MAE": mae, "R2": r2,
        "MAPE": compute_mape(yt, yp), "Nout_pct": compute_nout_percent(yt, yp),
        "Pearson_r": safe_pearsonr(yt, yp), "Spearman_rho": safe_spearmanr(yt, yp),
        "Kendall_tau": safe_kendalltau(yt, yp), "Bias": float(np.mean(yp - yt)),
    }


def metrics_row(prefix, metrics):
    """Flatten a metric dictionary into ``{prefix_RMSE: ..., ...}``."""
    return {f"{prefix}_{k}": metrics.get(k, np.nan) for k in METRIC_KEYS}

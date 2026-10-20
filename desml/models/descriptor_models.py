"""
Descriptor-based models: nuSVR (RBF / polynomial), random forest, XGBoost and
LightGBM (manuscript Section 2.5.1).

All four share the same interface:

``score_candidate``
    train on the inner fit set and return the validation RMSE of one
    hyperparameter candidate (used by the grid search, Section 2.6.3).
``fit_predict``
    train one model under one regime and one random-split seed and return the
    training / test predictions together with the metadata needed for the
    scatter plots.

Differences between the models are limited to three switches and are exactly
the ones described in the manuscript:

* nuSVR needs standardisation (``TiedScaler`` for X, ``StandardScaler`` for y)
  and the metrics are computed after the inverse transform; the tree models are
  scale-invariant and are trained on the raw values (Section 2.5.3).
* XGBoost and LightGBM carve a validation set out of the training data and use
  early stopping; nuSVR and the random forest are fitted on the whole training
  portion (Section 2.6.2).
* the permutation (slot-exchange) augmentation is applied to the fit part only,
  after the split, so that no information leaks into validation or test data
  (Section 2.4.1).
"""

import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR

from ..config import CONFIG
from ..features import TiedScaler, augment_swaps
from ..splits import carve_validation, regime_split

DESCRIPTOR_MODELS = ("nusvr_rbf", "nusvr_poly", "rf", "xgb", "lgb")


def needs_scaling(model: str) -> bool:
    """nuSVR is the only descriptor model that is standardised."""
    return model.lower().startswith("nusvr")


def needs_validation(model: str) -> bool:
    """XGBoost and LightGBM use a validation set for early stopping."""
    return model.lower() in ("xgb", "lgb")


# ------------------------------------------------------------ constructors
def _nusvr_fixed(kernel: str) -> dict:
    """Fixed nuSVR settings of Table 4."""
    fx = dict(kernel=kernel, max_iter=CONFIG["NUSVR_MAX_ITER"],
              cache_size=CONFIG["NUSVR_CACHE_SIZE"], tol=CONFIG["NUSVR_TOL"])
    if kernel == "poly":
        fx["degree"] = CONFIG["NUSVR_DEGREE"]
        fx["coef0"] = CONFIG["NUSVR_COEF0"]
    return fx


def _xgb_fixed(seed: int, n_estimators: int) -> dict:
    """Fixed XGBoost settings of Table 4."""
    return dict(
        n_estimators=n_estimators, subsample=CONFIG["XGB_SUBSAMPLE"],
        reg_alpha=CONFIG["XGB_REG_ALPHA"], reg_lambda=CONFIG["XGB_REG_LAMBDA"],
        objective="reg:squarederror", tree_method=CONFIG["XGB_TREE_METHOD"],
        device=CONFIG["XGB_DEVICE"], verbosity=CONFIG["XGB_VERBOSITY"],
        random_state=seed, n_jobs=CONFIG["N_JOBS"])


def _lgb_fixed(seed: int, n_estimators: int) -> dict:
    """Fixed LightGBM settings of Table 4."""
    return dict(
        n_estimators=n_estimators, subsample=CONFIG["LGB_SUBSAMPLE"],
        subsample_freq=CONFIG["LGB_SUBSAMPLE_FREQ"], reg_alpha=CONFIG["LGB_REG_ALPHA"],
        reg_lambda=CONFIG["LGB_REG_LAMBDA"], objective="regression", metric="rmse",
        verbose=CONFIG["LGB_VERBOSE"], random_state=seed, n_jobs=CONFIG["N_JOBS"])


def _make_estimator(model: str, params: dict, seed: int, n_estimators: int = None,
                    early_stopping: bool = False):
    model = model.lower()
    if model == "nusvr_rbf":
        return NuSVR(**_nusvr_fixed("rbf"), **params)
    if model == "nusvr_poly":
        return NuSVR(**_nusvr_fixed("poly"), **params)
    if model == "rf":
        return RandomForestRegressor(random_state=seed, n_jobs=CONFIG["N_JOBS"], **params)
    if model == "xgb":
        import xgboost as xgb
        extra = ({"early_stopping_rounds": CONFIG["XGB_EARLY_STOPPING_ROUNDS"]}
                 if early_stopping else {})
        return xgb.XGBRegressor(**_xgb_fixed(seed, n_estimators), **extra, **params)
    if model == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**_lgb_fixed(seed, n_estimators), **params)
    raise ValueError(f"unknown descriptor model: {model}")


def _fit_estimator(model: str, est, X_fit, y_fit, w_fit, eval_set=None):
    """Fit, using the early-stopping interface of the given library."""
    model = model.lower()
    if model == "lgb" and eval_set is not None:
        import lightgbm as lgb
        est.fit(X_fit, y_fit, sample_weight=w_fit, eval_set=eval_set,
                callbacks=[lgb.early_stopping(CONFIG["LGB_EARLY_STOPPING_ROUNDS"],
                                              verbose=False)])
    elif model == "xgb" and eval_set is not None:
        est.fit(X_fit, y_fit, sample_weight=w_fit, eval_set=eval_set, verbose=False)
    else:
        est.fit(X_fit, y_fit, sample_weight=w_fit)
    return est


# ---------------------------------------------------------- grid search
def score_candidate(model, X_fit_raw, y_fit_raw, swap_fit, X_val_raw, y_val_raw,
                    params, gs_seed, tied_idx_pairs):
    """Validation RMSE of one hyperparameter candidate (original log x scale).

    The boosted trees are scored with a fixed number of estimators
    (``*_GS_N_ESTIMATORS``) and without early stopping; the final training uses
    the larger limit together with early stopping (Section 2.6.3).
    """
    model = model.lower()
    n_est = CONFIG["XGB_GS_N_ESTIMATORS"] if model == "xgb" else (
        CONFIG["LGB_GS_N_ESTIMATORS"] if model == "lgb" else None)

    if needs_scaling(model):
        # scalers are fitted on the inner fit set only; scoring is done after
        # the inverse transform, on the original scale
        sx = TiedScaler(tied_idx_pairs).fit(X_fit_raw)
        sy = StandardScaler().fit(y_fit_raw.reshape(-1, 1))
        X_fit = sx.transform(X_fit_raw)
        y_fit = sy.transform(y_fit_raw.reshape(-1, 1)).ravel()
        Xa, ya, wa = augment_swaps(X_fit, y_fit, swap_fit, tied_idx_pairs)
        est = _fit_estimator(model, _make_estimator(model, params, gs_seed), Xa, ya, wa)
        yp = sy.inverse_transform(
            est.predict(sx.transform(X_val_raw)).reshape(-1, 1)).ravel()
    else:
        Xa, ya, wa = augment_swaps(X_fit_raw, y_fit_raw, swap_fit, tied_idx_pairs)
        est = _fit_estimator(model, _make_estimator(model, params, gs_seed, n_est), Xa, ya, wa)
        yp = est.predict(X_val_raw)
    return float(np.sqrt(mean_squared_error(y_val_raw, yp)))


# ------------------------------------------------------- final training
def fit_predict(model, regime, X_all, y_all, meta_df, is_des, data_seed,
                best_params, tied_idx_pairs, swappable_all):
    """Train one configuration and predict the training and test sets."""
    model = model.lower()
    sp = regime_split(regime, is_des, len(meta_df), data_seed)
    train_idx, test_idx = sp["train_idx"], sp["test_idx"]
    scatter_train_idx = sp["scatter_train_idx"]

    if needs_validation(model):
        fit_idx, val_idx = carve_validation(regime, train_idx, is_des, data_seed)
    else:
        fit_idx, val_idx = train_idx, np.array([], dtype=int)

    swp = swappable_all[fit_idx] if swappable_all is not None else None
    sx = sy = None
    if needs_scaling(model):
        sx = TiedScaler(tied_idx_pairs).fit(X_all[fit_idx])
        sy = StandardScaler().fit(y_all[fit_idx].reshape(-1, 1))
        X_fit = sx.transform(X_all[fit_idx])
        y_fit = sy.transform(y_all[fit_idx].reshape(-1, 1)).ravel()
    else:
        X_fit, y_fit = X_all[fit_idx], y_all[fit_idx]
    Xa, ya, wa = augment_swaps(X_fit, y_fit, swp, tied_idx_pairs)

    n_est = (CONFIG["XGB_FINAL_MAX_ESTIMATORS"] if model == "xgb" else
             CONFIG["LGB_FINAL_MAX_ESTIMATORS"] if model == "lgb" else None)
    est = _make_estimator(model, best_params, data_seed, n_est,
                          early_stopping=needs_validation(model))
    eval_set = [(X_all[val_idx], y_all[val_idx])] if needs_validation(model) else None

    t0 = time.time()
    _fit_estimator(model, est, Xa, ya, wa, eval_set=eval_set)
    fit_elapsed = time.time() - t0

    def _predict(idx):
        if needs_scaling(model):
            return sy.inverse_transform(
                est.predict(sx.transform(X_all[idx])).reshape(-1, 1)).ravel()
        return est.predict(X_all[idx])

    if model == "xgb":
        info = f"(best_iteration={getattr(est, 'best_iteration', None)}, n_val={len(val_idx)})"
    elif model == "lgb":
        info = f"(best_iteration={getattr(est, 'best_iteration_', None)}, n_val={len(val_idx)})"
    elif needs_scaling(model):
        info = f"(n_SV={int(getattr(est, 'support_', np.array([])).shape[0])})"
    else:
        info = ""

    return {
        "data_seed": data_seed, "fit_elapsed": fit_elapsed, "info": info,
        "meta_tr": meta_df.iloc[scatter_train_idx].reset_index(drop=True),
        "meta_te": meta_df.iloc[test_idx].reset_index(drop=True),
        "y_tr": y_all[scatter_train_idx], "y_te": y_all[test_idx],
        "yp_tr": _predict(scatter_train_idx), "yp_te": _predict(test_idx),
        "feature_importance": getattr(est, "feature_importances_", None),
        "n_train_aug": len(ya), "n_fit_raw": len(fit_idx), "n_val": len(val_idx),
        "model": est,
    }

"""
Leave-one-system-out evaluation (manuscript Section 2.8.3).

A "system" is one combination of API x HBA x HBD.  Every system with at least
``MIN_SYSTEM_COUNT`` (35) DES data points is held out in turn: the whole system
becomes the test set and the model is trained on the remaining systems.  Each
constituent molecule still appears in the training data through other systems,
so this measures generalisation to an unseen *combination* of known
constituents, not to a new region of descriptor space.

Besides the usual metrics the module computes, for each of the three condition
axes (temperature, water content, HBA:HBD ratio):

* the measured and predicted slope and their agreement in sign,
* the rank correlation along the axis and the correlation of the residual with
  the axis (a flat residual means the dependence is captured),
* the same quantities on slices where the other two axes are held fixed,
* the conditional RMSE in ``N_BIN`` bins along the axis,

and flags every test point whose condition lies outside the range covered by
the fit data (``extrap``), because those points are extrapolations rather than
interpolations and are highlighted in the figures.

The hyperparameters are not re-searched per fold: the values obtained in the
DES-only random-split search (Section 2.6.3) are reused, which is a mild
optimistic bias and is stated as such in the manuscript.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import AXES, CONFIG, SLOT_RENAME
from .features import (TiedScaler, augment_swaps, compute_condition_design)
from .metrics import safe_pearsonr, safe_spearmanr

LOSO_REGIMES = ("des_only", "joint")


def add_comp_ratio(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Composition axis: the water-free HBD fraction x_HBD/(x_HBA+x_HBD)."""
    xa = meta_df["x_HBA"].values.astype(float)
    xb = meta_df["x_HBD"].values.astype(float)
    denom = xa + xb
    cr = np.where(denom > 1e-12, xb / np.where(denom > 1e-12, denom, 1.0), np.nan)
    out = meta_df.copy()
    out["comp_ratio"] = cr
    return out


def system_key(drg, hba, hbd) -> str:
    """Identifier of one system, e.g. ``DAP__ChCl__TEG``."""
    return f"{drg}__{hba}__{hbd}"


def enumerate_target_systems(meta_df, min_count, logger):
    """DES systems with at least ``min_count`` points (the LOSO targets)."""
    des = meta_df[meta_df["system_type"].astype(str).str.upper() == "DES"]
    g = (des.groupby(["DRG", "HBA", "HBD"]).size()
         .reset_index(name="count").sort_values("count", ascending=False))
    targets = g[g["count"] >= min_count].reset_index(drop=True)
    logger.log(f"\n[Systems] {len(targets)} of {len(g)} DES systems have "
               f"count >= {min_count} and are used as LOSO targets")
    for _, r in targets.iterrows():
        logger.log(f"    {r['DRG']:>6} / {r['HBA']:>8} / {r['HBD']:>6} : {int(r['count'])} pts")
    return targets, g


def loso_split(regime, is_des, sys_mask):
    """Indices of one LOSO fold.

    The held-out system becomes the test set; ``des_only`` trains on the other
    DES systems, ``joint`` adds the whole of the general solvent data.  The
    validation set for early stopping is 10 % of the training pool (of its DES
    part only in the Joint regime) with the fixed seed ``LOSO_VAL_SEED``.
    """
    n = len(sys_mask)
    all_idx = np.arange(n)
    test_idx = all_idx[sys_mask]
    if regime == "des_only":
        train_pool = all_idx[is_des & ~sys_mask]
    elif regime == "joint":
        train_pool = all_idx[~sys_mask]              # other DES systems + general solvents
    else:
        raise ValueError(f"regime not supported by LOSO: {regime}")

    vf = CONFIG["VAL_FRACTION"]
    vseed = CONFIG["LOSO_VAL_SEED"]
    if regime == "joint" and CONFIG["VAL_DES_ONLY_IN_JOINT"]:
        des_local = np.where(is_des[train_pool])[0]
        gen_local = np.where(~is_des[train_pool])[0]
        fit_des, val_des = train_test_split(des_local, test_size=vf, random_state=vseed)
        fit_idx = train_pool[np.concatenate([fit_des, gen_local])]
        val_idx = train_pool[val_des]
    else:
        pos = np.arange(len(train_pool))
        fit_pos, val_pos = train_test_split(pos, test_size=vf, random_state=vseed)
        fit_idx, val_idx = train_pool[fit_pos], train_pool[val_pos]
    return test_idx, train_pool, fit_idx, val_idx


def extrapolation_flags(meta_df, fit_idx, test_idx):
    """Per-axis flag telling whether a test point lies outside the fit range."""
    extrap = {}
    fit_meta = meta_df.iloc[fit_idx]
    for axis, _, _ in AXES:
        fv = pd.to_numeric(fit_meta[axis], errors="coerce").values
        fv = fv[np.isfinite(fv)]
        tv = pd.to_numeric(meta_df.iloc[test_idx][axis], errors="coerce").values
        if len(fv) == 0:
            extrap[axis] = np.zeros(len(test_idx), bool)
        else:
            lo, hi = np.nanmin(fv), np.nanmax(fv)
            extrap[axis] = ((tv < lo - CONFIG["EXTRAP_TOL"]) |
                            (tv > hi + CONFIG["EXTRAP_TOL"]))
    extrap_any = np.zeros(len(test_idx), bool)
    for a in extrap:
        extrap_any = extrap_any | np.nan_to_num(extrap[a], nan=False).astype(bool)
    return extrap, extrap_any


def desc_template(X_df_row, feat_groups):
    """Descriptor columns of one row (constant within a system)."""
    cond = set(feat_groups["cond_cols"])
    return {c: float(X_df_row[c]) for c in feat_groups["all_cols"] if c not in cond}


def others_for(axis, meta_te):
    """Values of the two condition axes other than ``axis``."""
    cols = [a for a, _, _ in AXES if a != axis]
    return np.column_stack([pd.to_numeric(meta_te[c], errors="coerce").values for c in cols])


def _slope(x, y):
    if len(x) < 2 or np.unique(x).size < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def axis_dependency_metrics(axis_name, axis_vals, others, y, yp, cfg):
    """Dependence metrics along one condition axis.

    ``others`` holds the values of the other two axes, shape (n, 2).  Returns
    ``None`` when the axis does not vary within the held-out system.
    """
    av = np.asarray(axis_vals, float)
    if np.unique(np.round(av, 8)).size < 2:
        return None
    resid = yp - y
    m = {
        "axis": axis_name, "n_unique": int(np.unique(np.round(av, 8)).size),
        "slope_meas": _slope(av, y), "slope_pred": _slope(av, yp),
        "spearman_meas": safe_spearmanr(av, y), "spearman_pred": safe_spearmanr(av, yp),
        "resid_axis_pearson": safe_pearsonr(av, resid),
        "resid_axis_spearman": safe_spearmanr(av, resid),
    }
    m["slope_abs_err"] = (abs(m["slope_pred"] - m["slope_meas"])
                          if np.isfinite(m["slope_pred"]) and np.isfinite(m["slope_meas"]) else np.nan)
    m["sign_agree_overall"] = (float(np.sign(m["slope_meas"]) == np.sign(m["slope_pred"]))
                               if np.isfinite(m["slope_meas"]) and np.isfinite(m["slope_pred"]) else np.nan)

    # slices: round the other two axes, hold them fixed and measure the slope
    rd = cfg["SLICE_ROUND"]
    okey = np.column_stack([np.round(others[:, 0], 6), np.round(others[:, 1], 6)])
    keys = {}
    for i, k in enumerate(map(tuple, okey)):
        keys.setdefault(k, []).append(i)
    sm_list, sp_list, sign_list = [], [], []
    for k, idx in keys.items():
        idx = np.array(idx)
        if len(idx) < cfg["MIN_SLICE_POINTS"] or np.unique(np.round(av[idx], 8)).size < 2:
            continue
        sm = _slope(av[idx], y[idx]); sp = _slope(av[idx], yp[idx])
        if np.isfinite(sm) and np.isfinite(sp):
            sm_list.append(sm); sp_list.append(sp); sign_list.append(np.sign(sm) == np.sign(sp))
    m["n_slices"] = len(sm_list)
    m["slice_slope_meas"] = float(np.mean(sm_list)) if sm_list else np.nan
    m["slice_slope_pred"] = float(np.mean(sp_list)) if sp_list else np.nan
    m["slice_slope_abs_err"] = (float(np.mean(np.abs(np.array(sp_list) - np.array(sm_list))))
                                if sm_list else np.nan)
    m["slice_sign_agree"] = float(np.mean(sign_list)) if sign_list else np.nan

    # conditional RMSE in bins along the axis
    try:
        edges = np.linspace(av.min(), av.max(), cfg["N_BIN"] + 1)
        binid = np.clip(np.digitize(av, edges[1:-1]), 0, cfg["N_BIN"] - 1)
        rmses = []
        for b in range(cfg["N_BIN"]):
            mb = binid == b
            if mb.sum() >= 2:
                rmses.append(float(np.sqrt(mean_squared_error(y[mb], yp[mb]))))
        m["binned_rmse_max"] = float(np.max(rmses)) if rmses else np.nan
        m["binned_rmse_mean"] = float(np.mean(rmses)) if rmses else np.nan
    except Exception:
        m["binned_rmse_max"] = m["binned_rmse_mean"] = np.nan
    return m


# =====================================================================
# Per-fold training.  One function per model family; all three return the
# same dictionary, including a ``predictor`` callable that the figure code
# uses to draw the condition curves of Figs. 7-9.
# =====================================================================
def _condition_rows(desc_template_dict, cond_keys_sel, points):
    """Design rows for a grid of conditions of one fixed system."""
    rows = []
    for (T_K, x2, x3, xw) in points:
        cd = compute_condition_design(float(T_K), float(x2), float(x3), float(xw))
        row = dict(desc_template_dict)
        for k in cond_keys_sel:
            row[SLOT_RENAME.get(k, k)] = float(cd[k])
        rows.append(row)
    return rows


def fit_predict_descriptor(model_name, regime, X_all, y_all, meta_df, is_des, sys_mask,
                           best_params, tied_idx_pairs, swappable_all, logger):
    """One LOSO fold with a descriptor-based model (XGB in the published study)."""
    from .models.descriptor_models import (_fit_estimator, _make_estimator,
                                           needs_scaling, needs_validation)

    test_idx, train_pool, fit_idx, val_idx = loso_split(regime, is_des, sys_mask)
    if not needs_validation(model_name):
        fit_idx = np.concatenate([fit_idx, val_idx])
        val_idx = np.array([], dtype=int)

    sx = sy = None
    if needs_scaling(model_name):
        sx = TiedScaler(tied_idx_pairs).fit(X_all[fit_idx])
        sy = StandardScaler().fit(y_all[fit_idx].reshape(-1, 1))
        X_fit = sx.transform(X_all[fit_idx])
        y_fit = sy.transform(y_all[fit_idx].reshape(-1, 1)).ravel()
    else:
        X_fit, y_fit = X_all[fit_idx], y_all[fit_idx]

    swp = swappable_all[fit_idx] if swappable_all is not None else None
    Xa, ya, wa = augment_swaps(X_fit, y_fit, swp, tied_idx_pairs)

    n_est = (CONFIG["XGB_FINAL_MAX_ESTIMATORS"] if model_name == "xgb" else
             CONFIG["LGB_FINAL_MAX_ESTIMATORS"] if model_name == "lgb" else None)
    est = _make_estimator(model_name, best_params, CONFIG["LOSO_VAL_SEED"], n_est,
                          early_stopping=needs_validation(model_name))
    eval_set = ([(X_all[val_idx], y_all[val_idx])]
                if needs_validation(model_name) else None)
    t0 = time.time()
    _fit_estimator(model_name, est, Xa, ya, wa, eval_set=eval_set)
    elapsed = time.time() - t0

    def _predict(X):
        if needs_scaling(model_name):
            return sy.inverse_transform(
                est.predict(sx.transform(X)).reshape(-1, 1)).ravel()
        return est.predict(X)

    yp_te = _predict(X_all[test_idx])
    extrap, extrap_any = extrapolation_flags(meta_df, fit_idx, test_idx)

    def predictor(desc_tmpl, cond_keys_sel, all_cols, points):
        rows = _condition_rows(desc_tmpl, cond_keys_sel, points)
        Xg = pd.DataFrame(rows)[all_cols].values.astype(np.float64)
        return _predict(Xg)

    return {
        "predictor": predictor, "model": est,
        "test_idx": test_idx, "fit_idx": fit_idx, "val_idx": val_idx,
        "train_pool": train_pool, "y_te": y_all[test_idx], "yp_te": yp_te,
        "extrap": extrap, "extrap_any": extrap_any,
        "best_iteration": getattr(est, "best_iteration",
                                  getattr(est, "best_iteration_", None)),
        "elapsed": elapsed, "n_fit": len(fit_idx), "n_val": len(val_idx),
    }


def fit_predict_mlp(regime, X_all, y_all, meta_df, is_des, sys_mask, best_params,
                    tied_idx_pairs, swappable_all, device, logger):
    """One LOSO fold with the MLP: five weight seeds averaged into an ensemble."""
    from .models.mlp import predict_mlp, train_mlp

    test_idx, train_pool, fit_idx, val_idx = loso_split(regime, is_des, sys_mask)

    # the scalers see the fit data only - never the held-out system
    sx = TiedScaler(tied_idx_pairs).fit(X_all[fit_idx])
    sy = StandardScaler().fit(y_all[fit_idx].reshape(-1, 1))
    X_in = sx.transform(X_all[fit_idx])
    y_in = sy.transform(y_all[fit_idx].reshape(-1, 1)).ravel()
    X_va = sx.transform(X_all[val_idx])
    y_va = sy.transform(y_all[val_idx].reshape(-1, 1)).ravel()
    swp = swappable_all[fit_idx] if swappable_all is not None else None
    Xa, ya, wa = augment_swaps(X_in, y_in, swp, tied_idx_pairs)
    Xte_s = sx.transform(X_all[test_idx])

    t0 = time.time()
    models, best_epochs, preds = [], [], []
    for ws in CONFIG["WEIGHT_SEEDS"]:
        m, be, _, _ = train_mlp(Xa, ya, X_va, y_va, hp=best_params, weight_seed=ws,
                                device=device, sample_weight=wa, tag=f"{regime} w{ws}")
        models.append(m)
        best_epochs.append(be)
        preds.append(predict_mlp(m, Xte_s, device))
    elapsed = time.time() - t0
    yp_te = sy.inverse_transform(np.mean(preds, axis=0).reshape(-1, 1)).ravel()
    extrap, extrap_any = extrapolation_flags(meta_df, fit_idx, test_idx)

    def predictor(desc_tmpl, cond_keys_sel, all_cols, points):
        rows = _condition_rows(desc_tmpl, cond_keys_sel, points)
        Xg = pd.DataFrame(rows)[all_cols].values.astype(np.float64)
        Xs = sx.transform(Xg)
        p = np.mean([predict_mlp(m, Xs, device) for m in models], axis=0)
        return sy.inverse_transform(p.reshape(-1, 1)).ravel()

    return {
        "predictor": predictor, "model": models,
        "test_idx": test_idx, "fit_idx": fit_idx, "val_idx": val_idx,
        "train_pool": train_pool, "y_te": y_all[test_idx], "yp_te": yp_te,
        "extrap": extrap, "extrap_any": extrap_any,
        "best_iteration": int(np.mean(best_epochs)) if best_epochs else None,
        "elapsed": elapsed, "n_fit": len(fit_idx), "n_val": len(val_idx),
    }


def fit_predict_gnn(ctx, regime, X_all, y_all, meta_df, is_des, sys_mask, best_params,
                    tied_idx_pairs, swappable_all, device, logger):
    """One LOSO fold with GNN+MLP (Concat); five weight seeds averaged."""
    from .models.gnn import augment_swaps_graph, predict_gnn, train_gnn

    gb = ctx.graph_bank
    DRG, C2, C3, W2, W3, WW = ctx.packs
    arch, enc_hp = ctx.arch, ctx.encoder_hp
    combine, water = ctx.graph_combine, ctx.water_data
    hd, hdo, bs = best_params["dims"], best_params["dropouts"], best_params["batch_size"]

    test_idx, train_pool, fit_idx, val_idx = loso_split(regime, is_des, sys_mask)

    sx = TiedScaler(tied_idx_pairs).fit(X_all[fit_idx])
    sy = StandardScaler().fit(y_all[fit_idx].reshape(-1, 1))
    X_in = sx.transform(X_all[fit_idx])
    y_in = sy.transform(y_all[fit_idx].reshape(-1, 1)).ravel()
    X_va = sx.transform(X_all[val_idx])
    y_va = sy.transform(y_all[val_idx].reshape(-1, 1)).ravel()

    swp = swappable_all[fit_idx] if swappable_all is not None else None
    Xa, drga, c2a, c3a, ya, wa = augment_swaps_graph(
        X_in, DRG[fit_idx], C2[fit_idx], C3[fit_idx], y_in, swp, tied_idx_pairs)
    z = np.zeros(len(ya), np.float64)
    pin = (drga, c2a, c3a, z, z, z)
    pva = (DRG[val_idx], C2[val_idx], C3[val_idx], W2[val_idx], W3[val_idx], WW[val_idx])
    pack_te = (DRG[test_idx], C2[test_idx], C3[test_idx],
               W2[test_idx], W3[test_idx], WW[test_idx])
    Xte_s = sx.transform(X_all[test_idx])

    t0 = time.time()
    models, best_epochs, preds = [], [], []
    for ws in CONFIG["WEIGHT_SEEDS"]:
        m, be = train_gnn(gb, pin, Xa, ya, pva, X_va, y_va, arch, enc_hp, hd, hdo, bs,
                          ws, device, combine, water, sample_weight=wa,
                          tag=f"{regime} w{ws}")
        models.append(m)
        best_epochs.append(be)
        preds.append(predict_gnn(m, gb, pack_te, Xte_s, device, combine, water))
    elapsed = time.time() - t0
    yp_te = sy.inverse_transform(np.mean(preds, axis=0).reshape(-1, 1)).ravel()
    extrap, extrap_any = extrapolation_flags(meta_df, fit_idx, test_idx)

    # molecule codes of the held-out system (constant within the system)
    drg_c, c2_c, c3_c = DRG[test_idx[0]], C2[test_idx[0]], C3[test_idx[0]]

    def predictor(desc_tmpl, cond_keys_sel, all_cols, points):
        rows = _condition_rows(desc_tmpl, cond_keys_sel, points)
        Xg = pd.DataFrame(rows)[all_cols].values.astype(np.float64)
        Xs = sx.transform(Xg)
        m_pts = len(points)
        pack = (np.array([drg_c] * m_pts, dtype=object),
                np.array([c2_c] * m_pts, dtype=object),
                np.array([c3_c] * m_pts, dtype=object),
                np.array([float(p[1]) for p in points]),
                np.array([float(p[2]) for p in points]),
                np.array([float(p[3]) for p in points]))
        p = np.mean([predict_gnn(mm, gb, pack, Xs, device, combine, water)
                     for mm in models], axis=0)
        return sy.inverse_transform(p.reshape(-1, 1)).ravel()

    return {
        "predictor": predictor, "model": models,
        "test_idx": test_idx, "fit_idx": fit_idx, "val_idx": val_idx,
        "train_pool": train_pool, "y_te": y_all[test_idx], "yp_te": yp_te,
        "extrap": extrap, "extrap_any": extrap_any,
        "best_iteration": int(np.mean(best_epochs)) if best_epochs else None,
        "elapsed": elapsed, "n_fit": len(fit_idx), "n_val": len(val_idx),
    }

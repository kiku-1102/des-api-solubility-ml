"""
Figures (manuscript Figs. 3, 5 and 7-9).

Two kinds of figure are produced by the runs themselves:

``save_scatter``
    measured versus predicted log x with the metric box, written for every
    configuration of Studies I and II - the panels of Figs. 3 and 5.
``combined_figure``
    the per-system LOSO summary of Figs. 7-9 and S5: the dependence of log x on
    temperature, water content and HBA:HBD ratio (measured points, predicted
    points and the model curve), the residuals against the same three axes, and
    the measured-versus-predicted scatter plot.  Test points that lie outside
    the range of the training conditions are highlighted.

Aggregated figures and tables (the ranking tables and the bar charts of the
manuscript) are assembled outside this repository from the per-run CSV files.
"""

import re
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AXES, CONFIG
from .metrics import compute_all_metrics, safe_pearsonr, safe_spearmanr

#: colour used to highlight extrapolated test points
EX_COLOR = "#7E2F8E"


def predict_conditions(predictor, desc_template, cond_keys_sel, all_cols, points):
    """Predict log x for a grid of conditions of one fixed system.

    ``predictor`` is a callable ``(desc_template, cond_keys_sel, all_cols,
    points) -> ndarray`` supplied by the model that is being plotted, so that
    the figure code is independent of the model family.  ``desc_template`` holds
    the descriptor columns, which are constant within a system; ``points`` is a
    list of ``(T_K, x_slot2, x_slot3, x_water)`` tuples.
    """
    return predictor(desc_template, cond_keys_sel, all_cols, points)


def save_scatter(y_train_true, y_train_pred, y_test_true, y_test_pred,
                 path: Path, title: str,
                 m_train: Optional[Dict] = None, m_test: Optional[Dict] = None,
                 hp_text: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
    ar = CONFIG["SCATTER_RANGE"]
    if len(y_train_true) > 0:
        ax.scatter(y_train_true, y_train_pred, s=10, alpha=0.5, c="blue",
                   label=f"Train n={len(y_train_true)}")
    if len(y_test_true) > 0:
        ax.scatter(y_test_true, y_test_pred, s=14, alpha=0.8, c="red",
                   label=f"Test n={len(y_test_true)}")
    ax.plot(ar, ar, "--", lw=1, color="black", alpha=0.6)
    ax.set_xlim(*ar); ax.set_ylim(*ar)
    ax.set_xlabel("Experimental log(x)"); ax.set_ylabel("Predicted log(x)")
    ax.set_title(title, fontsize=10); ax.legend(loc="upper left", fontsize=9)

    lines = []
    if m_train and m_train.get("n", 0) >= 2:
        lines.append(
            f"Train R2={m_train.get('R2', np.nan):.4f}  RMSE={m_train.get('RMSE', np.nan):.4f}\n"
            f"      MAPE={m_train.get('MAPE', np.nan):.4f}%  %Nout={m_train.get('Nout_pct', np.nan):.4f}\n"
            f"      rho={m_train.get('Spearman_rho', np.nan):.4f}  tau={m_train.get('Kendall_tau', np.nan):.4f}"
        )
    if m_test and m_test.get("n", 0) >= 2:
        lines.append(
            f"Test  R2={m_test.get('R2', np.nan):.4f}  RMSE={m_test.get('RMSE', np.nan):.4f}\n"
            f"      MAPE={m_test.get('MAPE', np.nan):.4f}%  %Nout={m_test.get('Nout_pct', np.nan):.4f}\n"
            f"      rho={m_test.get('Spearman_rho', np.nan):.4f}  tau={m_test.get('Kendall_tau', np.nan):.4f}"
        )
    if lines:
        ax.text(0.97, 0.04, "\n\n".join(lines), transform=ax.transAxes,
                va="bottom", ha="right", fontsize=7.5,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
                family="monospace")
    if hp_text:
        ax.text(0.03, 0.82, hp_text, transform=ax.transAxes,
                va="top", ha="left", fontsize=7.0,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="lightgray"),
                family="monospace")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s)).strip("_") or "x"


def _fit_window(dmin, dmax, span=2.0, margin=0.08):
    """Shift a fixed-width window so that it contains [dmin, dmax].
    The window is only widened when the data range exceeds ``span``."""
    if not (np.isfinite(dmin) and np.isfinite(dmax)):
        return (-span / 2, span / 2)
    if (dmax - dmin) + 2 * margin <= span:
        mid = 0.5 * (dmin + dmax)
        return (mid - span / 2, mid + span / 2)
    return (dmin - margin, dmax + margin)


def _draw_scatter(ax, y, yp, extrap_any, rng, sub_title, hp_text=None):
    """Draw the measured-versus-predicted scatter plot on ``ax``."""
    lo, hi = rng
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, alpha=0.6)
    ok = ~extrap_any
    ax.scatter(y[ok], yp[ok], s=26, c="#1f77b4", edgecolor="k", lw=0.3,
               alpha=0.8, label=f"in-range (n={int(ok.sum())})")
    if extrap_any.any():
        ax.scatter(y[extrap_any], yp[extrap_any], s=46, marker="D", c=EX_COLOR,
                   edgecolor="k", lw=0.4, alpha=0.9, label=f"extrapolation (n={int(extrap_any.sum())})")
    m = compute_all_metrics(y, yp)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", "box")
    ax.set_xlabel("Experimental logS"); ax.set_ylabel("Predicted logS")
    ax.set_title(sub_title, fontsize=10); ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    box = (f"R2={m.get('R2', np.nan):.4f}  RMSE={m.get('RMSE', np.nan):.4f}\n"
           f"MAE={m.get('MAE', np.nan):.4f}  MAPE={m.get('MAPE', np.nan):.2f}%\n"
           f"%Nout={m.get('Nout_pct', np.nan):.2f}\n"
           f"rho={m.get('Spearman_rho', np.nan):.4f}  tau={m.get('Kendall_tau', np.nan):.4f}")
    ax.text(0.97, 0.04, box, transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
            family="monospace")
    if hp_text:
        ax.text(0.035, 0.88, hp_text, transform=ax.transAxes, va="top", ha="left", fontsize=6.8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="lightgray"),
                family="monospace", linespacing=1.25)


def _scatter_extrap(y, yp, extrap_any, path, title, rng, hp_text=None):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
    _draw_scatter(ax, y, yp, extrap_any, rng, title, hp_text)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _draw_residuals(axes, meta_te, y, yp, extrap):
    """Residuals against each condition axis; the y window has a fixed span of 2."""
    resid = yp - y
    win = _fit_window(float(np.nanmin(resid)), float(np.nanmax(resid)), span=2.0)
    for ax, (col, lab, _) in zip(axes, AXES):
        av = pd.to_numeric(meta_te[col], errors="coerce").values
        ex = np.nan_to_num(extrap.get(col, np.zeros(len(av), bool)), nan=False).astype(bool)
        ax.axhspan(-0.5, 0.5, color="green", alpha=0.06)
        ax.axhline(0, color="gray", lw=1, ls="--")
        ax.axhline(0.5, color="green", lw=0.7, ls=":", alpha=0.6)
        ax.axhline(-0.5, color="green", lw=0.7, ls=":", alpha=0.6)
        ax.scatter(av[~ex], resid[~ex], s=22, c="#1f77b4", edgecolor="k", lw=0.3, alpha=0.8)
        if ex.any():
            ax.scatter(av[ex], resid[ex], s=40, marker="D", c=EX_COLOR, edgecolor="k", lw=0.4)
        r = safe_pearsonr(av, resid)
        ax.set_ylim(*win)   # fixed span, shifted
        ax.set_xlabel(lab, fontsize=8); ax.set_ylabel("residual (pred-meas)", fontsize=8)
        ax.set_title(f"resid vs {col}  (Pearson={r:.3f})", fontsize=8)
        ax.grid(alpha=0.3)
    return win


def _residual_vs_axes(meta_te, y, yp, extrap, path, title):
    fig, axes = plt.subplots(1, len(AXES), figsize=(4.2 * len(AXES), 3.8), dpi=130)
    _draw_residuals(axes, meta_te, y, yp, extrap)
    fig.suptitle(title + "   [y span=2 fixed (shifted to fit) / green band: |resid|<0.5]",
                 fontsize=10, y=1.03)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _draw_dependency(ax, fig, P, vals, nuniq, med, y, yp, extrap,
                     model, desc_tmpl, cond_keys_sel, all_cols, lab, line_lw=0.8):
    """Draw the dependence of log x on the primary axis P.
       Point colour encodes the secondary axis S (its most frequent levels);
       measured values are circles, predictions crosses with a residual stick,
       and the model curve is drawn for every existing (S, Th) combination."""
    import matplotlib as mpl
    cmap = mpl.cm.viridis
    # the colour (secondary) axis is fixed per primary axis for consistency:
    #   T_K -> coloured by water content, x_WATER -> by temperature,
    #   comp_ratio -> by water content
    COLOR_BY = {"T_K": "x_WATER", "x_WATER": "T_K", "comp_ratio": "x_WATER"}
    S = COLOR_BY.get(P, next(a for a, _, _ in AXES if a != P))
    if nuniq.get(S, 0) < 2:   # fall back to a varying axis
        alt = [a for a, _, _ in AXES if a != P and a != S and nuniq.get(a, 0) >= 2]
        if alt:
            S = alt[0]
    Th = next(a for a, _, _ in AXES if a != P and a != S)
    pv, sv, thv = vals[P], vals[S], vals[Th]
    ex = np.nan_to_num(extrap.get(P, np.zeros(len(pv), bool)), nan=False).astype(bool)
    sf = sv[np.isfinite(sv)]
    smin, smax = (float(sf.min()), float(sf.max())) if len(sf) else (0.0, 1.0)
    norm = mpl.colors.Normalize(vmin=smin, vmax=(smax if smax > smin else smin + 1e-9))

    def quad(pp, ss, tt):
        cond = {P: pp, S: ss, Th: tt}
        T = cond["T_K"]; xw = cond["x_WATER"]; r = cond["comp_ratio"]
        r = 0.0 if not np.isfinite(r) else r
        xw = 0.0 if not np.isfinite(xw) else xw
        x3 = r * (1.0 - xw); x2 = (1.0 - r) * (1.0 - xw)
        return (T, x2, x3, xw)

    for i in range(len(pv)):
        c = cmap(norm(sv[i])) if np.isfinite(sv[i]) else (0.5, 0.5, 0.5, 1)
        ax.plot([pv[i], pv[i]], [y[i], yp[i]], color="0.6", lw=0.5, alpha=0.6, zorder=1)
        ec = EX_COLOR if ex[i] else "k"; elw = 1.6 if ex[i] else 0.4
        ax.scatter(pv[i], y[i], s=44, color=c, marker="o", edgecolor=ec, lw=elw, zorder=3)
        ax.scatter(pv[i], yp[i], s=50, color=c, marker="X", edgecolor=ec, lw=elw, zorder=3)

    # model curves for every existing (S, Th) combination
    grid = np.linspace(np.nanmin(pv), np.nanmax(pv), 40)
    fin = np.isfinite(sv) & np.isfinite(thv)
    combos = sorted(set(zip(np.round(sv[fin], 6).tolist(), np.round(thv[fin], 6).tolist())))
    if not combos:
        combos = [(round(float(s), 6), med[Th]) for s in np.unique(np.round(sf, 6))]
    for (slev, thlev) in combos:
        try:
            pts = [quad(gp, slev, thlev) for gp in grid]
            yc = predict_conditions(model, desc_tmpl, cond_keys_sel, all_cols, pts)
            ax.plot(grid, yc, "-", color=cmap(norm(slev)), lw=line_lw, alpha=0.8, zorder=2)
        except Exception:
            pass

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    _short = {"T_K": "T [K]", "x_WATER": "x_water", "comp_ratio": "HBA:HBD"}
    cb.ax.set_title(_short.get(S, lab[S]), fontsize=8, pad=3)   # label above the colour bar
    ax.set_xlabel(lab[P], fontsize=9); ax.set_ylabel("logS", fontsize=9)
    ax.set_title(f"{P}  (color={S})", fontsize=8.5)
    ax.grid(alpha=0.3)
    return bool(ex.any())


def _dependency_legend_handles(has_extrap, line_lw=0.8):
    """Legend handles shared by the dependence panels."""
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markeredgecolor="k", label="measured", markersize=8),
         Line2D([0], [0], marker="X", color="w", markerfacecolor="0.4", markeredgecolor="k", label="predicted", markersize=9),
         Line2D([0], [0], color="0.4", lw=max(line_lw, 1.2), label="model curves (all cond.)")]
    if has_extrap:
        h.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
                        markeredgecolor=EX_COLOR, markeredgewidth=1.6, label="extrapolation", markersize=8))
    return h


def _dependency_curves(model, meta_te, y, yp, extrap, desc_tmpl, cond_keys_sel, all_cols, path, title):
    lab = {a: l for a, l, _ in AXES}
    vals = {a: pd.to_numeric(meta_te[a], errors="coerce").values for a, _, _ in AXES}
    nuniq = {a: int(np.unique(np.round(vals[a][np.isfinite(vals[a])], 8)).size) for a in vals}
    varying = [a for a, _, _ in AXES if nuniq[a] >= 2]
    if not varying:
        return
    med = {a: float(np.nanmedian(vals[a])) for a in vals}
    fig, axes = plt.subplots(1, len(varying), figsize=(5.2 * len(varying), 4.7), dpi=130, squeeze=False)
    axes = axes[0]
    has_ex = False
    for ax, P in zip(axes, varying):
        e = _draw_dependency(ax, fig, P, vals, nuniq, med, y, yp, extrap,
                             model, desc_tmpl, cond_keys_sel, all_cols, lab)
        has_ex = has_ex or e
    fig.legend(handles=_dependency_legend_handles(has_ex), loc="upper center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle(title, fontsize=11, y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _combined_figure(model, meta_te, y, yp, extrap, extrap_any,
                     desc_tmpl, cond_keys_sel, all_cols, rng, hp_text, path, title):
    """Per-system summary figure (manuscript Figs. 7-9 and S5).
       Left: dependence curves (top) with the residuals aligned underneath,
       one column per condition axis; right: measured versus predicted."""
    lab = {a: l for a, l, _ in AXES}
    vals = {a: pd.to_numeric(meta_te[a], errors="coerce").values for a, _, _ in AXES}
    nuniq = {a: int(np.unique(np.round(vals[a][np.isfinite(vals[a])], 8)).size) for a in vals}
    med = {a: float(np.nanmedian(vals[a])) for a in vals}
    axcol = {a: j for j, (a, _, _) in enumerate(AXES)}     # T_K=0, x_WATER=1, comp_ratio=2

    fig = plt.figure(figsize=(15.5, 6.7), dpi=140)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0],
                          width_ratios=[1.0, 1.0, 1.0, 1.95],
                          hspace=0.5, wspace=0.38,
                          left=0.045, right=0.985, top=0.82, bottom=0.10)

    # dependence panels (row 0), one column per axis
    has_ex = False
    for a, _, _ in AXES:
        if nuniq[a] >= 2:
            ax = fig.add_subplot(gs[0, axcol[a]])
            e = _draw_dependency(ax, fig, a, vals, nuniq, med, y, yp, extrap,
                                 model, desc_tmpl, cond_keys_sel, all_cols, lab, line_lw=0.7)
            has_ex = has_ex or e

    # residuals (row 1), aligned with the columns above
    rax = [fig.add_subplot(gs[1, j]) for j in range(3)]
    _draw_residuals(rax, meta_te, y, yp, extrap)

    # scatter plot on the right, spanning both rows
    sax = fig.add_subplot(gs[0:2, 3])
    _draw_scatter(sax, y, yp, extrap_any, rng, "Predicted vs Experimental", hp_text)

    # one shared legend placed above the dependence panels
    fig.legend(handles=_dependency_legend_handles(has_ex, 0.7), loc="upper center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.33, 0.925), frameon=True)
    fig.suptitle(title + "  - LOSO summary", fontsize=13, y=0.975)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def combined_figure(predictor, meta_te, y, yp, extrap, extrap_any, desc_tmpl,
                    cond_keys_sel, all_cols, rng, hp_text, path, title):
    """Public name of the per-system LOSO summary figure (Figs. 7-9, S5)."""
    return _combined_figure(predictor, meta_te, y, yp, extrap, extrap_any, desc_tmpl,
                            cond_keys_sel, all_cols, rng, hp_text, path, title)


def dependency_curves(predictor, meta_te, y, yp, extrap, desc_tmpl, cond_keys_sel,
                      all_cols, path, title):
    """Public name of the stand-alone dependence-curve figure."""
    return _dependency_curves(predictor, meta_te, y, yp, extrap, desc_tmpl,
                              cond_keys_sel, all_cols, path, title)


def residual_vs_axes(meta_te, y, yp, extrap, path, title):
    """Public name of the stand-alone residual figure."""
    return _residual_vs_axes(meta_te, y, yp, extrap, path, title)


def scatter_extrap(y, yp, extrap_any, path, title, rng, hp_text=None):
    """Public name of the stand-alone LOSO scatter plot."""
    return _scatter_extrap(y, yp, extrap_any, path, title, rng, hp_text)


def slug(text: str) -> str:
    """File-name-safe version of a system key such as ``DAP/ChCl/TEG``."""
    return _slug(text)

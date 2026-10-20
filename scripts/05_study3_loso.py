#!/usr/bin/env python3
"""
Step 5 - Study III: leave-one-system-out generalisation
(manuscript Section 2.8.3).

A "system" is one combination of API x HBA x HBD.  Every system with at least
35 DES data points is held out in turn: the whole system becomes the test set
and the model is trained on the remaining systems, so the evaluation asks
whether the solubility of an entirely unseen combination can be predicted from
data on other systems.  The input representation is fixed to the concatenated
one and two training regimes are compared (DES only and Joint); the manuscript
reports three models - XGB, MLP and GATv2+MLP - giving six configurations.

The hyperparameters are those of step 2 (DES-only random split); they are not
re-searched per fold.

Usage
-----
    python scripts/05_study3_loso.py --model xgb   --regime des_only
    python scripts/05_study3_loso.py --model mlp   --regime joint
    python scripts/05_study3_loso.py --model gatv2 --regime des_only

Outputs (results/study3/<model>/<regime>/)
------------------------------------------
    summary_per_system.csv      metrics of every held-out system (Tables S4, S5)
    summary_aggregate.csv       mean and standard deviation over all systems (Fig. 6)
    axis_dependency.csv         per-axis slope / rank / residual diagnostics
    systems_coverage.csv        all DES systems and their sample counts
    per_system/<system>/        predictions.csv and the single-panel figures
    combined_figures/<system>.png   the per-system summary of Figs. 7-9 and S5
    execution.log
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desml.config import AXES, CONFIG, configure, results_dir
from desml.logging_utils import Logger
from desml.loso import (add_comp_ratio, axis_dependency_metrics, desc_template,
                        enumerate_target_systems, fit_predict_descriptor,
                        fit_predict_gnn, fit_predict_mlp, others_for, system_key)
from desml.metrics import compute_all_metrics
from desml.pipeline import prepare_descriptor_dataset, prepare_graph_dataset
from desml.plots import (combined_figure, dependency_curves, residual_vs_axes,
                         scatter_extrap, slug)
from desml.study import load_best_hp

MODELS = ("xgb", "mlp", "gatv2", "gin", "nusvr_rbf", "nusvr_poly", "rf", "lgb")
REGIMES = ("des_only", "joint")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--regime", choices=list(REGIMES), default="des_only")
    p.add_argument("--min-system-count", type=int, default=CONFIG["MIN_SYSTEM_COUNT"],
                   help="minimum number of DES points of a target system (default 35)")
    p.add_argument("--data-dir", default=CONFIG["DATA_DIR"])
    p.add_argument("--results-dir", default=CONFIG["RESULTS_DIR"])
    p.add_argument("--molecule-source", choices=["xyz", "smiles"],
                   default=CONFIG["MOLECULE_SOURCE"])
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    return p.parse_args()


def main():
    args = parse_args()
    configure(DATA_DIR=args.data_dir, RESULTS_DIR=args.results_dir,
              MOLECULE_SOURCE=args.molecule_source, USE_GPU=not args.cpu,
              MIN_SYSTEM_COUNT=args.min_system_count,
              SCATTER_RANGE=(-6.0, 0.0))
    model = args.model.lower()
    regime = args.regime
    is_graph = model in ("gin", "gatv2")

    out_dir = results_dir() / "study3" / model / regime
    (out_dir / "per_system").mkdir(parents=True, exist_ok=True)
    (out_dir / "combined_figures").mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "execution.log")

    try:
        logger.log("=" * 70)
        logger.log(f"Study III - LOSO   model={model}  regime={regime}  "
                   f"min_system_count={CONFIG['MIN_SYSTEM_COUNT']}")
        logger.log("=" * 70)
        with open(out_dir / "00_config.json", "w", encoding="utf-8") as f:
            json.dump({**CONFIG, "model": model, "regime": regime, "input": "concat"},
                      f, indent=2, ensure_ascii=False)
        t_total = time.time()

        device = ctx = None
        if is_graph:
            from desml.models.mlp import get_device
            device = get_device()
            arch = "GIN" if model == "gin" else "GATv2"
            data, ctx = prepare_graph_dataset(logger, arch, "concat", "concat")
            best_params, enc_hp = load_best_hp(model, "concat", results_dir(), logger)
            ctx.encoder_hp = enc_hp
            ctx.water_data = ctx.graph_bank.water
            logger.log(f"  Device: {device}   Configuration: {ctx.describe()}")
            # the design matrix of the graph run is built by build_gnn_data; the
            # concatenated feature table is needed as well for the condition curves
            desc_data = prepare_descriptor_dataset(logger, "concat")
            X_df, feat_groups = desc_data["X_df"], desc_data["feat_groups"]
        else:
            if model == "mlp":
                from desml.models.mlp import get_device
                device = get_device()
                logger.log(f"  Device: {device}")
            data = prepare_descriptor_dataset(logger, "concat")
            best_params, enc_hp = load_best_hp(model, "concat", results_dir(), logger)
            X_df, feat_groups = data["X_df"], data["feat_groups"]

        X_all, y = data["X"], data["y"]
        meta_df = add_comp_ratio(data["meta_df"])
        is_des = data["is_des"]
        tied_idx_pairs = data["tied_idx_pairs"]
        swappable_all = data["swappable_all"]
        all_cols = feat_groups["all_cols"]
        cond_keys_sel = data["manifest"]["condition"]

        targets, all_sizes = enumerate_target_systems(
            meta_df, CONFIG["MIN_SYSTEM_COUNT"], logger)
        all_sizes.to_csv(out_dir / "systems_coverage.csv", index=False, encoding="utf-8-sig")
        if len(targets) == 0:
            logger.log("[WARN] no target system found - lower --min-system-count")
            return

        sys_rows, axis_rows = [], []
        logger.log("\n" + "=" * 70)
        logger.log(f"LOSO over {len(targets)} systems")
        logger.log("=" * 70)

        for _, sysr in targets.iterrows():
            drg, hba, hbd = sysr["DRG"], sysr["HBA"], sysr["HBD"]
            cnt = int(sysr["count"])
            skey = system_key(drg, hba, hbd)
            sdir = out_dir / "per_system" / slug(skey)
            sdir.mkdir(parents=True, exist_ok=True)
            sys_mask = ((meta_df["DRG"] == drg) & (meta_df["HBA"] == hba)
                        & (meta_df["HBD"] == hbd) & is_des).values
            logger.log(f"\n[{skey}] n={cnt}")

            if is_graph:
                res = fit_predict_gnn(ctx, regime, X_all, y, meta_df, is_des, sys_mask,
                                      best_params, tied_idx_pairs, swappable_all,
                                      device, logger)
            elif model == "mlp":
                res = fit_predict_mlp(regime, X_all, y, meta_df, is_des, sys_mask,
                                      best_params, tied_idx_pairs, swappable_all,
                                      device, logger)
            else:
                res = fit_predict_descriptor(model, regime, X_all, y, meta_df, is_des,
                                             sys_mask, best_params, tied_idx_pairs,
                                             swappable_all, logger)

            test_idx = res["test_idx"]
            y_te, yp_te = res["y_te"], res["yp_te"]
            extrap, extrap_any = res["extrap"], res["extrap_any"]
            meta_te = meta_df.iloc[test_idx].reset_index(drop=True)
            metrics = compute_all_metrics(y_te, yp_te)

            # fold validity: are the API and the HBD of the held-out system still
            # present in the training data through other systems?
            tp_des = meta_df.iloc[res["train_pool"]]
            tp_des = tp_des[tp_des["system_type"].astype(str).str.upper() == "DES"]
            drug_in_train = bool((tp_des["DRG"] == drg).any())
            hbd_in_train = bool((tp_des["HBD"] == hbd).any())

            for axis, _, _ in AXES:
                av = pd.to_numeric(meta_te[axis], errors="coerce").values
                m = axis_dependency_metrics(axis, av, others_for(axis, meta_te),
                                            y_te, yp_te, CONFIG)
                if m is None:
                    continue
                m.update({"system": skey, "DRG": drg, "HBA": hba, "HBD": hbd})
                if axis == "T_K":   # physical expectation: solubility rises with T
                    m["physical_T_sign_ok"] = (float(m["slope_pred"] > 0)
                                               if np.isfinite(m["slope_pred"]) else np.nan)
                axis_rows.append(m)

            rng = CONFIG["SCATTER_RANGE"]
            title = f"{drg}/{hba}/{hbd} [{model} / {regime}] n={cnt}"
            hp_text = (f"{model} - hyperparameters reused from step 2\n"
                       f"{best_params}\n"
                       f"best_iteration~{res['best_iteration']}")
            tmpl = desc_template(X_df.iloc[test_idx[0]], feat_groups)
            predictor = res["predictor"]

            scatter_extrap(y_te, yp_te, extrap_any, sdir / "scatter.png", title, rng, hp_text)
            residual_vs_axes(meta_te, y_te, yp_te, extrap, sdir / "residual_vs_axes.png", title)
            dependency_curves(predictor, meta_te, y_te, yp_te, extrap, tmpl,
                              cond_keys_sel, all_cols, sdir / "dependency_curves.png", title)
            combined_figure(predictor, meta_te, y_te, yp_te, extrap, extrap_any, tmpl,
                            cond_keys_sel, all_cols, rng, hp_text,
                            out_dir / "combined_figures" / f"{slug(skey)}.png", title)

            pred = meta_te.copy()
            pred["logS_pred"] = yp_te
            pred["residual"] = yp_te - y_te
            for axis, _, _ in AXES:
                pred[f"extrap_{axis}"] = np.nan_to_num(
                    extrap.get(axis, np.zeros(len(pred), bool)), nan=False).astype(int)
            pred["extrap_any"] = extrap_any.astype(int)
            pred.to_csv(sdir / "predictions.csv", index=False, encoding="utf-8-sig")

            sys_rows.append({
                "system": skey, "DRG": drg, "HBA": hba, "HBD": hbd, "n_test": cnt,
                "drug_in_train": drug_in_train, "hbd_in_train": hbd_in_train,
                "n_extrap": int(extrap_any.sum()),
                "RMSE": metrics["RMSE"], "MAE": metrics["MAE"], "R2": metrics["R2"],
                "MAPE": metrics["MAPE"], "Nout_pct": metrics["Nout_pct"],
                "Pearson_r": metrics["Pearson_r"], "Spearman_rho": metrics["Spearman_rho"],
                "Kendall_tau": metrics["Kendall_tau"], "Bias": metrics["Bias"],
                "n_fit": res["n_fit"], "n_val": res["n_val"],
                "best_iteration": res["best_iteration"], "fit_sec": res["elapsed"],
            })
            logger.log(f"    RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} "
                       f"rho={metrics['Spearman_rho']:.4f} "
                       f"extrapolated={int(extrap_any.sum())}/{cnt} "
                       f"({res['elapsed']:.1f}s)")

        sys_df = pd.DataFrame(sys_rows)
        sys_df.to_csv(out_dir / "summary_per_system.csv", index=False, encoding="utf-8-sig")
        if axis_rows:
            pd.DataFrame(axis_rows).to_csv(out_dir / "axis_dependency.csv",
                                           index=False, encoding="utf-8-sig")

        num_cols = [c for c in sys_df.columns if sys_df[c].dtype.kind in "fi"]
        agg = {"model": model, "regime": regime, "n_systems": len(sys_df)}
        for c in num_cols:
            agg[f"{c}_mean"] = float(sys_df[c].mean())
            agg[f"{c}_std"] = float(sys_df[c].std(ddof=0))
        pd.DataFrame([agg]).to_csv(out_dir / "summary_aggregate.csv", index=False,
                                   encoding="utf-8-sig")

        logger.log("\n" + "=" * 70)
        logger.log(f"Mean over {len(sys_df)} systems: "
                   f"RMSE={agg['RMSE_mean']:.4f} +- {agg['RMSE_std']:.4f}, "
                   f"rho={agg['Spearman_rho_mean']:.4f} +- {agg['Spearman_rho_std']:.4f}")
        logger.log(f"  Results written to {out_dir}")
        logger.log(f"  Total elapsed: {time.time() - t_total:.1f} s")
        logger.log("=" * 70)
    finally:
        logger.close()


if __name__ == "__main__":
    main()

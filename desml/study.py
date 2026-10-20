"""
Runner shared by Study I (``scripts/03``) and Study II (``scripts/04``).

Both studies train the same models on the same features and differ only in the
training regime (manuscript Section 2.8):

* Study I  - ``prior``: all data pooled and split 8:2, the scheme of the
  previous study, so that the two can be compared on the same footing.
* Study II - ``des_only`` / ``joint`` / ``pretrain_ft``: only DES systems are
  used as test data, so the accuracy for DESs is assessed directly.

Every configuration is evaluated over five random splits (seeds 42, 123, 456,
789 and 1024).  For the neural networks five weight-initialisation seeds are
used in addition, giving 25 runs; the five weight seeds of one data seed are
averaged into an ensemble prediction and the reported metrics are the mean over
the five data seeds (Section 2.6.1).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG
from .grids import hp_to_string
from .logging_utils import Logger
from .metrics import METRIC_KEYS, compute_all_metrics, metrics_row
from .plots import save_scatter

NEURAL_MODELS = ("mlp", "gin", "gatv2")


def load_best_hp(model: str, representation: str, results_dir: Path, logger: Logger):
    """Read the hyperparameters written by ``scripts/02_hyperparameter_search.py``.

    For the graph models the head and the encoder come from the two-stage
    search in the graph-only configuration and are shared by Eqs. (4)-(7).
    """
    model = model.lower()
    if model in ("gin", "gatv2"):
        base = results_dir / "hyperparameters" / model / "graph_only"
        head_path = base / "best_head_hp.json"
        enc_path = base / "best_encoder_hp.json"
        for p in (head_path, enc_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found - run\n"
                    f"  python scripts/02_hyperparameter_search.py --model {model}\n"
                    f"first.")
        head = json.load(open(head_path, encoding="utf-8"))
        enc = json.load(open(enc_path, encoding="utf-8"))
        logger.log(f"  Head    : {hp_to_string('mlp', head)}  ({head_path})")
        logger.log(f"  Encoder : {enc}  ({enc_path})")
        return head, enc
    path = results_dir / "hyperparameters" / model / representation / "best_hp.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run\n"
            f"  python scripts/02_hyperparameter_search.py --model {model} "
            f"--input {representation}\nfirst.")
    hp = json.load(open(path, encoding="utf-8"))
    logger.log(f"  Hyperparameters: {hp_to_string(model, hp)}  ({path})")
    return hp, None


def _write_seed_outputs(out_dir, tag, y_tr, yp_tr, y_te, yp_te, meta_tr, meta_te,
                        title, hp_text):
    """Scatter plot and test predictions of one seed."""
    seed_dir = out_dir / tag
    seed_dir.mkdir(parents=True, exist_ok=True)
    m_tr = compute_all_metrics(y_tr, yp_tr)
    m_te = compute_all_metrics(y_te, yp_te)
    save_scatter(y_tr, yp_tr, y_te, yp_te, seed_dir / "scatter.png", title,
                 m_tr, m_te, hp_text)
    pred = meta_te.copy()
    pred["logS_pred"] = yp_te
    pred["residual"] = yp_te - y_te
    pred.to_csv(seed_dir / "predictions_test.csv", index=False, encoding="utf-8-sig")
    return m_tr, m_te


def run_study(model, representation, regime, data, out_dir, logger,
              device=None, ctx=None, hp=None, encoder_hp=None):
    """Train and evaluate one configuration over all seeds; write all outputs.

    ``data`` comes from ``pipeline.prepare_descriptor_dataset`` or
    ``pipeline.prepare_graph_dataset``.  Returns the summary DataFrame.
    """
    model = model.lower()
    X, y = data["X"], data["y"]
    meta_df, is_des = data["meta_df"], data["is_des"]
    tied, swappable = data["tied_idx_pairs"], data["swappable_all"]
    hp_text = (f"{model} [{representation}/{regime}] "
               f"{hp_to_string('mlp' if model in NEURAL_MODELS else model, hp)}")

    per_run_rows, per_seed_rows = [], []
    ov = {"y_tr": [], "yp_tr": [], "y_te": [], "yp_te": []}
    t_start = time.time()

    for data_seed in CONFIG["DATA_SEEDS"]:
        logger.log(f"\n--- data seed {data_seed} ---")
        if model in NEURAL_MODELS:
            preds_tr, preds_te = [], []
            for weight_seed in CONFIG["WEIGHT_SEEDS"]:
                if model == "mlp":
                    from .models.mlp import evaluate_one_pair_regime
                    res = evaluate_one_pair_regime(
                        regime, X, y, meta_df, is_des, data_seed, weight_seed,
                        hp, device, tied, swappable)
                else:
                    from .models.gnn import evaluate_one_pair_regime as gnn_eval
                    res = gnn_eval(ctx, regime, X, y, meta_df, is_des, data_seed,
                                   weight_seed, hp, device, tied, swappable)
                sti, ti = res["scatter_train_idx"], res["test_idx"]
                preds_tr.append(res["yp_tr"])
                preds_te.append(res["yp_te"])
                m_tr = compute_all_metrics(y[sti], res["yp_tr"])
                m_te = compute_all_metrics(y[ti], res["yp_te"])
                row = {"data_seed": data_seed, "weight_seed": weight_seed,
                       "fit_sec": res["fit_elapsed"], "info": res["info"]}
                row.update(metrics_row("train", m_tr))
                row.update(metrics_row("test", m_te))
                per_run_rows.append(row)
                logger.log(f"  w{weight_seed}: test RMSE={m_te['RMSE']:.4f} "
                           f"MAPE={m_te['MAPE']:.4f}% rho={m_te['Spearman_rho']:.4f} "
                           f"{res['info']}")
            yp_tr = np.mean(preds_tr, axis=0)
            yp_te = np.mean(preds_te, axis=0)
            meta_tr = meta_df.iloc[sti].reset_index(drop=True)
            meta_te = meta_df.iloc[ti].reset_index(drop=True)
            y_tr, y_te = y[sti], y[ti]
            info = f"ensemble of {len(CONFIG['WEIGHT_SEEDS'])} weight seeds"
        else:
            from .models.descriptor_models import fit_predict
            res = fit_predict(model, regime, X, y, meta_df, is_des, data_seed,
                              hp, tied, swappable)
            y_tr, yp_tr = res["y_tr"], res["yp_tr"]
            y_te, yp_te = res["y_te"], res["yp_te"]
            meta_tr, meta_te = res["meta_tr"], res["meta_te"]
            info = res["info"]

        m_tr, m_te = _write_seed_outputs(
            out_dir, f"seed{data_seed}", y_tr, yp_tr, y_te, yp_te, meta_tr, meta_te,
            f"{model} [{representation}/{regime}] seed {data_seed}", hp_text)
        if model not in NEURAL_MODELS and res.get("feature_importance") is not None:
            pd.DataFrame({"feature": data["feature_names"],
                          "importance": res["feature_importance"]}).to_csv(
                out_dir / f"seed{data_seed}" / "feature_importance.csv",
                index=False, encoding="utf-8-sig")
        row = {"data_seed": data_seed, "info": info}
        row.update(metrics_row("train", m_tr))
        row.update(metrics_row("test", m_te))
        per_seed_rows.append(row)
        logger.log(f"  seed {data_seed}: test RMSE={m_te['RMSE']:.4f} "
                   f"MAE={m_te['MAE']:.4f} MAPE={m_te['MAPE']:.4f}% "
                   f"%Nout={m_te['Nout_pct']:.4f} rho={m_te['Spearman_rho']:.4f}")
        ov["y_tr"].append(y_tr); ov["yp_tr"].append(yp_tr)
        ov["y_te"].append(y_te); ov["yp_te"].append(yp_te)

    # ---------------------------------------------------------- aggregation
    if per_run_rows:
        pd.DataFrame(per_run_rows).to_csv(out_dir / "metrics_per_run.csv",
                                          index=False, encoding="utf-8-sig")
    seed_df = pd.DataFrame(per_seed_rows)
    seed_df.to_csv(out_dir / "metrics_per_seed.csv", index=False, encoding="utf-8-sig")

    summary = {"model": model, "input": representation, "regime": regime,
               "n_seeds": len(CONFIG["DATA_SEEDS"])}
    for split in ("train", "test"):
        for k in METRIC_KEYS:
            col = f"{split}_{k}"
            summary[f"{col}_mean"] = float(seed_df[col].mean())
            summary[f"{col}_std"] = float(seed_df[col].std(ddof=0))
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    ay_tr = np.concatenate(ov["y_tr"]); ayp_tr = np.concatenate(ov["yp_tr"])
    ay_te = np.concatenate(ov["y_te"]); ayp_te = np.concatenate(ov["yp_te"])
    save_scatter(ay_tr, ayp_tr, ay_te, ayp_te, out_dir / "overall_scatter.png",
                 f"{model} [{representation}/{regime}] all seeds",
                 compute_all_metrics(ay_tr, ayp_tr), compute_all_metrics(ay_te, ayp_te),
                 hp_text)

    logger.log("\n" + "=" * 70)
    logger.log(f"Mean over {len(CONFIG['DATA_SEEDS'])} data seeds (test):")
    for k in METRIC_KEYS:
        logger.log(f"    {k:<13}= {summary[f'test_{k}_mean']:.4f} "
                   f"+- {summary[f'test_{k}_std']:.4f}")
    logger.log(f"  Total elapsed: {time.time() - t_start:.1f} s")
    logger.log("=" * 70)
    return summary_df

#!/usr/bin/env python3
"""
Step 2 - hyperparameter optimisation (manuscript Section 2.6.3, Table 4).

The search is performed once, on the DES systems alone (the DES-only regime),
on the single random split with seed 42 and without cross-validation.  The test
part of that split is sealed off, and the candidates are scored on an inner
validation set carved out of the training part.  The winning values are then
used unchanged in Studies I, II and III and in every training regime, so that
all comparisons are made under identical hyperparameter conditions.

Models
------
``nusvr_rbf`` ``nusvr_poly`` ``rf`` ``xgb`` ``lgb`` ``mlp``
    one grid, searched separately for each input representation
    (``--input concat`` / ``--input mixture``).
``gin`` ``gatv2``
    two stages in the graph-only configuration: stage Ia searches the MLP head
    with the encoder fixed at its default, stage Ib searches the encoder with
    the best head fixed.  The resulting head and encoder are shared by all four
    configurations of Eqs. (4)-(7) of the same architecture, so ``--input`` is
    not used here.

The boosted trees are scored with a fixed number of estimators
(``XGB/LGB_GS_N_ESTIMATORS`` = 1000) and without early stopping; the final
training in steps 3-5 uses up to 3000 estimators with early stopping.

Usage
-----
    python scripts/02_hyperparameter_search.py --model xgb   --input concat
    python scripts/02_hyperparameter_search.py --model mlp   --input mixture
    python scripts/02_hyperparameter_search.py --model gatv2

Outputs (results/hyperparameters/<model>/<input>/)
-------------------------------------------------
    best_hp.json                 winning hyperparameters (read by steps 3-5)
    grid_search_results.csv      every candidate with its validation RMSE
    best_head_hp.json            GNN only: winning head (stage Ia)
    best_encoder_hp.json         GNN only: winning encoder (stage Ib)
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

from desml.config import CONFIG, configure, results_dir
from desml.grids import (build_encoder_grid, build_grid, default_encoder,
                         enc_hp_to_id, enc_hp_to_string, hp_to_id, hp_to_string)
from desml.logging_utils import Logger
from desml.pipeline import prepare_descriptor_dataset, prepare_graph_dataset
from desml.splits import grid_search_split

DESCRIPTOR_MODELS = ("nusvr_rbf", "nusvr_poly", "rf", "xgb", "lgb")
GRAPH_MODELS = ("gin", "gatv2")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   choices=list(DESCRIPTOR_MODELS) + ["mlp"] + list(GRAPH_MODELS))
    p.add_argument("--input", choices=["concat", "mixture"], default="concat",
                   help="input representation (ignored for gin / gatv2)")
    p.add_argument("--data-dir", default=CONFIG["DATA_DIR"])
    p.add_argument("--results-dir", default=CONFIG["RESULTS_DIR"])
    p.add_argument("--molecule-source", choices=["xyz", "smiles"],
                   default=CONFIG["MOLECULE_SOURCE"])
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    return p.parse_args()


def search_descriptor_model(model, data, logger, out_dir):
    """Grid search for a descriptor-based model (and the plain MLP)."""
    from desml.models.descriptor_models import score_candidate

    X, y = data["X"], data["y"]
    des_idx = np.where(data["is_des"])[0]
    Xd, yd = X[des_idx], y[des_idx]
    swap = (data["swappable_all"][des_idx] if data["swappable_all"] is not None else None)
    tied = data["tied_idx_pairs"]

    val_fraction = CONFIG["GS_VAL_FRACTION_DESCRIPTOR"]
    idx_in, idx_va, n_tr42, n_te42 = grid_search_split(len(yd), val_fraction)
    grid = build_grid(model)
    logger.log("\n" + "=" * 70)
    logger.log(f"Grid search: {model} ({len(grid)} combinations, DES only, seed "
               f"{CONFIG['GRID_SEARCH_DATA_SEED']}, no CV)")
    logger.log(f"  seed-42 split: train={n_tr42} (test={n_te42} sealed off)")
    logger.log(f"  inner fit={len(idx_in)}  inner validation={len(idx_va)} "
               f"({val_fraction:.0%} of the training part)")
    logger.log("=" * 70)

    rows = []
    for i, params in enumerate(grid):
        t0 = time.time()
        rmse = score_candidate(model, Xd[idx_in], yd[idx_in],
                               swap[idx_in] if swap is not None else None,
                               Xd[idx_va], yd[idx_va], params,
                               CONFIG["GRID_SEARCH_DATA_SEED"], tied)
        elapsed = time.time() - t0
        row = {"rank": -1, "val_RMSE": rmse, "elapsed_sec": elapsed,
               "hp_id": hp_to_id(model, params)}
        row.update({f"param_{k}": v for k, v in params.items()})
        rows.append(row)
        logger.log(f"  [{i + 1:2d}/{len(grid)}] {hp_to_string(model, params):54s} "
                   f"val_RMSE={rmse:.4f}  ({elapsed:.1f}s)")

    df = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    best = next(p for p in grid if hp_to_id(model, p) == df.iloc[0]["hp_id"])
    logger.log(f"\n  Best: {hp_to_string(model, best)}  val_RMSE={df.iloc[0]['val_RMSE']:.4f}")
    df.to_csv(out_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "best_hp.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return best


def search_mlp(data, logger, out_dir, device):
    """Grid search for the MLP (validation fraction 0.1, weight seed fixed)."""
    from desml.models.mlp import predict_mlp, train_mlp
    from desml.features import TiedScaler, augment_swaps
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler

    X, y = data["X"], data["y"]
    des_idx = np.where(data["is_des"])[0]
    Xd, yd = X[des_idx], y[des_idx]
    swap = (data["swappable_all"][des_idx] if data["swappable_all"] is not None else None)
    tied = data["tied_idx_pairs"]

    idx_in, idx_va, n_tr42, n_te42 = grid_search_split(len(yd), CONFIG["GS_VAL_FRACTION_NN"])
    grid = build_grid("mlp")
    w_seed = CONFIG["GRID_SEARCH_WEIGHT_SEED"]
    logger.log("\n" + "=" * 70)
    logger.log(f"Grid search: MLP ({len(grid)} combinations, DES only, seed "
               f"{CONFIG['GRID_SEARCH_DATA_SEED']}, weight seed {w_seed}, no CV)")
    logger.log(f"  seed-42 split: train={n_tr42} (test={n_te42} sealed off)")
    logger.log(f"  inner fit={len(idx_in)}  inner validation={len(idx_va)}")
    logger.log("=" * 70)

    rows = []
    for i, hp in enumerate(grid):
        t0 = time.time()
        sx = TiedScaler(tied).fit(Xd[idx_in])
        sy = StandardScaler().fit(yd[idx_in].reshape(-1, 1))
        X_in = sx.transform(Xd[idx_in])
        X_va = sx.transform(Xd[idx_va])
        y_in = sy.transform(yd[idx_in].reshape(-1, 1)).ravel()
        y_va = sy.transform(yd[idx_va].reshape(-1, 1)).ravel()
        Xa, ya, wa = augment_swaps(X_in, y_in, swap[idx_in] if swap is not None else None, tied)
        model, best_epoch, _, _ = train_mlp(Xa, ya, X_va, y_va, hp=hp, weight_seed=w_seed,
                                            device=device, sample_weight=wa, tag="gs")
        yp = sy.inverse_transform(
            predict_mlp(model, X_va, device).reshape(-1, 1)).ravel()
        rmse = float(np.sqrt(mean_squared_error(yd[idx_va], yp)))
        elapsed = time.time() - t0
        rows.append({"rank": -1, "val_RMSE": rmse, "best_epoch": best_epoch,
                     "elapsed_sec": elapsed, "hp_id": hp_to_id("mlp", hp),
                     "dims": str(hp["dims"]), "dropouts": str(hp["dropouts"]),
                     "batch_size": hp["batch_size"]})
        logger.log(f"  [{i + 1:2d}/{len(grid)}] {hp_to_string('mlp', hp):46s} "
                   f"val_RMSE={rmse:.4f} (ep={best_epoch}, {elapsed:.1f}s)")

    df = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    best = next(h for h in grid if hp_to_id("mlp", h) == df.iloc[0]["hp_id"])
    logger.log(f"\n  Best: {hp_to_string('mlp', best)}  val_RMSE={df.iloc[0]['val_RMSE']:.4f}")
    df.to_csv(out_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "best_hp.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return best


def search_gnn(arch, data, ctx, logger, out_dir, device):
    """Two-stage search of the GNN head and encoder (stages Ia and Ib)."""
    from desml.models.gnn import score_candidate as gnn_score

    X, y = data["X"], data["y"]
    des_idx = np.where(data["is_des"])[0]
    Xd, yd = X[des_idx], y[des_idx]
    packs = tuple(a[des_idx] for a in ctx.packs)
    swap = (data["swappable_all"][des_idx] if data["swappable_all"] is not None else None)
    tied = data["tied_idx_pairs"]
    gb = ctx.graph_bank
    water = gb.water
    combine = ctx.graph_combine
    w_seed = CONFIG["GRID_SEARCH_WEIGHT_SEED"]

    idx_in, idx_va, n_tr42, n_te42 = grid_search_split(len(yd), CONFIG["GS_VAL_FRACTION_NN"])

    # ---- stage Ia: head, encoder fixed at its default ----
    enc0 = default_encoder(arch)
    head_grid = build_grid("mlp")
    logger.log("\n" + "=" * 70)
    logger.log(f"Stage Ia: head grid search ({len(head_grid)} combinations, "
               f"encoder fixed = {enc_hp_to_string(enc0, arch)})")
    logger.log(f"  inner fit={len(idx_in)}  inner validation={len(idx_va)}")
    logger.log("=" * 70)
    rows = []
    for i, h in enumerate(head_grid):
        t0 = time.time()
        rmse = gnn_score(gb, idx_in, idx_va, Xd, yd, packs, swap, arch, enc0, h,
                         device, tied, combine, water, w_seed)
        elapsed = time.time() - t0
        rows.append({"rank": -1, "val_RMSE": rmse, "elapsed_sec": elapsed,
                     "hp_id": hp_to_id("mlp", h), "dims": str(h["dims"]),
                     "dropouts": str(h["dropouts"]), "batch_size": h["batch_size"]})
        logger.log(f"  [{i + 1:2d}/{len(head_grid)}] {hp_to_string('mlp', h):46s} "
                   f"val_RMSE={rmse:.4f} ({elapsed:.1f}s)")
    head_df = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    head_df["rank"] = np.arange(1, len(head_df) + 1)
    head_hp = next(h for h in head_grid if hp_to_id("mlp", h) == head_df.iloc[0]["hp_id"])
    logger.log(f"\n  Best head: {hp_to_string('mlp', head_hp)}  "
               f"val_RMSE={head_df.iloc[0]['val_RMSE']:.4f}")
    head_df.to_csv(out_dir / "head_grid_search_results.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "best_head_hp.json", "w", encoding="utf-8") as f:
        json.dump(head_hp, f, indent=2)

    # ---- stage Ib: encoder, head fixed at the stage-Ia winner ----
    enc_grid = build_encoder_grid(arch)
    logger.log("\n" + "=" * 70)
    logger.log(f"Stage Ib: encoder grid search ({arch}, {len(enc_grid)} combinations, "
               f"head fixed = {hp_to_string('mlp', head_hp)})")
    logger.log("=" * 70)
    rows = []
    for i, e in enumerate(enc_grid):
        t0 = time.time()
        rmse = gnn_score(gb, idx_in, idx_va, Xd, yd, packs, swap, arch, e, head_hp,
                         device, tied, combine, water, w_seed)
        elapsed = time.time() - t0
        row = {"rank": -1, "val_RMSE": rmse, "elapsed_sec": elapsed,
               "enc_id": enc_hp_to_id(e, arch), "enc_json": json.dumps(e)}
        row.update(e)
        rows.append(row)
        logger.log(f"  [{i + 1:2d}/{len(enc_grid)}] {enc_hp_to_string(e, arch):46s} "
                   f"val_RMSE={rmse:.4f} ({elapsed:.1f}s)")
    enc_df = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    enc_df["rank"] = np.arange(1, len(enc_df) + 1)
    enc_hp = json.loads(enc_df.iloc[0]["enc_json"])
    logger.log(f"\n  Best encoder: {enc_hp_to_string(enc_hp, arch)}  "
               f"val_RMSE={enc_df.iloc[0]['val_RMSE']:.4f}")
    enc_df.to_csv(out_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "best_encoder_hp.json", "w", encoding="utf-8") as f:
        json.dump(enc_hp, f, indent=2)
    return head_hp, enc_hp


def main():
    args = parse_args()
    configure(DATA_DIR=args.data_dir, RESULTS_DIR=args.results_dir,
              MOLECULE_SOURCE=args.molecule_source,
              USE_GPU=not args.cpu)
    model = args.model.lower()
    sub = "graph_only" if model in GRAPH_MODELS else args.input
    out_dir = results_dir() / "hyperparameters" / model / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "execution.log")

    try:
        logger.log("=" * 70)
        logger.log(f"Step 2 - hyperparameter search: model={model} input={sub}")
        logger.log("=" * 70)
        t0 = time.time()
        if model in GRAPH_MODELS:
            from desml.models.mlp import get_device
            device = get_device()
            arch = "GIN" if model == "gin" else "GATv2"
            logger.log(f"  Device: {device}")
            data, ctx = prepare_graph_dataset(logger, arch, "concat", "none")
            search_gnn(arch, data, ctx, logger, out_dir, device)
        elif model == "mlp":
            from desml.models.mlp import get_device
            device = get_device()
            logger.log(f"  Device: {device}")
            data = prepare_descriptor_dataset(logger, args.input)
            search_mlp(data, logger, out_dir, device)
        else:
            data = prepare_descriptor_dataset(logger, args.input)
            search_descriptor_model(model, data, logger, out_dir)
        logger.log(f"\n  Results written to {out_dir}")
        logger.log(f"  Elapsed: {time.time() - t0:.1f} s")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

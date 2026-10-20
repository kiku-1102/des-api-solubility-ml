#!/usr/bin/env python3
"""
Step 4 - Study II: evaluation restricted to DES systems
(manuscript Section 2.8.2).

Only the DES data are split 8:2, so that the accuracy for the DES systems - the
target of this work - is assessed directly instead of being diluted by the
general solvents (which make up about four fifths of a pooled test set).  The
input representation is fixed to the concatenated one, which performed better
in Study I, and three training regimes are compared:

``des_only``     train on the DES training portion alone
``joint``        add the whole of the general solvent data to the training set
``pretrain_ft``  pre-train on the general solvents, then fine-tune all layers
                 on the DES training portion (neural networks only)

The principal metrics of this study are RMSE and Spearman's rho.

Usage
-----
    python scripts/04_study2_des.py --model xgb --regime des_only
    python scripts/04_study2_des.py --model gatv2 --use-descriptors --regime joint
    python scripts/04_study2_des.py --model mlp --regime pretrain_ft

Outputs (results/study2/<model>[_desc]/concat/<regime>/)
------------------------------------------------------
    metrics_summary.csv     mean and standard deviation over the five splits
    metrics_per_seed.csv    one row per random split
    metrics_per_run.csv     neural networks: one row per (split, weight seed)
    seed<NN>/scatter.png    measured vs predicted (panels of Fig. 5)
    seed<NN>/predictions_test.csv
    overall_scatter.png     all five splits pooled
    execution.log
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desml.config import CONFIG, configure, results_dir
from desml.logging_utils import Logger
from desml.pipeline import prepare_descriptor_dataset, prepare_graph_dataset
from desml.study import load_best_hp, run_study

REGIMES = ("des_only", "joint", "pretrain_ft")
NEURAL_ONLY_REGIMES = ("pretrain_ft",)
GRAPH_MODELS = ("gin", "gatv2")
ALL_MODELS = ("nusvr_rbf", "nusvr_poly", "rf", "xgb", "lgb", "mlp") + GRAPH_MODELS


def add_common_args(p):
    p.add_argument("--data-dir", default=CONFIG["DATA_DIR"])
    p.add_argument("--results-dir", default=CONFIG["RESULTS_DIR"])
    p.add_argument("--molecule-source", choices=["xyz", "smiles"],
                   default=CONFIG["MOLECULE_SOURCE"])
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    return p


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(ALL_MODELS))
    p.add_argument("--regime", choices=list(REGIMES), default="des_only",
                   help="training regime (Section 2.6.1)")
    p.add_argument("--use-descriptors", action="store_true",
                   help="graph models only: combine the graph embeddings with the "
                        "RDKit descriptors (GNN+MLP, Eqs. 6 and 7) instead of "
                        "using the graph alone (Eqs. 4 and 5)")
    add_common_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    configure(DATA_DIR=args.data_dir, RESULTS_DIR=args.results_dir,
              MOLECULE_SOURCE=args.molecule_source, USE_GPU=not args.cpu)
    model = args.model.lower()
    is_graph = model in GRAPH_MODELS
    regime = args.regime
    if regime in NEURAL_ONLY_REGIMES and model not in ("mlp", "gin", "gatv2"):
        raise SystemExit(f"regime {regime} is available for the neural networks only")
    if args.use_descriptors and not is_graph:
        raise SystemExit("--use-descriptors applies to the graph models only")

    tag = f"{model}_desc" if (is_graph and args.use_descriptors) else model
    out_dir = results_dir() / "study2" / tag / "concat" / regime
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "execution.log")

    try:
        logger.log("=" * 70)
        logger.log(f"Study II - DES-only evaluation (regime = {regime})")
        logger.log(f"  model={model}  input=concat"
                   + (f"  descriptors={args.use_descriptors}" if is_graph else ""))
        logger.log("=" * 70)
        t0 = time.time()
        with open(out_dir / "00_config.json", "w", encoding="utf-8") as f:
            json.dump({**CONFIG, "model": model, "input": "concat",
                       "regime": regime, "use_descriptors": bool(args.use_descriptors)},
                      f, indent=2, ensure_ascii=False)

        device = ctx = None
        if is_graph:
            from desml.models.mlp import get_device
            device = get_device()
            arch = "GIN" if model == "gin" else "GATv2"
            desc_combine = "concat" if args.use_descriptors else "none"
            data, ctx = prepare_graph_dataset(logger, arch, "concat", desc_combine)
            hp, enc_hp = load_best_hp(model, "concat", results_dir(), logger)
            ctx.encoder_hp = enc_hp
            ctx.water_data = ctx.graph_bank.water
            logger.log(f"  Device: {device}   Configuration: {ctx.describe()}")
        else:
            if model == "mlp":
                from desml.models.mlp import get_device
                device = get_device()
                logger.log(f"  Device: {device}")
            data = prepare_descriptor_dataset(logger, "concat")
            hp, enc_hp = load_best_hp(model, "concat", results_dir(), logger)

        run_study(model, "concat", regime, data, out_dir, logger,
                  device=device, ctx=ctx, hp=hp, encoder_hp=enc_hp)
        logger.log(f"  Results written to {out_dir}")
        logger.log(f"  Elapsed: {time.time() - t0:.1f} s")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

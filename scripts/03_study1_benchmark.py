#!/usr/bin/env python3
"""
Step 3 - Study I: benchmark comparison with the previous study
(manuscript Section 2.8.1).

All data - DES systems together with the general single-component and binary
solvent systems - are split 8:2 at random, which is the evaluation scheme of
the previous study (the ``prior`` regime).  Twenty configurations are compared:
six descriptor-based models, each with the concatenated and the mixture input
vector, and the two graph architectures, each as a graph-only model and as
GNN+MLP with both input representations.

The metrics reported in the manuscript for this study are MAPE and the outlier
fraction %Nout, so that they can be compared with the previous study; all other
metrics are written as well.

Usage
-----
    python scripts/03_study1_benchmark.py --model xgb --input concat
    python scripts/03_study1_benchmark.py --model gatv2 --input concat --use-descriptors
    python scripts/03_study1_benchmark.py --model gatv2 --input mixture   # graph only

The 20 configurations of Table 5 are:

    nusvr_rbf, nusvr_poly, rf, xgb, lgb, mlp   x  --input concat | mixture
    gin, gatv2   x  --input concat | mixture   (graph only)
    gin, gatv2   x  --input concat | mixture   --use-descriptors  (GNN+MLP)

Outputs (results/study1/<model>[_desc]/<input>/prior/)
------------------------------------------------------
    metrics_summary.csv     mean and standard deviation over the five splits
    metrics_per_seed.csv    one row per random split
    metrics_per_run.csv     neural networks: one row per (split, weight seed)
    seed<NN>/scatter.png    measured vs predicted (panels of Fig. 3)
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

REGIME = "prior"
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
    p.add_argument("--input", choices=["concat", "mixture"], default="concat",
                   help="input representation (Eq. 1 or Eqs. 2-3)")
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
    if args.use_descriptors and not is_graph:
        raise SystemExit("--use-descriptors applies to the graph models only")

    tag = f"{model}_desc" if (is_graph and args.use_descriptors) else model
    out_dir = results_dir() / "study1" / tag / args.input / REGIME
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "execution.log")

    try:
        logger.log("=" * 70)
        logger.log(f"Study I - benchmark comparison (regime = {REGIME})")
        logger.log(f"  model={model}  input={args.input}"
                   + (f"  descriptors={args.use_descriptors}" if is_graph else ""))
        logger.log("=" * 70)
        t0 = time.time()
        with open(out_dir / "00_config.json", "w", encoding="utf-8") as f:
            json.dump({**CONFIG, "model": model, "input": args.input,
                       "regime": REGIME, "use_descriptors": bool(args.use_descriptors)},
                      f, indent=2, ensure_ascii=False)

        device = ctx = None
        if is_graph:
            from desml.models.mlp import get_device
            device = get_device()
            arch = "GIN" if model == "gin" else "GATv2"
            desc_combine = args.input if args.use_descriptors else "none"
            data, ctx = prepare_graph_dataset(logger, arch, args.input, desc_combine)
            hp, enc_hp = load_best_hp(model, args.input, results_dir(), logger)
            ctx.encoder_hp = enc_hp
            ctx.water_data = ctx.graph_bank.water
            logger.log(f"  Device: {device}   Configuration: {ctx.describe()}")
        else:
            if model == "mlp":
                from desml.models.mlp import get_device
                device = get_device()
                logger.log(f"  Device: {device}")
            data = prepare_descriptor_dataset(logger, args.input)
            hp, enc_hp = load_best_hp(model, args.input, results_dir(), logger)

        run_study(model, args.input, REGIME, data, out_dir, logger,
                  device=device, ctx=ctx, hp=hp, encoder_hp=enc_hp)
        logger.log(f"  Results written to {out_dir}")
        logger.log(f"  Elapsed: {time.time() - t0:.1f} s")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

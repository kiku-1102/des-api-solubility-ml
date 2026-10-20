#!/usr/bin/env python3
"""
Step 1 - molecular descriptors and feature selection (manuscript Section 2.3).

Computes the 202 RDKit descriptors for every molecule in the data set and
filters them independently in three pools (API / solvent / condition) with a
variance filter followed by a correlation filter.  The surviving feature names
are written to ``feature_manifest.json``; every later step reads that file and
never re-derives the feature list.

Usage
-----
    python scripts/01_feature_selection.py
    python scripts/01_feature_selection.py --data-dir data --results-dir results

Outputs (results/feature_selection/)
------------------------------------
    feature_manifest.json                 surviving features (read by steps 2-5)
    00_config.json                        settings of this run
    01_*_variance_distribution.png        variance histogram per pool
    02_*_removed_by_variance.csv          features dropped by the variance filter
    03_*_removed_by_correlation.csv       features dropped by the correlation filter
    03_*_correlation_pairs.csv            which pair caused each deletion
    04_api_descriptors.csv                surviving API descriptors
    04_solvent_descriptors.csv            surviving solvent descriptors (both slots)
    04_condition_features.csv             surviving condition features
    04_final_features.csv                 full input-vector feature list
    04_final_features_categories.csv      the same list with pool / slot labels
    05_*_heatmap_before|after.png         correlation heat maps
    execution.log                         full log of the run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desml.config import CONFIG, configure, results_dir
from desml.data import load_all_data
from desml.descriptors import RDKIT_DESC_NAMES, check_descriptor_count, compute_all_descriptors
from desml.feature_selection import (BASE_CONDITION_KEYS, CONDITION_KEYS,
                                     DERIVED_CONDITION_KEYS,
                                     assemble_final_feature_names,
                                     build_condition_table,
                                     build_drg_descriptor_table,
                                     build_solvent_descriptor_table, filter_pool,
                                     plot_pool_heatmap, plot_variance_distribution_v2,
                                     save_list_csv, save_pairs_csv)
from desml.logging_utils import Logger
from desml.molecules import build_all_mols


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=CONFIG["DATA_DIR"],
                   help="directory with the CSV files and the structures (default: data)")
    p.add_argument("--results-dir", default=CONFIG["RESULTS_DIR"],
                   help="output directory (default: results)")
    p.add_argument("--molecule-source", choices=["xyz", "smiles"],
                   default=CONFIG["MOLECULE_SOURCE"],
                   help="build the molecules from the xyz files (published route) "
                        "or from the SMILES column of molecules.csv")
    p.add_argument("--variance-threshold", type=float, default=CONFIG["VARIANCE_THRESHOLD"])
    p.add_argument("--correlation-threshold", type=float, default=CONFIG["CORRELATION_THRESHOLD"])
    return p.parse_args()


def main():
    args = parse_args()
    configure(DATA_DIR=args.data_dir, RESULTS_DIR=args.results_dir,
              MOLECULE_SOURCE=args.molecule_source,
              VARIANCE_THRESHOLD=args.variance_threshold,
              CORRELATION_THRESHOLD=args.correlation_threshold)

    out_dir = results_dir() / "feature_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "execution.log")
    var_thr = CONFIG["VARIANCE_THRESHOLD"]
    corr_thr = CONFIG["CORRELATION_THRESHOLD"]

    try:
        logger.log("=" * 70)
        logger.log("Step 1 - RDKit descriptors and feature selection")
        logger.log("=" * 70)
        logger.log(f"  Data directory     : {Path(CONFIG['DATA_DIR']).resolve()}")
        logger.log(f"  Output             : {out_dir}")
        logger.log(f"  Molecule source    : {CONFIG['MOLECULE_SOURCE']}")
        logger.log(f"  Variance threshold : {var_thr:.2e}")
        logger.log(f"  Correlation thresh.: |r| > {corr_thr}")
        logger.log(f"  RDKit descriptors  : {len(RDKIT_DESC_NAMES)} (BCUT2D excluded)")
        logger.log(f"  Condition features : {len(CONDITION_KEYS)}")
        logger.log(f"    base    : {BASE_CONDITION_KEYS}")
        logger.log(f"    derived : {DERIVED_CONDITION_KEYS}")
        check_descriptor_count(logger)

        with open(out_dir / "00_config.json", "w", encoding="utf-8") as f:
            json.dump({**CONFIG, "condition_keys": CONDITION_KEYS,
                       "n_rdkit_descriptors": len(RDKIT_DESC_NAMES)},
                      f, indent=2, ensure_ascii=False)

        t0 = time.time()
        df_all = load_all_data(logger)
        mols = build_all_mols(df_all, logger)
        descriptors = compute_all_descriptors(mols, logger)

        logger.log("\n[4] Building the three descriptor pools...")
        drg_tbl = build_drg_descriptor_table(df_all, descriptors, logger)
        solvent_tbl = build_solvent_descriptor_table(df_all, descriptors, logger)
        cond_df, y_all, meta_all = build_condition_table(df_all, logger)

        logger.log("\n[5] Filtering each pool independently...")
        pools = {
            "api": filter_pool(drg_tbl, var_thr, corr_thr, logger, "API"),
            "solvent": filter_pool(solvent_tbl, var_thr, corr_thr, logger, "SOLVENT"),
            "condition": filter_pool(cond_df, var_thr, corr_thr, logger, "CONDITION"),
        }

        final = assemble_final_feature_names(
            drg_kept=pools["api"]["final_kept"],
            solvent_kept=pools["solvent"]["final_kept"],
            condition_kept=pools["condition"]["final_kept"])

        for name, res in pools.items():
            save_list_csv(res["removed_by_variance"],
                          out_dir / f"02_{name}_removed_by_variance.csv")
            save_list_csv(res["removed_by_correlation"],
                          out_dir / f"03_{name}_removed_by_correlation.csv")
            save_pairs_csv(res["correlation_pairs"],
                           out_dir / f"03_{name}_correlation_pairs.csv")
        save_list_csv(pools["api"]["final_kept"],
                      out_dir / "04_api_descriptors.csv", "descriptor")
        save_list_csv(pools["solvent"]["final_kept"],
                      out_dir / "04_solvent_descriptors.csv", "descriptor")
        save_list_csv(pools["condition"]["final_kept"],
                      out_dir / "04_condition_features.csv")
        save_list_csv(final["all_slot_form"], out_dir / "04_final_features.csv")

        cat_rows = ([{"feature": d, "category": "API_RDKit", "slot": "API"} for d in final["drg"]]
                    + [{"feature": d, "category": "Solvent_RDKit", "slot": "slot2"} for d in final["slot2"]]
                    + [{"feature": d, "category": "Solvent_RDKit", "slot": "slot3"} for d in final["slot3"]]
                    + [{"feature": d, "category": "Condition", "slot": "-"} for d in final["condition"]])
        pd.DataFrame(cat_rows).to_csv(out_dir / "04_final_features_categories.csv",
                                      index=False, encoding="utf-8-sig")

        with open(out_dir / CONFIG["FEATURE_MANIFEST"], "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)

        logger.log("\n[6] Writing figures...")
        plot_variance_distribution_v2(drg_tbl, var_thr,
                                      out_dir / "01_api_variance_distribution.png",
                                      "API pool variance distribution")
        plot_variance_distribution_v2(solvent_tbl, var_thr,
                                      out_dir / "01_solvent_variance_distribution.png",
                                      "Solvent pool variance distribution")
        plot_pool_heatmap(pools["api"]["table_after_variance"],
                          out_dir / "05_api_heatmap_before.png",
                          "API correlation (after variance filter)")
        plot_pool_heatmap(pools["api"]["table_final"],
                          out_dir / "05_api_heatmap_after.png",
                          "API correlation (final)")
        plot_pool_heatmap(pools["solvent"]["table_after_variance"],
                          out_dir / "05_solvent_heatmap_before.png",
                          "Solvent correlation (after variance filter)")
        plot_pool_heatmap(pools["solvent"]["table_final"],
                          out_dir / "05_solvent_heatmap_after.png",
                          "Solvent correlation (final)")

        logger.log("\n" + "=" * 70)
        logger.log("Summary (initial -> after variance -> final)")
        logger.log("=" * 70)
        summary_rows = []
        for name, tbl in [("api", drg_tbl), ("solvent", solvent_tbl), ("condition", cond_df)]:
            res = pools[name]
            n0 = len(tbl.columns)
            n1 = len(res["kept_after_variance"])
            n2 = len(res["final_kept"])
            logger.log(f"  {name:>9}: {n0:>4} -> {n1:>4} -> {n2:>4} "
                       f"(variance -{n0 - n1}, correlation -{n1 - n2})")
            summary_rows.append({"pool": name, "initial": n0, "after_variance": n1,
                                 "final": n2, "removed_by_variance": n0 - n1,
                                 "removed_by_correlation": n1 - n2})
        pd.DataFrame(summary_rows).to_csv(out_dir / "06_summary.csv", index=False,
                                          encoding="utf-8-sig")
        n_input = (len(final["drg"]) + 2 * len(final["solvent_descriptors"])
                   + len(final["condition"]))
        logger.log(f"\n  Concatenated input dimension: {n_input} "
                   f"(API {len(final['drg'])} + 2 x solvent {len(final['solvent_descriptors'])} "
                   f"+ conditions {len(final['condition'])})")
        logger.log(f"  Manifest written to {out_dir / CONFIG['FEATURE_MANIFEST']}")
        logger.log(f"  Elapsed: {time.time() - t0:.1f} s")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

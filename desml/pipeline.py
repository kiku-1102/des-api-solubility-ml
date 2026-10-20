"""
Glue between the data files and the models.

``prepare_descriptor_dataset`` performs the steps that every run of Studies
I-III has in common - load the three CSV files, build the molecules, compute
the 202 RDKit descriptors, read the feature manifest and assemble the design
matrix in the requested input representation - and returns everything the
models need.  ``prepare_graph_dataset`` does the same for the graph models and
additionally builds the graph bank and the per-row molecule codes.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .config import CONFIG, results_dir
from .data import load_all_data
from .descriptors import compute_all_descriptors
from .features import (build_feature_table_concat, build_feature_table_mixture,
                       load_feature_manifest)
from .logging_utils import Logger
from .molecules import build_all_mols

INPUT_REPRESENTATIONS = ("concat", "mixture")


def manifest_path() -> Path:
    """Location of the manifest written by scripts/01_feature_selection.py."""
    return results_dir() / "feature_selection" / CONFIG["FEATURE_MANIFEST"]


def load_descriptors(logger: Logger):
    """Data table plus the descriptor dictionary of every molecule."""
    df_all = load_all_data(logger)
    mols = build_all_mols(df_all, logger)
    descriptors = compute_all_descriptors(mols, logger)
    return df_all, mols, descriptors


def prepare_descriptor_dataset(logger: Logger, representation: str) -> Dict:
    """Design matrix for the descriptor-based models.

    Returns a dictionary with the design matrix ``X``, the target ``y``, the
    meta table, the feature names, the tied column pairs (empty for the
    mixture representation, which is order-invariant by construction) and the
    ``swappable`` flag of every row.
    """
    if representation not in INPUT_REPRESENTATIONS:
        raise ValueError(f"unknown input representation: {representation}")
    df_all, mols, descriptors = load_descriptors(logger)
    manifest = load_feature_manifest(manifest_path(), logger)

    tied_idx_pairs = None
    if representation == "concat":
        X_df, y, meta_df, feat_groups = build_feature_table_concat(
            df_all, descriptors, manifest, logger)
        col_idx = {c: i for i, c in enumerate(feat_groups["all_cols"])}
        tied_idx_pairs = [(col_idx[a], col_idx[b]) for a, b in feat_groups["tied_pairs"]]
    else:
        X_df, y, meta_df = build_feature_table_mixture(
            df_all, descriptors, manifest, logger)
        feat_groups = None

    swappable_all = (meta_df["swappable"].values.astype(bool)
                     if "swappable" in meta_df.columns else None)
    is_des = (meta_df["system_type"].astype(str).str.upper() == "DES").values
    return {
        "df_all": df_all, "mols": mols, "descriptors": descriptors,
        "manifest": manifest, "X_df": X_df, "X": X_df.values.astype(np.float64),
        "y": y, "meta_df": meta_df, "feat_groups": feat_groups,
        "feature_names": list(X_df.columns), "tied_idx_pairs": tied_idx_pairs,
        "swappable_all": swappable_all, "is_des": is_des,
    }


def prepare_graph_dataset(logger: Logger, arch: str, graph_combine: str,
                          desc_combine: str) -> Tuple[Dict, object]:
    """Design matrix, graph bank and context for a GNN configuration."""
    from .graphs import GraphBank
    from .models.gnn import GnnContext, build_gnn_data

    df_all, mols, descriptors = load_descriptors(logger)
    manifest = load_feature_manifest(manifest_path(), logger)
    graph_bank = GraphBank(mols, logger)
    (X, y, meta_df, DRG, C2, C3, W2, W3, WW,
     swappable_all, tied_idx_pairs) = build_gnn_data(
        df_all, descriptors, manifest, graph_combine, desc_combine, logger)
    is_des = (meta_df["system_type"].astype(str).str.upper() == "DES").values

    ctx = GnnContext(arch=arch, graph_combine=graph_combine, desc_combine=desc_combine,
                     graph_bank=graph_bank, packs=(DRG, C2, C3, W2, W3, WW))
    data = {
        "df_all": df_all, "mols": mols, "descriptors": descriptors,
        "manifest": manifest, "X": X, "y": y, "meta_df": meta_df,
        "tied_idx_pairs": tied_idx_pairs, "swappable_all": swappable_all,
        "is_des": is_des,
    }
    return data, ctx

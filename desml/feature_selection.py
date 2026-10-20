"""
Feature selection in three independent pools (manuscript Section 2.3.2).

The descriptors are filtered separately in

1. the **API pool**      - descriptors of the drug molecules, one row per molecule,
2. the **solvent pool**  - HBAs, HBDs and the non-aqueous components of the
   general solvents pooled into one set (the same set is applied to slot2 and
   slot3, so the two slots keep identical dimension and meaning),
3. the **condition pool** - the ten candidate condition features, computed for
   every data row.

Separating the pools prevents features with physically different meanings - a
drug descriptor and a solvent descriptor, say - from eliminating each other.
Two filters are applied in sequence to each pool: a variance filter
(threshold 1e-12) and a correlation filter (|r| > 0.9, one member of the most
correlated pair removed at a time with the mean correlations recomputed after
every deletion).  With RDKit 2023.09.6 this leaves 76 API descriptors, 64
solvent descriptors and 6 condition features (Section 3.1).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CONFIG, NONE_CODE
from .data import is_water_code
from .descriptors import RDKIT_DESC_NAMES
from .logging_utils import Logger

T_REFERENCE = CONFIG["T_REFERENCE"]

#: the ten candidate condition features (four base + six derived)
BASE_CONDITION_KEYS = ["T_K", "x_HBA", "x_HBD", "x_WATER"]
DERIVED_CONDITION_KEYS = ["inv_T", "log_T", "inv_T_diff", "log_water",
                          "has_water", "n_components"]
CONDITION_KEYS = BASE_CONDITION_KEYS + DERIVED_CONDITION_KEYS


def is_none_code(code: str) -> bool:
    """True for the empty-slot placeholder."""
    return str(code).strip().upper() == NONE_CODE


def is_real_solvent(code: str) -> bool:
    """True for a real solvent molecule (the placeholder and water excluded)."""
    if is_none_code(code):
        return False
    if is_water_code(code):
        return False
    return True


def compute_condition_features_v2(row: pd.Series) -> Dict[str, float]:
    """The ten candidate condition features of one data point (Section 2.3.1)."""
    T_K = float(row["T_K"])
    x_HBA = float(row["x_HBA"])
    x_HBD = float(row["x_HBD"])
    x_WATER = float(row["x_WATER"])

    feat = {
        "T_K": T_K,
        "x_HBA": x_HBA,
        "x_HBD": x_HBD,
        "x_WATER": x_WATER,
    }

    # temperature terms
    feat["inv_T"] = 1.0 / T_K if T_K > 0 else 0.0
    feat["log_T"] = float(np.log(T_K)) if T_K > 0 else 0.0
    feat["inv_T_diff"] = feat["inv_T"] - 1.0 / T_REFERENCE

    # water terms
    feat["log_water"] = float(np.log(x_WATER + CONFIG["LOG_WATER_EPS"]))
    feat["has_water"] = 1.0 if x_WATER > 1e-10 else 0.0

    # number of solvent components
    n_comp = sum(1 for x in [x_HBA, x_HBD, x_WATER] if x > 1e-10)
    feat["n_components"] = float(n_comp)

    return feat


def build_drg_descriptor_table(
    df: pd.DataFrame,
    descriptors: Dict[str, Dict[str, float]],
    logger: Logger,
) -> pd.DataFrame:
    """API pool: one row per unique drug molecule.

    Using one row per molecule removes any influence of composition or
    frequency of occurrence, so the correlation filter sees purely
    descriptor-to-descriptor relationships (Section 2.3.2).
    """
    drg_codes = sorted({
        str(c).strip() for c in df["DRG"].astype(str).str.strip().unique()
    })
    # drug codes are always real molecules; the guard is defensive
    drg_codes = [c for c in drg_codes if c in descriptors and not is_none_code(c)]

    rows = []
    index = []
    for code in drg_codes:
        d = descriptors[code]
        rows.append({k: float(d.get(k, 0.0)) for k in RDKIT_DESC_NAMES})
        index.append(code)

    tbl = pd.DataFrame(rows, index=index)
    logger.log(f"    DRG pool: {len(tbl)} unique molecules "
               f"x {len(tbl.columns)} descriptors")
    return tbl


def build_solvent_descriptor_table(
    df: pd.DataFrame,
    descriptors: Dict[str, Dict[str, float]],
    logger: Logger,
) -> pd.DataFrame:
    """Solvent pool: one row per unique solvent molecule.

    HBAs, HBDs and the non-aqueous components of the general solvents are
    pooled into a single set, because the same descriptor set is applied to
    slot2 and slot3 (Section 2.3.2).  Water is excluded: it has no molecular
    slot and enters through the condition features only.
    """
    solvent_codes = set()
    for col in ["HBA", "HBD"]:
        for c in df[col].astype(str).str.strip().unique():
            if is_real_solvent(c):
                solvent_codes.add(str(c).strip())

    solvent_codes = sorted(c for c in solvent_codes if c in descriptors)

    rows = []
    index = []
    for code in solvent_codes:
        d = descriptors[code]
        rows.append({k: float(d.get(k, 0.0)) for k in RDKIT_DESC_NAMES})
        index.append(code)

    tbl = pd.DataFrame(rows, index=index)
    logger.log(f"    Solvent pool: {len(tbl)} unique molecules "
               f"x {len(tbl.columns)} descriptors")
    logger.log(f"      Solvents: {index}")
    return tbl


def build_condition_table(
    df: pd.DataFrame,
    logger: Logger,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Condition pool: computed for every data row (not uniquified)."""
    cond_rows = []
    labels = []
    meta_rows = []
    skip = 0

    for idx, row in df.iterrows():
        drg = str(row["DRG"]).strip()
        hba = str(row["HBA"]).strip()
        hbd = str(row["HBD"]).strip()

        cond = compute_condition_features_v2(row)
        cond_rows.append({k: float(cond[k]) for k in CONDITION_KEYS})
        labels.append(float(row["logS"]))
        meta_rows.append({
            "row_index": int(idx),
            "DRG": drg, "HBA": hba, "HBD": hbd,
            "T_C": float(row["T_C"]), "T_K": float(row["T_K"]),
            "x_HBA": float(row["x_HBA"]), "x_HBD": float(row["x_HBD"]),
            "x_WATER": float(row["x_WATER"]),
            "logS": float(row["logS"]),
            "system_type": str(row["system_type"]),
        })

    cond_df = pd.DataFrame(cond_rows)
    y = np.asarray(labels, dtype=np.float64)
    meta_df = pd.DataFrame(meta_rows)

    finite_mask = np.isfinite(cond_df.values).all(axis=1) & np.isfinite(y)
    n_dropped = int((~finite_mask).sum())
    cond_df = cond_df.loc[finite_mask].reset_index(drop=True)
    y = y[finite_mask]
    meta_df = meta_df.loc[finite_mask].reset_index(drop=True)

    if n_dropped > 0:
        logger.log(f"    [WARN] {n_dropped} rows dropped (NaN/Inf in conditions)")
    logger.log(f"    Condition table: {len(cond_df)} rows "
               f"x {len(cond_df.columns)} features")

    return cond_df, y, meta_df


def variance_filter(
    tbl: pd.DataFrame, threshold: float, logger: Logger, label: str
) -> Tuple[List[str], List[str]]:
    """Drop near-constant columns; returns (kept, removed)."""
    variances = tbl.var(axis=0)
    mask = variances > threshold
    kept = tbl.columns[mask].tolist()
    removed = tbl.columns[~mask].tolist()
    logger.log(f"    [{label}] variance: {len(tbl.columns)} -> {len(kept)} "
               f"(removed {len(removed)})")
    return kept, removed


def correlation_filter_iterative(
    tbl: pd.DataFrame, threshold: float, logger: Logger, label: str
) -> Tuple[List[str], List[str], List[Dict]]:
    """Correlation filter with recomputed mean correlations (Section 2.3.2).

    Repeatedly:
      1. compute the correlation matrix and the mean correlation of every
         column, using the surviving columns only;
      2. pick the single pair with the highest |r| above the threshold
         (ties broken lexicographically by column name, for reproducibility);
      3. drop whichever member of the pair has the larger mean correlation,
         i.e. the one that is more redundant with the remaining features;
      4. stop when no pair exceeds the threshold.

    Note that a pair is removed when |r| is strictly greater than the
    threshold, so a pair at exactly |r| = 0.9 is kept.
    """
    if tbl.shape[1] <= 1:
        return tbl.columns.tolist(), [], []

    remaining = list(tbl.columns)
    to_drop_order: List[str] = []
    pairs_removed: List[Dict] = []

    while len(remaining) > 1:
        sub = tbl[remaining]
        corr = sub.corr().abs().fillna(0.0)
        mean_corr = corr.mean()  # recomputed over the surviving columns
        corr_vals = corr.values
        cols = list(corr.columns)
        m = len(cols)

        # find the most correlated pair among the surviving columns
        best_r = -1.0
        best_a = best_b = None
        for i in range(m):
            for j in range(i + 1, m):
                r = corr_vals[i, j]
                if r > best_r:
                    best_r, best_a, best_b = r, cols[i], cols[j]
                elif r == best_r and best_a is not None:
                    # ties: lexicographic order of the column-name pair
                    if (cols[i], cols[j]) < (best_a, best_b):
                        best_a, best_b = cols[i], cols[j]

        # stop when no pair exceeds the threshold
        if best_a is None or best_r <= threshold:
            break

        # drop the more redundant member (ties: name order)
        ma, mb = mean_corr[best_a], mean_corr[best_b]
        if ma > mb or (ma == mb and best_a <= best_b):
            drop, keep = best_a, best_b
        else:
            drop, keep = best_b, best_a

        remaining.remove(drop)
        to_drop_order.append(drop)
        pairs_removed.append({
            "step": len(to_drop_order),
            "kept": keep,
            "dropped": drop,
            "r": float(best_r),
            "mean_corr_kept_at_step": float(mean_corr[keep]),
            "mean_corr_dropped_at_step": float(mean_corr[drop]),
        })

    logger.log(f"    [{label}] correlation (iterative): "
               f"{len(tbl.columns)} -> {len(remaining)} "
               f"(removed {len(to_drop_order)})")
    return remaining, to_drop_order, pairs_removed


def filter_pool(
    tbl: pd.DataFrame, var_thr: float, corr_thr: float,
    logger: Logger, label: str,
) -> Dict[str, object]:
    """Apply the variance filter and then the correlation filter to one pool."""
    logger.log(f"\n  --- Pool: {label} ---")
    kept_var, removed_var = variance_filter(tbl, var_thr, logger, label)
    tbl_var = tbl[kept_var]
    kept_corr, removed_corr, pairs = correlation_filter_iterative(
        tbl_var, corr_thr, logger, label
    )
    return {
        "kept_after_variance": kept_var,
        "removed_by_variance": removed_var,
        "final_kept": kept_corr,
        "removed_by_correlation": removed_corr,
        "correlation_pairs": pairs,
        "table_after_variance": tbl_var,
        "table_final": tbl_var[kept_corr],
    }


def plot_pool_heatmap(
    tbl: pd.DataFrame, output_path: Path, title: str, max_features: int = 100
):
    """Correlation heat map of one pool."""
    if tbl.shape[1] == 0:
        return
    n = min(len(tbl.columns), max_features)
    if len(tbl.columns) > max_features:
        variances = tbl.var(axis=0).sort_values(ascending=False)
        sub = tbl[variances.head(n).index]
        title += f" (top {n} by variance)"
    else:
        sub = tbl
    corr = sub.corr()
    fig_size = max(8, min(20, len(sub.columns) * 0.15))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=120)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    if len(sub.columns) <= 50:
        ax.set_xticks(range(len(sub.columns)))
        ax.set_yticks(range(len(sub.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(corr.index, fontsize=6)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_variance_distribution_v2(
    tbl: pd.DataFrame, threshold: float, output_path: Path, title: str
):
    if tbl.shape[1] == 0:
        return
    variances = tbl.var(axis=0)
    log_var = np.log10(variances.replace(0, 1e-30) + 1e-30)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.hist(log_var, bins=40, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(np.log10(threshold), color="red", linestyle="--",
               label=f"Threshold = {threshold:.0e}")
    ax.set_xlabel("log10(Variance)")
    ax.set_ylabel("Number of features")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_list_csv(names: List[str], output_path: Path, colname: str = "feature"):
    pd.DataFrame({colname: names}).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )


def save_pairs_csv(pairs: List[Dict], output_path: Path):
    if pairs:
        pd.DataFrame(pairs).to_csv(output_path, index=False, encoding="utf-8-sig")


def assemble_final_feature_names(drg_kept: List[str], solvent_kept: List[str],
                                 condition_kept: List[str]) -> Dict[str, List[str]]:
    """Model-side feature names built from the surviving descriptors.

    ``DRG_RDKit_<desc>`` for the API pool and ``slot2_RDKit_<desc>`` /
    ``slot3_RDKit_<desc>`` for the solvent pool (the same descriptor set in both
    slots); the condition features keep their own names.
    """
    drg_feats = [f"DRG_RDKit_{d}" for d in drg_kept]
    slot2_feats = [f"slot2_RDKit_{d}" for d in solvent_kept]
    slot3_feats = [f"slot3_RDKit_{d}" for d in solvent_kept]
    return {
        "drg": drg_feats,
        "slot2": slot2_feats,
        "slot3": slot3_feats,
        "condition": list(condition_kept),
        "drg_descriptors": list(drg_kept),            # bare descriptor names
        "solvent_descriptors": list(solvent_kept),    # bare names, shared by both slots
        "all_slot_form": drg_feats + slot2_feats + slot3_feats + list(condition_kept),
    }

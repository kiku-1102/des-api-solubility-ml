"""
Input-vector construction (manuscript Sections 2.3.2, 2.4 and 2.5.3).

This module turns the solubility table plus the descriptor dictionary into the
numerical design matrices used by every model:

* :func:`build_feature_table_concat`  - Eq. (1), the concatenated vector
* :func:`build_feature_table_mixture` - Eqs. (2) and (3), the mixture vector
* :func:`compute_eff_desc_df`         - the mixture solvent block aligned with
  the concatenated meta table (needed for the GNN+MLP mixture form, Eq. 7)
* :class:`TiedScaler`                 - tied standardisation of slot2/slot3
* :func:`augment_swaps`               - slot-exchange augmentation (Table 2)

The surviving feature names are read from the manifest written by
``scripts/01_feature_selection.py``; the module never re-derives them from
column-name parsing.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import CONFIG, DESIGN_CONDITION_KEYS, NONE_CODE, SLOT_RENAME
from .data import is_water_code
from .logging_utils import Logger


def load_feature_manifest(manifest_path: Path, logger: Logger) -> Dict[str, List[str]]:
    """Read the three surviving feature lists written by the selection step.

    The manifest holds ``drg_descriptors`` (API pool), ``solvent_descriptors``
    (solvent pool, applied to slot2 and slot3 alike) and ``condition``
    (condition pool).  Using the lists directly - instead of parsing column
    names - makes it impossible to mix up the API and solvent blocks.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"feature manifest not found: {manifest_path}\n"
            f"  run scripts/01_feature_selection.py first.")
    man = json.load(open(manifest_path, encoding="utf-8"))

    drg = list(man.get("drg_descriptors", []))
    solv = list(man.get("solvent_descriptors", []))
    cond = list(man.get("condition", []))
    if not drg or not solv or not cond:
        raise ValueError(
            f"manifest has empty lists: drg={len(drg)} solvent={len(solv)} "
            f"condition={len(cond)}  ({manifest_path})")

    unknown = [c for c in cond if c not in DESIGN_CONDITION_KEYS]
    if unknown:
        logger.log(f"  [WARN] unexpected condition keys: {unknown} (used as given)")

    logger.log(f"\n  Feature manifest: {manifest_path}")
    logger.log(f"    API descriptors     : {len(drg)}")
    logger.log(f"    Solvent descriptors : {len(solv)} (shared by slot2 and slot3)")
    logger.log(f"    Condition features  : {len(cond)}  {cond}")
    return {"drg_descriptors": drg, "solvent_descriptors": solv, "condition": cond}


def compute_condition_design(T_K: float, xa: float, xb: float, xw: float) -> Dict[str, float]:
    """All ten candidate condition features under the fixed role naming.

    In the concatenated representation ``xa``/``xb`` are the amounts of slot2 /
    slot3 and the column names are renamed afterwards (``SLOT_RENAME``); in the
    mixture representation they are x_HBA / x_HBD directly.
    """
    T_REF = CONFIG["T_REFERENCE"]
    inv_T = 1.0 / T_K if T_K > 0 else 0.0
    return {
        "T_K": float(T_K),
        "x_HBA": float(xa),
        "x_HBD": float(xb),
        "x_WATER": float(xw),
        "inv_T": inv_T,
        "log_T": float(np.log(T_K)) if T_K > 0 else 0.0,
        "inv_T_diff": inv_T - 1.0 / T_REF,
        "has_water": 1.0 if xw > 1e-10 else 0.0,
        "log_water": float(np.log(xw + CONFIG["LOG_WATER_EPS"])),
        "n_components": float(sum(1 for x in (xa, xb, xw) if x > 1e-10)),
    }


def assign_component_slots(row):
    """Map one row onto (code_slot2, code_slot3, x_slot2, x_slot3, x_water, swappable).

    Water never occupies a molecular slot; it is collected in ``x_water``.
      * DES      - slot2 = HBA, slot3 = HBD, roles fixed (swappable = False)
      * non-DES  - the non-aqueous components fill the slots in column order,
                   so a second, slot-exchanged view may be generated
                   (swappable = True, manuscript Table 2)
    """
    st = str(row["system_type"]).upper()
    hba, hbd = str(row["HBA"]).strip(), str(row["HBD"]).strip()
    x_hba, x_hbd, x_water = float(row["x_HBA"]), float(row["x_HBD"]), float(row["x_WATER"])

    if st == "DES":
        return hba, hbd, x_hba, x_hbd, x_water, False

    comps = []
    for code, x in [(hba, x_hba), (hbd, x_hbd)]:
        if code == NONE_CODE or is_water_code(code):
            continue
        comps.append((code, x))
    code2 = comps[0][0] if len(comps) >= 1 else NONE_CODE
    x2 = comps[0][1] if len(comps) >= 1 else 0.0
    code3 = comps[1][0] if len(comps) >= 2 else NONE_CODE
    x3 = comps[1][1] if len(comps) >= 2 else 0.0
    swappable = (len(comps) >= 1)
    return code2, code3, x2, x3, x_water, swappable


def build_feature_table_concat(df, descriptors, manifest, logger):
    """Concatenated input vector, manuscript Eq. (1).

        [API descriptors] || [slot2 descriptors] || [slot3 descriptors] || [conditions]

    slot2 and slot3 share one descriptor set (the solvent pool), so the two
    slots have identical dimension and meaning and may be exchanged.  Water is
    represented by the condition features only.

    Returns ``(X_df, y, meta_df, feat_groups)``; ``feat_groups["tied_pairs"]``
    lists the slot2/slot3 column pairs used both for the permutation
    augmentation and for the tied standardisation.
    """
    logger.log("\n[4] Building the concatenated input vector (Eq. 1)...")
    drg_descs = manifest["drg_descriptors"]
    solv_descs = manifest["solvent_descriptors"]
    cond_keys_sel = manifest["condition"]                 # fixed role naming

    drg_cols = [f"DRG_RDKit_{d}" for d in drg_descs]
    slot2_cols = [f"slot2_RDKit_{d}" for d in solv_descs]
    slot3_cols = [f"slot3_RDKit_{d}" for d in solv_descs]
    cond_cols = [SLOT_RENAME.get(k, k) for k in cond_keys_sel]  # renamed to slot names
    all_cols = drg_cols + slot2_cols + slot3_cols + cond_cols

    feat_rows, labels, meta_rows = [], [], []
    skip = 0
    for _, row in df.iterrows():
        drg = str(row["DRG"]).strip()
        if drg not in descriptors:
            skip += 1; continue
        code2, code3, x2, x3, xw, swappable = assign_component_slots(row)
        if code2 != NONE_CODE and code2 not in descriptors:
            skip += 1; continue
        if code3 != NONE_CODE and code3 not in descriptors:
            skip += 1; continue

        cond_design = compute_condition_design(float(row["T_K"]), x2, x3, xw)
        d2 = descriptors.get(code2, {}) if code2 != NONE_CODE else {}
        d3 = descriptors.get(code3, {}) if code3 != NONE_CODE else {}

        feat = {}
        for d in drg_descs:
            feat[f"DRG_RDKit_{d}"] = float(descriptors[drg].get(d, 0.0))
        for d in solv_descs:
            feat[f"slot2_RDKit_{d}"] = float(d2.get(d, 0.0))
            feat[f"slot3_RDKit_{d}"] = float(d3.get(d, 0.0))
        for k in cond_keys_sel:
            feat[SLOT_RENAME.get(k, k)] = float(cond_design[k])

        feat_rows.append(feat)
        labels.append(float(row["logS"]))
        meta_rows.append({
            "DRG": drg,
            "code_slot2": code2, "code_slot3": code3,        # used by the graph models
            "x_slot2": x2, "x_slot3": x3, "x_water": xw,
            "swappable": bool(swappable),
            "system_type": str(row["system_type"]),
            "T_C": float(row["T_C"]), "T_K": float(row["T_K"]),
            "logS": float(row["logS"]),
            "HBA": str(row["HBA"]).strip(), "HBD": str(row["HBD"]).strip(),
            "x_HBA": float(row["x_HBA"]), "x_HBD": float(row["x_HBD"]),
            "x_WATER": float(row["x_WATER"]),
        })

    X_df = pd.DataFrame(feat_rows)[all_cols]
    y = np.asarray(labels, dtype=np.float64)
    meta_df = pd.DataFrame(meta_rows)

    finite = np.isfinite(X_df.values).all(axis=1) & np.isfinite(y)
    n_drop = int((~finite).sum())
    X_df = X_df.loc[finite].reset_index(drop=True)
    y = y[finite]
    meta_df = meta_df.loc[finite].reset_index(drop=True)

    # column pairs for swapping / tied scaling: descriptor pairs (+ composition pair)
    tied_pairs = list(zip(slot2_cols, slot3_cols))
    if "x_HBA" in cond_keys_sel and "x_HBD" in cond_keys_sel:
        tied_pairs = tied_pairs + [("x_slot2", "x_slot3")]
    feat_groups = {
        "drg_cols": drg_cols,
        "slot2_cols": slot2_cols,
        "slot3_cols": slot3_cols,
        "cond_cols": cond_cols,
        "tied_pairs": tied_pairs,     # swapped together and standardised jointly
        "all_cols": all_cols,
    }

    if skip > 0:
        logger.log(f"  [WARN] {skip} rows skipped (missing descriptor)")
    if n_drop > 0:
        logger.log(f"  [WARN] {n_drop} rows dropped (NaN/Inf)")
    n_swap = int(meta_df["swappable"].sum()) if len(meta_df) else 0
    logger.log(f"  Samples: {len(meta_df)}, Features: {X_df.shape[1]} "
               f"(DRG {len(drg_cols)} + slot2 {len(slot2_cols)} + slot3 {len(slot3_cols)} "
               f"+ cond {len(cond_cols)})")
    logger.log(f"  Swappable rows (non-DES): {n_swap}/{len(meta_df)}  (DES roles are fixed)")
    logger.log(f"  Condition cols: {cond_cols}")
    return X_df, y, meta_df, feat_groups


def build_feature_table_mixture(df, descriptors, manifest, logger):
    logger.log("\n[4] Building the mixture input vector (Eqs. 2 and 3)...")
    drg_descs = manifest["drg_descriptors"]
    solv_descs = manifest["solvent_descriptors"]
    cond_keys_sel = manifest["condition"]   # fixed role naming, used as is

    water_code = None
    for col in ["HBA", "HBD"]:
        for c in df[col].astype(str).str.strip().unique():
            if is_water_code(c):
                water_code = c; break
        if water_code is not None:
            break
    if water_code is None:
        water_code = "water"

    feat_rows, labels, meta_rows = [], [], []
    skip = 0
    for _, row in df.iterrows():
        drg = str(row["DRG"]).strip()
        hba = str(row["HBA"]).strip()
        hbd = str(row["HBD"]).strip()
        if drg not in descriptors or hba not in descriptors:
            skip += 1; continue
        if hbd not in descriptors and hbd != NONE_CODE:
            skip += 1; continue
        x_HBA = float(row["x_HBA"]); x_HBD = float(row["x_HBD"]); x_WATER = float(row["x_WATER"])

        feat = {}
        for d in drg_descs:
            feat[f"DRG_RDKit_{d}"] = float(descriptors[drg].get(d, 0.0))
        hba_d = descriptors[hba]
        hbd_d = descriptors[hbd] if hbd != NONE_CODE else None
        water_d = descriptors.get(water_code) if x_WATER > 1e-10 else None
        for d in solv_descs:
            v_hba = float(hba_d.get(d, 0.0)) if x_HBA > 1e-10 else 0.0
            v_hbd = float(hbd_d.get(d, 0.0)) if (hbd_d is not None and x_HBD > 1e-10) else 0.0
            v_wat = float(water_d.get(d, 0.0)) if (water_d is not None and x_WATER > 1e-10) else 0.0
            feat[f"Eff_RDKit_{d}"] = x_HBA * v_hba + x_HBD * v_hbd + x_WATER * v_wat
        cond_design = compute_condition_design(float(row["T_K"]), x_HBA, x_HBD, x_WATER)
        for k in cond_keys_sel:
            feat[k] = float(cond_design[k])

        feat_rows.append(feat)
        labels.append(float(row["logS"]))
        meta_rows.append({
            "DRG": drg, "HBA": hba, "HBD": hbd,
            "T_C": float(row["T_C"]), "T_K": float(row["T_K"]),
            "x_HBA": x_HBA, "x_HBD": x_HBD, "x_WATER": x_WATER,
            "logS": float(row["logS"]), "system_type": str(row["system_type"]),
        })

    X_df = pd.DataFrame(feat_rows)
    y = np.asarray(labels, dtype=np.float64)
    meta_df = pd.DataFrame(meta_rows)
    finite = np.isfinite(X_df.values).all(axis=1) & np.isfinite(y)
    n_drop = int((~finite).sum())
    X_df = X_df.loc[finite].reset_index(drop=True)
    y = y[finite]
    meta_df = meta_df.loc[finite].reset_index(drop=True)
    if skip > 0: logger.log(f"  [WARN] {skip} rows skipped (missing descriptor)")
    if n_drop > 0: logger.log(f"  [WARN] {n_drop} rows dropped (NaN/Inf)")
    logger.log(f"  Samples: {len(meta_df)}, Features: {X_df.shape[1]} "
               f"(DRG {len(drg_descs)} + Eff {len(solv_descs)} + cond {len(cond_keys_sel)})")
    return X_df, y, meta_df


class TiedScaler:
    """Standardiser with tied statistics for the slot2/slot3 columns.

    Each pair in ``tied_idx_pairs`` is standardised with statistics pooled over
    both columns, so that exchanging slot2 and slot3 is still a symmetry after
    standardisation (manuscript Section 2.5.3).  With ``tied_idx_pairs=None``
    the behaviour is that of ``StandardScaler`` (ddof = 0).

    Always fit on the training part only - never on validation or test data.
    """
    def __init__(self, tied_idx_pairs=None):
        self.tied = tied_idx_pairs
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0).astype(np.float64)
        std = X.std(axis=0).astype(np.float64)        # ddof = 0, as in StandardScaler
        if self.tied is not None:
            for i, j in self.tied:
                pooled = np.concatenate([X[:, i], X[:, j]])
                m = float(pooled.mean()); s = float(pooled.std())
                mean[i] = mean[j] = m
                std[i] = std[j] = s
        std[std < 1e-12] = 1.0
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def augment_swaps(X, y, swappable, tied_idx_pairs):
    """Augment the swappable rows with their slot-exchanged view (Table 2).

    Returns ``(X_aug, y_aug, sample_weight)``.  The weight is 1/(number of
    views): 1.0 for a row with a fixed role (DES) and 0.5 for each of the two
    views of a row whose slot order is arbitrary, so that one physical
    measurement always contributes a total weight of one.

    Apply only to the training part, after the split, to avoid leakage.
    """
    n = X.shape[0]
    if tied_idx_pairs is None or swappable is None or not np.any(swappable):
        return X, y, np.ones(n, dtype=np.float64)

    sw = np.asarray(swappable, dtype=bool)
    X_ns, y_ns = X[~sw], y[~sw]
    w_ns = np.ones((~sw).sum(), dtype=np.float64)

    X_s, y_s = X[sw], y[sw]
    X_sw = X_s.copy()
    ii = [i for i, j in tied_idx_pairs]
    jj = [j for i, j in tied_idx_pairs]
    X_sw[:, ii], X_sw[:, jj] = X_s[:, jj], X_s[:, ii]

    X_aug = np.concatenate([X_ns, X_s, X_sw], axis=0)
    y_aug = np.concatenate([y_ns, y_s, y_s], axis=0)
    w_aug = np.concatenate([
        w_ns,
        0.5 * np.ones(len(y_s), dtype=np.float64),
        0.5 * np.ones(len(y_s), dtype=np.float64),
    ], axis=0)
    return X_aug, y_aug, w_aug


def compute_eff_desc_df(meta_df, descriptors, manifest):
    """Mole-fraction weighted solvent descriptors for the rows of ``meta_df``.

    Same expression as ``build_feature_table_mixture`` (Eq. 3), but aligned row
    by row with the meta table produced by the concatenated builder, so that
    the graph models can use the mixture representation on the descriptor side
    (manuscript Eq. 7).
    """
    solv_descs = manifest["solvent_descriptors"]
    water_code = None
    for code in descriptors.keys():
        if is_water_code(code):
            water_code = code; break
    rows = []
    for _, m in meta_df.iterrows():
        hba = str(m["HBA"]).strip(); hbd = str(m["HBD"]).strip()
        x_HBA = float(m["x_HBA"]); x_HBD = float(m["x_HBD"]); x_WATER = float(m["x_WATER"])
        hba_d = descriptors.get(hba, {})
        hbd_d = descriptors.get(hbd) if hbd != NONE_CODE else None
        water_d = descriptors.get(water_code) if (water_code is not None and x_WATER > 1e-10) else None
        r = {}
        for d in solv_descs:
            v_hba = float(hba_d.get(d, 0.0)) if x_HBA > 1e-10 else 0.0
            v_hbd = float(hbd_d.get(d, 0.0)) if (hbd_d is not None and x_HBD > 1e-10) else 0.0
            v_wat = float(water_d.get(d, 0.0)) if (water_d is not None and x_WATER > 1e-10) else 0.0
            r[f"Eff_RDKit_{d}"] = x_HBA * v_hba + x_HBD * v_hbd + x_WATER * v_wat
        rows.append(r)
    eff_cols = [f"Eff_RDKit_{d}" for d in solv_descs]
    return pd.DataFrame(rows, columns=eff_cols), eff_cols

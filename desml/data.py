"""
Loading of the solubility data set (manuscript Section 2.1).

Three CSV files are read and concatenated into one long table:

==================  ======  ==============================================
file                rows    content
==================  ======  ==============================================
solubility_des      1,589   API in DES (HBA + HBD, optionally with water)
solubility_pure     2,064   API in a single-component solvent
solubility_binary   4,332   API in a binary solvent mixture
==================  ======  ==============================================

All three are mapped onto the same column layout so that the downstream code
treats a DES and a general solvent identically::

    DRG  HBA  HBD  T_C  T_K  logS  x_HBA  x_HBD  x_WATER  system_type

For a single-component or binary system the non-aqueous components are placed
in the HBA / HBD columns; water never occupies a molecular slot and is carried
by ``x_WATER`` alone (Section 2.3.2).  ``logS`` is the common logarithm of the
mole-fraction solubility, log x.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import CONFIG, NONE_CODE
from .logging_utils import Logger


def is_water_code(code: str) -> bool:
    """True if a component code denotes water."""
    return str(code).strip().lower() == "water"


def load_des_csv(csv_path: Path, logger: Logger) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"DES CSV not found: {csv_path}")
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    col_map = {}
    for c in raw.columns:
        cn = c.strip().lower()
        if cn == "drg": col_map["DRG"] = c
        elif cn == "hba": col_map["HBA"] = c
        elif cn == "hbd": col_map["HBD"] = c
        elif cn.startswith("t[") or cn == "t_c": col_map["T_C"] = c
        elif cn.startswith("log(x1)") or cn == "logs": col_map["logS"] = c
        elif cn in ("x2*", "x2", "x_hba"): col_map["x_HBA"] = c
        elif cn in ("x3*", "x3", "x_hbd"): col_map["x_HBD"] = c
        elif cn in ("x4*", "x4", "x_water"): col_map["x_WATER"] = c

    df = pd.DataFrame({
        "DRG": raw[col_map["DRG"]].astype(str).str.strip(),
        "HBA": raw[col_map["HBA"]].astype(str).str.strip(),
        "HBD": raw[col_map["HBD"]].astype(str).str.strip(),
        "T_C": pd.to_numeric(raw[col_map["T_C"]], errors="coerce"),
        "logS": pd.to_numeric(raw[col_map["logS"]], errors="coerce"),
        "x_HBA": pd.to_numeric(raw[col_map["x_HBA"]], errors="coerce").fillna(0.0),
        "x_HBD": pd.to_numeric(raw[col_map["x_HBD"]], errors="coerce").fillna(0.0),
        "x_WATER": pd.to_numeric(raw[col_map["x_WATER"]], errors="coerce").fillna(0.0),
    })
    df = df.dropna(subset=["DRG", "HBA", "HBD", "T_C", "logS"]).reset_index(drop=True)
    df["T_K"] = df["T_C"] + 273.15
    df["system_type"] = "DES"
    logger.log(f"  [OK] DES: {len(df)} rows")
    return df


def load_pure_csv(csv_path: Path, logger: Logger) -> pd.DataFrame:
    if not csv_path.exists():
        logger.log(f"  [WARN] Pure CSV not found")
        return pd.DataFrame()
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    col_map = {}
    for c in raw.columns:
        cn = c.strip().lower()
        if cn == "drg": col_map["DRG"] = c
        elif cn == "solvent": col_map["solvent"] = c
        elif cn.startswith("t[") or cn == "t_c": col_map["T_C"] = c
        elif cn.startswith("log(x1)") or cn == "logs": col_map["logS"] = c

    rows = []
    for _, row in raw.iterrows():
        sol = str(row[col_map["solvent"]]).strip()
        rows.append({
            "DRG": str(row[col_map["DRG"]]).strip(),
            "HBA": sol, "HBD": NONE_CODE,
            "T_C": pd.to_numeric(row[col_map["T_C"]], errors="coerce"),
            "logS": pd.to_numeric(row[col_map["logS"]], errors="coerce"),
            "x_HBA": 0.0 if is_water_code(sol) else 1.0,
            "x_HBD": 0.0,
            "x_WATER": 1.0 if is_water_code(sol) else 0.0,
            "system_type": "PURE",
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["DRG", "HBA", "HBD", "T_C", "logS"]).reset_index(drop=True)
    df["T_K"] = df["T_C"] + 273.15
    logger.log(f"  [OK] Pure: {len(df)} rows")
    return df


def load_binary_csv(csv_path: Path, logger: Logger) -> pd.DataFrame:
    if not csv_path.exists():
        logger.log(f"  [WARN] Binary CSV not found")
        return pd.DataFrame()
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    col_map = {}
    for c in raw.columns:
        cn = c.strip().lower()
        if cn == "drg": col_map["DRG"] = c
        elif cn == "solvent_1": col_map["solvent_1"] = c
        elif cn == "solvent_2": col_map["solvent_2"] = c
        elif cn.startswith("t[") or cn == "t_c": col_map["T_C"] = c
        elif cn.startswith("log(x1)") or cn == "logs": col_map["logS"] = c
        elif cn in ("x2*", "x2"): col_map["x2"] = c
        elif cn in ("x3*", "x3"): col_map["x3"] = c

    rows = []
    for _, row in raw.iterrows():
        s1, s2 = str(row[col_map["solvent_1"]]).strip(), str(row[col_map["solvent_2"]]).strip()
        x1 = float(pd.to_numeric(row[col_map["x2"]], errors="coerce"))
        x2 = float(pd.to_numeric(row[col_map["x3"]], errors="coerce"))
        w1, w2 = is_water_code(s1), is_water_code(s2)
        if w1 and w2:
            hba, hbd, xh, xd, xw = s1, NONE_CODE, 1.0, 0.0, 1.0
        elif w1:
            hba, hbd, xh, xd, xw = s2, s1, x2, x1, x1
        elif w2:
            hba, hbd, xh, xd, xw = s1, s2, x1, x2, x2
        else:
            pair = sorted([(s1, x1), (s2, x2)], key=lambda t: t[0].lower())
            (hba, xh), (hbd, xd) = pair[0], pair[1]
            xw = 0.0
        rows.append({
            "DRG": str(row[col_map["DRG"]]).strip(),
            "HBA": hba, "HBD": hbd,
            "T_C": pd.to_numeric(row[col_map["T_C"]], errors="coerce"),
            "logS": pd.to_numeric(row[col_map["logS"]], errors="coerce"),
            "x_HBA": xh, "x_HBD": xd, "x_WATER": xw,
            "system_type": "BINARY",
        })
    df = pd.DataFrame(rows)
    for c in ["x_HBA", "x_HBD", "x_WATER"]:
        df[c] = df[c].fillna(0.0)
    df = df.dropna(subset=["DRG", "HBA", "HBD", "T_C", "logS"]).reset_index(drop=True)
    df["T_K"] = df["T_C"] + 273.15
    logger.log(f"  [OK] Binary: {len(df)} rows")
    return df


def load_all_data(logger: Logger, data_dir: Path = None) -> pd.DataFrame:
    """Read the three CSV files and return one concatenated table."""
    base = Path(data_dir) if data_dir is not None else Path(CONFIG["DATA_DIR"]).resolve()
    logger.log("\n[1] Loading CSV data...")
    frames = []
    df_des = load_des_csv(base / CONFIG["DES_CSV"], logger)
    if len(df_des) > 0:
        frames.append(df_des)
    df_pure = load_pure_csv(base / CONFIG["PURE_CSV"], logger)
    if len(df_pure) > 0:
        frames.append(df_pure)
    df_bin = load_binary_csv(base / CONFIG["BINARY_CSV"], logger)
    if len(df_bin) > 0:
        frames.append(df_bin)
    if not frames:
        raise RuntimeError(f"no data loaded from {base}")
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["DRG", "T_C", "logS"]
    ).reset_index(drop=True)
    logger.log(f"\n  Total: {len(df_all)} rows")
    for st in ["DES", "PURE", "BINARY"]:
        n = int((df_all["system_type"] == st).sum())
        logger.log(f"    {st}: {n}")
    return df_all


def collect_required_codes(df: pd.DataFrame) -> List[str]:
    """Unique molecule codes that need a structure (water excluded)."""
    codes = set()
    for col in ["DRG", "HBA", "HBD"]:
        codes.update(df[col].astype(str).str.strip().tolist())
    codes.discard(NONE_CODE)
    codes = {c for c in codes if not is_water_code(c)}
    return sorted(codes)

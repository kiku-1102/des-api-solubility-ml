#!/usr/bin/env python3
"""
Step 7 (utility) - export the molecule table and check the two input routes.

The published descriptors were computed from the geometry files in
``data/structures`` (``Chem.MolFromXYZFile`` followed by
``rdDetermineBonds.DetermineBonds``).  This script

1. writes ``data/molecules.csv`` with the canonical SMILES of every molecule
   derived from those geometries (code, role, SMILES), and
2. compares, descriptor by descriptor, the values obtained from the geometry
   with the values obtained by parsing that SMILES.

Use it to confirm that the SMILES route (``--molecule-source smiles`` in steps
1-6) reproduces the published descriptors before relying on it, and as a
readable record of which molecule each code stands for.

Usage
-----
    python scripts/07_export_smiles.py
    python scripts/07_export_smiles.py --output data/molecules.csv --tolerance 1e-8

Outputs
-------
    data/molecules.csv                     code, role, SMILES
    results/molecule_check/comparison.csv  per-molecule maximum deviation
    results/molecule_check/mismatches.csv  descriptors that differ, if any
    results/molecule_check/execution.log
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desml.config import CONFIG, configure, data_dir, results_dir
from desml.data import collect_required_codes, is_water_code, load_all_data
from desml.descriptors import RDKIT_DESC_NAMES, compute_rdkit_descriptors
from desml.logging_utils import Logger
from desml.molecules import (CHOLINE_CHLORIDE_CODES, build_choline_chloride_mol,
                             build_mol_from_smiles, build_water_mol, find_xyz_file,
                             load_mol_from_xyz)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=CONFIG["DATA_DIR"])
    p.add_argument("--results-dir", default=CONFIG["RESULTS_DIR"])
    p.add_argument("--output", default=None,
                   help="where to write the molecule table (default: <data-dir>/molecules.csv)")
    p.add_argument("--tolerance", type=float, default=1e-8,
                   help="absolute tolerance of the descriptor comparison")
    return p.parse_args()


def role_of(code, df):
    roles = []
    if (df["DRG"].astype(str).str.strip() == code).any():
        roles.append("API")
    des = df[df["system_type"] == "DES"]
    if (des["HBA"].astype(str).str.strip() == code).any():
        roles.append("HBA")
    if (des["HBD"].astype(str).str.strip() == code).any():
        roles.append("HBD")
    other = df[df["system_type"] != "DES"]
    if ((other["HBA"].astype(str).str.strip() == code).any()
            or (other["HBD"].astype(str).str.strip() == code).any()):
        roles.append("solvent")
    return "/".join(roles) if roles else "unused"


def main():
    args = parse_args()
    configure(DATA_DIR=args.data_dir, RESULTS_DIR=args.results_dir)
    out_dir = results_dir() / "molecule_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "execution.log")

    try:
        logger.log("=" * 70)
        logger.log("Step 7 - export SMILES and compare the xyz and SMILES routes")
        logger.log("=" * 70)
        df_all = load_all_data(logger)
        codes = collect_required_codes(df_all)
        structure_dir = data_dir() / CONFIG["STRUCTURE_DIR"]

        rows, comparison, mismatches = [], [], []
        for code in codes:
            path = find_xyz_file(code, structure_dir)
            if path is None:
                logger.log(f"  [WARN] no geometry for {code}; skipped")
                continue
            if code.upper() in CHOLINE_CHLORIDE_CODES:
                mol_xyz = build_choline_chloride_mol(path)
            else:
                mol_xyz = load_mol_from_xyz(path, total_charge=0)
            smiles = Chem.MolToSmiles(mol_xyz)
            rows.append({"code": code, "role": role_of(code, df_all), "smiles": smiles})

            d_xyz = compute_rdkit_descriptors(mol_xyz)
            try:
                d_smi = compute_rdkit_descriptors(build_mol_from_smiles(smiles))
            except Exception as exc:
                logger.log(f"  [WARN] SMILES route failed for {code}: {exc}")
                continue
            diffs = {k: abs(d_xyz[k] - d_smi.get(k, np.nan)) for k in RDKIT_DESC_NAMES}
            worst = max(diffs, key=lambda k: (np.nan_to_num(diffs[k], nan=np.inf)))
            comparison.append({"code": code, "smiles": smiles,
                               "max_abs_diff": diffs[worst], "worst_descriptor": worst,
                               "n_differing": int(sum(1 for v in diffs.values()
                                                      if not (v <= args.tolerance)))})
            for k, v in diffs.items():
                if not (v <= args.tolerance):
                    mismatches.append({"code": code, "descriptor": k,
                                       "from_xyz": d_xyz[k], "from_smiles": d_smi.get(k),
                                       "abs_diff": v})

        # water has no geometry file and is built from SMILES in both routes
        if any(is_water_code(c) for c in
               pd.concat([df_all["HBA"], df_all["HBD"]]).astype(str).str.strip().unique()):
            rows.append({"code": "water", "role": "solvent/co-solvent", "smiles": "O"})

        out_path = Path(args.output) if args.output else data_dir() / CONFIG["MOLECULE_CSV"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("code").to_csv(out_path, index=False,
                                                      encoding="utf-8-sig")
        comp_df = pd.DataFrame(comparison)
        comp_df.to_csv(out_dir / "comparison.csv", index=False, encoding="utf-8-sig")
        if mismatches:
            pd.DataFrame(mismatches).to_csv(out_dir / "mismatches.csv", index=False,
                                            encoding="utf-8-sig")

        logger.log(f"\n  Molecule table written to {out_path} ({len(rows)} molecules)")
        if len(comp_df):
            n_bad = int((comp_df["n_differing"] > 0).sum())
            logger.log(f"  Descriptor comparison: {len(comp_df) - n_bad}/{len(comp_df)} "
                       f"molecules identical within {args.tolerance:g}")
            if n_bad:
                logger.log(f"  [WARN] {n_bad} molecules differ between the two routes - "
                           f"see mismatches.csv.  Keep --molecule-source xyz for the "
                           f"published numbers.")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

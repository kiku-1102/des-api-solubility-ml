#!/usr/bin/env python3
"""
Step 6 - cost of computing the features (manuscript Section 2.5.5).

Measures how long the RDKit descriptor calculation takes, so that it can be
compared approximately with the quantum-chemical (COSMO-RS sigma-potential)
features of the previous study.  The procedure that is timed is the standard
cheminformatics input route: take the SMILES string of a molecule, build the
molecule object and compute all 202 descriptors (before feature selection).

As in the manuscript the measurement is single-threaded, the CPU time is
recorded, a warm-up pass is run first to remove first-call overhead, and the
mean over five repetitions is reported.  The comparison in the paper is
restricted to the 15 drug molecules, matching the published COSMO-family
timings; the timing over all unique molecules and the timing of the xyz route
(read the geometry, perceive the bonds, compute the descriptors) are reported
as additional reference values.

Usage
-----
    python scripts/06_descriptor_timing.py
    python scripts/06_descriptor_timing.py --repeats 5 --no-xyz-route

Outputs (results/descriptor_timing/)
------------------------------------
    timing.csv        one row per measured route
    environment.json  CPU, thread settings and library versions
    execution.log
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

# Single-threaded measurement (manuscript Section 2.5.5).  These variables must
# be set before NumPy / RDKit are imported, which is why the import block below
# comes after them; an environment that already pins them is overridden on
# purpose, so that the reported CPU time is always for one core.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desml.config import CONFIG, configure, results_dir
from desml.data import collect_required_codes, is_water_code, load_all_data
from desml.descriptors import RDKIT_DESC_NAMES, compute_rdkit_descriptors
from desml.logging_utils import Logger
from desml.molecules import (CHOLINE_CHLORIDE_CODES, build_choline_chloride_mol,
                             find_xyz_file, load_mol_from_xyz, load_molecule_table)

WATER_SMILES = "O"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeats", type=int, default=5, help="timed repetitions (default 5)")
    p.add_argument("--warmup", type=int, default=1, help="warm-up passes (default 1)")
    p.add_argument("--no-xyz-route", action="store_true",
                   help="skip the additional timing of the xyz route")
    p.add_argument("--data-dir", default=CONFIG["DATA_DIR"])
    p.add_argument("--results-dir", default=CONFIG["RESULTS_DIR"])
    return p.parse_args()


def mol_from_xyz(code, structure_dir):
    path = find_xyz_file(code, structure_dir)
    if path is None:
        raise FileNotFoundError(f"xyz not found for {code}")
    if code.upper() in CHOLINE_CHLORIDE_CODES:
        return build_choline_chloride_mol(path)
    return load_mol_from_xyz(path, total_charge=0)


def smiles_for_codes(codes, data_dir, logger):
    """SMILES of each code: from molecules.csv if present, else from the geometry.

    Generating the SMILES is preprocessing and is not part of the measurement.
    """
    smiles = {}
    try:
        table = load_molecule_table(data_dir)
        smiles.update({c: s for c, s in zip(table["code"], table["smiles"].astype(str))
                       if isinstance(s, str) and s.strip()})
        logger.log(f"  SMILES read from {CONFIG['MOLECULE_CSV']}: {len(smiles)}")
    except FileNotFoundError:
        logger.log(f"  [INFO] {CONFIG['MOLECULE_CSV']} not found; deriving SMILES from xyz")

    structure_dir = Path(data_dir) / CONFIG["STRUCTURE_DIR"]
    missing = [c for c in codes if c not in smiles]
    for code in missing:
        try:
            smiles[code] = Chem.MolToSmiles(mol_from_xyz(code, structure_dir))
        except Exception as exc:
            logger.log(f"  [WARN] no SMILES for {code}: {exc}")
    return {c: smiles[c] for c in codes if c in smiles}


def time_smiles_route(smiles_list, repeats, warmup):
    """SMILES -> Mol -> all descriptors."""
    for _ in range(warmup):
        for s in smiles_list:
            compute_rdkit_descriptors(Chem.MolFromSmiles(s))
    wall, cpu = [], []
    for _ in range(repeats):
        t0w, t0c = time.perf_counter(), time.process_time()
        for s in smiles_list:
            compute_rdkit_descriptors(Chem.MolFromSmiles(s))
        wall.append(time.perf_counter() - t0w)
        cpu.append(time.process_time() - t0c)
    return wall, cpu


def time_xyz_route(codes, structure_dir, repeats, warmup):
    """xyz -> Mol (bond perception) -> all descriptors."""
    for _ in range(warmup):
        for code in codes:
            try:
                compute_rdkit_descriptors(mol_from_xyz(code, structure_dir))
            except Exception:
                pass
    wall, cpu = [], []
    for _ in range(repeats):
        t0w, t0c = time.perf_counter(), time.process_time()
        for code in codes:
            try:
                compute_rdkit_descriptors(mol_from_xyz(code, structure_dir))
            except Exception:
                pass
        wall.append(time.perf_counter() - t0w)
        cpu.append(time.process_time() - t0c)
    return wall, cpu


def summarize(name, n_mol, wall, cpu, logger):
    w_mean = statistics.mean(wall)
    w_sd = statistics.stdev(wall) if len(wall) > 1 else 0.0
    c_mean = statistics.mean(cpu)
    c_sd = statistics.stdev(cpu) if len(cpu) > 1 else 0.0
    logger.log(f"\n  [{name}]  molecules = {n_mol},  repeats = {len(wall)}")
    logger.log(f"    total CPU time : {c_mean * 1000:.2f} +- {c_sd * 1000:.2f} ms   "
               f"(wall {w_mean * 1000:.2f} +- {w_sd * 1000:.2f} ms)")
    logger.log(f"    per molecule   : {c_mean / n_mol * 1000:.4f} ms (CPU)   "
               f"{w_mean / n_mol * 1000:.4f} ms (wall)")
    return {"route": name, "n_molecules": n_mol, "repeats": len(wall),
            "cpu_total_ms_mean": c_mean * 1000, "cpu_total_ms_sd": c_sd * 1000,
            "wall_total_ms_mean": w_mean * 1000, "wall_total_ms_sd": w_sd * 1000,
            "cpu_per_molecule_ms": c_mean / n_mol * 1000,
            "wall_per_molecule_ms": w_mean / n_mol * 1000}


def environment_info():
    import numpy
    import rdkit
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "n_descriptors": len(RDKIT_DESC_NAMES),
        "thread_env": {v: os.environ.get(v) for v in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def main():
    args = parse_args()
    configure(DATA_DIR=args.data_dir, RESULTS_DIR=args.results_dir)
    out_dir = results_dir() / "descriptor_timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "execution.log")

    try:
        logger.log("=" * 70)
        logger.log("Step 6 - RDKit descriptor timing (single thread, CPU time)")
        logger.log("=" * 70)
        env = environment_info()
        for k, v in env.items():
            logger.log(f"  {k}: {v}")
        with open(out_dir / "environment.json", "w", encoding="utf-8") as f:
            json.dump(env, f, indent=2, ensure_ascii=False)

        df_all = load_all_data(logger)
        api_codes = sorted({c for c in df_all["DRG"].astype(str).str.strip().unique()
                            if c and c.lower() != "nan" and not is_water_code(c)})
        all_codes = [c for c in collect_required_codes(df_all) if c.lower() != "nan"]
        logger.log(f"\n  APIs ({len(api_codes)}): {api_codes}")
        logger.log(f"  All unique molecules ({len(all_codes)})")

        data_dir = Path(CONFIG["DATA_DIR"]).resolve()
        structure_dir = data_dir / CONFIG["STRUCTURE_DIR"]
        smiles_api = smiles_for_codes(api_codes, data_dir, logger)
        smiles_all = smiles_for_codes(all_codes, data_dir, logger)

        rows = []
        rows.append(summarize("SMILES route, APIs only (manuscript value)",
                              len(smiles_api),
                              *time_smiles_route(list(smiles_api.values()),
                                                 args.repeats, args.warmup),
                              logger=logger))
        rows.append(summarize("SMILES route, all unique molecules", len(smiles_all),
                              *time_smiles_route(list(smiles_all.values()),
                                                 args.repeats, args.warmup),
                              logger=logger))
        if not args.no_xyz_route:
            rows.append(summarize("xyz route, APIs only", len(api_codes),
                                  *time_xyz_route(api_codes, structure_dir,
                                                  args.repeats, args.warmup),
                                  logger=logger))

        pd.DataFrame(rows).to_csv(out_dir / "timing.csv", index=False, encoding="utf-8-sig")
        logger.log(f"\n  Results written to {out_dir}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

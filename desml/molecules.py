"""
Construction of the RDKit molecule objects (manuscript Section 2.3.1).

Two routes are provided and both end in the same dictionary
``{molecule code: rdkit.Chem.Mol}``:

``xyz`` (default, the route used for the published results)
    ``data/structures/<code>.xyz`` is read with ``Chem.MolFromXYZBlock`` and the
    bonds are perceived with ``rdDetermineBonds.DetermineBonds``.  Choline
    chloride is a single object made of the choline cation (total charge +1)
    and a chloride anion joined with ``Chem.CombineMols``; the chloride is
    therefore part of the molecule even though RDKit does not count it in the
    default hydrogen-bond-acceptor count.

``smiles``
    the SMILES column of ``data/molecules.csv`` is parsed instead.  This route
    needs no geometry files and is convenient for new molecules, but it is a
    re-derivation of the graph rather than the published input; use
    ``scripts/07_export_smiles.py`` to verify that both routes give identical
    descriptors before relying on it.

Water never has a geometry file: it is built from the SMILES "O" and embedded
with a fixed seed so that the Gasteiger charges are reproducible.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from rdkit.Chem import rdDetermineBonds
except ImportError:                                    # pragma: no cover
    rdDetermineBonds = None

from .config import CONFIG, NONE_CODE
from .data import collect_required_codes, is_water_code
from .logging_utils import Logger

#: codes that denote choline chloride and need the CombineMols treatment
CHOLINE_CHLORIDE_CODES = {"CH", "CHCL"}


def find_xyz_file(code: str, structure_dir: Path) -> Optional[Path]:
    """Case-insensitive lookup of ``<code>.xyz`` inside the structure directory."""
    structure_dir = Path(structure_dir)
    if not structure_dir.exists():
        return None
    for p in structure_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".xyz" and p.stem.lower() == code.lower():
            return p
    return None


def load_mol_from_xyz(xyz_path: Path, total_charge: int = 0):
    """Read a geometry file and perceive the bonds.

    The file is read in Python and passed to RDKit as a block rather than by
    path, because ``MolFromXYZFile`` cannot open paths containing non-ASCII
    characters on some platforms.  Gasteiger charges are computed here because
    several RDKit descriptors (``MaxPartialCharge`` and friends) need them.
    """
    mol = Chem.MolFromXYZBlock(Path(xyz_path).read_text(encoding="utf-8"))
    if mol is None:
        raise RuntimeError(f"could not parse the geometry file: {xyz_path}")

    # bond perception; fall back to connectivity only if it fails
    if rdDetermineBonds is not None:
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=total_charge)
        except Exception:
            try:
                rdDetermineBonds.DetermineBonds(mol)
            except Exception:
                try:
                    rdDetermineBonds.DetermineConnectivity(mol)
                except Exception:
                    pass
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    return mol


def build_water_mol():
    """Water, built from SMILES and embedded with a fixed seed."""
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42) == 0:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    return mol


def build_choline_chloride_mol(ch_xyz_path: Path):
    """Choline chloride as one object: choline cation (+1) joined with [Cl-]."""
    ch_mol = load_mol_from_xyz(ch_xyz_path, total_charge=+1)
    cl_mol = Chem.MolFromSmiles("[Cl-]")
    combined = Chem.CombineMols(ch_mol, cl_mol)
    for step in (lambda: Chem.SanitizeMol(combined),
                 lambda: Chem.GetSSSR(combined),
                 lambda: AllChem.ComputeGasteigerCharges(combined)):
        try:
            step()
        except Exception:
            pass
    return combined


def load_molecule_table(data_dir: Path = None) -> pd.DataFrame:
    """Read ``molecules.csv``.

    Required columns are ``code`` and ``smiles``; ``role`` is written by
    ``scripts/07_export_smiles.py`` and any further column (a full compound
    name, for instance) is ignored.
    """
    base = Path(data_dir) if data_dir is not None else Path(CONFIG["DATA_DIR"]).resolve()
    path = base / CONFIG["MOLECULE_CSV"]
    if not path.exists():
        raise FileNotFoundError(f"molecule table not found: {path}")
    tbl = pd.read_csv(path, encoding="utf-8-sig")
    tbl.columns = [c.strip().lower() for c in tbl.columns]
    for col in ("code", "smiles"):
        if col not in tbl.columns:
            raise ValueError(f"{path} must contain a '{col}' column")
    tbl["code"] = tbl["code"].astype(str).str.strip()
    return tbl


def build_mol_from_smiles(smiles: str):
    """Parse a SMILES string and prepare it exactly like the xyz route."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"MolFromSmiles failed: {smiles}")
    mol = Chem.AddHs(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    return mol


def build_all_mols(df: pd.DataFrame, logger: Logger, data_dir: Path = None,
                   source: str = None) -> Dict[str, Optional[object]]:
    """Build every molecule required by the data table.

    Returns ``{code: Mol}``; the empty slot code maps to ``None`` and every
    code that denotes water maps to the same water molecule.
    """
    base = Path(data_dir) if data_dir is not None else Path(CONFIG["DATA_DIR"]).resolve()
    source = (source or CONFIG["MOLECULE_SOURCE"]).lower()
    if source not in ("xyz", "smiles"):
        raise ValueError(f"MOLECULE_SOURCE must be 'xyz' or 'smiles', got {source!r}")

    logger.log(f"\n[2] Building RDKit Mol objects (source={source})...")
    required = collect_required_codes(df)

    mols = {NONE_CODE: None}
    water_codes = set()
    for col in ["HBA", "HBD"]:
        for c in df[col].astype(str).str.strip().unique():
            if is_water_code(c):
                water_codes.add(c)
    if water_codes:
        water_mol = build_water_mol()
        for wc in water_codes:
            mols[wc] = water_mol

    if source == "xyz":
        structure_dir = base / CONFIG["STRUCTURE_DIR"]
        missing, xyz_paths = [], {}
        for code in required:
            p = find_xyz_file(code, structure_dir)
            if p is None:
                missing.append(code)
            else:
                xyz_paths[code] = p
        if missing:
            logger.log(f"  [ERROR] Missing XYZ files in {structure_dir}: {missing}")
            raise FileNotFoundError(f"missing XYZ files: {missing}")
        for code, xyz_path in xyz_paths.items():
            if code.upper() in CHOLINE_CHLORIDE_CODES:
                mols[code] = build_choline_chloride_mol(xyz_path)
            else:
                mols[code] = load_mol_from_xyz(xyz_path, total_charge=0)
    else:
        tbl = load_molecule_table(base)
        smiles_map = dict(zip(tbl["code"], tbl["smiles"].astype(str)))
        missing = [c for c in required if c not in smiles_map]
        if missing:
            raise FileNotFoundError(f"codes absent from {CONFIG['MOLECULE_CSV']}: {missing}")
        for code in required:
            mols[code] = build_mol_from_smiles(smiles_map[code])

    logger.log(f"  Built {len(mols)} molecules")
    return mols

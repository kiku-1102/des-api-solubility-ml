"""
RDKit physicochemical descriptors (manuscript Section 2.3.1).

``RDKIT_DESC_NAMES`` is ``Descriptors.descList`` minus the eight BCUT2D
entries, i.e. 202 descriptors with the pinned RDKit 2023.09.6.  Because the
descriptor list grows between RDKit releases, ``check_descriptor_count()``
compares the number against ``N_EXPECTED_DESCRIPTORS`` and warns (or raises)
when a different RDKit is used: the published feature selection counts
(76 API / 64 solvent descriptors) only reproduce with the pinned version.
"""

from typing import Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from .config import BCUT2D_SKIP, N_EXPECTED_DESCRIPTORS

#: descriptor names used in this study (BCUT2D removed), in RDKit order
RDKIT_DESC_NAMES = [name for name, _ in Descriptors.descList if name not in BCUT2D_SKIP]


def check_descriptor_count(logger=None, strict: bool = False) -> int:
    """Verify that the installed RDKit yields the expected number of descriptors."""
    n = len(RDKIT_DESC_NAMES)
    if n != N_EXPECTED_DESCRIPTORS:
        from rdkit import __version__ as rdkit_version
        msg = (f"  [WARN] {n} RDKit descriptors available (expected "
               f"{N_EXPECTED_DESCRIPTORS}); installed RDKit is {rdkit_version}. "
               f"Feature-selection counts will differ from the published ones.")
        if strict:
            raise RuntimeError(msg.strip())
        if logger is not None:
            logger.log(msg)
        else:
            print(msg)
    return n


def compute_rdkit_descriptors(mol: Optional[Chem.Mol]) -> Dict[str, float]:
    """All descriptors of one molecule; non-finite values and failures give 0.0."""
    if mol is None:
        return {name: 0.0 for name in RDKIT_DESC_NAMES}
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
    except Exception:
        pass
    result = {}
    for name, func in Descriptors.descList:
        if name in BCUT2D_SKIP:
            continue
        try:
            v = float(func(mol))
            result[name] = v if np.isfinite(v) else 0.0
        except Exception:
            result[name] = 0.0
    return result


def compute_all_descriptors(mols, logger) -> Dict[str, Dict[str, float]]:
    """Descriptor dictionary for every molecule code."""
    logger.log("\n[3] Computing RDKit descriptors...")
    check_descriptor_count(logger)
    all_desc = {code: compute_rdkit_descriptors(mol) for code, mol in mols.items()}
    logger.log(f"  Computed for {len(all_desc)} molecules ({len(RDKIT_DESC_NAMES)} descriptors each)")
    return all_desc

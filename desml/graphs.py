"""
Molecular graph representation for the GNN models (manuscript Section 2.5.2).

Atom (node) features are 29-dimensional:
one-hot atomic number (H, B, C, N, O, F, P, S, Cl, Br, I + other), degree 0-5,
formal charge, one-hot hybridisation (sp, sp2, sp3, sp3d, sp3d2 + other),
aromaticity flag, in-ring flag, number of bonded hydrogens and the Gasteiger
partial charge clipped to +-2.

Bond (edge) features are 7-dimensional: one-hot bond order (single, double,
triple, aromatic + other), conjugation flag and in-ring flag.  Because the
features are predominantly one-hot encodings they are not standardised
(Section 2.5.2).  An empty solvent slot is represented by a single-atom dummy
graph whose mask is zero, so it contributes nothing to the embedding.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

from .config import NONE_CODE
from .data import is_water_code
from .molecules import build_water_mol

_ALLOWED_Z = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
_HYB_LIST = [Chem.rdchem.HybridizationType.SP,
             Chem.rdchem.HybridizationType.SP2,
             Chem.rdchem.HybridizationType.SP3,
             Chem.rdchem.HybridizationType.SP3D,
             Chem.rdchem.HybridizationType.SP3D2]
_BOND_TYPES = [Chem.rdchem.BondType.SINGLE,
               Chem.rdchem.BondType.DOUBLE,
               Chem.rdchem.BondType.TRIPLE,
               Chem.rdchem.BondType.AROMATIC]

#: 11 + 1 atomic numbers, 6 degrees, formal charge, 5 + 1 hybridisations,
#: aromaticity, ring, hydrogens, Gasteiger charge = 29
ATOM_FDIM = len(_ALLOWED_Z) + 1 + 6 + 1 + (len(_HYB_LIST) + 1) + 1 + 1 + 1 + 1
#: 4 + 1 bond orders, conjugation, ring = 7
BOND_FDIM = len(_BOND_TYPES) + 1 + 1 + 1


def _onehot(value, choices) -> List[float]:
    vec = [0.0] * (len(choices) + 1)
    if value in choices:
        vec[choices.index(value)] = 1.0
    else:
        vec[-1] = 1.0
    return vec

def _onehot_int(value: int, n: int) -> List[float]:
    vec = [0.0] * n
    v = int(value)
    v = 0 if v < 0 else (n - 1 if v >= n else v)
    vec[v] = 1.0
    return vec

def atom_features(atom) -> List[float]:
    f: List[float] = []
    f += _onehot(atom.GetAtomicNum(), _ALLOWED_Z)
    f += _onehot_int(atom.GetDegree(), 6)
    f += [float(atom.GetFormalCharge())]
    f += _onehot(atom.GetHybridization(), _HYB_LIST)
    f += [float(atom.GetIsAromatic())]
    f += [float(atom.IsInRing())]
    f += [float(atom.GetTotalNumHs(includeNeighbors=True))]
    gc = 0.0
    if atom.HasProp("_GasteigerCharge"):
        try:
            gc = float(atom.GetDoubleProp("_GasteigerCharge"))
        except Exception:
            gc = 0.0
    if not np.isfinite(gc):
        gc = 0.0
    f += [float(np.clip(gc, -2.0, 2.0))]
    return f

def bond_features(bond) -> List[float]:
    f: List[float] = []
    f += _onehot(bond.GetBondType(), _BOND_TYPES)
    f += [float(bond.GetIsConjugated())]
    f += [float(bond.IsInRing())]
    return f

def mol_to_graph(mol) -> Data:
    n_atoms = mol.GetNumAtoms()
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    if x.shape[0] == 0:
        x = torch.zeros((1, ATOM_FDIM), dtype=torch.float32)
    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        src += [i, j]; dst += [j, i]; eattr += [bf, bf]
    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FDIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def _dummy_graph() -> Data:
    return Data(x=torch.zeros((1, ATOM_FDIM), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, BOND_FDIM), dtype=torch.float32))

class GraphBank:
    """Cache of ``code -> PyG Data``; the water graph is kept for Eqs. (5) and (7)."""
    def __init__(self, mols: Dict[str, Optional[object]], logger=None):
        self.cache: Dict[str, Data] = {}
        self.dummy = _dummy_graph()
        self.water = None
        n_ok = 0
        for code, mol in mols.items():
            if code == NONE_CODE or mol is None:
                continue
            try:
                g = mol_to_graph(mol)
                self.cache[code] = g
                n_ok += 1
                if is_water_code(code):
                    self.water = g
            except Exception as e:
                if logger:
                    logger.log(f"  [WARN] graph build failed for {code}: {e}")
                self.cache[code] = self.dummy
        if self.water is None:
            self.water = mol_to_graph(build_water_mol())
        if logger:
            logger.log(f"  GraphBank: {n_ok} graphs cached "
                       f"(atom_fdim={ATOM_FDIM}, bond_fdim={BOND_FDIM})")

    def get(self, code: str) -> Tuple[Data, float]:
        code = str(code).strip()
        if code == NONE_CODE or code not in self.cache:
            return self.dummy, 0.0
        return self.cache[code], 1.0

    def build_batch(self, codes: List[str], device: str):
        datas, masks = [], []
        for c in codes:
            d, mk = self.get(c)
            datas.append(d); masks.append(mk)
        batch = Batch.from_data_list(datas).to(device)
        mask = torch.tensor(masks, dtype=torch.float32, device=device)
        return batch, mask

    def water_batch(self, n: int, device: str):
        batch = Batch.from_data_list([self.water] * n).to(device)
        mask = torch.ones(n, dtype=torch.float32, device=device)
        return batch, mask

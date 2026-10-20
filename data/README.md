# Data

## Provenance

The solubility data used in this study were derived from the data set reported
by Cysewski et al. (ref. [15] of the manuscript), with deduplication and
reformatting (unification of compound names, assignment of abbreviations, and
removal of auxiliary columns). The processed data set used in this study is
available in this repository, in the files listed below. Please cite the
original publication when you use these data.

The DES subset was obtained from the 1,618 points of the previous study by
removing 29 duplicated entries, so that all records are unique (1,589 points).

## Files

| file | rows | content |
|---|---|---|
| `solubility_des.csv`    | 1,589 | API in a DES (HBA + HBD, optionally with water) |
| `solubility_pure.csv`   | 2,064 | API in a single-component solvent |
| `solubility_binary.csv` | 4,332 | API in a binary solvent mixture |
| `molecules.csv`         | - | molecule code, role and canonical SMILES |
| `structures/*.xyz`      | - | one geometry per molecule code |

Total: 7,985 data points, 15 APIs, 2 HBAs, 12 HBDs and 46 general solvents.

### Column layout

`solubility_des.csv`

| column | meaning |
|---|---|
| `DRG` | API code (see `molecules.csv` and Table 1 of the paper) |
| `HBA` | hydrogen-bond acceptor code |
| `HBD` | hydrogen-bond donor code |
| `T_C` | temperature [°C] |
| `logS` | common logarithm of the mole-fraction solubility, log x |
| `x_HBA`, `x_HBD`, `x_WATER` | mole fractions of the solvent components |

`solubility_pure.csv`: `DRG`, `solvent`, `T_C`, `logS`.

`solubility_binary.csv`: `DRG`, `solvent_1`, `solvent_2`, `x2`, `x3`, `T_C`,
`logS` (`x2` and `x3` are the mole fractions of solvent 1 and solvent 2).

The loaders in `desml/data.py` accept both these names and the column names of
the original tables (`T[C]`, `log(x1)`, `x2*`, `x3*`, `x4*`), so a table
exported directly from the source can be used unchanged.

## Molecule structures

`structures/<code>.xyz` holds one geometry per molecule code; the file stem is
the code used in the CSV files (case-insensitive). The descriptors of the
published results were computed from these geometries: RDKit reads the
coordinates and perceives the bonds with `rdDetermineBonds.DetermineBonds`.

Two codes are special:

* **choline chloride** (`CH`) - the geometry file holds the choline cation
  (total charge +1); the chloride anion is added as `[Cl-]` and the two are
  joined with `Chem.CombineMols`, so the salt is a single molecule object. Note
  that RDKit does not count the chloride in its default hydrogen-bond-acceptor
  descriptor, although it is a strong acceptor.
* **water** - has no geometry file: it is built from the SMILES `O`, embedded
  with a fixed random seed and assigned Gasteiger charges. Water never occupies
  a molecular slot of the input vector; it enters through the condition
  features (`x_water`, the water-presence flag and ln(x_water + eps)).

`molecules.csv` lists the canonical SMILES derived from these geometries.
`scripts/07_export_smiles.py` regenerates that table and verifies, descriptor
by descriptor, that parsing the SMILES gives the same values as the geometry
route; use `--molecule-source smiles` in steps 1-6 only after that check
passes.

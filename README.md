# Machine-learning prediction of API solubility in deep eutectic solvents

Code accompanying

> K. Kikuchi, K. Kaneko, K. Fujimoto, T. Matsuyama, J. Ida,
> *Predicting drug solubility in deep eutectic solvents for unseen component
> combinations without quantum-chemical computation*
> (submitted to **Journal of Cheminformatics**).

The study predicts the mole-fraction solubility (log x) of poorly water-soluble
APIs in deep eutectic solvents (DESs) from RDKit physicochemical descriptors and
molecular graphs - that is, **without any quantum-chemical calculation**. Two
ways of presenting a multicomponent solvent to a model (a concatenated vector
and a mole-fraction weighted "mixture" vector) and ten model variants are
compared in three experimental designs:

| study | question | evaluation |
|---|---|---|
| **Study I**   | how do low-cost descriptors compare with the quantum-chemistry-based previous study? | all data split 8:2 at random (the scheme of the previous study) |
| **Study II**  | how accurate are the models for DES systems themselves? | only DES data split 8:2, three training regimes |
| **Study III** | can an entirely unseen system (API x HBA x HBD) be predicted? | leave-one-system-out (LOSO) |

---

## 1. What is in this repository

```
.
├── data/                      solubility data, molecule table and geometries
│   ├── solubility_des.csv         DES systems          (1,589 points)
│   ├── solubility_pure.csv        single-component solvents (2,064 points)
│   ├── solubility_binary.csv      binary solvent mixtures   (4,332 points)
│   ├── molecules.csv              code -> role -> canonical SMILES
│   ├── structures/                <code>.xyz, one geometry per molecule
│   └── README.md                  provenance and licence of the data
├── desml/                     the library shared by every step
│   ├── config.py                  all settings and thresholds in one place
│   ├── data.py                    reading and harmonising the three CSV files
│   ├── molecules.py               geometry/SMILES -> RDKit molecules
│   ├── descriptors.py             the 202 RDKit descriptors
│   ├── feature_selection.py       three-pool variance and correlation filters
│   ├── features.py                Concat / Mixture input vectors, TiedScaler
│   ├── graphs.py                  atom and bond features, molecular graphs
│   ├── splits.py                  training regimes and validation splits
│   ├── metrics.py                 RMSE, MAE, R2, MAPE, %Nout, r, rho, tau
│   ├── grids.py                   the search spaces of Table 4
│   ├── models/                    nuSVR, RF, XGB, LGB, MLP, GIN, GATv2
│   ├── study.py                   the runner shared by Studies I and II
│   ├── loso.py                    the leave-one-system-out machinery
│   └── plots.py                   the figures produced by the runs
├── scripts/                   the pipeline, in execution order (01 -> 07)
├── environment.yml            conda environment with pinned versions
├── requirements.txt           the same pins for pip
└── docs/
    ├── workflow.png               Fig. 2 of the manuscript
    └── REPRODUCE.md               step-by-step reproduction guide
```

## 2. Installation

```bash
git clone https://github.com/kiku-1102/des-api-solubility-ml.git
cd des-api-solubility-ml
conda env create -f environment.yml     # creates the environment "desml"
conda activate desml
python -c "import desml, rdkit; from desml.descriptors import RDKIT_DESC_NAMES; print(len(RDKIT_DESC_NAMES), 'descriptors')"
```

The last command must print **202 descriptors**. RDKit adds descriptors between
releases, so the pinned version (2023.09.6) is part of the reproduction recipe;
with a newer RDKit the number differs and the feature-selection counts of the
paper will not be reproduced. Every script warns when the count is not 202.

A CUDA GPU is optional. The neural networks were trained on an RTX 4080 SUPER
with automatic mixed precision; on a CPU they run as well, only slower
(pass `--cpu` to force it).

## 3. The pipeline

The numbered scripts follow Fig. 2 of the manuscript, top to bottom. Each one
writes its own `execution.log`, a `00_config.json` with the exact settings and
CSV files with every metric, so a run can be inspected afterwards.

```
01_feature_selection.py   ->  results/feature_selection/feature_manifest.json
02_hyperparameter_search.py -> results/hyperparameters/<model>/<input>/best_hp.json
03_study1_benchmark.py    ->  results/study1/...
04_study2_des.py          ->  results/study2/...
05_study3_loso.py         ->  results/study3/...
06_descriptor_timing.py   ->  results/descriptor_timing/timing.csv
07_export_smiles.py       ->  data/molecules.csv + route comparison (utility)
```

### Step 1 - descriptors and feature selection (Section 2.3)

```bash
python scripts/01_feature_selection.py
```

Computes the 202 RDKit descriptors (`Descriptors.descList` minus the eight
BCUT2D entries, which cannot be computed for the choline-chloride object) for
every molecule, then filters them **independently in three pools** - API,
solvent and condition - with a variance filter (threshold 1e-12) followed by a
correlation filter (a pair is removed when |r| > 0.9, the more redundant member
of the most correlated pair first, with the mean correlations recomputed after
every deletion). Expected result: **76 API + 64 solvent descriptors and 6
condition features**, i.e. a concatenated input vector of 76 + 2x64 + 6 = 210
columns. The surviving names go into `feature_manifest.json`, which every later
step reads.

### Step 2 - hyperparameter optimisation (Section 2.6.3, Table 4)

```bash
python scripts/02_hyperparameter_search.py --model xgb --input concat
python scripts/02_hyperparameter_search.py --model mlp --input concat
python scripts/02_hyperparameter_search.py --model gatv2            # two-stage
```

The search is run **once**, on the DES systems alone, on the single split with
seed 42 and without cross-validation; the winning values are then used
unchanged in all three studies and all regimes, so every comparison is made
under identical hyperparameter conditions. For the graph models the head is
searched first with a default encoder (stage Ia) and the encoder afterwards
with the best head (stage Ib), in the graph-only configuration; both are shared
by all four graph configurations of that architecture.

Run it once per model and per input representation you intend to use
(`--input concat` and `--input mixture` are searched separately; the graph
models ignore `--input`).

### Step 3 - Study I (Section 2.8.1)

```bash
python scripts/03_study1_benchmark.py --model xgb   --input concat
python scripts/03_study1_benchmark.py --model gatv2 --input concat --use-descriptors
```

All data are pooled and split 8:2 (the `prior` regime). The 20 configurations
of Table 5 are the six descriptor models x two input representations, plus the
two graph architectures x two representations, each with and without the RDKit
descriptors (`--use-descriptors` switches between Eqs. 4/5 and Eqs. 6/7).

### Step 4 - Study II (Section 2.8.2)

```bash
python scripts/04_study2_des.py --model xgb   --regime des_only
python scripts/04_study2_des.py --model gatv2 --regime joint --use-descriptors
python scripts/04_study2_des.py --model mlp   --regime pretrain_ft
```

Only the DES data are split 8:2; the input is fixed to the concatenated
representation. `pretrain_ft` (pre-train on the general solvents, then
fine-tune all layers on DES) is available for the neural networks only.

### Step 5 - Study III, LOSO (Section 2.8.3)

```bash
python scripts/05_study3_loso.py --model xgb   --regime des_only
python scripts/05_study3_loso.py --model mlp   --regime des_only
python scripts/05_study3_loso.py --model gatv2 --regime des_only
```

Every system with at least 35 DES points is held out in turn. Besides the
metrics per system, the script writes the per-system summary figure of
Figs. 7-9 (dependence on temperature, water content and HBA:HBD ratio, the
residuals and the measured-versus-predicted plot) and flags test points that
lie outside the range of the training conditions.

### Step 6 - feature-computation cost (Section 2.5.5)

```bash
python scripts/06_descriptor_timing.py
```

Single-threaded CPU time of "SMILES -> molecule -> 202 descriptors", one
warm-up pass and five timed repetitions, for the 15 APIs.

## 4. Reproducing the whole study

`docs/REPRODUCE.md` lists every command of the published runs in order, together
with the output each one produces. In short:

```bash
python scripts/01_feature_selection.py
for m in nusvr_rbf nusvr_poly rf xgb lgb mlp; do
  for i in concat mixture; do
    python scripts/02_hyperparameter_search.py --model $m --input $i
  done
done
python scripts/02_hyperparameter_search.py --model gin
python scripts/02_hyperparameter_search.py --model gatv2
# ... then steps 3-5 for the configurations you need
```

**Aggregated tables and figures.** The ranking tables (Tables 5, 6, S2, S3) and
the aggregate bar charts (Figs. 4 and 6) of the manuscript were assembled
outside this repository from the `metrics_summary.csv` /
`summary_aggregate.csv` files that the runs write; the scripts here stop at
those CSV files and at the per-run figures.

## 5. Settings that matter

| setting | value | where |
|---|---|---|
| descriptors | 202 (descList minus BCUT2D) | `desml/descriptors.py` |
| variance filter | 1e-12 | `CONFIG["VARIANCE_THRESHOLD"]` |
| correlation filter | drop when \|r\| > 0.9 | `CONFIG["CORRELATION_THRESHOLD"]` |
| train/test split | 8:2, seeds 42, 123, 456, 789, 1024 | `CONFIG["DATA_SEEDS"]` |
| validation (early stopping) | 10 % of the training part | `CONFIG["VAL_FRACTION"]` |
| weight-initialisation seeds | 1-5 (neural networks) | `CONFIG["WEIGHT_SEEDS"]` |
| grid-search split | seed 42; inner validation 20 % (descriptor models) / 10 % (neural networks) | `CONFIG["GS_VAL_FRACTION_*"]` |
| boosted trees | 1000 estimators during the search, up to 3000 with early stopping (50) afterwards | `CONFIG["XGB_*"]`, `CONFIG["LGB_*"]` |
| neural networks | Adam, lr 1.5e-4, weight decay 1e-6, max 500 epochs, patience 50, gradient clipping 5.0 | `CONFIG["MLP_*"]` |
| LOSO target systems | at least 35 DES points | `CONFIG["MIN_SYSTEM_COUNT"]` |

Everything lives in `desml/config.py`; the search spaces are in
`desml/grids.py`, which asserts the grid sizes of Table 4.

## 6. Data

See `data/README.md`. The solubility data used in this study were derived from
the data set reported by Cysewski et al., with deduplication and reformatting
(unification of compound names, assignment of abbreviations, and removal of
auxiliary columns). The processed data set used in this study is available in
this repository. Please cite the original source as well:

> P. Cysewski, T. Jeliński, M. Przybyłek, Exploration of the solubility
> hyperspace of selected active pharmaceutical ingredients in choline- and
> betaine-based deep eutectic solvents: machine learning modeling and
> experimental validation, *Molecules* **29** (2024) 4894.
> https://doi.org/10.3390/molecules29204894

## 7. Licence and citation

The code is released under the MIT licence (`LICENSE`). If you use it, please
cite the paper above; `CITATION.cff` contains the machine-readable metadata.

## 8. Contact

Kodai Kikuchi - kkodai@soka.ac.jp - Soka University.
Questions about the code are best raised as a GitHub issue.

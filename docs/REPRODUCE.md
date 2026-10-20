# Reproducing the study step by step

Everything below assumes the environment of `environment.yml` is active and that
you are in the repository root. Each command writes into `results/` and prints
the same information to its own `execution.log`.

Check first that the installed RDKit gives the expected descriptor count:

```bash
python -c "from desml.descriptors import RDKIT_DESC_NAMES; print(len(RDKIT_DESC_NAMES))"
# -> 202
```

## 0. Overview

```
01 feature selection          once          -> feature_manifest.json
02 hyperparameter search      per model     -> best_hp.json
03 Study I   (prior)          per config    -> results/study1/...
04 Study II  (DES only)       per config    -> results/study2/...
05 Study III (LOSO)           per config    -> results/study3/...
06 descriptor timing          once          -> results/descriptor_timing/
```

Steps 3-5 all read the manifest of step 1 and the hyperparameters of step 2, so
those two must be run first. Steps 3, 4 and 5 are independent of each other.

## 1. Feature selection (Section 2.3, Section 3.1)

```bash
python scripts/01_feature_selection.py
```

Expected summary at the end of the log:

```
        api:  202 ->  135 ->   76 (variance -67, correlation -59)
    solvent:  202 ->  130 ->   64 (variance -72, correlation -66)
  condition:   10 ->   10 ->    6 (variance -0,  correlation -4)

  Concatenated input dimension: 210 (API 76 + 2 x solvent 64 + conditions 6)
```

The six surviving condition features are `T_K`, `x_HBA`, `x_HBD`, `x_WATER`,
`has_water` and `n_components`.

## 2. Hyperparameter search (Section 2.6.3, Table 4)

One run per model and input representation. The descriptor models are quick;
the MLP takes roughly an hour per representation on a GPU and the two graph
architectures several hours each (48 head candidates + 24 or 16 encoder
candidates, every candidate trained with early stopping).

```bash
for m in nusvr_rbf nusvr_poly rf xgb lgb mlp; do
  for i in concat mixture; do
    python scripts/02_hyperparameter_search.py --model "$m" --input "$i"
  done
done

# graph models: two-stage search in the graph-only configuration,
# shared by Eqs. (4)-(7) of the same architecture
python scripts/02_hyperparameter_search.py --model gin
python scripts/02_hyperparameter_search.py --model gatv2
```

## 3. Study I - the 20 configurations of Table 5

```bash
# descriptor-based models (12 configurations)
for m in nusvr_rbf nusvr_poly rf xgb lgb mlp; do
  for i in concat mixture; do
    python scripts/03_study1_benchmark.py --model "$m" --input "$i"
  done
done

# graph-only models, Eqs. (4) and (5)  (4 configurations)
for m in gin gatv2; do
  for i in concat mixture; do
    python scripts/03_study1_benchmark.py --model "$m" --input "$i"
  done
done

# GNN+MLP, Eqs. (6) and (7)  (4 configurations)
for m in gin gatv2; do
  for i in concat mixture; do
    python scripts/03_study1_benchmark.py --model "$m" --input "$i" --use-descriptors
  done
done
```

The MAPE and %Nout columns of `results/study1/*/metrics_summary.csv` are the
numbers of Table 5; `seed42/scatter.png` of the four best configurations are the
panels of Fig. 3.

## 4. Study II - the 19 configurations of Table 6

```bash
# DES only and Joint for every model
for m in nusvr_rbf nusvr_poly rf xgb lgb mlp; do
  for r in des_only joint; do
    python scripts/04_study2_des.py --model "$m" --regime "$r"
  done
done
for m in gin gatv2; do
  for r in des_only joint; do
    python scripts/04_study2_des.py --model "$m" --regime "$r" --use-descriptors
  done
done

# transfer learning, neural networks only
python scripts/04_study2_des.py --model mlp   --regime pretrain_ft
python scripts/04_study2_des.py --model gin   --regime pretrain_ft --use-descriptors
python scripts/04_study2_des.py --model gatv2 --regime pretrain_ft --use-descriptors
```

RMSE and Spearman's rho in `metrics_summary.csv` are the numbers of Table 6;
`seed42/scatter.png` of the four best configurations are the panels of Fig. 5.

## 5. Study III - LOSO (six configurations)

```bash
for m in xgb mlp gatv2; do
  for r in des_only joint; do
    python scripts/05_study3_loso.py --model "$m" --regime "$r"
  done
done
```

Each run holds out every system with at least 35 DES points in turn and writes

* `summary_per_system.csv` - the per-system metrics of Tables S4 and S5,
* `summary_aggregate.csv`  - the means plotted in Fig. 6,
* `axis_dependency.csv`    - the slope / rank / residual diagnostics per axis,
* `combined_figures/<system>.png` - the per-system summary of Figs. 7-9 and S5.

The GATv2 runs are the slowest part of the study: five weight seeds are trained
per held-out system.

## 6. Feature-computation cost (Section 2.5.5)

```bash
python scripts/06_descriptor_timing.py
```

Single-threaded CPU time of SMILES -> molecule -> 202 descriptors for the 15
APIs, one warm-up pass plus five timed repetitions.

## 7. Utility - molecule table and route check

```bash
python scripts/07_export_smiles.py
```

Regenerates `data/molecules.csv` from the geometries and compares the
descriptors of the geometry route with those of the SMILES route. The published
numbers come from the geometry route (`--molecule-source xyz`, the default).

## Notes on exact reproduction

* **Random seeds.** The five data splits (42, 123, 456, 789, 1024) and the five
  weight initialisations (1-5) are fixed in `desml/config.py`. GPU training is
  not bit-wise deterministic, so neural-network metrics can differ in the last
  decimals between machines; the tree models and nuSVR are deterministic.
* **Hyperparameters in Study III.** The LOSO folds reuse the hyperparameters
  found on the DES-only random split. The held-out systems were part of the
  training data of that search, which is a mild optimistic bias; it is stated
  in the manuscript and no per-fold search is performed.
* **Grid-search validation fraction.** Candidates are scored on 20 % of the
  training part for the descriptor models and on 10 % for the neural networks
  (`GS_VAL_FRACTION_DESCRIPTOR` / `GS_VAL_FRACTION_NN`); the final training
  always uses the 10 % validation split of Section 2.6.2.
* **Aggregated tables and figures.** Tables 5, 6, S2 and S3 and the bar charts
  of Figs. 4 and 6 were assembled from the `metrics_summary.csv` and
  `summary_aggregate.csv` files outside this repository.

"""
Central configuration.

Every number in this file is a setting of the published study.  Values that
appear in Table 4 of the manuscript ("principal fixed settings") are marked
with the model they belong to.  Scripts never hard-code settings: they read
them from ``CONFIG`` and may override the few run-specific entries
(input representation, training regime, directories) through ``configure()``.
"""

from pathlib import Path

# ---------------------------------------------------------------- constants
NONE_CODE = "NONE"                 # placeholder used when a solvent slot is empty

#: condition features that the design can produce (manuscript Section 2.3.1).
#: Ten candidates enter the condition pool of the feature selection; six of
#: them survive (Section 3.1).
DESIGN_CONDITION_KEYS = [
    "T_K", "x_HBA", "x_HBD", "x_WATER",
    "inv_T", "log_T", "inv_T_diff", "has_water", "log_water", "n_components",
]
COMPOSITION_KEYS = {"x_HBA", "x_HBD", "x_WATER"}
#: in the concatenated representation the composition features are renamed so
#: that slot2/slot3 stay exchange-symmetric (manuscript Section 2.4.1).
SLOT_RENAME = {"x_HBA": "x_slot2", "x_HBD": "x_slot3", "x_WATER": "x_water"}

#: the eight BCUT2D descriptors are excluded for every molecule because they
#: cannot be computed for the choline-chloride CombineMols object
#: (manuscript Section 2.3.1).
BCUT2D_SKIP = {
    "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW",
}

#: number of RDKit descriptors expected after removing BCUT2D.  RDKit changes
#: ``Descriptors.descList`` between releases, so the pinned version in
#: environment.yml (2023.09.6) is part of the reproduction recipe.
N_EXPECTED_DESCRIPTORS = 202


#: condition axes analysed in the LOSO study (column, axis label, short name)
AXES = [
    ("T_K", "Temperature T [K]", "temperature"),
    ("x_WATER", "Water content x_water", "water"),
    ("comp_ratio", "HBA:HBD ratio x_HBD/(x_HBA+x_HBD)", "composition"),
]


CONFIG = {
    # ----------------------------------------------------------- input data
    "DATA_DIR": "data",
    "RESULTS_DIR": "results",
    "DES_CSV": "solubility_des.csv",
    "PURE_CSV": "solubility_pure.csv",
    "BINARY_CSV": "solubility_binary.csv",
    "MOLECULE_CSV": "molecules.csv",          # code -> role -> canonical SMILES
    "STRUCTURE_DIR": "structures",            # *.xyz, one file per molecule code
    "MOLECULE_SOURCE": "xyz",                 # "xyz" (published route) or "smiles"
    "FEATURE_MANIFEST": "feature_manifest.json",

    # --------------------------------------- feature selection (Section 2.3.2)
    "VARIANCE_THRESHOLD": 1e-12,              # drop near-constant features
    "CORRELATION_THRESHOLD": 0.9,             # drop one of a pair with |r| > 0.9

    # ------------------------------------------- data splitting (Section 2.6)
    "TEST_SIZE": 0.2,                         # 8:2 train/test split
    "VAL_FRACTION": 0.10,                     # 9:1 carve-out for early stopping
    "VAL_DES_ONLY_IN_JOINT": True,            # Joint: validation from DES only
    "DATA_SEEDS": [42, 123, 456, 789, 1024],  # five random splits
    "WEIGHT_SEEDS": [1, 2, 3, 4, 5],          # five weight initialisations (NN)
    "GRID_SEARCH_DATA_SEED": 42,              # single split used for the search
    "GRID_SEARCH_WEIGHT_SEED": 1,             # weight seed fixed during search

    # ------------------------------- hyperparameter search (Section 2.6.3)
    # The scoring split of the grid search differs between the two model
    # families in the published runs and is kept as it was:
    #   descriptor models without early stopping / boosted trees -> 20 %
    #   neural networks (MLP, GNN)                               -> 10 %
    "GS_VAL_FRACTION_DESCRIPTOR": 0.2,
    "GS_VAL_FRACTION_NN": 0.10,

    # ------------------------------------------------ miscellaneous settings
    "T_REFERENCE": 298.15,                    # reference T of inv_T_diff [K]
    "LOG_WATER_EPS": 1e-3,                    # epsilon of ln(x_water + eps)
    "SCATTER_RANGE": (-9.0, 0.0),             # axis range of the scatter plots
    "N_JOBS": -1,

    # ------------------------------------------------- LOSO (Section 2.8.3)
    "MIN_SYSTEM_COUNT": 35,                   # minimum DES points of a target system
    "LOSO_VAL_SEED": 42,                      # fixed seed of the 9:1 validation split
    "EXTRAP_TOL": 1e-9,                       # tolerance of the extrapolation flag
    "SLICE_ROUND": {"T_K": 0, "x_WATER": 3, "comp_ratio": 3},
    "MIN_SLICE_POINTS": 3,
    "N_BIN": 4,                               # bins of the per-axis conditional RMSE

    # ------------------------------------------------- nuSVR (Table 4)
    "NUSVR_MAX_ITER": -1,
    "NUSVR_CACHE_SIZE": 1000,
    "NUSVR_TOL": 1e-3,
    "NUSVR_DEGREE": 4,                        # polynomial kernel only
    "NUSVR_COEF0": 1.0,                       # polynomial kernel only

    # ------------------------------------------------- XGBoost (Table 4)
    "XGB_GS_N_ESTIMATORS": 1000,              # fixed tree count during the search
    "XGB_FINAL_MAX_ESTIMATORS": 3000,         # upper limit of the final training
    "XGB_EARLY_STOPPING_ROUNDS": 50,
    "XGB_SUBSAMPLE": 0.8,
    "XGB_REG_ALPHA": 0.0,
    "XGB_REG_LAMBDA": 1.0,
    "XGB_TREE_METHOD": "hist",
    "XGB_DEVICE": "cpu",
    "XGB_VERBOSITY": 0,

    # ------------------------------------------------- LightGBM (Table 4)
    "LGB_GS_N_ESTIMATORS": 1000,
    "LGB_FINAL_MAX_ESTIMATORS": 3000,
    "LGB_EARLY_STOPPING_ROUNDS": 50,
    "LGB_SUBSAMPLE": 0.8,
    "LGB_SUBSAMPLE_FREQ": 1,
    "LGB_REG_ALPHA": 0.0,
    "LGB_REG_LAMBDA": 1.0,
    "LGB_VERBOSE": -1,

    # ---------------------------------------- MLP / GNN head (Table 4)
    "MLP_LR": 1.5e-4,
    "MLP_WEIGHT_DECAY": 1e-6,
    "MLP_MAX_EPOCHS": 500,
    "MLP_PATIENCE": 50,
    "MLP_GRAD_CLIP": 5.0,
    "FINETUNE_LR": 1.5e-4,                    # Pretrain Ft: all-layer fine-tuning
    "USE_GPU": True,                          # falls back to CPU automatically
    "USE_AMP": True,                          # only used on CUDA devices
    "PRINT_EACH_EPOCH": False,
}


def configure(**kwargs):
    """Override CONFIG entries (used by the command-line front ends)."""
    for key, value in kwargs.items():
        if value is None:
            continue
        if key not in CONFIG:
            raise KeyError(f"unknown configuration key: {key}")
        CONFIG[key] = value
    return CONFIG


def data_dir() -> Path:
    return Path(CONFIG["DATA_DIR"]).resolve()


def results_dir() -> Path:
    return Path(CONFIG["RESULTS_DIR"]).resolve()


def structure_dir() -> Path:
    return data_dir() / CONFIG["STRUCTURE_DIR"]

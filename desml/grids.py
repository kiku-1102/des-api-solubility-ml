"""
Hyperparameter search spaces and fixed settings (manuscript Table 4).

Every grid in this module is the one that was searched for the published
results; the totals are asserted so that an accidental edit is caught:

=================  ============  ==========================================
model              combinations  searched hyperparameters
=================  ============  ==========================================
nuSVR (RBF)        48            C x gamma x nu
nuSVR (poly)       48            C x gamma x nu   (degree = 4, coef0 = 1.0)
RF                 24            n_estimators x max_depth x min_samples_leaf
                                 x max_features
XGB                54            learning_rate x max_depth x min_child_weight
                                 x colsample_bytree
LGB                54            learning_rate x num_leaves x
                                 min_child_samples x colsample_bytree
MLP / GNN head     48            hidden layers x per-layer dropout x batch
GIN encoder        24            hidden x layers x pool x dropout
GATv2 encoder      16            hidden x layers x heads x dropout
=================  ============  ==========================================

The search is run once, on the DES-only regime with split seed 42 and without
cross-validation; the winning values are then used unchanged in every study
and every regime (Section 2.6.3).
"""

from itertools import product
from typing import Dict, List

MODELS = ("nusvr_rbf", "nusvr_poly", "rf", "xgb", "lgb", "mlp", "gin", "gatv2")


# ------------------------------------------------------------------ nuSVR
def build_nusvr_grid(kernel: str) -> List[Dict]:
    if kernel == "rbf":
        Cs, gammas, nus = [0.1, 1.0, 10.0, 100.0], ["scale", 0.01, 0.1, 1.0], [0.3, 0.5, 0.7]
    elif kernel == "poly":
        Cs, gammas, nus = [1.0, 5.0, 10.0, 50.0], ["scale", 0.001, 0.01, 0.1], [0.3, 0.5, 0.7]
    else:
        raise ValueError(f"invalid nuSVR kernel: {kernel} (rbf/poly)")
    return [{"C": c, "gamma": g, "nu": nu} for c in Cs for g in gammas for nu in nus]


# --------------------------------------------------------------------- RF
def build_rf_grid() -> List[Dict]:
    grid = []
    for n_est in [300, 500]:
        for md in [None, 20]:
            for msl in [1, 2, 5]:
                for mf in ["sqrt", 0.3]:
                    grid.append({"n_estimators": n_est, "max_depth": md,
                                 "min_samples_leaf": msl, "max_features": mf})
    return grid


# -------------------------------------------------------------------- XGB
def build_xgb_grid() -> List[Dict]:
    grid = []
    for lr in [0.03, 0.05, 0.1]:
        for md in [4, 6, 8]:
            for mcw in [1, 3, 5]:
                for cs in [0.7, 0.9]:
                    grid.append({"learning_rate": lr, "max_depth": md,
                                 "min_child_weight": mcw, "colsample_bytree": cs})
    return grid


# -------------------------------------------------------------------- LGB
def build_lgb_grid() -> List[Dict]:
    grid = []
    for lr in [0.03, 0.05, 0.1]:
        for nl in [31, 63, 127]:
            for mcs in [10, 20, 50]:
                for cs in [0.7, 0.9]:
                    grid.append({"learning_rate": lr, "num_leaves": nl,
                                 "min_child_samples": mcs, "colsample_bytree": cs})
    return grid


# ------------------------------------------------- MLP (and the GNN head)
def build_mlp_grid() -> List[Dict]:
    """Four architectures x per-layer dropout {0, 0.1} x batch size {32, 64}.

    The dropout rate is searched independently for every hidden layer, which
    gives 2**2 x 2 = 8 candidates for each two-layer architecture and
    2**3 x 2 = 16 for each three-layer one, 48 in total.
    """
    architectures = [[512, 256], [256, 128], [512, 256, 128], [256, 128, 64]]
    grid = []
    for dims in architectures:
        for dps in product([0.0, 0.1], repeat=len(dims)):
            for bs in [32, 64]:
                grid.append({"n_layers": len(dims), "dims": list(dims),
                             "dropouts": list(dps), "batch_size": bs})
    return grid


build_head_grid = build_mlp_grid           # the GNN head uses the same grid


# ------------------------------------------------------- GNN encoders
def build_encoder_grid(arch: str) -> List[Dict]:
    grid = []
    if arch == "GIN":
        for hidden in [128, 256]:
            for nl in [2, 3, 5]:
                for pool in ["mean", "add"]:
                    for do in [0.0, 0.1]:
                        grid.append({"hidden": hidden, "n_layers": nl,
                                     "pool": pool, "dropout": do})
    elif arch == "GATv2":
        for hidden in [128, 256]:
            for nl in [2, 3]:
                for heads in [4, 8]:
                    for do in [0.0, 0.1]:
                        grid.append({"hidden": hidden, "n_layers": nl, "heads": heads,
                                     "dropout": do, "pool": "mean", "residual": True})
    else:
        raise ValueError(f"unknown GNN architecture: {arch} (GIN/GATv2)")
    return grid


def default_encoder(arch: str) -> Dict:
    """Encoder kept fixed while the head is searched (stage Ia of Section 2.6.3)."""
    if arch == "GIN":
        return {"hidden": 128, "n_layers": 3, "pool": "mean", "dropout": 0.1}
    return {"hidden": 128, "n_layers": 2, "heads": 4, "dropout": 0.1,
            "pool": "mean", "residual": True}


def build_grid(model: str) -> List[Dict]:
    """Search space of one model name (see ``MODELS``)."""
    model = model.lower()
    if model == "nusvr_rbf":
        return build_nusvr_grid("rbf")
    if model == "nusvr_poly":
        return build_nusvr_grid("poly")
    if model == "rf":
        return build_rf_grid()
    if model == "xgb":
        return build_xgb_grid()
    if model == "lgb":
        return build_lgb_grid()
    if model in ("mlp", "gin", "gatv2"):
        return build_mlp_grid()
    raise ValueError(f"unknown model: {model} (expected one of {MODELS})")


# ------------------------------------------------------------- formatting
def hp_to_string(model: str, p: Dict) -> str:
    """Human-readable hyperparameter string for the log files."""
    model = model.lower()
    if model.startswith("nusvr"):
        return f"C={p['C']} gamma={p['gamma']} nu={p['nu']}"
    if model == "rf":
        return (f"n_est={p['n_estimators']} max_depth={p['max_depth']} "
                f"min_leaf={p['min_samples_leaf']} max_feat={p['max_features']}")
    if model == "xgb":
        return (f"lr={p['learning_rate']} max_depth={p['max_depth']} "
                f"min_child_w={p['min_child_weight']} colsample={p['colsample_bytree']}")
    if model == "lgb":
        return (f"lr={p['learning_rate']} num_leaves={p['num_leaves']} "
                f"min_child_samples={p['min_child_samples']} colsample={p['colsample_bytree']}")
    return f"dims={p['dims']} dropouts={p['dropouts']} batch={p['batch_size']}"


def hp_to_id(model: str, p: Dict) -> str:
    """Short identifier used as a dictionary key and in file names."""
    model = model.lower()
    if model.startswith("nusvr"):
        return f"C{p['C']}_g{p['gamma']}_nu{p['nu']}"
    if model == "rf":
        return (f"n{p['n_estimators']}_d{p['max_depth']}"
                f"_l{p['min_samples_leaf']}_f{p['max_features']}")
    if model == "xgb":
        return (f"lr{p['learning_rate']}_d{p['max_depth']}"
                f"_w{p['min_child_weight']}_c{p['colsample_bytree']}")
    if model == "lgb":
        return (f"lr{p['learning_rate']}_nl{p['num_leaves']}"
                f"_mcs{p['min_child_samples']}_c{p['colsample_bytree']}")
    d = "-".join(map(str, p["dims"]))
    dp = "-".join(f"{x:g}" for x in p["dropouts"])
    return f"d{d}_dp{dp}_b{p['batch_size']}"


def enc_hp_to_string(e: Dict, arch: str) -> str:
    if arch == "GIN":
        return f"hid={e['hidden']} L={e['n_layers']} pool={e['pool']} do={e['dropout']}"
    return (f"hid={e['hidden']} L={e['n_layers']} heads={e['heads']} "
            f"do={e['dropout']} res={e.get('residual', True)}")


def enc_hp_to_id(e: Dict, arch: str) -> str:
    if arch == "GIN":
        return f"h{e['hidden']}_L{e['n_layers']}_{e['pool']}_d{e['dropout']}"
    return (f"h{e['hidden']}_L{e['n_layers']}_H{e['heads']}_d{e['dropout']}"
            f"_r{int(e.get('residual', True))}")


# grid sizes of Table 4 - a guard against accidental edits
_EXPECTED_SIZES = {"nusvr_rbf": 48, "nusvr_poly": 48, "rf": 24, "xgb": 54,
                   "lgb": 54, "mlp": 48, "gin": 48, "gatv2": 48}
for _m, _n in _EXPECTED_SIZES.items():
    assert len(build_grid(_m)) == _n, f"{_m}: grid size changed"
assert len(build_encoder_grid("GIN")) == 24
assert len(build_encoder_grid("GATv2")) == 16

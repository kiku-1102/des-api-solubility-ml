"""
Training regimes, validation carve-out and the grid-search split
(manuscript Sections 2.6.1, 2.6.2 and 2.6.3).

Four training regimes are supported:

``prior``
    all data (DES + general solvents) pooled and split 8:2 - the scheme of the
    previous study, used by Study I.
``des_only``
    the DES systems alone split 8:2.
``joint``
    the DES training portion pooled with the whole of the general solvent data;
    the test set is the DES test portion.
``pretrain_ft``
    pre-training on the general solvents, then fine-tuning all layers on the
    DES training portion (neural networks only).

Because ``des_only``, ``joint`` and ``pretrain_ft`` derive their DES split from
the same seed, their test sets are identical and can be compared directly.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from .config import CONFIG

REGIMES = ("prior", "des_only", "joint", "pretrain_ft")


def regime_split(regime, is_des, n, data_seed):
    """Index sets of one training regime for one random-split seed."""
    all_idx = np.arange(n)
    des_idx = all_idx[is_des]
    gen_idx = all_idx[~is_des]
    ts = CONFIG["TEST_SIZE"]
    if regime == "prior":
        tr, te = train_test_split(all_idx, test_size=ts, random_state=data_seed)
        return {"kind": "single", "train_idx": tr, "test_idx": te, "scatter_train_idx": tr}
    des_tr, des_te = train_test_split(des_idx, test_size=ts, random_state=data_seed)
    if regime == "des_only":
        return {"kind": "single", "train_idx": des_tr, "test_idx": des_te,
                "scatter_train_idx": des_tr}
    if regime == "joint":
        train_idx = np.concatenate([des_tr, gen_idx])
        return {"kind": "single", "train_idx": train_idx, "test_idx": des_te,
                "scatter_train_idx": des_tr}
    if regime == "pretrain_ft":
        return {"kind": "pretrain", "pre_idx": gen_idx, "ft_idx": des_tr,
                "test_idx": des_te, "scatter_train_idx": des_tr}
    raise ValueError(f"unknown regime: {regime} (expected one of {REGIMES})")


def carve_validation(regime, train_idx, is_des, data_seed):
    """Split 10 % of the training data off for early stopping.

    In the Joint regime the validation points are taken from the DES training
    portion only, because the test target is the DES data; the whole of the
    general solvent data stays in the fit set (Section 2.6.2).

    Returns ``(idx_fit, idx_val)``.
    """
    vf = CONFIG["VAL_FRACTION"]
    if regime == "joint" and CONFIG["VAL_DES_ONLY_IN_JOINT"]:
        des_local = np.where(is_des[train_idx])[0]
        gen_local = np.where(~is_des[train_idx])[0]
        fit_des, val_des = train_test_split(des_local, test_size=vf, random_state=data_seed)
        in_local = np.concatenate([fit_des, gen_local])
        va_local = val_des
    else:
        pos = np.arange(len(train_idx))
        in_local, va_local = train_test_split(pos, test_size=vf, random_state=data_seed)
    return train_idx[in_local], train_idx[va_local]


def grid_search_split(n, val_fraction):
    """Split used to score the hyperparameter candidates (Section 2.6.3).

    The test part of the seed-42 split is sealed off first, so no candidate is
    ever scored on test data; the remaining training part is then divided into
    an inner fit set and an inner validation set.

    ``val_fraction`` is 0.2 for the descriptor models (nuSVR, RF, XGB, LGB) and
    0.1 for the neural networks, reproducing the published runs; the values
    live in ``CONFIG["GS_VAL_FRACTION_DESCRIPTOR"]`` and
    ``CONFIG["GS_VAL_FRACTION_NN"]``.

    Returns ``(inner_fit_idx, inner_val_idx, n_train_42, n_test_42)``.
    """
    gs_seed = CONFIG["GRID_SEARCH_DATA_SEED"]
    full = np.arange(n)
    tr42, te42 = train_test_split(full, test_size=CONFIG["TEST_SIZE"], random_state=gs_seed)
    in_pos, va_pos = train_test_split(np.arange(len(tr42)), test_size=val_fraction,
                                      random_state=gs_seed)
    return tr42[in_pos], tr42[va_pos], len(tr42), len(te42)

"""
Multilayer perceptron (manuscript Section 2.5.1) and the training loop shared
with the GNN head.

Each hidden layer is Linear -> ReLU -> Dropout, the output layer has a single
unit and the network is optimised with Adam on a weighted mean-squared-error
loss.  Training uses early stopping on a validation set carved out of the
training data, automatic mixed precision on CUDA devices, and gradient-norm
clipping; learning rate, weight decay, epoch limit, patience and the clipping
norm are the fixed settings of Table 4.
"""

import copy
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..config import CONFIG
from ..features import TiedScaler, augment_swaps
from ..splits import carve_validation, regime_split


def set_seed(seed: int):
    """Seed Python, NumPy and torch (CPU and CUDA)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """CUDA when available and enabled, otherwise CPU."""
    return "cuda" if (CONFIG["USE_GPU"] and torch.cuda.is_available()) else "cpu"


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, dims: List[int], dropouts: List[float]):
        super().__init__()
        layers = []
        prev = input_dim
        for d, dp in zip(dims, dropouts):
            layers += [nn.Linear(prev, d), nn.ReLU()]
            if dp and dp > 0:
                layers.append(nn.Dropout(dp))
            prev = d
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_in, y_in, X_va, y_va, hp, weight_seed, device, sample_weight=None,
              init_state=None, lr=None, tag="", verbose=False):
    """Train with a weighted mean-squared-error loss and early stopping.

    The sample weights come from the slot-exchange augmentation (1.0 for a row
    with a fixed role, 0.5 for each of two views).  The state with the lowest
    validation loss is kept and restored at the end.  Passing ``init_state``
    warm-starts the network, which is how the Pretrain Ft regime fine-tunes all
    layers (Section 2.6.1).

    Returns ``(best_model, best_epoch, train_history, val_history)``.
    """
    torch.manual_seed(weight_seed)
    np.random.seed(weight_seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(weight_seed)

    input_dim = X_in.shape[1]
    model = MLPNet(input_dim, hp["dims"], hp["dropouts"]).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)

    lr = lr if lr is not None else CONFIG["MLP_LR"]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CONFIG["MLP_WEIGHT_DECAY"])
    use_amp = (device == "cuda" and CONFIG["USE_AMP"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    Xt = torch.tensor(X_in, dtype=torch.float32)
    yt = torch.tensor(y_in, dtype=torch.float32)
    wt = torch.tensor(sample_weight if sample_weight is not None else np.ones(len(y_in)),
                      dtype=torch.float32)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=device)
    yv = torch.tensor(y_va, dtype=torch.float32, device=device)

    n = len(yt)
    bs = int(hp["batch_size"])
    best_val = np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0
    train_hist, val_hist = [], []

    for epoch in range(CONFIG["MLP_MAX_EPOCHS"]):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, bs):
            bidx = perm[i:i + bs]
            xb = Xt[bidx].to(device); yb = yt[bidx].to(device); wb = wt[bidx].to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb)
                loss = (wb * (pred - yb) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["MLP_GRAD_CLIP"])
            scaler.step(opt); scaler.update()
            ep_loss += float(loss.detach()) * len(bidx)
        ep_loss /= n

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                vp = model(Xv)
                vloss = float(((vp - yv) ** 2).mean())
        train_hist.append(ep_loss); val_hist.append(vloss)
        if verbose and CONFIG["PRINT_EACH_EPOCH"]:
            print(f"    [{tag}] epoch {epoch+1} train={ep_loss:.4f} val={vloss:.4f}")

        if vloss < best_val - 1e-7:
            best_val = vloss; best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CONFIG["MLP_PATIENCE"]:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, train_hist, val_hist


def predict_mlp(model, X, device) -> np.ndarray:
    model.eval()
    use_amp = (device == "cuda" and CONFIG["USE_AMP"])
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(Xt)
        return out.float().cpu().numpy().ravel()


def _scale_fit_transform(X_all, y_all, idx_in, idx_va, tied_idx_pairs, swappable_all):
    """Standardise on ``idx_in`` only, then augment; the validation set is
    transformed with the same scalers (no leakage)."""
    sx = TiedScaler(tied_idx_pairs).fit(X_all[idx_in])
    sy = StandardScaler().fit(y_all[idx_in].reshape(-1, 1))
    X_in = sx.transform(X_all[idx_in]); X_va = sx.transform(X_all[idx_va])
    y_in = sy.transform(y_all[idx_in].reshape(-1, 1)).ravel()
    y_va = sy.transform(y_all[idx_va].reshape(-1, 1)).ravel()
    swap_in = swappable_all[idx_in] if swappable_all is not None else None
    Xa, ya, wa = augment_swaps(X_in, y_in, swap_in, tied_idx_pairs)
    return sx, sy, Xa, ya, wa, X_va, y_va


def evaluate_one_pair_regime(regime, X_all, y_all, meta_df, is_des, data_seed, weight_seed,
                             best_hp, device, tied_idx_pairs, swappable_all, verbose=False):
    """Train and predict one (data seed, weight seed) pair.

    Predictions are returned on the original log x scale; the caller averages
    the five weight seeds into the ensemble prediction (Section 2.6.1).
    """
    sp = regime_split(regime, is_des, len(meta_df), data_seed)
    scatter_train_idx = sp["scatter_train_idx"]
    test_idx = sp["test_idx"]
    t0 = time.time()

    if sp["kind"] == "single":
        train_idx = sp["train_idx"]
        idx_in, idx_va = carve_validation(regime, train_idx, is_des, data_seed)
        sx, sy, Xa, ya, wa, X_va, y_va = _scale_fit_transform(
            X_all, y_all, idx_in, idx_va, tied_idx_pairs, swappable_all)
        model, best_epoch, _, _ = train_mlp(
            Xa, ya, X_va, y_va, hp=best_hp, weight_seed=weight_seed, device=device,
            sample_weight=wa, tag=f"{regime} d{data_seed} w{weight_seed}", verbose=verbose)
        info = f"(best_epoch={best_epoch}, n_val={len(idx_va)})"

    else:  # pretrain_ft: pre-train on the general solvents, then fine-tune on DES
        pre_idx, ft_idx = sp["pre_idx"], sp["ft_idx"]
        # ---- pre-training: the X scaler is fitted on the general solvent data
        pin, pva = carve_validation("pretrain", pre_idx, is_des, data_seed)
        sx = TiedScaler(tied_idx_pairs).fit(X_all[pre_idx])
        sy_pre = StandardScaler().fit(y_all[pin].reshape(-1, 1))
        X_pin = sx.transform(X_all[pin]); X_pva = sx.transform(X_all[pva])
        y_pin = sy_pre.transform(y_all[pin].reshape(-1, 1)).ravel()
        y_pva = sy_pre.transform(y_all[pva].reshape(-1, 1)).ravel()
        swp_p = swappable_all[pin] if swappable_all is not None else None
        Xpa, ypa, wpa = augment_swaps(X_pin, y_pin, swp_p, tied_idx_pairs)
        pre_model, pre_ep, _, _ = train_mlp(
            Xpa, ypa, X_pva, y_pva, hp=best_hp, weight_seed=weight_seed, device=device,
            sample_weight=wpa, tag=f"pretrain d{data_seed} w{weight_seed}", verbose=verbose)
        pre_state = copy.deepcopy(pre_model.state_dict())
        del pre_model
        if device == "cuda":
            torch.cuda.empty_cache()
        # ---- fine-tuning of all layers: X scaler kept, y re-standardised on DES
        fin, fva = carve_validation("pretrain", ft_idx, is_des, data_seed)
        sy = StandardScaler().fit(y_all[fin].reshape(-1, 1))
        X_fin = sx.transform(X_all[fin]); X_fva = sx.transform(X_all[fva])
        y_fin = sy.transform(y_all[fin].reshape(-1, 1)).ravel()
        y_fva = sy.transform(y_all[fva].reshape(-1, 1)).ravel()
        swp_f = swappable_all[fin] if swappable_all is not None else None   # DES: no swap
        Xfa, yfa, wfa = augment_swaps(X_fin, y_fin, swp_f, tied_idx_pairs)
        model, ft_ep, _, _ = train_mlp(
            Xfa, yfa, X_fva, y_fva, hp=best_hp, weight_seed=weight_seed, device=device,
            sample_weight=wfa, init_state=pre_state, lr=CONFIG["FINETUNE_LR"],
            tag=f"finetune d{data_seed} w{weight_seed}", verbose=verbose)
        info = f"(pre_ep={pre_ep}, ft_ep={ft_ep})"

    fit_elapsed = time.time() - t0
    yp_tr = sy.inverse_transform(
        predict_mlp(model, sx.transform(X_all[scatter_train_idx]), device).reshape(-1, 1)).ravel()
    yp_te = sy.inverse_transform(
        predict_mlp(model, sx.transform(X_all[test_idx]), device).reshape(-1, 1)).ravel()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "data_seed": data_seed, "weight_seed": weight_seed,
        "fit_elapsed": fit_elapsed, "info": info,
        "scatter_train_idx": scatter_train_idx, "test_idx": test_idx,
        "yp_tr": yp_tr, "yp_te": yp_te,
    }

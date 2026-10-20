"""
Graph neural networks: GIN and GATv2 encoders with an MLP head
(manuscript Section 2.5.2).

The GIN encoder uses GINEConv layers that take the edge features into account
(each layer is Linear -> ReLU -> Linear with a learnable ``train_eps``),
followed by batch normalisation, ReLU, dropout and a residual connection, and
aggregates the graph with mean or sum pooling.  The GATv2 encoder uses
multi-head GATv2Conv layers (self-loops, edge features, hidden dimension
divisible by the number of heads) with batch normalisation, ELU, dropout,
residual connections and mean pooling.

Four configurations are reproduced, exactly the four equations of the paper::

    graph_combine  desc_combine   manuscript
    -------------  -------------  ------------------------------------------
    concat         none           Eq. (4)  GNN (graph only) / Concat
    mixture        none           Eq. (5)  GNN (graph only) / Mixture
    concat         concat         Eq. (6)  GNN+MLP / Concat
    mixture        mixture        Eq. (7)  GNN+MLP / Mixture

The hyperparameters are searched in the graph-only, DES-only configuration in
two stages (head first, then encoder) and are shared by all four
configurations of the same architecture (Section 2.6.3).
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, GINEConv, global_add_pool, global_mean_pool

from ..config import CONFIG
from ..features import (TiedScaler, build_feature_table_concat, compute_eff_desc_df)
from ..graphs import ATOM_FDIM, BOND_FDIM, GraphBank
from ..models.mlp import set_seed
from ..splits import carve_validation, regime_split

#: the four configurations of Eqs. (4)-(7)
GNN_CONFIGURATIONS = {
    ("concat", "none"): "Eq. (4) GNN (graph only) / Concat",
    ("mixture", "none"): "Eq. (5) GNN (graph only) / Mixture",
    ("concat", "concat"): "Eq. (6) GNN+MLP / Concat",
    ("mixture", "mixture"): "Eq. (7) GNN+MLP / Mixture",
}


@dataclass
class GnnContext:
    """Everything a GNN run needs besides the design matrix itself."""

    arch: str                      # "GIN" or "GATv2"
    graph_combine: str             # "concat" or "mixture"
    desc_combine: str              # "concat", "mixture" or "none"
    graph_bank: GraphBank = None
    packs: Tuple = None            # (DRG, C2, C3, W2, W3, WW)
    encoder_hp: Dict = field(default_factory=dict)
    water_data: object = None

    def describe(self) -> str:
        key = (self.graph_combine, self.desc_combine)
        return GNN_CONFIGURATIONS.get(key, f"{self.graph_combine}/{self.desc_combine}")


class GINEncoder(nn.Module):
    def __init__(self, atom_fdim, bond_fdim, hidden, n_layers, dropout, pool):
        super().__init__()
        self.atom_encoder = nn.Linear(atom_fdim, hidden)
        self.bond_encoder = nn.Linear(bond_fdim, hidden)
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(),
                                nn.Linear(2 * hidden, hidden))
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.pool = global_add_pool if pool == "add" else global_mean_pool

    def forward(self, data: Batch) -> torch.Tensor:
        x = self.atom_encoder(data.x)
        ea = self.bond_encoder(data.edge_attr) if data.edge_attr.numel() > 0 \
            else data.edge_attr.new_zeros((0, x.shape[1]))
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, data.edge_index, ea); h = bn(h); h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        return self.pool(x, data.batch)


class GATv2Encoder(nn.Module):
    def __init__(self, atom_fdim, bond_fdim, hidden, n_layers, heads, dropout, residual, pool):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by the number of heads"
        self.atom_encoder = nn.Linear(atom_fdim, hidden)
        self.bond_encoder = nn.Linear(bond_fdim, hidden)
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GATv2Conv(
                in_channels=hidden, out_channels=hidden // heads, heads=heads,
                concat=True, edge_dim=hidden, add_self_loops=True,
                fill_value=0.0, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout; self.residual = residual
        self.pool = global_add_pool if pool == "add" else global_mean_pool

    def forward(self, data: Batch) -> torch.Tensor:
        x = self.atom_encoder(data.x)
        ea = self.bond_encoder(data.edge_attr) if data.edge_attr.numel() > 0 \
            else data.edge_attr.new_zeros((0, x.shape[1]))
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, data.edge_index, ea); h = bn(h); h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if self.residual else h
        return self.pool(x, data.batch)


def make_encoder(arch: str, e: Dict) -> nn.Module:
    if arch == "GIN":
        return GINEncoder(ATOM_FDIM, BOND_FDIM, int(e["hidden"]), int(e["n_layers"]),
                          float(e["dropout"]), e["pool"])
    if arch == "GATv2":
        return GATv2Encoder(ATOM_FDIM, BOND_FDIM, int(e["hidden"]), int(e["n_layers"]),
                            int(e["heads"]), float(e["dropout"]),
                            bool(e.get("residual", True)), e.get("pool", "mean"))
    raise ValueError(f"Unknown GNN_ARCH: {arch}")


def _build_head(in_dim, dims, dropouts):
    layers = []; prev = in_dim
    for d, do in zip(dims, dropouts):
        layers += [nn.Linear(prev, d), nn.ReLU()]
        if do and do > 0:
            layers.append(nn.Dropout(do))
        prev = d
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class GNNRegressor(nn.Module):
    """Graph encoder(s) plus an MLP head (manuscript Eqs. 4-7).

    ``combine="concat"``
        [enc(API) || enc(slot2) || enc(slot3)] || tabular features
    ``combine="mixture"``
        [enc(API) || (x2 enc(slot2) + x3 enc(slot3) + xw enc(water))] || tabular

    The encoder weights are shared across all component slots.  The tabular
    part is the condition features alone (graph-only), the concatenated
    descriptors (Eq. 6) or the mole-fraction weighted descriptors (Eq. 7).
    """
    def __init__(self, arch, enc_hp, tab_dim, head_dims, head_dropouts, combine):
        super().__init__()
        self.encoder = make_encoder(arch, enc_hp)
        self.gnn_out = int(enc_hp["hidden"])
        self.combine = combine
        n_slots = 2 if combine == "mixture" else 3
        self.head = _build_head(n_slots * self.gnn_out + tab_dim, head_dims, head_dropouts)

    def forward(self, b_drg, m_drg, b_s2, m_s2, b_s3, m_s3, x_tab,
                b_w=None, m_w=None, w2=None, w3=None, ww=None):
        e_drg = self.encoder(b_drg) * m_drg.unsqueeze(1)
        e_s2 = self.encoder(b_s2) * m_s2.unsqueeze(1)
        e_s3 = self.encoder(b_s3) * m_s3.unsqueeze(1)
        if self.combine == "mixture":
            e_w = self.encoder(b_w) * m_w.unsqueeze(1)
            e_mix = (w2.unsqueeze(1) * e_s2 + w3.unsqueeze(1) * e_s3 + ww.unsqueeze(1) * e_w)
            g = torch.cat([e_drg, e_mix], dim=1)
        else:
            g = torch.cat([e_drg, e_s2, e_s3], dim=1)
        fused = torch.cat([g, x_tab], dim=1) if (x_tab is not None and x_tab.shape[1] > 0) else g
        return self.head(fused)


def _minibatch(gb, pack, X_tab, y, idx, device, combine, water_data):
    drg, c2, c3, w2, w3, ww = pack
    b_drg, m_drg = gb.build_batch([drg[i] for i in idx], device)
    b_s2, m_s2 = gb.build_batch([c2[i] for i in idx], device)
    b_s3, m_s3 = gb.build_batch([c3[i] for i in idx], device)
    xt = torch.tensor(X_tab[idx].astype(np.float32), device=device)
    yt = None if y is None else torch.tensor(y[idx].astype(np.float32), device=device).unsqueeze(1)
    extra = {}
    if combine == "mixture":
        b_w, m_w = gb.water_batch(len(idx), device)
        extra = dict(b_w=b_w, m_w=m_w,
                     w2=torch.tensor(w2[idx].astype(np.float32), device=device),
                     w3=torch.tensor(w3[idx].astype(np.float32), device=device),
                     ww=torch.tensor(ww[idx].astype(np.float32), device=device))
    return b_drg, m_drg, b_s2, m_s2, b_s3, m_s3, xt, yt, extra


def train_gnn(gb, pack_tr, X_tr, y_tr, pack_va, X_va, y_va,
              arch, enc_hp, head_dims, head_dropouts, batch_size, weight_seed, device,
              combine, water_data, sample_weight=None, init_state=None, lr=None,
              tag="", verbose=False):
    set_seed(weight_seed)
    model = GNNRegressor(arch, enc_hp, X_tr.shape[1], head_dims, head_dropouts, combine).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(),
                           lr=(CONFIG["MLP_LR"] if lr is None else lr),
                           weight_decay=CONFIG["MLP_WEIGHT_DECAY"])
    use_amp = CONFIG["USE_AMP"] and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    bs = int(batch_size); n_tr, n_va = len(y_tr), len(y_va)
    if sample_weight is None:
        sample_weight = np.ones(n_tr, np.float64)
    sw_t = torch.tensor(sample_weight.astype(np.float32), device=device)
    best_val, best_epoch, best_state, bad = float("inf"), 0, None, 0
    val_chunks = [np.arange(s, min(s + 512, n_va)) for s in range(0, n_va, 512)]

    for ep in range(1, CONFIG["MLP_MAX_EPOCHS"] + 1):
        model.train()
        perm = np.random.permutation(n_tr)
        for st in range(0, n_tr, bs):
            idx = perm[st:st + bs]
            if len(idx) <= 1:           # BatchNorm1d needs a batch of >= 2
                continue
            b_drg, m_drg, b_s2, m_s2, b_s3, m_s3, xt, yt, ex = _minibatch(
                gb, pack_tr, X_tr, y_tr, idx, device, combine, water_data)
            wb = sw_t[idx]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(b_drg, m_drg, b_s2, m_s2, b_s3, m_s3, xt, **ex)
                se = ((pred - yt) ** 2).squeeze(1)
                loss = (wb * se).sum() / wb.sum().clamp_min(1e-8)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["MLP_GRAD_CLIP"])
            scaler.step(opt); scaler.update()

        model.eval()
        se_sum, ns = 0.0, 0
        with torch.no_grad():
            for vi in val_chunks:
                b_drg, m_drg, b_s2, m_s2, b_s3, m_s3, xt, yt, ex = _minibatch(
                    gb, pack_va, X_va, y_va, vi, device, combine, water_data)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(b_drg, m_drg, b_s2, m_s2, b_s3, m_s3, xt, **ex)
                se_sum += float(((pred - yt) ** 2).sum()); ns += len(vi)
        val_loss = se_sum / max(ns, 1)
        if verbose and CONFIG["PRINT_EACH_EPOCH"]:
            print(f"    [{tag}] ep {ep:3d} val={val_loss:.6f}", flush=True)
        if val_loss < best_val - 1e-9:
            best_val, best_epoch, bad = val_loss, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= CONFIG["MLP_PATIENCE"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def predict_gnn(model, gb, pack, X_tab, device, combine, water_data, batch=512) -> np.ndarray:
    model.eval()
    n = X_tab.shape[0]
    use_amp = CONFIG["USE_AMP"] and device == "cuda"
    preds = []
    with torch.no_grad():
        for st in range(0, n, batch):
            idx = np.arange(st, min(st + batch, n))
            b_drg, m_drg, b_s2, m_s2, b_s3, m_s3, xt, _, ex = _minibatch(
                gb, pack, X_tab, None, idx, device, combine, water_data)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(b_drg, m_drg, b_s2, m_s2, b_s3, m_s3, xt, **ex)
            preds.append(pred.squeeze(-1).cpu().float().numpy().astype(np.float64))
    return np.concatenate(preds) if preds else np.zeros(0)


def augment_swaps_graph(X, drg, c2, c3, y, swappable, tied_idx_pairs):
    """Slot-exchange augmentation for the graph models without mixture weights.

    The graph codes of slot2/slot3 and the tied tabular columns are exchanged
    together; used by the hyperparameter search (graph-only configuration).
    """
    n = len(y)
    swappable = np.zeros(n, bool) if swappable is None else np.asarray(swappable, bool)
    if not swappable.any():
        return X, drg, c2, c3, y, np.ones(n, np.float64)
    sw = swappable; nsw = ~sw
    pX = [X[nsw], X[sw], X[sw].copy()]
    if tied_idx_pairs:
        ii = [a for a, b in tied_idx_pairs]; jj = [b for a, b in tied_idx_pairs]
        tmp = pX[2][:, ii].copy(); pX[2][:, ii] = pX[2][:, jj]; pX[2][:, jj] = tmp
    pdrg = [drg[nsw], drg[sw], drg[sw]]
    pc2 = [c2[nsw], c2[sw], c3[sw]]      # swapped view: c2<->c3
    pc3 = [c3[nsw], c3[sw], c2[sw]]
    py = [y[nsw], y[sw], y[sw]]
    pw = [np.ones(int(nsw.sum()), np.float64),
          0.5 * np.ones(int(sw.sum()), np.float64),
          0.5 * np.ones(int(sw.sum()), np.float64)]
    return (np.concatenate(pX), np.concatenate(pdrg), np.concatenate(pc2),
            np.concatenate(pc3), np.concatenate(py), np.concatenate(pw))


def augment_swaps_graph_weighted(X, drg, c2, c3, w2, w3, ww, y, swappable, tied_idx_pairs):
    """Slot-exchange augmentation that also exchanges the mixture weights.

    In the swapped view the graph codes c2/c3 and the mole fractions w2/w3 are
    exchanged simultaneously (the water fraction is unchanged), and the tied
    columns of the tabular block are exchanged as well.  This is the version
    used for every configuration of Eqs. (4)-(7).
    """
    n = len(y)
    swappable = np.zeros(n, bool) if swappable is None else np.asarray(swappable, bool)
    if not swappable.any():
        return (X, drg, c2, c3, w2, w3, ww, y, np.ones(n, np.float64))
    sw = swappable; nsw = ~sw
    pX = [X[nsw], X[sw], X[sw].copy()]
    if tied_idx_pairs:
        ii = [a for a, b in tied_idx_pairs]; jj = [b for a, b in tied_idx_pairs]
        tmp = pX[2][:, ii].copy(); pX[2][:, ii] = pX[2][:, jj]; pX[2][:, jj] = tmp
    pdrg = [drg[nsw], drg[sw], drg[sw]]
    pc2 = [c2[nsw], c2[sw], c3[sw]]      # swapped view: c2<->c3
    pc3 = [c3[nsw], c3[sw], c2[sw]]
    pw2 = [w2[nsw], w2[sw], w3[sw]]      # swapped view: w2<->w3
    pw3 = [w3[nsw], w3[sw], w2[sw]]
    pww = [ww[nsw], ww[sw], ww[sw]]
    py = [y[nsw], y[sw], y[sw]]
    pw = [np.ones(int(nsw.sum()), np.float64),
          0.5 * np.ones(int(sw.sum()), np.float64),
          0.5 * np.ones(int(sw.sum()), np.float64)]
    cat = np.concatenate
    return (cat(pX), cat(pdrg), cat(pc2), cat(pc3), cat(pw2), cat(pw3), cat(pww),
            cat(py), cat(pw))


def build_gnn_data(df, descriptors, manifest, graph_combine, desc_combine, logger):
    """Build the graph codes, mixture weights and tabular block of a GNN run.

    The concatenated table is built first to obtain the meta information
    (component codes, mole fractions, tied column pairs).  The graph side is
    then combined inside the model according to ``graph_combine``, while the
    tabular side is selected by ``desc_combine``:

    ``concat``   [API || slot2 || slot3] + conditions   (Eq. 6)
    ``mixture``  [API || weighted sum]   + conditions   (Eq. 7)
    ``none``     conditions only                        (Eqs. 4 and 5)
    """
    X_df, y, meta_df, fg = build_feature_table_concat(df, descriptors, manifest, logger)
    cond_cols = fg["cond_cols"]
    drg_cols = fg["drg_cols"]
    comp_pair = [("x_slot2", "x_slot3")] if ("x_slot2" in cond_cols and "x_slot3" in cond_cols) else []

    if desc_combine == "concat":
        cols = fg["all_cols"]
        X = X_df[cols].values.astype(np.float64)
        tied = fg["tied_pairs"]
    elif desc_combine == "mixture":
        eff_df, eff_cols = compute_eff_desc_df(meta_df, descriptors, manifest)
        cols = drg_cols + eff_cols + cond_cols
        X = np.concatenate([X_df[drg_cols].values, eff_df.values, X_df[cond_cols].values],
                           axis=1).astype(np.float64)
        tied = comp_pair            # the weighted sum is already order-invariant
    else:  # none: graph-only configuration
        cols = cond_cols
        X = X_df[cols].values.astype(np.float64)
        tied = comp_pair
    col_idx = {c: i for i, c in enumerate(cols)}
    tied_idx_pairs = [(col_idx[a], col_idx[b]) for a, b in tied]
    DRG = meta_df["DRG"].values.astype(object)
    C2 = meta_df["code_slot2"].values.astype(object)
    C3 = meta_df["code_slot3"].values.astype(object)
    W2 = meta_df["x_slot2"].values.astype(np.float64)
    W3 = meta_df["x_slot3"].values.astype(np.float64)
    WW = meta_df["x_water"].values.astype(np.float64)
    swappable = meta_df["swappable"].values.astype(bool)
    logger.log(f"  GNN data [graph={graph_combine}/desc={desc_combine}]: samples={len(meta_df)}, "
               f"tabular={X.shape[1]}, tied_pairs={len(tied_idx_pairs)}, "
               f"swappable={int(swappable.sum())}")
    return X, y, meta_df, DRG, C2, C3, W2, W3, WW, swappable, tied_idx_pairs


def score_candidate(gb, idx_in, idx_va, X_all, y_all, packs, swappable_all,
                    arch, enc_hp, head_hp, device, tied_idx_pairs, combine, water_data,
                    w_seed):
    """Validation RMSE of one head / encoder candidate (original log x scale)."""
    DRG, C2, C3, W2, W3, WW = packs
    sx = TiedScaler(tied_idx_pairs).fit(X_all[idx_in]); sy = StandardScaler().fit(y_all[idx_in].reshape(-1, 1))
    X_in = sx.transform(X_all[idx_in]); X_va = sx.transform(X_all[idx_va])
    y_in = sy.transform(y_all[idx_in].reshape(-1, 1)).ravel()
    y_va = sy.transform(y_all[idx_va].reshape(-1, 1)).ravel()
    if combine == "mixture":
        pin = (DRG[idx_in], C2[idx_in], C3[idx_in], W2[idx_in], W3[idx_in], WW[idx_in])
        ya, wa = y_in, None
        Xa = X_in
    else:
        swp = swappable_all[idx_in] if swappable_all is not None else None
        Xa, drga, c2a, c3a, ya, wa = augment_swaps_graph(
            X_in, DRG[idx_in], C2[idx_in], C3[idx_in], y_in, swp, tied_idx_pairs)
        z = np.zeros(len(ya), np.float64)
        pin = (drga, c2a, c3a, z, z, z)
    pva = (DRG[idx_va], C2[idx_va], C3[idx_va], W2[idx_va], W3[idx_va], WW[idx_va])
    model, _ = train_gnn(gb, pin, Xa, ya, pva, X_va, y_va, arch, enc_hp,
                         head_hp["dims"], head_hp["dropouts"], head_hp["batch_size"],
                         w_seed, device, combine, water_data, sample_weight=wa, tag="gs")
    yp = sy.inverse_transform(predict_gnn(model, gb, pva, X_va, device, combine, water_data).reshape(-1, 1)).ravel()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return float(np.sqrt(mean_squared_error(y_all[idx_va], yp)))


def evaluate_one_pair_regime(ctx, regime, X_all, y_all, meta_df, is_des, data_seed,
                             weight_seed, head_hp, device, tied_idx_pairs, swappable_all,
                             verbose=False):
    """Train and predict one (data seed, weight seed) pair of a GNN configuration."""
    gb = ctx.graph_bank
    DRG, C2, C3, W2, W3, WW = ctx.packs
    arch, enc_hp, combine, water = ctx.arch, ctx.encoder_hp, ctx.graph_combine, ctx.water_data
    tab_mode = ctx.desc_combine
    # augmentation is needed unless both sides are order-invariant (mixture)
    need_swap = (combine == "concat") or (tab_mode == "concat")
    hd, hdo, bs = head_hp["dims"], head_hp["dropouts"], head_hp["batch_size"]
    sp = regime_split(regime, is_des, len(meta_df), data_seed)
    scatter_train_idx, test_idx = sp["scatter_train_idx"], sp["test_idx"]
    t0 = time.time()

    def _prep_train(idx_in, idx_va, sx, sy):
        X_in = sx.transform(X_all[idx_in]); X_va = sx.transform(X_all[idx_va])
        y_in = sy.transform(y_all[idx_in].reshape(-1, 1)).ravel()
        y_va = sy.transform(y_all[idx_va].reshape(-1, 1)).ravel()
        if need_swap:
            swp = swappable_all[idx_in] if swappable_all is not None else None
            (Xa, drga, c2a, c3a, w2a, w3a, wwa, ya, wa) = augment_swaps_graph_weighted(
                X_in, DRG[idx_in], C2[idx_in], C3[idx_in],
                W2[idx_in], W3[idx_in], WW[idx_in], y_in, swp, tied_idx_pairs)
            if combine == "mixture":
                pin = (drga, c2a, c3a, w2a, w3a, wwa)   # mixture graph side: swapped weights
            else:
                z = np.zeros(len(ya), np.float64); pin = (drga, c2a, c3a, z, z, z)
        else:
            pin = (DRG[idx_in], C2[idx_in], C3[idx_in], W2[idx_in], W3[idx_in], WW[idx_in])
            Xa, ya, wa = X_in, y_in, None
        pva = (DRG[idx_va], C2[idx_va], C3[idx_va], W2[idx_va], W3[idx_va], WW[idx_va])
        return pin, Xa, ya, wa, pva, X_va, y_va

    if sp["kind"] == "single":
        idx_in, idx_va = carve_validation(regime, sp["train_idx"], is_des, data_seed)
        sx = TiedScaler(tied_idx_pairs).fit(X_all[idx_in])
        sy = StandardScaler().fit(y_all[idx_in].reshape(-1, 1))
        pin, Xa, ya, wa, pva, X_va, y_va = _prep_train(idx_in, idx_va, sx, sy)
        model, best_epoch = train_gnn(gb, pin, Xa, ya, pva, X_va, y_va, arch, enc_hp,
                                      hd, hdo, bs, weight_seed, device, combine, water,
                                      sample_weight=wa, tag=f"{regime} d{data_seed} w{weight_seed}",
                                      verbose=verbose)
        info = f"(best_epoch={best_epoch}, n_val={len(idx_va)})"
    else:  # pretrain_ft
        pre_idx, ft_idx = sp["pre_idx"], sp["ft_idx"]
        pin_i, pva_i = carve_validation("pretrain", pre_idx, is_des, data_seed)
        sx = TiedScaler(tied_idx_pairs).fit(X_all[pre_idx])          # X scaler: general solvents
        sy_pre = StandardScaler().fit(y_all[pin_i].reshape(-1, 1))
        pin, Xa, ya, wa, pva, X_va, y_va = _prep_train(pin_i, pva_i, sx, sy_pre)
        pre_model, pre_ep = train_gnn(gb, pin, Xa, ya, pva, X_va, y_va, arch, enc_hp,
                                      hd, hdo, bs, weight_seed, device, combine, water,
                                      sample_weight=wa, tag=f"pretrain d{data_seed} w{weight_seed}",
                                      verbose=verbose)
        pre_state = copy.deepcopy(pre_model.state_dict()); del pre_model
        if device == "cuda":
            torch.cuda.empty_cache()
        fin_i, fva_i = carve_validation("pretrain", ft_idx, is_des, data_seed)
        sy = StandardScaler().fit(y_all[fin_i].reshape(-1, 1))        # y re-standardised on DES
        pin, Xa, ya, wa, pva, X_va, y_va = _prep_train(fin_i, fva_i, sx, sy)
        model, ft_ep = train_gnn(gb, pin, Xa, ya, pva, X_va, y_va, arch, enc_hp,
                                 hd, hdo, bs, weight_seed, device, combine, water,
                                 sample_weight=wa, init_state=pre_state, lr=CONFIG["FINETUNE_LR"],
                                 tag=f"finetune d{data_seed} w{weight_seed}", verbose=verbose)
        info = f"(pre_ep={pre_ep}, ft_ep={ft_ep})"

    fit_elapsed = time.time() - t0
    pack_tr = (DRG[scatter_train_idx], C2[scatter_train_idx], C3[scatter_train_idx],
               W2[scatter_train_idx], W3[scatter_train_idx], WW[scatter_train_idx])
    pack_te = (DRG[test_idx], C2[test_idx], C3[test_idx],
               W2[test_idx], W3[test_idx], WW[test_idx])
    yp_tr = sy.inverse_transform(predict_gnn(model, gb, pack_tr, sx.transform(X_all[scatter_train_idx]),
                                             device, combine, water).reshape(-1, 1)).ravel()
    yp_te = sy.inverse_transform(predict_gnn(model, gb, pack_te, sx.transform(X_all[test_idx]),
                                             device, combine, water).reshape(-1, 1)).ravel()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"data_seed": data_seed, "weight_seed": weight_seed, "fit_elapsed": fit_elapsed,
            "info": info, "scatter_train_idx": scatter_train_idx, "test_idx": test_idx,
            "yp_tr": yp_tr, "yp_te": yp_te}

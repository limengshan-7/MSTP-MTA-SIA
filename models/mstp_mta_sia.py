"""Eqs. (3)-(9): pooled resolutions, TCN/BiLSTM, two-stage attention, MHA.

This implements the current paper formulation, not the legacy ensemble. It
does NOT add the legacy base-scale residual, static class-probability features,
clustering, XGBoost, auxiliary temporal penalties, or learned client-ID vectors.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


def downsample_time(x, factor):
    """Nonoverlapping means of [B,T,D]; ceil(T/factor), no zero-pad dilution."""
    if factor < 1:
        raise ValueError("factor must be positive")
    if factor == 1:
        return x
    length = x.shape[1]
    pad = (-length) % factor
    padded = F.pad(x, (0, 0, 0, pad))
    summed = padded.reshape(x.shape[0], -1, factor, x.shape[-1]).sum(dim=2)
    counts = torch.full((summed.shape[1],), factor, device=x.device, dtype=x.dtype)
    if pad:
        counts[-1] = factor - pad
    return summed / counts[None, :, None]


def positional_encoding(positions, width):
    """Sinusoidal encoding at original-round coordinates (pooled for a scale)."""
    rates = torch.exp(torch.arange(0, width, 2, device=positions.device,
                                   dtype=positions.dtype) * (-math.log(10000.0) / width))
    angles = positions[:, None] * rates[None, :]
    pe = torch.zeros((len(positions), width), device=positions.device,
                     dtype=positions.dtype)
    pe[:, 0::2] = angles.sin()
    pe[:, 1::2] = angles[:, :width // 2].cos()
    return pe


class TemporalConvBlock(nn.Module):
    """Legacy-style residual, symmetric dilated TCN within the observed window."""
    def __init__(self, width, hidden=128):
        super().__init__()
        layers, channels = [], width
        for dilation in (1, 2, 4):
            layers.extend([nn.Conv1d(channels, hidden, 3, padding=dilation,
                                     dilation=dilation),
                           nn.BatchNorm1d(hidden), nn.GELU()])
            channels = hidden
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden, width)

    def forward(self, x):
        return x + self.proj(self.net(x.transpose(1, 2)).transpose(1, 2))


class AttentionPool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(width, width), nn.Tanh(),
                                   nn.Linear(width, 1, bias=False))

    def forward(self, x):
        weights = self.score(x).softmax(dim=1)
        return (weights * x).sum(dim=1), weights.squeeze(-1)


class TemporalBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["hidden_dim"]
        self.tcn = TemporalConvBlock(width, cfg["tcn_width"]) if cfg["tcn"] else nn.Identity()
        self.lstm = nn.LSTM(width, width // 2, num_layers=cfg["lstm_layers"],
                            batch_first=True, bidirectional=True,
                            dropout=cfg["dropout"] if cfg["lstm_layers"] > 1 else 0.0)
        self.pool = AttentionPool(width) if cfg["temporal_attention"] else None

    def forward(self, x):
        h, _ = self.lstm(self.tcn(x))
        if self.pool is not None:
            return self.pool(h)
        weights = h.new_full((h.shape[0], h.shape[1]), 1.0 / h.shape[1])
        return h.mean(dim=1), weights


class MSTPMTASIA(nn.Module):
    """[B,K,T,6] -> [B,K] logits; CE applies softmax over candidate CLIENTS."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.scales = tuple(cfg["scales"])
        width = cfg["hidden_dim"]
        self.input_projection = nn.Linear(6, width, bias=False)  # W_e in Eq. (4)
        self.branches = nn.ModuleList([TemporalBranch(cfg) for _ in self.scales])
        self.scale_pool = AttentionPool(width) if cfg["scale_attention"] else None
        self.client_attention = (nn.MultiheadAttention(width, cfg["heads"], batch_first=True)
                                 if cfg["cross_client_attention"] else None)
        self.client_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                  nn.Dropout(cfg["dropout"]), nn.Linear(width, 1))

    def forward(self, x, return_details=False):
        if x.ndim != 4 or x.shape[-1] != 6:
            raise ValueError("Expected [B,K,T,6]")
        batch, clients, length, _ = x.shape
        flat = x.reshape(batch * clients, length, 6)
        positions = torch.arange(length, dtype=x.dtype, device=x.device).view(1, -1, 1)
        summaries, temporal_weights = [], []
        for scale, branch in zip(self.scales, self.branches):
            z = self.input_projection(downsample_time(flat, scale))
            if self.cfg["positional_encoding"]:
                p = downsample_time(positions, scale).flatten()
                z = z + positional_encoding(p, z.shape[-1])[None, :, :]
            summary, alpha = branch(z)
            summaries.append(summary)
            temporal_weights.append(alpha.reshape(batch, clients, -1))
        stacked = torch.stack(summaries, dim=1)
        if self.scale_pool is not None:
            fused, beta = self.scale_pool(stacked)
        else:
            fused = stacked.mean(dim=1)
            beta = stacked.new_full(stacked.shape[:2], 1.0 / len(self.scales))
        z = fused.reshape(batch, clients, -1)
        if self.client_attention is not None:
            update, _ = self.client_attention(z, z, z, need_weights=False)
            z = self.client_norm(z + update)
        else:
            z = self.client_norm(z)  # retain norm in the no-cross-attention control
        logits = self.head(z).squeeze(-1)
        if return_details:
            return logits, {"temporal_weights": temporal_weights,
                            "scale_weights": beta.reshape(batch, clients, -1)}
        return logits


class TemporalAverage(nn.Module):
    """Projected features/time encodings, mean over time, shared client scorer."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        width = cfg["hidden_dim"]
        self.project = nn.Linear(6, width, bias=False)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                  nn.Dropout(cfg["dropout"]), nn.Linear(width, 1))

    def forward(self, x):
        z = self.project(x)
        if self.cfg["positional_encoding"]:
            positions = torch.arange(x.shape[2], dtype=x.dtype, device=x.device)
            z = z + positional_encoding(positions, z.shape[-1])[None, None, :, :]
        return self.head(z.mean(dim=2)).squeeze(-1)


class DilatedResidual(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.net(x)


class StandardTCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        width = cfg["baseline_width"]
        self.project = nn.Linear(6, width, bias=False)
        self.tcn = nn.Sequential(*[DilatedResidual(width, d, cfg["dropout"])
                                   for d in (1, 2, 4)])
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                  nn.Dropout(cfg["dropout"]), nn.Linear(width, 1))

    def forward(self, x):
        b, k, t, _ = x.shape
        z = self.project(x.reshape(b * k, t, 6))
        if self.cfg["positional_encoding"]:
            positions = torch.arange(t, dtype=x.dtype, device=x.device)
            z = z + positional_encoding(positions, z.shape[-1])[None, :, :]
        z = self.tcn(z.transpose(1, 2)).mean(dim=-1)
        return self.head(z).reshape(b, k)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_tcn_width(target):
    # For StandardTCN above: W_e=6w; 6 convs=(18w^2+6w);
    # shared head=w^2+2w+1. Candidate widths are fixed before seeing data.
    return min(range(8, 1025, 8), key=lambda w: abs(19 * w * w + 14 * w + 1 - target))


def build_model(method, cfg):
    if method in ("mstp", "single"):
        return MSTPMTASIA(cfg)
    if method == "average":
        return TemporalAverage(cfg)
    if method == "tcn":
        return StandardTCN(cfg)
    raise ValueError(f"Unknown method: {method}")

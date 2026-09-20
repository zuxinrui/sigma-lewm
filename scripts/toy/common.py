"""Shared pieces for the Trinity-MI toy experiments (CPU only).

Losses are verbatim copies of the implemented forms:
  * SIGReg      -- stable_worldmodel/wm/loss.py::SIGReg (fewer projections)
  * symmetric_dcl_loss -- aligned_champion_losses.py (backward edge, I_B)
  * infonce_idm_loss   -- aligned_champion_losses.py::dcl_idm_loss without the model wrapper (inverse edge, I_A)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("TOY_THREADS", "3")))

# ---------------------------------------------------------------- palette / style
FORWARD = "#3E6F9E"
BACKWARD = "#6E5F8C"
INVERSE = "#B4585A"
ACTION = "#B07A2E"
INK = "#1F2933"
MUTED = "#6B7280"
GREY = "#A8AEB8"
LIGHT = "#E6E8EB"
TEAL = "#598BE7"
ARM_COLORS = {
    "F+SIG": GREY,
    "F+B+SIG": BACKWARD,
    "F+I+SIG": INVERSE,
    "F+B+I+SIG": TEAL,
    "F+Bmse+Imse+SIG": "#6B7280",
    "F+Ishuf+SIG": "#C4C9D2",
}
# display order: ablations and controls first, our full method LAST (bottom row / rightmost bar)
ARM_ORDER = ["F+SIG", "F+B+SIG", "F+I+SIG", "F+Bmse+Imse+SIG", "F+Ishuf+SIG", "F+B+I+SIG"]
ARM_LABEL = {
    "F+SIG": "F+SIG (LeWM)",
    "F+B+SIG": "F+B+SIG",
    "F+I+SIG": "F+I+SIG",
    "F+B+I+SIG": "F+B+I+SIG (ours)",
    "F+Bmse+Imse+SIG": "F+B$_{mse}$+I$_{mse}$+SIG",
    "F+Ishuf+SIG": "F+I$_{shuf}$+SIG (ctrl)",
}
ICLR_W = 5.5

# sequential colormap for heatmaps: white -> the paper's teal accent (print- and greyscale-safe)
TEAL_FILL = "#E7EFFC"
TEAL_LIGHT = "#A8C2F0"
TEAL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "paper_blue", [(0.00, "#FFFFFF"), (0.45, TEAL_FILL), (0.80, TEAL_LIGHT), (1.00, TEAL)])   # theme blue only, no green


def style():
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.monospace": ["DejaVu Sans Mono", "Liberation Mono", "Courier New"],
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "grid.color": LIGHT,
            "grid.linewidth": 0.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def savefig(fig, stem: str):
    fig.savefig(stem + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- losses (copied)
class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (copied; num_proj reduced for toys)."""

    def __init__(self, knots=17, num_proj=64):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        # proj: (T, B, D)  (we pass (1, B, D))
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def symmetric_dcl_loss(query, key, *, temperature: float):
    """Symmetric decoupled contrastive loss (copied from aligned_champion_losses.py)."""
    query = F.normalize(query.reshape(-1, query.shape[-1]).float(), dim=-1)
    key = F.normalize(key.reshape(-1, key.shape[-1]).float(), dim=-1)
    logits = query @ key.T / float(temperature)
    diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    negatives = logits.masked_fill(diagonal, float("-inf"))
    positive = logits.diagonal()
    return (-positive + 0.5 * (negatives.logsumexp(dim=1) + negatives.logsumexp(dim=0))).mean()


def infonce_idm_loss(query, key, raw_action, *, temperature: float, duplicate_cosine: float = 0.98):
    """One-way InfoNCE with near-duplicate action mask (dcl_idm_loss without the model wrapper)."""
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    logits = query @ key.T / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    with torch.no_grad():
        na = F.normalize(raw_action.float(), dim=-1)
        duplicates = na @ na.T > float(duplicate_cosine)
        duplicates.fill_diagonal_(False)
    logits = logits.masked_fill(duplicates, -1e4)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------- small nets
def mlp(din, dout, hidden=256, depth=2, act=nn.GELU):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, dout))
    return nn.Sequential(*layers)


def proj_head(din, dout=32, hidden=128):
    return nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, dout))


# ---------------------------------------------------------------- probes
def ridge_r2(X, Y, Xte, Yte, lam=1e-3):
    """Closed-form ridge from X to Y; returns per-column R^2 on held-out (Xte, Yte)."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xn = (X - mu) / sd
    Xn1 = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)
    A = Xn1.T @ Xn1 + lam * np.eye(Xn1.shape[1])
    W = np.linalg.solve(A, Xn1.T @ Y)
    Xt = np.concatenate([(np.asarray(Xte) - mu) / sd, np.ones((len(Xte), 1))], 1)
    pred = Xt @ W
    Yte = np.asarray(Yte, np.float64)
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    return 1 - ss_res / ss_tot, W, (mu, sd)


def mlp_r2(X, Y, Xte, Yte, steps=600, hidden=128, seed=0):
    """Small MLP probe (non-linear recoverability)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
    Xte = torch.as_tensor(np.asarray(Xte), dtype=torch.float32)
    Yte = torch.as_tensor(np.asarray(Yte), dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    net = mlp(X.shape[1], Y.shape[1], hidden=hidden, depth=2)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    n = len(X)
    for _ in range(steps):
        idx = torch.randint(0, n, (512,), generator=g)
        loss = F.mse_loss(net((X[idx] - mu) / sd), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net((Xte - mu) / sd)
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    return (1 - ss_res / ss_tot).numpy()

"""
model_cnn.py
=============
1D-CNN classifier operating on phase-folded light curve views, inspired by
the Astronet / ExoMiner dual-input architecture (global + local view), but
simplified for fast training within a hackathon timeline.

Input:
  - global_view: (B, 1, 201) — full orbital phase, coarse binning
  - local_view:  (B, 1, 61)  — zoomed around the transit, fine binning
  - aux_features: (B, n_aux) — classical BLS/vetting features (SNR, odd-even, etc.)

Output: 5-class softmax over
  {PLANET_TRANSIT, ECLIPSING_BINARY, STELLAR_VARIABILITY, STARSPOT_ACTIVITY, INSTRUMENT_NOISE}

Why dual-branch: the global view captures overall periodicity/shape context
(e.g. secondary eclipses elsewhere in phase), while the local view captures
fine transit-shape detail (V-shape vs flat-bottom, ingress/egress smoothness)
that's diagnostic for planet vs. eclipsing-binary discrimination.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 5
CLASS_NAMES = ["PLANET_TRANSIT", "ECLIPSING_BINARY", "STELLAR_VARIABILITY", "STARSPOT_ACTIVITY", "INSTRUMENT_NOISE"]


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, pool=2, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool) if pool else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.pool is not None:
            x = self.pool(x)
        return self.drop(x)


class GlobalLocalBranch(nn.Module):
    """A small CNN tower for one input view (global or local)."""

    def __init__(self, in_length: int, channels=(16, 32, 64), kernel_size=5, dropout=0.1):
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch, kernel_size=kernel_size, pool=2, dropout=dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        # compute flattened output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_length)
            out = self.blocks(dummy)
            self.out_features = out.numel()

    def forward(self, x):
        x = self.blocks(x)
        return x.flatten(1)


class ExoplanetCNN(nn.Module):
    """
    Dual-branch 1D-CNN + auxiliary classical-feature fusion head.

    n_aux_features: number of classical BLS/vetting features concatenated
                    before the final classification head (enables the network
                    to combine learned shape features with domain knowledge
                    like odd-even depth diff, secondary eclipse significance, etc.)
    """

    def __init__(self, global_len=201, local_len=61, n_aux_features=10,
                 channels=(16, 32, 64), dropout=0.25):
        super().__init__()
        self.global_branch = GlobalLocalBranch(global_len, channels=channels, dropout=dropout * 0.5)
        self.local_branch = GlobalLocalBranch(local_len, channels=channels, dropout=dropout * 0.5)

        fused_dim = self.global_branch.out_features + self.local_branch.out_features + n_aux_features

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, N_CLASSES),
        )
        self.n_aux_features = n_aux_features

    def forward(self, global_view, local_view, aux_features):
        # global_view, local_view: (B, L) -> (B, 1, L)
        g = self.global_branch(global_view.unsqueeze(1))
        l = self.local_branch(local_view.unsqueeze(1))
        fused = torch.cat([g, l, aux_features], dim=1)
        logits = self.head(fused)
        return logits

    def forward_with_embedding(self, global_view, local_view, aux_features):
        """Returns logits AND the pre-head fused embedding (for explainability / nearest-neighbor)."""
        g = self.global_branch(global_view.unsqueeze(1))
        l = self.local_branch(local_view.unsqueeze(1))
        fused = torch.cat([g, l, aux_features], dim=1)
        logits = self.head(fused)
        return logits, fused


class TemperatureScaler(nn.Module):
    """Post-hoc confidence calibration (Guo et al. 2017). Learns a single scalar
    temperature T applied to logits before softmax, fit on a held-out validation
    set by minimizing NLL. Produces calibrated confidence scores (not just
    overconfident softmax outputs) — important for a research tool where
    confidence numbers will be trusted by scientists."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature) + 1e-3

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr=0.01, max_iter=200):
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


if __name__ == "__main__":
    model = ExoplanetCNN(n_aux_features=10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    g = torch.randn(4, 201)
    l = torch.randn(4, 61)
    aux = torch.randn(4, 10)
    out = model(g, l, aux)
    print("Output shape:", out.shape)

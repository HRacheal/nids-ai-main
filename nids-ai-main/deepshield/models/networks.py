"""PyTorch architectures for DeepShield AI.

All models take the preprocessed float32 matrix produced by
`deepshield.data.preprocess.Preprocessor` (numeric block first, then missing
flags, then one-hot categories) and are built from a JSON-serialisable
config, so a checkpoint can always be re-instantiated for inference.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _dense_block(sizes: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Feed-forward classifier: [Linear-BatchNorm-ReLU-Dropout] x k + Linear."""

    def __init__(self, n_in: int, n_out: int, hidden=(256, 128), dropout: float = 0.3):
        super().__init__()
        self.body = _dense_block([n_in, *hidden], dropout)
        self.head = nn.Linear(hidden[-1], n_out)

    def forward(self, x):
        return self.head(self.body(x))


class CNN1D(nn.Module):
    """1D convolution over the ordered numeric features + dense categorical branch.

    The numeric block (length n_numeric) is treated as a one-channel signal.
    Kernels of width k see k ADJACENT features; because the preprocessor
    orders features by meaning (bytes next to packets next to loads, the
    TCP handshake timings together, the ct_* counters together), a filter
    can learn local interactions such as bytes-per-packet or handshake
    timing patterns, and weight sharing reuses one filter along the vector.
    `feature_perm` (an index permutation of the numeric block) supports the
    random-order ablation that tests whether this adjacency matters.
    The one-hot categorical block has no meaningful order, so it goes
    through a dense layer instead.
    """

    def __init__(self, n_numeric: int, n_categorical: int, n_out: int,
                 channels=(32, 64), kernel: int = 3, dropout: float = 0.3,
                 feature_perm: list[int] | None = None):
        super().__init__()
        self.n_numeric = n_numeric
        if feature_perm is not None:
            self.register_buffer("perm", torch.as_tensor(feature_perm, dtype=torch.long))
        else:
            self.perm = None
        c1, c2 = channels
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel, padding=pad), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel, padding=pad), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c2, kernel, padding=pad), nn.BatchNorm1d(c2), nn.ReLU(),
        )
        self.cat = nn.Sequential(nn.Linear(n_categorical, 32), nn.ReLU()) if n_categorical else None
        self.head = nn.Sequential(
            nn.Linear(2 * c2 + (32 if n_categorical else 0), 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_out))

    def forward(self, x):
        num = x[:, : self.n_numeric]
        if self.perm is not None:
            num = num[:, self.perm]
        h = self.conv(num.unsqueeze(1))
        parts = [h.mean(dim=2), h.amax(dim=2)]          # global average + max pooling
        if self.cat is not None:
            parts.append(self.cat(x[:, self.n_numeric:]))
        return self.head(torch.cat(parts, dim=1))


class Autoencoder(nn.Module):
    """Symmetric dense autoencoder; anomaly score = mean squared reconstruction error."""

    def __init__(self, n_in: int, hidden=(128, 64), bottleneck: int = 16, dropout: float = 0.0):
        super().__init__()
        enc = [n_in, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(enc[:-1], enc[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
        self.encoder = nn.Sequential(*layers, nn.Linear(enc[-1], bottleneck))
        dec = [bottleneck, *hidden[::-1]]
        layers = []
        for a, b in zip(dec[:-1], dec[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.decoder = nn.Sequential(*layers, nn.Linear(dec[-1], n_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        return ((self(x) - x) ** 2).mean(dim=1)


class FlowLSTM(nn.Module):
    """Classifies the LAST flow of a per-host window of flows.

    Input: (B, L, F) windows LEFT-padded with zeros plus true lengths. Windows
    are re-aligned to the left and packed, so padding never enters the
    recurrence; the final hidden state summarises the host's recent flows
    up to and including the target flow.
    """

    def __init__(self, n_in: int, n_out: int, proj: int = 64, hidden: int = 64,
                 layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_in, proj), nn.ReLU())
        self.lstm = nn.LSTM(proj, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x, lengths):
        B, L, _ = x.shape
        shift = (L - lengths).unsqueeze(1)                          # left padding size
        idx = (torch.arange(L, device=x.device).unsqueeze(0) + shift).clamp(max=L - 1)
        x = torch.gather(x, 1, idx.unsqueeze(2).expand_as(x))       # real steps first
        packed = pack_padded_sequence(self.proj(x), lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1])


ARCHITECTURES = {"MLP": MLP, "CNN1D": CNN1D, "Autoencoder": Autoencoder, "FlowLSTM": FlowLSTM}


def build(arch: str, params: dict) -> nn.Module:
    return ARCHITECTURES[arch](**params)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""Model architectures for the DM2026 Asg-3 0.85 attempt.

Provides:
  - PatchTST: Transformer over patches of the 300-row sequence
  - TCN: Temporal Convolutional Network with dilated convolutions
  - ResNet1D: deep residual 1D CNN
  - InceptionTime: re-exported from train_inceptiontime_oof for unified usage

Each model:
  - takes (B, 6, 300) input
  - emits (main_logits, aux_logits) where main is (B, 6) and aux is (B,) for binary class-2 head.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


N_CLASSES = 6
IN_CHANNELS = 6


# -------------------- PatchTST Transformer --------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PatchTST(nn.Module):
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        seq_len: int = 300,
        patch_len: int = 10,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        ff: int = 512,
        dropout: float = 0.2,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.patch_embed = nn.Linear(patch_len * in_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.n_patches + 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head_share = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_main = nn.Linear(d_model, n_classes)
        self.head_aux = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T)
        b, c, t = x.shape
        # Reshape to patches: (B, n_patches, patch_len * C)
        x = x.unfold(-1, self.patch_len, self.patch_len)  # (B, C, n_patches, patch_len)
        x = x.permute(0, 2, 1, 3).contiguous().view(b, self.n_patches, c * self.patch_len)
        x = self.patch_embed(x)  # (B, n_patches, d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = self.encoder(x)  # (B, n_patches, d_model)
        z = x.mean(dim=1)  # global pool over patches
        h = self.head_share(z)
        return self.head_main(h), self.head_aux(h).squeeze(-1)


# -------------------- TCN --------------------


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.bn1(self.chomp1(self.conv1(x)))))
        out = self.dropout2(self.relu2(self.bn2(self.chomp2(self.conv2(out)))))
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCN(nn.Module):
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        channels: Iterable[int] = (64, 64, 64, 64, 64, 64),
        kernel_size: int = 3,
        dilations: Iterable[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.2,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        channels = list(channels)
        dilations = list(dilations)
        assert len(channels) == len(dilations)
        layers = []
        prev = in_channels
        for c, d in zip(channels, dilations):
            layers.append(TemporalBlock(prev, c, kernel_size, d, dropout))
            prev = c
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_share = nn.Sequential(
            nn.Linear(prev, prev),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_main = nn.Linear(prev, n_classes)
        self.head_aux = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tcn(x)
        z = self.gap(x).squeeze(-1)
        h = self.head_share(z)
        return self.head_main(h), self.head_aux(h).squeeze(-1)


# -------------------- ResNet1D --------------------


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = None
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        n_classes: int = N_CLASSES,
        layers: Iterable[int] = (2, 2, 2, 2),
        channels: Iterable[int] = (64, 128, 192, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = list(layers)
        channels = list(channels)
        assert len(layers) == len(channels) == 4
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        block_layers = []
        prev = 64
        for i, (n_blocks, ch) in enumerate(zip(layers, channels)):
            stride = 1 if i == 0 else 2
            for j in range(n_blocks):
                s = stride if j == 0 else 1
                block_layers.append(BasicBlock1D(prev, ch, stride=s, dropout=dropout))
                prev = ch
        self.body = nn.Sequential(*block_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_share = nn.Sequential(
            nn.Linear(prev, prev),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_main = nn.Linear(prev, n_classes)
        self.head_aux = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.body(x)
        z = self.gap(x).squeeze(-1)
        h = self.head_share(z)
        return self.head_main(h), self.head_aux(h).squeeze(-1)


# -------------------- Model factory --------------------


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "patchtst":
        return PatchTST(**kwargs)
    if name == "tcn":
        return TCN(**kwargs)
    if name == "resnet1d":
        return ResNet1D(**kwargs)
    if name == "inception":
        # Import lazily to avoid circular import
        from train_inceptiontime_oof import InceptionTime
        return InceptionTime(**kwargs)
    raise ValueError(f"Unknown model name: {name}")

import os
import json
import math
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn  # PyTorch framework QoL
import torch.nn.functional as F  # PyTorch framework QoL


class FeatureNorm1D(nn.Module):  # PyTorch nn.Module base (framework QoL)
    def __init__(self, dim: int, eps: float = 1e-5):
        # Initializes a persistent 1D feature normalizer with stored stats
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(int(dim)), persistent=True)  # PyTorch buffer utility
        self.register_buffer("std", torch.ones(int(dim)), persistent=True)  # PyTorch buffer utility
        self.register_buffer("initialized", torch.tensor(False), persistent=True)  # PyTorch buffer utility

    @torch.no_grad()  # PyTorch no_grad decorator (framework QoL)
    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        # Sets normalization mean/std values (typically from training fold only)
        mean = mean.detach().flatten().to(self.mean.device)
        std = std.detach().flatten().to(self.std.device)

        if mean.numel() != self.mean.numel() or std.numel() != self.std.numel():
            raise ValueError(
                f"Stats shape mismatch: mean {mean.numel()} std {std.numel()} expected {self.mean.numel()}"
            )

        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(self.eps))
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalizes x using stored stats if initialized
        if not bool(self.initialized.item()):
            return x
        return (x - self.mean) / (self.std + self.eps)


class SpecNorm2D(nn.Module):  # PyTorch nn.Module base (framework QoL)
    def __init__(self, eps: float = 1e-5):
        # Initializes spectrogram normalizer with epsilon for stability
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Normalizes spectrogram either globally or with a length mask
        if lengths is None:
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True).clamp_min(self.eps)
            return (x - mean) / std

        if lengths.dim() != 1 or lengths.numel() != x.size(0):
            raise ValueError(f"lengths must be [B], got {tuple(lengths.shape)} for B={x.size(0)}")

        B, _, Freq, T = x.shape
        device = x.device
        lengths = lengths.to(device=device, dtype=torch.long).clamp_min(1).clamp_max(T)

        t_idx = torch.arange(T, device=device).view(1, 1, 1, T)  # PyTorch tensor utility
        mask = (t_idx < lengths.view(B, 1, 1, 1)).to(x.dtype)

        denom = (mask * Freq).sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        sum_x = (x * mask).sum(dim=(2, 3), keepdim=True)
        mean = sum_x / denom

        var = ((x - mean) ** 2 * mask).sum(dim=(2, 3), keepdim=True) / denom
        std = torch.sqrt(var + self.eps)
        return (x - mean) / std


class CNNBlock64(nn.Module):  # PyTorch nn.Module base (framework QoL)
    def __init__(self, in_channels: int, dropout: float = 0.4):
        # Defines a 2-layer conv block with pooling and dropout
        super().__init__()
        self.net = nn.Sequential(  # PyTorch Sequential (framework QoL)
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Dropout(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Runs the conv block on the input tensor
        return self.net(x)


class CRNNBranch(nn.Module):  # PyTorch nn.Module base (framework QoL)
    def __init__(
        self,
        mel_bins: int = 128,
        lstm_hidden_size: int = 128,
        cnn_dropout: float = 0.4,
        crnn_embed_dim: int = 128,
        min_time_frames: int = 8,
    ):
        # Defines the CRNN branch: CNN feature extractor + BiLSTM + projection
        super().__init__()
        self.mel_bins = int(mel_bins)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.crnn_embed_dim = int(crnn_embed_dim)
        self.min_time_frames = int(min_time_frames)

        self.cnn1 = CNNBlock64(in_channels=1, dropout=cnn_dropout)
        self.cnn2 = CNNBlock64(in_channels=64, dropout=cnn_dropout)
        self.cnn3 = CNNBlock64(in_channels=64, dropout=cnn_dropout)

        mel_reduced = (self.mel_bins + 8 - 1) // 8
        if mel_reduced < 1:
            raise ValueError(f"mel_bins too small for 3 pooling ops: {self.mel_bins} -> {mel_reduced}")
        if (self.mel_bins % 8) != 0:
            print(f"mel_bins={self.mel_bins} not divisible by 8. Using ceil mel_reduced={mel_reduced}")

        lstm_input_size = 64 * mel_reduced

        self.bilstm = nn.LSTM(  # PyTorch LSTM (framework QoL)
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.dense = nn.Linear(self.lstm_hidden_size * 2, self.crnn_embed_dim)

    @staticmethod
    def _ensure_min_time_frames(x: torch.Tensor, min_T: int) -> torch.Tensor:
        # Pads spectrogram time axis to a minimum length for stable pooling/LSTM
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B,1,F,T], got {tuple(x.shape)}")
        T = x.size(-1)
        if T >= min_T:
            return x
        pad_amt = min_T - T
        return F.pad(x, (0, pad_amt, 0, 0))  # PyTorch padding utility

    @staticmethod
    def _ceil_div(x: torch.Tensor, d: int) -> torch.Tensor:
        # Computes ceil(x / d) for integer tensors
        return (x + (d - 1)) // d

    def forward(self, spectrogram: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Produces a fixed-size embedding from a log-mel spectrogram (optionally length-aware)
        x = self._ensure_min_time_frames(spectrogram, self.min_time_frames)

        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)

        B, C, M, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C * M)

        if lengths is not None:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp_min(1)
            t_red = self._ceil_div(lengths, 8).clamp_min(1).clamp_max(T)

            packed = nn.utils.rnn.pack_padded_sequence(  # PyTorch packing utility
                x, t_red.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.bilstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(  # PyTorch unpacking utility
                packed_out, batch_first=True, total_length=T
            )

            t_idx = torch.arange(T, device=out.device).unsqueeze(0)
            mask = (t_idx < t_red.unsqueeze(1)).to(out.dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            out, _ = self.bilstm(x)
            pooled = out.mean(dim=1)

        emb = self.dense(pooled)
        return emb


class TwoLayerMLP(nn.Module):  # PyTorch nn.Module base (framework QoL)
    def __init__(
        self,
        input_dim: int = 88,
        hidden1: int = 128,
        hidden2: int = 64,
        out_dim: int = 64,
        dropout: float = 0.2,
    ):
        # Defines the eGeMAPS MLP branch that outputs an embedding
        super().__init__()
        self.net = nn.Sequential(  # PyTorch Sequential (framework QoL)
            nn.Linear(int(input_dim), int(hidden1)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(int(hidden1)),

            nn.Linear(int(hidden1), int(hidden2)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(int(hidden2)),

            nn.Linear(int(hidden2), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Produces an embedding from eGeMAPS features
        return self.net(x)


class FusionClassifier(nn.Module):  # PyTorch nn.Module base (framework QoL)
    def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = 2, dropout: float = 0.2):
        # Defines a small classifier head used for logmel/egemaps/hybrid modes
        super().__init__()
        self.net = nn.Sequential(  # PyTorch Sequential (framework QoL)
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Outputs class logits from an input embedding
        return self.net(x)


class HybridCRNNModel(nn.Module):  # PyTorch nn.Module base (framework QoL)
    def __init__(
        self,
        mode: str = "hybrid",
        mel_bins: int = 128,
        egemaps_input_dim: int = 88,
        num_classes: int = 2,

        lstm_hidden_size: int = 128,
        cnn_dropout: float = 0.4,
        crnn_embed_dim: int = 128,

        eg_hidden1: int = 128,
        eg_hidden2: int = 64,
        eg_embed_dim: int = 64,
        eg_dropout: float = 0.2,

        fusion_hidden: int = 128,
        fusion_dropout: float = 0.2,
    ):
        # Main model that supports hybrid/logmel-only/egemaps-only inference modes
        super().__init__()
        mode = str(mode).lower().strip()
        if mode not in {"hybrid", "logmel_only", "egemaps_only"}:
            raise ValueError("mode must be one of: 'hybrid', 'logmel_only', 'egemaps_only'")
        self.mode = mode

        self.spec_norm = SpecNorm2D()
        self.eg_norm = FeatureNorm1D(int(egemaps_input_dim))

        self.crnn = CRNNBranch(
            mel_bins=mel_bins,
            lstm_hidden_size=lstm_hidden_size,
            cnn_dropout=cnn_dropout,
            crnn_embed_dim=crnn_embed_dim,
            min_time_frames=8,
        )
        self.eg_mlp = TwoLayerMLP(
            input_dim=egemaps_input_dim,
            hidden1=eg_hidden1,
            hidden2=eg_hidden2,
            out_dim=eg_embed_dim,
            dropout=eg_dropout,
        )

        self.head_logmel = FusionClassifier(crnn_embed_dim, fusion_hidden, num_classes, fusion_dropout)
        self.head_eg = FusionClassifier(eg_embed_dim, fusion_hidden, num_classes, fusion_dropout)
        self.head_hybrid = FusionClassifier(crnn_embed_dim + eg_embed_dim, fusion_hidden, num_classes, fusion_dropout)

        self.config: Dict[str, Any] = {
            "mode": self.mode,
            "mel_bins": mel_bins,
            "egemaps_input_dim": egemaps_input_dim,
            "num_classes": num_classes,
            "lstm_hidden_size": lstm_hidden_size,
            "cnn_dropout": float(cnn_dropout),
            "crnn_embed_dim": crnn_embed_dim,
            "eg_hidden1": eg_hidden1,
            "eg_hidden2": eg_hidden2,
            "eg_embed_dim": eg_embed_dim,
            "eg_dropout": float(eg_dropout),
            "fusion_hidden": fusion_hidden,
            "fusion_dropout": float(fusion_dropout),
        }

    @torch.no_grad()  # PyTorch no_grad decorator (framework QoL)
    def set_egemaps_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        # Sets eGeMAPS normalization stats (mean/std)
        self.eg_norm.set_stats(mean, std)

    def forward(self, spectrogram: torch.Tensor, egemaps: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Forward pass returning logits according to selected mode
        if self.mode == "egemaps_only":
            eg = self.eg_norm(egemaps)
            eg_emb = self.eg_mlp(eg)
            return self.head_eg(eg_emb)

        x = self.spec_norm(spectrogram, lengths=lengths)
        crnn_emb = self.crnn(x, lengths=lengths)

        if self.mode == "logmel_only":
            return self.head_logmel(crnn_emb)

        eg = self.eg_norm(egemaps)
        eg_emb = self.eg_mlp(eg)

        fused = torch.cat([crnn_emb, eg_emb], dim=1)  # PyTorch tensor op
        return self.head_hybrid(fused)

    @torch.no_grad()  # PyTorch no_grad decorator (framework QoL)
    def predict_proba(self, spectrogram: torch.Tensor, egemaps: torch.Tensor) -> torch.Tensor:
        # Returns softmax probabilities for each class
        logits = self.forward(spectrogram, egemaps)
        return F.softmax(logits, dim=1)  # PyTorch softmax

    def save(self, filepath: str, additional_info: Optional[Dict[str, Any]] = None) -> None:
        # Saves model weights + config (and optional metadata) to disk
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        payload: Dict[str, Any] = {"model_state_dict": self.state_dict(), "config": self.config}
        if additional_info is not None:
            payload["additional_info"] = additional_info
        torch.save(payload, filepath)  # PyTorch serialization
        print(f"Model saved to: {filepath}")

        config_path = filepath.replace(".pth", "_config.json").replace(".pt", "_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None, strict: bool = True) -> Tuple["HybridCRNNModel", Dict[str, Any]]:
        # Loads a saved model checkpoint and returns (model, additional_info)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        ckpt = torch.load(filepath, map_location=device or "cpu")  # PyTorch checkpoint load
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)  # PyTorch state_dict load
        if device is not None:
            model = model.to(device)
        return model, ckpt.get("additional_info", {})


SimplifiedHybridModel = HybridCRNNModel

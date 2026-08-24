from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


PRITHVI_MEAN = torch.tensor([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0])
PRITHVI_STD = torch.tensor([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0])


class PrithviSegmentationHead(nn.Module):
    """Small trainable decoder over a frozen two-date Prithvi feature map."""

    def __init__(self, feature_channels: int = 384, auxiliary_channels: int = 8) -> None:
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.Conv2d(feature_channels, 96, kernel_size=3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(32 + auxiliary_channels + 1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(
        self,
        features: torch.Tensor,
        auxiliary: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.feature_projection(features)
        projected = F.interpolate(projected, size=auxiliary.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([projected, auxiliary, valid.float()], dim=1)
        return self.fusion(fused)


def load_prithvi_encoder(vendor_dir: Path, num_frames: int = 2) -> tuple[nn.Module, dict[str, Any]]:
    """Load the official NASA/IBM Prithvi-EO-2.0-tiny-TL encoder."""

    vendor_dir = vendor_dir.resolve()
    config_path = vendor_dir / "config.json"
    checkpoint_path = vendor_dir / "Prithvi_EO_V2_tiny_TL.pt"
    source_path = vendor_dir / "prithvi_mae.py"
    for required in (config_path, checkpoint_path, source_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing Prithvi asset: {required}")

    vendor_string = str(vendor_dir)
    if vendor_string not in sys.path:
        sys.path.insert(0, vendor_string)
    from prithvi_mae import PrithviMAE  # type: ignore

    raw_config = json.loads(config_path.read_text())
    pretrained = raw_config["pretrained_cfg"]
    keys = [
        "img_size",
        "patch_size",
        "in_chans",
        "embed_dim",
        "depth",
        "num_heads",
        "decoder_embed_dim",
        "decoder_depth",
        "decoder_num_heads",
        "mlp_ratio",
        "coords_encoding",
        "coords_scale_learn",
        "mask_ratio",
        "norm_pix_loss",
    ]
    kwargs = {key: pretrained[key] for key in keys}
    kwargs["num_frames"] = num_frames
    model = PrithviMAE(**kwargs)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Position embeddings depend on frame count. The official inference code
    # regenerates them for the requested temporal shape before strict loading.
    state["encoder.pos_embed"] = model.encoder.pos_embed
    state["decoder.decoder_pos_embed"] = model.decoder.decoder_pos_embed
    model.load_state_dict(state, strict=True)
    encoder = model.encoder
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    metadata = {
        "architecture": raw_config.get("architecture", "prithvi_eo_v2_tiny"),
        "checkpoint": str(checkpoint_path),
        "num_frames": num_frames,
        "bands": pretrained["bands"],
        "band_semantics": ["blue", "green", "red", "nir", "swir1", "swir2"],
        "mean": pretrained["mean"],
        "std": pretrained["std"],
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "license": "Apache-2.0",
        "upstream": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL",
    }
    return encoder, metadata


def encode_prithvi(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return the last Prithvi feature map as [B, 384, 14, 14]."""

    features = encoder.forward_features(images)
    return encoder.prepare_features_for_image_model([features[-1]])[-1]


def normalize_prithvi(reflectance: torch.Tensor) -> torch.Tensor:
    """Convert 0-1 surface reflectance to Prithvi's HLS normalization."""

    if reflectance.ndim != 5 or reflectance.shape[1] != 6:
        raise ValueError("Prithvi input must have shape [B, 6, T, H, W].")
    mean = PRITHVI_MEAN.to(reflectance.device, reflectance.dtype).view(1, 6, 1, 1, 1)
    std = PRITHVI_STD.to(reflectance.device, reflectance.dtype).view(1, 6, 1, 1, 1)
    scaled = torch.clamp(reflectance * 10000.0, min=-1000.0, max=16000.0)
    return (scaled - mean) / std

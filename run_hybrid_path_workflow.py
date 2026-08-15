#!/usr/bin/env python3
"""Hybrid tornado path workflow: geospatial validation + RF + tiny DNN.

This runner keeps the geospatial workflow as the source of truth and adds a
small CPU-safe U-Net style model for BEFORE/AFTER change segmentation. The DNN
is used only when it can train from usable labels; otherwise the workflow falls
back to the Random Forest probability.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tornado_hybrid_mpl")

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from rasterio.features import shapes
from scipy import ndimage as ndi
from shapely.geometry import shape
from torch.utils.data import DataLoader, TensorDataset

import run_rf_path_workflow as wf


PROJECT = Path(__file__).resolve().parent


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 16)
        self.enc2 = ConvBlock(16, 32)
        self.bottleneck = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


def cfg_for_hybrid() -> dict[str, Any]:
    cfg = wf.parse_config(PROJECT / "config.yaml")
    hybrid_out = cfg["data"].get("hybrid_output_dir", "outputs_hybrid_path")
    cfg["data"]["output_dir"] = hybrid_out
    return cfg


def make_training_label(case_id: str, before_tif: Path, after_tif: Path, cfg: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Return dense 1/0/-1 labels and a short provenance string."""

    before, after, _ = wf.read_stacks(before_tif, after_tif)
    valid = wf.valid_mask(before, after)
    label = np.full(valid.shape, -1, dtype="float32")
    nws = wf.find_nws_path(case_id, PROJECT / str(cfg["data"]["shapefile_dir"]))
    if nws:
        pos, excl = wf.rasterize_nws_label(
            nws,
            before_tif,
            int(cfg["model"]["positive_buffer_pixels"]),
            int(cfg["model"]["negative_exclusion_pixels"]),
        )
        label[valid & ~excl] = 0
        label[valid & pos] = 1
        if int(np.sum(label == 1)) >= 50 and int(np.sum(label == 0)) >= 200:
            return label, "NWS buffered path label"

    features = wf.feature_stack(before, after)
    mag = features[-1]
    values = mag[valid]
    if values.size < 500:
        return label, "rejected: too few valid pixels"
    high = np.nanquantile(values, 0.992)
    low = np.nanquantile(values, 0.45)
    label[valid & (mag <= low)] = 0
    label[valid & (mag >= high)] = 1
    return label, "image-change pseudo-label"


def sample_dnn_patches(
    pairs: list[wf.Pair],
    aligned: dict[str, tuple[Path, Path, Path, dict[str, Any]]],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(int(cfg["model"]["random_seed"]) + 7000)
    patch = int(cfg["dnn"]["patch_size"])
    patches_per_case = int(cfg["dnn"]["patches_per_case"])
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for pair in pairs:
        before_tif, after_tif = aligned[pair.case_id][0], aligned[pair.case_id][1]
        before, after, _ = wf.read_stacks(before_tif, after_tif)
        features = np.nan_to_num(wf.feature_stack(before, after), nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
        label, label_mode = make_training_label(pair.case_id, before_tif, after_tif, cfg)
        h, w = label.shape
        rows.append(
            {
                "case_id": pair.case_id,
                "label_mode": label_mode,
                "positive_pixels": int(np.sum(label == 1)),
                "negative_pixels": int(np.sum(label == 0)),
            }
        )
        if h < patch or w < patch or np.sum(label == 1) < 20 or np.sum(label == 0) < 100:
            continue
        positive_centers = np.argwhere(label == 1)
        labeled_centers = np.argwhere(label >= 0)
        for i in range(patches_per_case):
            centers = positive_centers if i < max(1, patches_per_case // 2) and len(positive_centers) else labeled_centers
            r, c = centers[rng.integers(0, len(centers))]
            y0 = int(np.clip(r - patch // 2, 0, h - patch))
            x0 = int(np.clip(c - patch // 2, 0, w - patch))
            lab = label[y0 : y0 + patch, x0 : x0 + patch]
            train_mask = lab >= 0
            if train_mask.mean() < 0.10:
                continue
            y = np.where(lab == 1, 1.0, 0.0).astype("float32")
            xs.append(features[:, y0 : y0 + patch, x0 : x0 + patch])
            ys.append(y[None, :, :])
            masks.append(train_mask[None, :, :].astype("float32"))

    if not xs:
        return (
            np.empty((0, 1, patch, patch), dtype="float32"),
            np.empty((0, 1, patch, patch), dtype="float32"),
            np.empty((0, 1, patch, patch), dtype="float32"),
            pd.DataFrame(rows),
        )
    return np.stack(xs), np.stack(ys), np.stack(masks), pd.DataFrame(rows)


def train_dnn(
    pairs: list[wf.Pair],
    aligned: dict[str, tuple[Path, Path, Path, dict[str, Any]]],
    cfg: dict[str, Any],
    model_dir: Path,
) -> tuple[TinyUNet | None, pd.DataFrame]:
    x, y, mask, label_df = sample_dnn_patches(pairs, aligned, cfg)
    label_df.to_csv(PROJECT / str(cfg["data"]["output_dir"]) / "reports" / "dnn_label_report.csv", index=False)
    if len(x) < 4:
        return None, pd.DataFrame([{"status": "skipped", "reason": "not enough DNN training patches"}])

    torch.manual_seed(int(cfg["model"]["random_seed"]))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    positive = float(np.sum((y == 1) & (mask == 1)))
    negative = float(np.sum((y == 0) & (mask == 1)))
    pos_weight = torch.tensor([min(max(negative / max(positive, 1.0), 1.0), 25.0)], dtype=torch.float32)
    model = TinyUNet(x.shape[1])
    loader = DataLoader(
        TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(mask)),
        batch_size=int(cfg["dnn"]["batch_size"]),
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["dnn"]["learning_rate"]))
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    rows = []
    model.train()
    for epoch in range(1, int(cfg["dnn"]["epochs"]) + 1):
        losses = []
        for xb, yb, mb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = (loss_fn(logits, yb) * mb).sum() / mb.sum().clamp_min(1)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        rows.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else np.nan, "patches": len(x)})

    if not rows or not np.isfinite(rows[-1]["loss"]):
        return None, pd.DataFrame(rows or [{"status": "skipped", "reason": "DNN loss was not finite"}])

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_channels": int(x.shape[1])}, model_dir / "tiny_unet_change_model.pt")
    return model, pd.DataFrame(rows)


def rf_probability(model, before_tif: Path, after_tif: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    before, after, profile = wf.read_stacks(before_tif, after_tif)
    valid = wf.valid_mask(before, after)
    features = wf.feature_stack(before, after)
    coords = np.argwhere(valid)
    prob = np.zeros(valid.shape, dtype="float32")
    chunk = 250000
    for start in range(0, len(coords), chunk):
        xy = coords[start : start + chunk]
        x = features[:, xy[:, 0], xy[:, 1]].T
        prob[xy[:, 0], xy[:, 1]] = model.predict_proba(x)[:, 1]
    return prob, valid, profile, features


def pad_tile(tile: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((tile.shape[0], size, size), dtype="float32")
    out[:, : tile.shape[1], : tile.shape[2]] = tile
    return out


def dnn_probability(model: TinyUNet | None, features: np.ndarray, valid: np.ndarray, cfg: dict[str, Any]) -> np.ndarray | None:
    if model is None:
        return None
    model.eval()
    tile = int(cfg["dnn"]["max_inference_tile"])
    overlap = int(cfg["dnn"]["inference_overlap"])
    stride = max(64, tile - overlap)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    h, w = valid.shape
    prob_sum = np.zeros((h, w), dtype="float32")
    count = np.zeros((h, w), dtype="float32")
    with torch.no_grad():
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                y1 = min(y0 + tile, h)
                x1 = min(x0 + tile, w)
                arr = pad_tile(features[:, y0:y1, x0:x1], tile)
                pred = torch.sigmoid(model(torch.tensor(arr[None, ...])))[0, 0].cpu().numpy()
                pred = pred[: y1 - y0, : x1 - x0]
                prob_sum[y0:y1, x0:x1] += pred.astype("float32")
                count[y0:y1, x0:x1] += 1
    out = np.divide(prob_sum, np.maximum(count, 1), out=np.zeros_like(prob_sum), where=count > 0)
    out[~valid] = 0
    return out


def write_prediction_products(
    case_id: str,
    before_tif: Path,
    after_tif: Path,
    rf_prob: np.ndarray,
    dnn_prob: np.ndarray | None,
    hybrid_prob: np.ndarray,
    valid: np.ndarray,
    profile: dict[str, Any],
    meta: dict[str, Any],
    cfg: dict[str, Any],
    write_probability_products: bool = True,
) -> dict[str, Any]:
    out = PROJECT / str(cfg["data"]["output_dir"])
    case_out = out / "cases" / case_id
    pred_out = out / "predictions" / case_id
    pred_out.mkdir(parents=True, exist_ok=True)
    clean = wf.clean_prediction(
        hybrid_prob,
        valid,
        float(cfg["model"]["probability_threshold"]),
        int(cfg["model"]["min_component_pixels"]),
    )
    nws = wf.find_nws_path(case_id, PROJECT / str(cfg["data"]["shapefile_dir"]))
    guide = None
    if nws is not None:
        try:
            guide, _ = wf.rasterize_nws_label(
                nws,
                before_tif,
                max(12, int(cfg["model"]["positive_buffer_pixels"]) * 2),
                int(cfg["model"]["negative_exclusion_pixels"]),
            )
            guide &= valid
        except Exception:
            guide = None
    corridor = wf.select_path_component(clean, guide)
    line = wf.centerline_from_mask(corridor, profile["transform"])
    tortuosity = wf.line_tortuosity(line)
    quality = wf.classify_quality(
        float(meta["valid_fraction"]),
        float(corridor.sum() / max(valid.sum(), 1)),
        line is not None,
        cfg,
        tortuosity,
    )
    published_line = line if quality != "Rejected" else None
    if write_probability_products:
        wf.write_raster(pred_out / "rf_probability.tif", rf_prob, profile, "float32", 0.0)
        if dnn_prob is not None:
            wf.write_raster(pred_out / "dnn_probability.tif", dnn_prob, profile, "float32", 0.0)
        wf.write_raster(pred_out / "predicted_probability.tif", hybrid_prob, profile, "float32", 0.0)
    wf.write_raster(pred_out / "predicted_damage_mask.tif", corridor.astype("uint8"), profile, "uint8", 0)
    polygons = [shape(g) for g, value in shapes(corridor.astype("uint8"), mask=corridor, transform=profile["transform"]) if value == 1]
    gpd.GeoDataFrame({"class": ["hybrid_prediction"] * len(polygons)}, geometry=polygons, crs=profile["crs"]).to_file(
        pred_out / "predicted_damage_corridor.geojson",
        driver="GeoJSON",
    )
    if published_line:
        gpd.GeoDataFrame({"class": ["hybrid_centerline"]}, geometry=[published_line], crs=profile["crs"]).to_file(
            pred_out / "predicted_path_centerline.geojson",
            driver="GeoJSON",
        )
    else:
        gpd.GeoDataFrame({"class": []}, geometry=[], crs=profile["crs"]).to_file(pred_out / "predicted_path_centerline.geojson", driver="GeoJSON")
    wf.make_maps(
        case_id,
        before_tif,
        after_tif,
        hybrid_prob,
        corridor,
        published_line,
        nws,
        case_out,
        pred_out,
        model_label="Hybrid DNN + RF",
        quality=quality,
    )
    metrics = {
        "case_id": case_id,
        "valid_fraction": float(meta["valid_fraction"]),
        "predicted_damage_fraction": float(corridor.sum() / max(valid.sum(), 1)),
        "centerline_available": published_line is not None,
        "nws_available": nws is not None,
        "path_tortuosity": tortuosity,
        "dnn_used": dnn_prob is not None,
        "confidence": quality,
    }
    (pred_out / "case_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def write_hybrid_reports(
    pairs: list[wf.Pair],
    aligned: dict[str, tuple[Path, Path, Path, dict[str, Any]]],
    metrics: list[dict[str, Any]],
    dnn_history: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    wf.write_reports(pairs, aligned, metrics, cfg)
    reports = PROJECT / str(cfg["data"]["output_dir"]) / "reports"
    dnn_history.to_csv(reports / "dnn_training_history.csv", index=False)
    pd.DataFrame(
        [
            {"model": "random_forest", "selected": False, "reason": "Stable fallback and feature baseline."},
            {"model": "tiny_unet_dnn", "selected": False, "reason": "Learns patch context from BEFORE/AFTER/difference channels when labels are usable."},
            {"model": "hybrid_dnn_rf", "selected": True, "reason": "Blends RF stability with DNN spatial context, then applies geospatial QC and centerline extraction."},
        ]
    ).to_csv(reports / "model_comparison.csv", index=False)
    (reports / "final_model_selection.md").write_text(
        "Selected model: Hybrid DNN + Random Forest.\n\n"
        "The DNN is a small U-Net style segmentation model trained on available NWS-buffer labels where possible and image-change pseudo-labels only when NWS labels have no readable overlap. "
        "The final corridor is extracted from blended RF/DNN probabilities and quality-controlled using valid overlap, fragmentation, and centerline availability. "
        "This does not guarantee a perfect path when the input rasters have missing or non-overlapping readable pixels.\n"
    )


def write_hybrid_docs(cfg: dict[str, Any]) -> None:
    out = PROJECT / str(cfg["data"]["output_dir"])
    (PROJECT / "README.md").write_text(
        "# Spatial Analysis Hybrid Tornado Path Workflow\n\n"
        "Run the hybrid model with:\n\n"
        "```bash\n"
        ".venv/bin/python run_hybrid_path_workflow.py\n"
        "```\n\n"
        "Outputs are in `outputs_hybrid_path`. Red is the predicted hybrid corridor/path. Cyan is the NWS reference path when available.\n"
    )
    (PROJECT / "RUN_NEW_DATASET.command").write_text(
        '#!/bin/zsh\ncd "$(dirname "$0")"\n.venv/bin/python run_hybrid_path_workflow.py\nopen outputs_hybrid_path/presentation 2>/dev/null || true\n'
    )
    os.chmod(PROJECT / "RUN_NEW_DATASET.command", 0o755)
    docs = PROJECT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "MODEL_METHODOLOGY.md").write_text(
        "# Model Methodology\n\n"
        "The current workflow is hybrid: rasterio/geopandas handle CRS, alignment, and NWS reprojection; Random Forest predicts per-pixel damage probability from spectral-change features; "
        "a compact U-Net style DNN learns patch-level spatial context; the final map blends both probabilities and extracts a cleaned corridor plus skeleton centerline.\n"
    )
    (docs / "SCIENTIFIC_LIMITATIONS.md").write_text(
        "# Scientific Limitations\n\n"
        "This workflow cannot produce a trustworthy tornado path where BEFORE/AFTER rasters have very low valid overlap or where NWS reference geometry falls outside readable imagery. "
        "Those cases are marked low confidence or rejected instead of being hidden.\n"
    )


def main() -> None:
    cfg = cfg_for_hybrid()
    out = PROJECT / str(cfg["data"]["output_dir"])
    if out.exists():
        shutil.rmtree(out)
    for folder in ["cases", "predictions", "reports", "models/final_model", "presentation"]:
        (out / folder).mkdir(parents=True, exist_ok=True)

    pairs = wf.discover_pairs(PROJECT / str(cfg["data"]["raster_dir"]))
    aligned = {p.case_id: wf.align_pair(p, out) for p in pairs}
    rf_model, label_df = wf.train_model(aligned, pairs, cfg)
    model_dir = out / "models" / "final_model"
    joblib.dump(rf_model, model_dir / "random_forest_path_model.joblib")
    dnn_model, dnn_history = train_dnn(pairs, aligned, cfg, model_dir)

    metrics = []
    for pair in pairs:
        before_tif, after_tif, _, meta = aligned[pair.case_id]
        rfp, valid, profile, features = rf_probability(rf_model, before_tif, after_tif)
        dnnp = dnn_probability(dnn_model, features, valid, cfg)
        if dnnp is None:
            hybrid = rfp
        else:
            w = float(cfg["dnn"]["blend_weight"])
            hybrid = ((1 - w) * rfp + w * dnnp).astype("float32")
            hybrid[~valid] = 0
        metrics.append(write_prediction_products(pair.case_id, before_tif, after_tif, rfp, dnnp, hybrid, valid, profile, meta, cfg))

    (model_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model": "Hybrid DNN + RandomForestClassifier",
                "random_forest_file": "random_forest_path_model.joblib",
                "dnn_file": "tiny_unet_change_model.pt" if dnn_model is not None else "",
                "dnn_trained": dnn_model is not None,
                "training_cases": label_df.loc[label_df["training_use_allowed"], "case_id"].tolist(),
                "all_cases": [p.case_id for p in pairs],
                "feature_schema": ["before_norm", "after_norm", "signed_diff", "absolute_diff", "change_magnitude"],
                "selected_threshold": cfg["model"]["probability_threshold"],
                "warning": "Low-valid-overlap rasters cannot support perfect path detection.",
            },
            indent=2,
        )
    )
    write_hybrid_reports(pairs, aligned, metrics, dnn_history, cfg)
    wf.make_ppt(
        [p.case_id for p in pairs],
        out / "presentation" / "hybrid_tornado_path_analysis.pptx",
        out,
        deck_title="Hybrid DNN + Random Forest Tornado Path Workflow",
        subtitle="Each predicted corridor and centerline is drawn directly on the full AFTER satellite image. Cyan shows the official NWS path when available.",
    )
    write_hybrid_docs(cfg)
    print("Hybrid workflow complete")
    print(out / "presentation" / "hybrid_tornado_path_analysis.pptx")


if __name__ == "__main__":
    main()

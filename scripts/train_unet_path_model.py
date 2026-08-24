#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi
from skimage.morphology import disk, remove_small_objects, skeletonize
from skimage.transform import resize
from sklearn.model_selection import GroupKFold
from torch import nn


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs_unet_v6"
LABEL_ROOT = PROJECT / "data" / "manual_labels" / "screenshot_verified"
CASE_ROOT = PROJECT / "outputs_all_cases" / "cases"
DATA_ROOTS = [PROJECT / "data_previous_full", PROJECT / "data_aug2"]
SEED = 20260814
MAX_DIMENSION = 640
PATCH = 128
BASE = 12
CV_EPOCHS = 8
FINAL_EPOCHS = 12
STEPS_PER_EPOCH = 36
BATCH = 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def case_key(path: Path) -> tuple[str, str] | None:
    match = re.search(r"(TOR\d+)_BEST_(BEFORE|AFTER)", path.name.upper())
    return (match.group(1), match.group(2)) if match else None


def discover_pairs() -> dict[str, dict[str, Path]]:
    candidates: dict[str, dict[str, list[Path]]] = {}
    for root in DATA_ROOTS:
        for path in root.glob("*.tif"):
            key = case_key(path)
            if key:
                candidates.setdefault(key[0], {}).setdefault(key[1], []).append(path)
    result = {}
    for case_id, roles in candidates.items():
        if "BEFORE" not in roles or "AFTER" not in roles:
            continue
        def choose(paths: list[Path]) -> Path:
            return sorted(paths, key=lambda p: (" 2" in p.stem, len(p.name), str(p)))[0]
        result[case_id] = {role: choose(paths) for role, paths in roles.items()}
    return result


def target_grid(case_id: str, max_dimension: int):
    with rasterio.open(CASE_ROOT / case_id / "model_probability.tif") as src:
        height, width = src.height, src.width
        scale = min(1.0, max_dimension / max(height, width))
        out_height = max(32, int(round(height * scale)))
        out_width = max(32, int(round(width * scale)))
        transform = src.transform * src.transform.scale(width / out_width, height / out_height)
        return out_height, out_width, src.crs, transform


def read_reprojected(path: Path, grid) -> np.ndarray:
    height, width, crs, transform = grid
    with rasterio.open(path) as src:
        destination = np.full((src.count, height, width), np.nan, dtype="float32")
        for band in range(src.count):
            reproject(
                source=rasterio.band(src, band + 1),
                destination=destination[band],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    return destination


def robust_pair(before: np.ndarray, after: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    before_out = np.zeros_like(before, dtype="float32")
    after_out = np.zeros_like(after, dtype="float32")
    for band in range(before.shape[0]):
        values = np.concatenate([before[band][valid], after[band][valid]])
        low, high = np.percentile(values[np.isfinite(values)], [2, 98])
        denom = max(float(high - low), 1e-6)
        before_out[band] = np.clip((before[band] - low) / denom, 0, 1)
        after_out[band] = np.clip((after[band] - low) / denom, 0, 1)
    return before_out, after_out


def read_single(path: Path, shape: tuple[int, int], nearest: bool = False) -> np.ndarray:
    with rasterio.open(path) as src:
        array = src.read(1).astype("float32")
    if array.shape != shape:
        array = resize(array, shape, order=0 if nearest else 1, preserve_range=True, anti_aliasing=not nearest).astype("float32")
    return array


def robust_channel(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = array[valid & np.isfinite(array)]
    if not len(values):
        return np.zeros_like(array, dtype="float32")
    low, high = np.percentile(values, [2, 98])
    return np.clip((array - low) / max(float(high - low), 1e-6), 0, 1).astype("float32")


def prepare_case(case_id: str, pair: dict[str, Path]) -> dict[str, np.ndarray]:
    cache_dir = OUTPUT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{case_id}.npz"
    if cache.exists():
        data = np.load(cache)
        return {key: data[key] for key in data.files}
    grid = target_grid(case_id, MAX_DIMENSION)
    before = read_reprojected(pair["BEFORE"], grid)
    after = read_reprojected(pair["AFTER"], grid)
    bands = min(before.shape[0], after.shape[0], 6)
    before, after = before[:bands], after[:bands]
    valid = np.all(np.isfinite(before), axis=0) & np.all(np.isfinite(after), axis=0)
    before, after = robust_pair(before, after, valid)
    signed = after - before
    absolute = np.abs(signed)
    magnitude = np.sqrt(np.mean(signed * signed, axis=0, keepdims=True))
    root = CASE_ROOT / case_id
    old_probability = read_single(root / "model_probability.tif", valid.shape)
    change_score = read_single(root / "change_score.tif", valid.shape)
    water = read_single(root / "stable_water_mask.tif", valid.shape, nearest=True) > 0.5
    old_probability = robust_channel(old_probability, valid)[None]
    change_score = robust_channel(change_score, valid)[None]
    channels = np.concatenate([before, after, signed, absolute, magnitude, old_probability, change_score, water[None].astype("float32")], axis=0)
    channels[:, ~valid] = 0
    result = {"x": channels.astype("float32"), "valid": valid.astype("uint8"), "water": water.astype("uint8")}
    label_path = LABEL_ROOT / case_id / "manual_damage_corridor_mask.tif"
    if label_path.exists():
        label = read_single(label_path, valid.shape, nearest=True) > 0.5
        result["y"] = (label & valid).astype("uint8")
    np.savez_compressed(cache, **result)
    return result


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int, base: int = BASE):
        super().__init__()
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        mid = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(mid), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        return self.out(self.d1(torch.cat([self.u1(d2), e1], dim=1)))


def random_patch(case: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, valid = case["x"], case["y"].astype(bool), case["valid"].astype(bool)
    height, width = y.shape
    if rng.random() < 0.72 and y.any():
        row, col = np.argwhere(y)[rng.integers(int(y.sum()))]
    else:
        points = np.argwhere(valid)
        row, col = points[rng.integers(len(points))]
    top = int(np.clip(row - rng.integers(PATCH // 4, 3 * PATCH // 4), 0, max(height - PATCH, 0)))
    left = int(np.clip(col - rng.integers(PATCH // 4, 3 * PATCH // 4), 0, max(width - PATCH, 0)))
    px = x[:, top:top + PATCH, left:left + PATCH]
    py = y[top:top + PATCH, left:left + PATCH]
    pv = valid[top:top + PATCH, left:left + PATCH]
    pad_h, pad_w = PATCH - px.shape[1], PATCH - px.shape[2]
    if pad_h or pad_w:
        px = np.pad(px, ((0, 0), (0, pad_h), (0, pad_w)))
        py = np.pad(py, ((0, pad_h), (0, pad_w)))
        pv = np.pad(pv, ((0, pad_h), (0, pad_w)))
    k = int(rng.integers(4)); px = np.rot90(px, k, axes=(1, 2)).copy(); py = np.rot90(py, k).copy(); pv = np.rot90(pv, k).copy()
    if rng.random() < 0.5: px = px[:, :, ::-1].copy(); py = py[:, ::-1].copy(); pv = pv[:, ::-1].copy()
    if rng.random() < 0.5: px = px[:, ::-1, :].copy(); py = py[::-1, :].copy(); pv = pv[::-1, :].copy()
    return px, py.astype("float32"), pv.astype("float32")


def loss_fn(logits, truth, valid):
    positive_weight = torch.tensor(8.0, device=logits.device)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, truth, reduction="none", pos_weight=positive_weight)
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    probability = torch.sigmoid(logits) * valid
    intersection = (probability * truth).sum((1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (probability.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1)).mean()
    return bce + dice


def train_model(train_cases: list[dict[str, np.ndarray]], epochs: int, seed: int) -> SmallUNet:
    seed_everything(seed)
    model = SmallUNet(train_cases[0]["x"].shape[0]).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for _ in range(STEPS_PER_EPOCH):
            patches = [random_patch(train_cases[int(rng.integers(len(train_cases)))], rng) for _ in range(BATCH)]
            x = torch.from_numpy(np.stack([p[0] for p in patches]))
            y = torch.from_numpy(np.stack([p[1] for p in patches]))[:, None]
            valid = torch.from_numpy(np.stack([p[2] for p in patches]))[:, None]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y, valid)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            running += float(loss)
        scheduler.step()
        print(f"epoch {epoch + 1}/{epochs} loss={running / STEPS_PER_EPOCH:.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def predict(model: SmallUNet, case: dict[str, np.ndarray]) -> np.ndarray:
    x = torch.from_numpy(case["x"][None])
    height, width = x.shape[-2:]
    pad_h = (8 - height % 8) % 8; pad_w = (8 - width % 8) % 8
    x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
    probability = torch.sigmoid(model(x))[0, 0, :height, :width].cpu().numpy().astype("float32")
    probability[~case["valid"].astype(bool)] = 0
    return probability


def metric(probability: np.ndarray, truth: np.ndarray, valid: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probability >= threshold) & valid
    truth = truth.astype(bool) & valid
    intersection = int((pred & truth).sum())
    return {
        "dice": 2 * intersection / max(int(pred.sum() + truth.sum()), 1),
        "iou": intersection / max(int((pred | truth).sum()), 1),
        "precision": intersection / max(int(pred.sum()), 1),
        "recall": intersection / max(int(truth.sum()), 1),
    }


def extract_paths(probability: np.ndarray, valid: np.ndarray, water: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, int]:
    mask = (probability >= threshold) & valid
    mask &= ~ndi.binary_dilation(water.astype(bool), structure=disk(2))
    mask = ndi.binary_closing(mask, structure=disk(3))
    mask = remove_small_objects(mask, min_size=max(20, int(valid.sum() * 0.00004)))
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype="uint8"))
    candidates = []
    for idx in range(1, count + 1):
        component = labels == idx
        points = np.argwhere(component)
        if len(points) < 20:
            continue
        covariance = np.cov((points - points.mean(0)).T)
        values = np.linalg.eigvalsh(covariance)
        elongation = np.sqrt(max(values[-1], 1e-6) / max(values[0], 1e-6))
        if elongation < 2.2:
            continue
        score = np.sqrt(values[-1]) * min(elongation, 18) * float(probability[component].mean())
        candidates.append((score, idx))
    candidates.sort(reverse=True)
    if not candidates:
        return np.zeros_like(mask), np.zeros_like(mask), 0
    selected = [idx for score, idx in candidates if score >= candidates[0][0] * 0.30][:4]
    corridor = np.isin(labels, selected)
    return corridor, skeletonize(corridor), len(selected)


def save_prediction(case_id: str, case: dict[str, np.ndarray], probability: np.ndarray, threshold: float, evaluation_type: str) -> dict:
    case_dir = OUTPUT / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    valid = case["valid"].astype(bool); water = case["water"].astype(bool)
    corridor, centerline, path_count = extract_paths(probability, valid, water, threshold)
    np.savez_compressed(case_dir / "prediction.npz", probability=probability, corridor=corridor, centerline=centerline, valid=valid, water=water)
    source = CASE_ROOT / case_id / "before_after.png"
    from PIL import Image
    composite = Image.open(source).convert("RGB"); width, height = composite.size
    after = np.asarray(composite.crop((width // 2, 0, width, height)))
    display = resize(centerline.astype("float32"), after.shape[:2], order=0, preserve_range=True, anti_aliasing=False) > 0.5
    fig, axis = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    axis.imshow(after)
    if display.any():
        axis.contour(ndi.binary_dilation(display, iterations=2), [0.5], colors=["#10231D"], linewidths=4)
        axis.contour(display, [0.5], colors=["#FFEA00"], linewidths=2.5)
    axis.set_title(f"{case_id}: U-Net model-generated tornado damage path", fontsize=19, fontweight="bold")
    axis.set_axis_off(); fig.savefig(case_dir / "model_final_path.png", dpi=180, facecolor="white"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(after); axes[0].set_title("Full AFTER image", fontweight="bold")
    im = axes[1].imshow(probability, cmap="inferno", vmin=0, vmax=1); axes[1].set_title("U-Net damage probability", fontweight="bold"); fig.colorbar(im, ax=axes[1], fraction=0.04)
    axes[2].imshow(after)
    if display.any(): axes[2].contour(display, [0.5], colors=["#FFEA00"], linewidths=2.5)
    axes[2].set_title(f"{path_count} model path(s)", fontweight="bold")
    for axis in axes: axis.set_axis_off()
    fig.savefig(case_dir / "model_prediction_panel.png", dpi=170, facecolor="white"); plt.close(fig)
    row = {"case_id": case_id, "evaluation_type": evaluation_type, "path_count": path_count, "threshold": threshold}
    if "y" in case:
        row.update(metric(probability, case["y"].astype(bool), valid, threshold))
    return row


def main() -> None:
    seed_everything(SEED)
    pairs = discover_pairs()
    all_ids = sorted(set(pairs) & {p.name for p in CASE_ROOT.iterdir() if p.is_dir()}, key=lambda value: int(value[3:]))
    labeled = [case_id for case_id in all_ids if (LABEL_ROOT / case_id / "manual_damage_corridor_mask.tif").exists()]
    print(f"cases={len(all_ids)} labeled={len(labeled)}", flush=True)
    cases = {case_id: prepare_case(case_id, pairs[case_id]) for case_id in all_ids}
    folds = GroupKFold(n_splits=3)
    oof: dict[str, np.ndarray] = {}
    cv_rows = []
    for fold, (train_idx, test_idx) in enumerate(folds.split(labeled, groups=labeled), 1):
        train_ids = [labeled[index] for index in train_idx]; test_ids = [labeled[index] for index in test_idx]
        print(f"fold={fold} train={train_ids} test={test_ids}", flush=True)
        model = train_model([cases[case_id] for case_id in train_ids], CV_EPOCHS, SEED + fold)
        for case_id in test_ids:
            oof[case_id] = predict(model, cases[case_id])
    thresholds = np.arange(0.15, 0.86, 0.05)
    summaries = []
    for threshold in thresholds:
        rows = [metric(oof[case_id], cases[case_id]["y"].astype(bool), cases[case_id]["valid"].astype(bool), float(threshold)) for case_id in labeled]
        summaries.append({"threshold": float(threshold), **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]}})
    summaries.sort(key=lambda row: row["dice"], reverse=True)
    threshold = float(summaries[0]["threshold"])
    for case_id in labeled:
        cv_rows.append({"case_id": case_id, "threshold": threshold, **metric(oof[case_id], cases[case_id]["y"].astype(bool), cases[case_id]["valid"].astype(bool), threshold)})
    (OUTPUT / "reports").mkdir(parents=True, exist_ok=True)
    for name, rows in [("threshold_selection.csv", summaries), ("event_level_cross_validation.csv", cv_rows)]:
        with (OUTPUT / "reports" / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"selected_threshold={threshold:.2f} mean_oof_dice={summaries[0]['dice']:.4f}", flush=True)
    final_model = train_model([cases[case_id] for case_id in labeled], FINAL_EPOCHS, SEED)
    model_dir = OUTPUT / "models" / "final_unet"; model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final_model.state_dict(), "in_channels": cases[labeled[0]]["x"].shape[0], "base": BASE}, model_dir / "model.pt")
    metadata = {"model": "SmallUNet", "training_cases": labeled, "unlabeled_inference_cases": [case_id for case_id in all_ids if case_id not in labeled], "selected_threshold": threshold, "grouped_validation": summaries[0], "manual_labels_visible_in_output": False, "labeled_case_predictions": "out-of-fold"}
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    inference_rows = []
    for case_id in all_ids:
        if case_id in oof:
            probability = oof[case_id]; evaluation_type = "out-of-fold validation"
        else:
            probability = predict(final_model, cases[case_id]); evaluation_type = "unlabeled model inference"
        print(f"saving {case_id} ({evaluation_type})", flush=True)
        inference_rows.append(save_prediction(case_id, cases[case_id], probability, threshold, evaluation_type))
    with (OUTPUT / "reports" / "all_case_inference.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in inference_rows for key in row})); writer.writeheader(); writer.writerows(inference_rows)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

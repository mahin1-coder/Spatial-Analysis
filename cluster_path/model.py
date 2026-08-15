from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.features import rasterize
from scipy import ndimage as ndi
from shapely.geometry import MultiLineString, box
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

from .core import (
    AnalysisData,
    CasePair,
    _corridor_geometry,
    _extent,
    _rgb,
    _write_geometry,
    _write_raster,
    build_features,
    centerline_from_corridor,
    cluster_change,
    find_nws_path,
    load_analysis_data,
    select_corridor,
    validate_against_nws,
)
from .multipath import (
    centerlines_from_corridor,
    evaluate_path_set,
    extract_multiple_corridors,
    stable_water_mask,
)


PATH_COLORS = ("#FFD400", "#FF4FD8", "#7CFF4F", "#FFFFFF", "#FF9F1C", "#B388FF")


def create_nws_label(nws_path: Path, data: AnalysisData) -> np.ndarray:
    frame = gpd.read_file(nws_path)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty or frame.crs is None:
        raise ValueError("NWS path is empty or has no CRS.")
    frame = frame.to_crs(data.crs)
    footprint = box(*data.bounds)
    frame = frame[frame.geometry.intersects(footprint)].copy()
    if frame.empty:
        raise ValueError("NWS path does not overlap the raster.")
    projected_crs = frame.estimate_utm_crs()
    projected = frame.to_crs(projected_crs)
    buffers = []
    for _, row in projected.iterrows():
        raw_width = row.get("width", np.nan)
        width_yards = float(raw_width) if raw_width is not None and np.isfinite(raw_width) and raw_width > 0 else 660.0
        radius_m = float(np.clip(width_yards * 0.9144 / 2.0, 180.0, 2500.0))
        buffers.append(row.geometry.buffer(radius_m, cap_style="round", join_style="round"))
    corridors = gpd.GeoSeries(buffers, crs=projected_crs).to_crs(data.crs)
    return rasterize(
        [(geometry, 1) for geometry in corridors],
        out_shape=data.valid.shape,
        transform=data.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)


def load_reference_label(
    pair: CasePair,
    data: AnalysisData,
    shapefile_root: Path,
    manual_label_root: Path | None = None,
) -> tuple[np.ndarray, str, bool]:
    """Prefer a reviewed manual raster label; otherwise use partial DAT data."""
    if manual_label_root is not None:
        manual_path = manual_label_root / pair.case_id / "manual_damage_corridor_mask.tif"
        if manual_path.exists():
            with rasterio.open(manual_path) as source:
                if (
                    source.crs != data.crs
                    or source.transform != data.transform
                    or source.shape != data.valid.shape
                ):
                    raise ValueError(f"{pair.case_id} manual label is not on the analysis grid.")
                return source.read(1).astype(bool) & data.valid, "user_verified_screenshot", True
    nws_path = find_nws_path(pair.case_id, shapefile_root)
    if nws_path is None:
        raise ValueError(f"{pair.case_id} has no usable reviewed or DAT label.")
    return create_nws_label(nws_path, data), "NOAA_DAT_partial_reference", False


def model_features(data: AnalysisData, clusters: int, seed: int) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    features, names, diagnostics = build_features(data)
    _, score, _ = cluster_change(features, names, data.valid, clusters, seed)
    combined = np.concatenate([features, score[..., None]], axis=-1).astype("float32")
    combined[~data.valid] = 0.0
    return combined, [*names, "unsupervised_damage_score"], diagnostics


def sample_training_pixels(
    pair: CasePair,
    data: AnalysisData,
    features: np.ndarray,
    label: np.ndarray,
    *,
    seed: int,
    positive_limit: int = 15_000,
    negative_limit: int = 30_000,
    complete_label: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + int(pair.case_id[3:]))
    positive_pool = np.argwhere(label & data.valid)
    exclusion = ndi.binary_dilation(label, iterations=14)
    # DAT may document only some paths in a scene. Do not turn every pixel
    # outside its corridor into a negative label. Use only low-change, distant
    # pixels as trusted negatives and leave unexplained change unlabeled.
    change_score = features[..., -1]
    stable_threshold = float(np.percentile(change_score[data.valid], 45.0))
    if complete_label:
        # Reviewed screenshot labels state that the marked paths are the
        # accepted paths for that scene, so both stable pixels and high-change
        # false positives outside the corridor are useful negatives.
        outside = data.valid & ~exclusion
        stable = np.argwhere(outside & (change_score <= stable_threshold))
        hard_threshold = float(np.percentile(change_score[data.valid], 78.0))
        hard = np.argwhere(outside & (change_score >= hard_threshold))
        negative_pool = np.vstack([stable, hard]) if len(stable) and len(hard) else np.argwhere(outside)
    else:
        trusted_negative = data.valid & ~exclusion & (change_score <= stable_threshold)
        negative_pool = np.argwhere(trusted_negative)
    if not len(positive_pool) or not len(negative_pool):
        raise ValueError(f"{pair.case_id} does not contain usable positive and negative training pixels.")
    positive = positive_pool[
        rng.choice(len(positive_pool), min(positive_limit, len(positive_pool)), replace=False)
    ]
    negative = negative_pool[
        rng.choice(len(negative_pool), min(negative_limit, len(negative_pool)), replace=False)
    ]
    coordinates = np.vstack([positive, negative])
    target = np.concatenate(
        [np.ones(len(positive), dtype="uint8"), np.zeros(len(negative), dtype="uint8")]
    )
    return features[coordinates[:, 0], coordinates[:, 1]], target


def fit_model(x: np.ndarray, y: np.ndarray, seed: int, kind: str = "extra_trees"):
    if kind == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=3,
            max_features=0.75,
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )
    elif kind == "random_forest":
        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported model kind: {kind}")
    model.fit(x, y)
    return model


def infer_probability(
    model,
    features: np.ndarray,
    valid: np.ndarray,
    block_rows: int = 256,
) -> np.ndarray:
    probability = np.zeros(valid.shape, dtype="float32")
    for row in range(0, valid.shape[0], block_rows):
        bottom = min(row + block_rows, valid.shape[0])
        local_valid = valid[row:bottom]
        if not local_valid.any():
            continue
        local_features = features[row:bottom][local_valid]
        probability[row:bottom][local_valid] = model.predict_proba(local_features)[:, 1].astype("float32")
    return probability


def postprocess_probability(
    probability: np.ndarray,
    valid: np.ndarray,
    percentile: float,
    water_mask: np.ndarray | None = None,
    max_paths: int = 6,
    max_gap_pixels: int = 45,
    min_water_fraction: float = 0.30,
    max_bridge_angle_degrees: float = 50.0,
    min_relative_path_score: float = 0.35,
    water_exclusion_buffer_pixels: int = 6,
    exclusion_mask: np.ndarray | None = None,
    crossing_mask: np.ndarray | None = None,
    max_feature_follow_fraction: float = 0.65,
    axis_filter_half_width_pixels: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return extract_multiple_corridors(
        probability,
        valid,
        percentile=percentile,
        water_mask=water_mask,
        max_paths=max_paths,
        min_area_fraction=0.00005,
        max_gap_pixels=max_gap_pixels,
        min_water_fraction=min_water_fraction,
        max_bridge_angle_degrees=max_bridge_angle_degrees,
        min_relative_path_score=min_relative_path_score,
        water_exclusion_buffer_pixels=water_exclusion_buffer_pixels,
        exclusion_mask=exclusion_mask,
        crossing_mask=crossing_mask,
        max_feature_follow_fraction=max_feature_follow_fraction,
        axis_filter_half_width_pixels=axis_filter_half_width_pixels,
    )


def mask_metrics(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    prediction = prediction & valid
    truth = truth & valid
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "dice": float(dice),
        "iou": float(tp / max(tp + fp + fn, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "false_negative_rate": float(fn / max(tp + fn, 1)),
    }


def save_model_case(
    pair: CasePair,
    data: AnalysisData,
    diagnostics: dict[str, np.ndarray],
    probability: np.ndarray,
    corridor: np.ndarray,
    output_root: Path,
    shapefile_root: Path,
    evaluation_type: str,
    postprocess_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case_dir = output_root / "cases" / pair.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    centerlines = centerlines_from_corridor(
        corridor,
        data.transform,
        straighten=bool((postprocess_diagnostics or {}).get("straighten_centerline", False)),
    )
    centerline = (
        centerlines[0]
        if len(centerlines) == 1
        else MultiLineString(centerlines) if centerlines else None
    )
    geometry = _corridor_geometry(corridor, data.transform)
    nws_path = find_nws_path(pair.case_id, shapefile_root)
    nws_metrics, nws_frame = validate_against_nws(nws_path, data, corridor, centerline)
    path_set_metrics, path_matches = evaluate_path_set(centerlines, nws_frame, data.crs)
    water = stable_water_mask(data)

    if postprocess_diagnostics is not None:
        (case_dir / "postprocessing_diagnostics.json").write_text(
            json.dumps(postprocess_diagnostics, indent=2),
            encoding="utf-8",
        )

    _write_raster(case_dir / "model_probability.tif", probability, data.profile, "float32", -9999.0)
    _write_raster(case_dir / "model_damage_mask.tif", corridor, data.profile, "uint8", 0)
    _write_raster(case_dir / "stable_water_mask.tif", water, data.profile, "uint8", 0)
    _write_geometry(case_dir / "model_damage_corridor.geojson", geometry, data.crs, "model_corridor")
    centerline_path = case_dir / "model_path_centerline.geojson"
    centerline_path.unlink(missing_ok=True)
    if centerlines:
        gpd.GeoDataFrame(
            {
                "type": ["model_centerline"] * len(centerlines),
                "path_id": list(range(1, len(centerlines) + 1)),
            },
            geometry=centerlines,
            crs=data.crs,
        ).to_file(centerline_path, driver="GeoJSON")

    extent = _extent(data)
    after_rgb = _rgb(diagnostics["after_normalized"])
    valid_probability = probability[data.valid]
    probability_limit = max(float(np.percentile(valid_probability, 99)), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(after_rgb, extent=extent, origin="upper")
    axes[0].set_title("Full AFTER image", fontweight="bold")
    probability_image = axes[1].imshow(
        np.ma.masked_where(~data.valid, probability),
        extent=extent,
        origin="upper",
        cmap="inferno",
        vmin=0,
        vmax=probability_limit,
    )
    axes[1].set_title("Imagery-only damage probability", fontweight="bold")
    fig.colorbar(probability_image, ax=axes[1], fraction=0.035, pad=0.02)
    axes[2].imshow(after_rgb, extent=extent, origin="upper")
    axes[2].imshow(
        np.ma.masked_where(~corridor, corridor),
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["#FF8C00"]),
        alpha=0.34,
    )
    for path_index, line in enumerate(centerlines):
        x, y = line.xy
        axes[2].plot(x, y, color=PATH_COLORS[path_index % len(PATH_COLORS)], linewidth=3.2)
    axes[2].set_title(f"{len(centerlines)} cleaned path corridor(s)", fontweight="bold")
    for axis in axes:
        axis.set_axis_off()
    fig.suptitle(f"{pair.case_id}: Model Prediction Stages", fontsize=18, fontweight="bold")
    fig.savefig(case_dir / "model_prediction_panel.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(15, 8), constrained_layout=True)
    axis.imshow(after_rgb, extent=extent, origin="upper")
    axis.imshow(
        np.ma.masked_where(~corridor, corridor),
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["#FF8C00"]),
        alpha=0.34,
    )
    for path_index, line in enumerate(centerlines):
        x, y = line.xy
        axis.plot(
            x,
            y,
            color=PATH_COLORS[path_index % len(PATH_COLORS)],
            linewidth=3.2,
            label=f"Predicted path {path_index + 1}",
        )
    if nws_frame is not None and not nws_frame.empty:
        nws_frame.plot(ax=axis, color="#00D9FF", linewidth=2.4, label="Official NWS path")
    axis.set_title(f"{pair.case_id}: Model Path on AFTER Image", fontsize=18, fontweight="bold")
    axis.set_axis_off()
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="lower left", framealpha=0.92)
    fig.savefig(case_dir / "model_final_path_map.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(15, 8), constrained_layout=True)
    axis.imshow(after_rgb, extent=extent, origin="upper")
    axis.imshow(
        np.ma.masked_where(~water, water),
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["#1E90FF"]),
        alpha=0.42,
    )
    axis.imshow(
        np.ma.masked_where(~corridor, corridor),
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["#FF8C00"]),
        alpha=0.30,
    )
    for path_index, line in enumerate(centerlines):
        x, y = line.xy
        axis.plot(x, y, color=PATH_COLORS[path_index % len(PATH_COLORS)], linewidth=3.0)
    axis.set_title(f"{pair.case_id}: Water-Aware Gap Diagnostic", fontsize=18, fontweight="bold")
    axis.set_axis_off()
    fig.savefig(case_dir / "water_gap_diagnostic.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame(path_matches).to_csv(case_dir / "per_path_nws_matching.csv", index=False)
    (case_dir / "path_set_metrics.json").write_text(
        json.dumps({**path_set_metrics, "matches": path_matches}, indent=2),
        encoding="utf-8",
    )

    result = {
        "case_id": pair.case_id,
        "evaluation_type": evaluation_type,
        "path_found": bool(centerlines),
        "predicted_path_count": len(centerlines),
        "nws_used_inside_inference": False,
        **path_set_metrics,
        **{f"nws_{key}": value for key, value in nws_metrics.items()},
        "water_gap_bridge_count": len((postprocess_diagnostics or {}).get("bridges", [])),
    }
    (case_dir / "model_case_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_samples_for_cases(
    pairs: list[CasePair],
    shapefile_root: Path,
    *,
    max_dimension: int,
    clusters: int,
    seed: int,
    positive_limit: int = 6_000,
    negative_limit: int = 12_000,
    manual_label_root: Path | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str], pd.DataFrame]:
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_names: list[str] = []
    label_rows: list[dict[str, Any]] = []
    for pair in pairs:
        data = load_analysis_data(pair, max_dimension=max_dimension)
        features, feature_names, _ = model_features(data, clusters, seed + int(pair.case_id[3:]))
        try:
            label, label_source, complete_label = load_reference_label(
                pair, data, shapefile_root, manual_label_root
            )
        except ValueError:
            continue
        samples[pair.case_id] = sample_training_pixels(
            pair,
            data,
            features,
            label,
            seed=seed,
            positive_limit=positive_limit,
            negative_limit=negative_limit,
            complete_label=complete_label,
        )
        label_rows.append(
            {
                "case_id": pair.case_id,
                "label_source": label_source,
                "label_is_complete": complete_label,
                "positive_pixels": int(label.sum()),
            }
        )
    return samples, feature_names, pd.DataFrame(label_rows)


def run_grouped_model_selection(
    pairs: list[CasePair],
    samples: dict[str, tuple[np.ndarray, np.ndarray]],
    shapefile_root: Path,
    event_groups: dict[str, str],
    *,
    max_dimension: int,
    clusters: int,
    model_kinds: list[str],
    percentiles: list[float],
    folds: int,
    seed: int,
    manual_label_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, float]:
    cases = sorted(samples, key=lambda case: int(case[3:]))
    groups = np.asarray([event_groups.get(case, case) for case in cases])
    unique_groups = np.unique(groups)
    n_splits = min(int(folds), len(unique_groups))
    if n_splits < 3:
        raise ValueError("At least three independent NWS event groups are required.")

    pair_lookup = {pair.case_id: pair for pair in pairs}
    splitter = GroupKFold(n_splits=n_splits)
    metric_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    indexes = np.arange(len(cases))
    for fold, (train_index, test_index) in enumerate(splitter.split(indexes, groups=groups), start=1):
        train_cases = [cases[index] for index in train_index]
        test_cases = [cases[index] for index in test_index]
        for case in train_cases:
            split_rows.append(
                {"fold": fold, "case_id": case, "event_group": event_groups.get(case, case), "role": "train"}
            )
        for case in test_cases:
            split_rows.append(
                {"fold": fold, "case_id": case, "event_group": event_groups.get(case, case), "role": "validation"}
            )

        x = np.vstack([samples[case][0] for case in train_cases])
        y = np.concatenate([samples[case][1] for case in train_cases])
        for kind in model_kinds:
            print(
                f"[model-cv] Fold {fold}/{n_splits}: fitting {kind} "
                f"on {len(y):,} pixels; validating {', '.join(test_cases)}",
                flush=True,
            )
            model = fit_model(x, y, seed + fold, kind=kind)
            for held_out in test_cases:
                pair = pair_lookup[held_out]
                data = load_analysis_data(pair, max_dimension=max_dimension)
                features, _, _ = model_features(data, clusters, seed + int(held_out[3:]))
                probability = infer_probability(model, features, data.valid)
                truth, truth_source, truth_complete = load_reference_label(
                    pair, data, shapefile_root, manual_label_root
                )
                for percentile in percentiles:
                    corridor, diagnostics = postprocess_probability(
                        probability,
                        data.valid,
                        percentile,
                        water_mask=stable_water_mask(data),
                        max_paths=1,
                        max_gap_pixels=0,
                    )
                    metrics = mask_metrics(corridor, truth, data.valid)
                    metric_rows.append(
                        {
                            "fold": fold,
                            "case_id": held_out,
                            "event_group": event_groups.get(held_out, held_out),
                            "model": kind,
                            "probability_percentile": float(percentile),
                            "training_cases": ";".join(train_cases),
                            "truth_source": truth_source,
                            "truth_is_complete": truth_complete,
                            **metrics,
                            "path_found": bool(corridor.any()),
                            "postprocess_reason": diagnostics.get("reason"),
                        }
                    )

    detailed = pd.DataFrame(metric_rows)
    comparison = (
        detailed.groupby(["model", "probability_percentile"], as_index=False)
        .agg(
            mean_dice=("dice", "mean"),
            mean_iou=("iou", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_false_negative_rate=("false_negative_rate", "mean"),
            validation_cases=("case_id", "nunique"),
        )
        .sort_values(["mean_dice", "mean_iou", "mean_recall"], ascending=False)
        .reset_index(drop=True)
    )
    winner = comparison.iloc[0]
    return detailed, comparison, pd.DataFrame(split_rows), str(winner["model"]), float(winner["probability_percentile"])


def run_leave_one_case_out(
    pairs: list[CasePair],
    samples: dict[str, tuple[np.ndarray, np.ndarray]],
    shapefile_root: Path,
    *,
    max_dimension: int,
    clusters: int,
    percentile: float,
    seed: int,
    manual_label_root: Path | None = None,
) -> pd.DataFrame:
    rows = []
    pair_lookup = {pair.case_id: pair for pair in pairs}
    for held_out in sorted(samples, key=lambda case: int(case[3:])):
        training_cases = [case for case in samples if case != held_out]
        x = np.vstack([samples[case][0] for case in training_cases])
        y = np.concatenate([samples[case][1] for case in training_cases])
        model = fit_model(x, y, seed)
        pair = pair_lookup[held_out]
        data = load_analysis_data(pair, max_dimension=max_dimension)
        features, _, _ = model_features(data, clusters, seed + int(held_out[3:]))
        probability = infer_probability(model, features, data.valid)
        corridor, diagnostics = postprocess_probability(
            probability,
            data.valid,
            percentile,
            water_mask=stable_water_mask(data),
        )
        label, label_source, label_complete = load_reference_label(
            pair, data, shapefile_root, manual_label_root
        )
        metrics = mask_metrics(corridor, label, data.valid)
        rows.append(
            {
                "case_id": held_out,
                "evaluation_type": "leave-one-tornado-out",
                "training_cases": ";".join(training_cases),
                "truth_source": label_source,
                "truth_is_complete": label_complete,
                **metrics,
                "path_found": bool(corridor.any()),
                "postprocess_reason": diagnostics.get("reason"),
            }
        )
    return pd.DataFrame(rows)


def save_model_bundle(
    model,
    feature_names: list[str],
    training_cases: list[str],
    output_dir: Path,
    config: dict[str, Any],
    validation_summary: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "tornado_path_model.joblib"
    temporary_path = output_dir / "tornado_path_model.joblib.tmp"
    joblib.dump(model, temporary_path, compress=3)
    temporary_path.replace(model_path)
    (output_dir / "feature_schema.json").write_text(
        json.dumps({"features": feature_names}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_type": type(model).__name__,
                "training_cases": training_cases,
                "label_source": "NWS paths buffered by reported width",
                "validation": "event-grouped cross-validation",
                "nws_used_at_inference": False,
                "configuration": config,
                "validation_summary": validation_summary or {},
                "known_limitation": "NWS labels are available for only a subset of cases; imagery-only cases remain unverified.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

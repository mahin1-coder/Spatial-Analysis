"""Phase 4 before/after statistical change analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from .config import ProjectConfig

LOGGER = logging.getLogger(__name__)


def _read_stack(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(masked=True).astype("float32")
        profile = src.profile.copy()
    return np.ma.filled(arr, np.nan), profile


def _write_diff(path: Path, diff: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(dtype="float32", nodata=np.nan, compress="deflate", BIGTIFF="IF_SAFER")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(diff.astype("float32"))


def _quicklook_band(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    low, high = np.nanpercentile(finite, [2, 98])
    if high <= low:
        return arr
    return np.clip((arr - low) / (high - low), 0, 1)


def _save_band_map(arr: np.ndarray, title: str, out_path: Path, cmap: str = "viridis") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.imshow(_quicklook_band(arr), cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_hist(before: np.ndarray, after: np.ndarray, diff: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for label, data, alpha in [("BEFORE", before, 0.5), ("AFTER", after, 0.5), ("DIFF", diff, 0.45)]:
        finite = data[np.isfinite(data)]
        if finite.size:
            sample = finite if finite.size <= 200000 else np.random.default_rng(42).choice(finite, 200000, replace=False)
            plt.hist(sample, bins=80, alpha=alpha, label=label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_corr(stack: np.ndarray, title: str, out_path: Path) -> None:
    vectors = []
    for band in stack:
        finite = band[np.isfinite(band)]
        vectors.append(finite)
    min_len = min((len(v) for v in vectors), default=0)
    if min_len < 2:
        return
    sample_len = min(min_len, 100000)
    rng = np.random.default_rng(42)
    sampled = []
    for vector in vectors:
        if len(vector) > sample_len:
            sampled.append(rng.choice(vector, sample_len, replace=False))
        else:
            sampled.append(vector[:sample_len])
    corr = np.corrcoef(np.vstack(sampled))
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(sampled)), [f"B{i+1}" for i in range(len(sampled))])
    plt.yticks(range(len(sampled)), [f"B{i+1}" for i in range(len(sampled))])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _band_stats(tor_id: str, band_index: int, before: np.ndarray, after: np.ndarray, diff: np.ndarray) -> dict[str, object]:
    abs_diff = np.abs(diff)
    finite_diff = diff[np.isfinite(diff)]
    finite_abs = abs_diff[np.isfinite(abs_diff)]
    return {
        "tornado_id": tor_id,
        "band": band_index,
        "before_mean": float(np.nanmean(before)),
        "after_mean": float(np.nanmean(after)),
        "mean_shift": float(np.nanmean(after) - np.nanmean(before)),
        "before_median": float(np.nanmedian(before)),
        "after_median": float(np.nanmedian(after)),
        "median_shift": float(np.nanmedian(after) - np.nanmedian(before)),
        "before_std": float(np.nanstd(before)),
        "after_std": float(np.nanstd(after)),
        "std_shift": float(np.nanstd(after) - np.nanstd(before)),
        "variance_shift": float(np.nanvar(after) - np.nanvar(before)),
        "absolute_mean_change": float(np.nanmean(finite_abs)) if finite_abs.size else np.nan,
        "diff_min": float(np.nanmin(finite_diff)) if finite_diff.size else np.nan,
        "diff_max": float(np.nanmax(finite_diff)) if finite_diff.size else np.nan,
        "diff_p05": float(np.nanpercentile(finite_diff, 5)) if finite_diff.size else np.nan,
        "diff_p50": float(np.nanpercentile(finite_diff, 50)) if finite_diff.size else np.nan,
        "diff_p95": float(np.nanpercentile(finite_diff, 95)) if finite_diff.size else np.nan,
        "spatial_concentration_top_5pct_abs_change": float(
            np.nansum(finite_abs[finite_abs >= np.nanpercentile(finite_abs, 95)]) / np.nansum(finite_abs)
        )
        if finite_abs.size and np.nansum(finite_abs) > 0
        else np.nan,
    }


def run_change_analysis(config: ProjectConfig) -> pd.DataFrame:
    """Generate difference rasters, plots, and change-statistics CSVs."""

    preprocessing_path = config.reports_dir / "preprocessing_summary.csv"
    if not preprocessing_path.exists():
        raise FileNotFoundError("Run preprocessing before analysis.")

    summary = pd.read_csv(preprocessing_path)
    rows: list[dict[str, object]] = []
    for item in summary.to_dict("records"):
        tor_id = item["tornado_id"]
        if item.get("status") != "OK":
            rows.append({"tornado_id": tor_id, "band": "", "analysis_status": "SKIPPED", "error": item.get("error", "")})
            continue
        try:
            before_path = Path(item["before_aligned_path"])
            after_path = Path(item["after_aligned_path"])
            before, profile = _read_stack(before_path)
            after, _ = _read_stack(after_path)
            diff = after - before

            plot_dir = config.outputs_dir / "plots" / tor_id
            diff_path = plot_dir / "difference_stack.tif"
            _write_diff(diff_path, diff, profile)

            for idx in range(before.shape[0]):
                band = idx + 1
                _save_band_map(before[idx], f"{tor_id} BEFORE Band {band}", plot_dir / f"before_band_{band}.png")
                _save_band_map(after[idx], f"{tor_id} AFTER Band {band}", plot_dir / f"after_band_{band}.png")
                _save_band_map(diff[idx], f"{tor_id} AFTER - BEFORE Band {band}", plot_dir / f"difference_band_{band}.png", cmap="coolwarm")
                _save_hist(before[idx], after[idx], diff[idx], f"{tor_id} Band {band} Distribution Shift", plot_dir / f"hist_band_{band}.png")
                band_row = _band_stats(tor_id, band, before[idx], after[idx], diff[idx])
                band_row.update({"analysis_status": "OK", "difference_raster": str(diff_path), "error": ""})
                rows.append(band_row)

            _save_corr(before, f"{tor_id} BEFORE Channel Correlation", plot_dir / "correlation_before.png")
            _save_corr(after, f"{tor_id} AFTER Channel Correlation", plot_dir / "correlation_after.png")
            _save_corr(diff, f"{tor_id} Difference Channel Correlation", plot_dir / "correlation_difference.png")
        except Exception as exc:
            LOGGER.warning("Analysis failed for %s: %s", tor_id, exc)
            rows.append({"tornado_id": tor_id, "band": "", "analysis_status": "FAILED", "error": str(exc)})

    stats = pd.DataFrame(rows)
    stats_path = config.reports_dir / "statistical_summary.csv"
    stats.to_csv(stats_path, index=False)

    if "absolute_mean_change" in stats.columns:
        ranked = (
            stats[stats["analysis_status"].eq("OK")]
            .groupby("tornado_id", as_index=False)["absolute_mean_change"]
            .mean()
            .sort_values("absolute_mean_change", ascending=False)
        )
    else:
        ranked = pd.DataFrame(columns=["tornado_id", "absolute_mean_change"])
    ranked.to_csv(config.reports_dir / "case_signal_ranking.csv", index=False)
    LOGGER.info("Wrote statistical summary: %s", stats_path)
    return stats

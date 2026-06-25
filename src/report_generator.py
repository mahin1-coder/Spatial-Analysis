"""Per-case exploratory data analysis (EDA) reports.

Reproduces the report structure used in the project's own prior EDA work
(Gloria Le, "Datasets Analyzing Tornado Research", 02/12/2026) - which
covered only TOR5 and TOR62 by hand - for every tornado case in the
dataset: per-channel BEFORE/AFTER comparison, a difference map (red =
increased after, blue = decreased after), distribution-shift histograms,
and an auto-generated interpretation summary.

Like the rest of this pipeline, corrupted/unreadable raster tiles are
never treated as valid data: readability is computed and reported
explicitly, and all statistics are NaN-aware so corrupted pixels simply
drop out of the calculation instead of being silently included.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import rasterio

from .config import ProjectConfig
from .preprocessing import pair_registered_rasters

LOGGER = logging.getLogger(__name__)


def _read_band_stack(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read all bands of a raster, leaving NaN wherever a tile fails to read.

    Returns (stack with shape (bands, H, W), per-pixel readable mask (H, W)
    that is True only where every band read successfully and is finite).
    """

    with rasterio.open(path) as src:
        stack = np.full((src.count, src.height, src.width), np.nan, dtype="float32")
        readable = np.zeros((src.height, src.width), dtype=bool)
        for _, window in src.block_windows(1):
            try:
                block = np.ma.filled(src.read(window=window, masked=True).astype("float32"), np.nan)
            except Exception:
                continue
            row_slice, col_slice = window.toslices()
            stack[:, row_slice, col_slice] = block
            readable[row_slice, col_slice] = np.isfinite(block).all(axis=0)
    return stack, readable


def _channel_stats(stack: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for c in range(stack.shape[0]):
        band = stack[c]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            rows.append({"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan})
        else:
            rows.append(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                }
            )
    return rows


def _correlation_matrix(stack: np.ndarray) -> np.ndarray:
    flat = stack.reshape(stack.shape[0], -1)
    valid_cols = np.isfinite(flat).all(axis=0)
    if valid_cols.sum() < 2:
        return np.full((stack.shape[0], stack.shape[0]), np.nan)
    return np.corrcoef(flat[:, valid_cols])


def _stretch(band: np.ndarray) -> np.ndarray:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band)
    low, high = np.nanpercentile(finite, [2, 98])
    if high <= low:
        return np.zeros_like(band)
    return np.clip((band - low) / (high - low), 0, 1)


def _add_title_page(pdf: PdfPages, tor_id: str, before_path: Path, after_path: Path, meta: dict[str, object]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.08, 0.92, f"{tor_id}: Before/After EDA Report", fontsize=22, fontweight="bold")
    fig.text(0.08, 0.87, before_path.name, fontsize=11, family="monospace")
    fig.text(0.08, 0.84, after_path.name, fontsize=11, family="monospace")

    lines = [
        f"Grid size: {meta['height']} x {meta['width']}    Channels: {meta['count']}    dtype: {meta['dtype']}",
        f"Readable pixels - BEFORE: {meta['before_readable_pct']:.1f}%    AFTER: {meta['after_readable_pct']:.1f}%"
        f"    Both: {meta['both_readable_pct']:.1f}%",
        "",
        "Per-channel statistics (NaN/unreadable pixels excluded):",
    ]
    y = 0.78
    for line in lines:
        fig.text(0.08, y, line, fontsize=12)
        y -= 0.035

    header = f"{'Ch':<4}{'Before min/max':<22}{'Before mean/std':<22}{'After min/max':<22}{'After mean/std':<22}"
    fig.text(0.08, y, header, fontsize=10, family="monospace", fontweight="bold")
    y -= 0.03
    for c, (b, a) in enumerate(zip(meta["before_stats"], meta["after_stats"])):
        row = (
            f"{c:<4}"
            + f"{b['min']:.4f}/{b['max']:.4f}".ljust(22)
            + f"{b['mean']:.4f}/{b['std']:.4f}".ljust(22)
            + f"{a['min']:.4f}/{a['max']:.4f}".ljust(22)
            + f"{a['mean']:.4f}/{a['std']:.4f}".ljust(22)
        )
        fig.text(0.08, y, row, fontsize=10, family="monospace")
        y -= 0.03

    fig.text(0.08, y - 0.02, "AFTER channel correlation matrix:", fontsize=12)
    y -= 0.055
    corr = meta["after_corr"]
    for r in range(corr.shape[0]):
        row_text = "  ".join(f"{corr[r, c]:+.2f}" for c in range(corr.shape[1]))
        fig.text(0.10, y, row_text, fontsize=10, family="monospace")
        y -= 0.03

    pdf.savefig(fig)
    plt.close(fig)


def _add_channel_grid(
    pdf: PdfPages, tor_id: str, title: str, before: np.ndarray, after: np.ndarray, n_channels: int
) -> None:
    fig, axes = plt.subplots(n_channels, 2, figsize=(9, 2.2 * n_channels))
    axes = np.atleast_2d(axes)
    for c in range(n_channels):
        axes[c, 0].imshow(_stretch(before[c]), cmap="gray")
        axes[c, 0].set_title(f"Channel {c} - Before" if c == 0 else f"Channel {c}", fontsize=10)
        axes[c, 0].axis("off")
        axes[c, 1].imshow(_stretch(after[c]), cmap="gray")
        axes[c, 1].set_title(f"Channel {c} - After" if c == 0 else f"Channel {c}", fontsize=10)
        axes[c, 1].axis("off")
    fig.suptitle(f"{tor_id}: {title}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def _add_difference_grid(pdf: PdfPages, tor_id: str, before: np.ndarray, after: np.ndarray, n_channels: int) -> None:
    diff = after - before
    cols = 3
    rows = (n_channels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2))
    axes = np.atleast_2d(axes)
    for c in range(rows * cols):
        ax = axes[c // cols][c % cols]
        if c >= n_channels:
            ax.axis("off")
            continue
        band = diff[c]
        finite = band[np.isfinite(band)]
        v = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        v = v if v > 0 else 1.0
        im = ax.imshow(band, cmap="bwr", vmin=-v, vmax=v)
        ax.set_title(f"Channel {c}", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{tor_id}: Difference (After - Before)  |  Red = increased after, Blue = decreased after",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _add_distribution_grid(pdf: PdfPages, tor_id: str, before: np.ndarray, after: np.ndarray, n_channels: int) -> None:
    cols = 3
    rows = (n_channels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2))
    axes = np.atleast_2d(axes)
    for c in range(rows * cols):
        ax = axes[c // cols][c % cols]
        if c >= n_channels:
            ax.axis("off")
            continue
        b = before[c][np.isfinite(before[c])]
        a = after[c][np.isfinite(after[c])]
        if b.size:
            ax.hist(b, bins=80, alpha=0.5, label="Before", color="#1F77B4")
        if a.size:
            ax.hist(a, bins=80, alpha=0.5, label="After", color="#E03131")
        ax.set_title(f"Channel {c}", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f"{tor_id}: Distribution Shift (Before vs After)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _add_interpretation_page(pdf: PdfPages, tor_id: str, meta: dict[str, object]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.08, 0.92, f"{tor_id}: Interpretation", fontsize=20, fontweight="bold")

    before_stats = meta["before_stats"]
    after_stats = meta["after_stats"]
    mean_deltas = [a["mean"] - b["mean"] for a, b in zip(after_stats, before_stats)]
    std_deltas = [a["std"] - b["std"] for a, b in zip(after_stats, before_stats)]
    finite_mean_deltas = [d for d in mean_deltas if np.isfinite(d)]
    strongest_mean_shift = int(np.nanargmax(np.abs(mean_deltas))) if finite_mean_deltas else None
    n_variance_up = sum(1 for d in std_deltas if np.isfinite(d) and d > 0)

    lines = [
        f"Readable coverage: {meta['both_readable_pct']:.1f}% of the scene has usable BEFORE+AFTER data "
        f"(the rest is corrupted/unreadable source imagery and is excluded from every statistic above).",
        "",
        f"{n_variance_up} of {len(std_deltas)} channels show higher variance after the event than before.",
    ]
    if strongest_mean_shift is not None:
        lines.append(
            f"Channel {strongest_mean_shift} shows the largest mean shift "
            f"({mean_deltas[strongest_mean_shift]:+.4f}), making it the most informative single channel for this case."
        )
    corr = meta["after_corr"]
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    if off_diag.size and np.isfinite(off_diag).any():
        avg_corr = float(np.nanmean(off_diag))
        lines.append(
            f"Average AFTER cross-channel correlation is {avg_corr:+.2f} - "
            + (
                "channels are moving together (simple, mostly redundant signal)."
                if avg_corr > 0.6
                else "channels are not all moving together (richer, more independent signal across bands)."
            )
        )

    y = 0.80
    for line in lines:
        fig.text(0.08, y, line, fontsize=13, wrap=True)
        y -= 0.07

    pdf.savefig(fig)
    plt.close(fig)


def generate_case_eda_report(tor_id: str, before_path: Path, after_path: Path, out_path: Path) -> dict[str, object]:
    """Build one PDF EDA report for a single BEFORE/AFTER pair, Gloria-report style."""

    before, before_readable = _read_band_stack(before_path)
    after, after_readable = _read_band_stack(after_path)
    n_channels = min(before.shape[0], after.shape[0])
    before, after = before[:n_channels], after[:n_channels]

    both_readable = before_readable & after_readable
    meta = {
        "height": before.shape[1],
        "width": before.shape[2],
        "count": n_channels,
        "dtype": str(before.dtype),
        "before_readable_pct": 100.0 * before_readable.mean(),
        "after_readable_pct": 100.0 * after_readable.mean(),
        "both_readable_pct": 100.0 * both_readable.mean(),
        "before_stats": _channel_stats(before),
        "after_stats": _channel_stats(after),
        "after_corr": _correlation_matrix(after),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        _add_title_page(pdf, tor_id, before_path, after_path, meta)
        _add_channel_grid(pdf, tor_id, "Before vs After (per channel)", before, after, n_channels)
        _add_difference_grid(pdf, tor_id, before, after, n_channels)
        _add_distribution_grid(pdf, tor_id, before, after, n_channels)
        _add_interpretation_page(pdf, tor_id, meta)

    LOGGER.info("EDA report written for %s -> %s", tor_id, out_path)
    return {
        "tornado_id": tor_id,
        "report_path": str(out_path),
        "both_readable_pct": meta["both_readable_pct"],
    }


def generate_all_eda_reports(config: ProjectConfig) -> list[dict[str, object]]:
    """Build one EDA report per tornado case, matching the project's prior manual EDA work."""

    pairs = pair_registered_rasters(config)
    out_root = config.outputs_dir / "reports" / "eda"
    results: list[dict[str, object]] = []
    for pair in pairs.to_dict("records"):
        tor_id = pair["tornado_id"]
        if pair["pair_status"] != "OK":
            results.append({"tornado_id": tor_id, "status": "SKIPPED", "error": pair.get("warnings", "")})
            continue
        out_path = out_root / f"{tor_id}_eda_report.pdf"
        try:
            result = generate_case_eda_report(tor_id, Path(pair["before_path"]), Path(pair["after_path"]), out_path)
            result["status"] = "OK"
            results.append(result)
        except Exception as exc:
            LOGGER.warning("EDA report failed for %s: %s", tor_id, exc)
            results.append({"tornado_id": tor_id, "status": "FAILED", "error": str(exc)})
    return results

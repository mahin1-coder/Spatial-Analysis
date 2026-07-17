"""Batch prediction for folders containing BEFORE/AFTER image pairs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .pairing import find_image_pairs, write_pairing_report

LOGGER = logging.getLogger(__name__)

def find_before_after_pairs(folder: Path) -> pd.DataFrame:
    """Find BEFORE/AFTER pairs in a folder tree using filename conventions."""

    df = find_image_pairs(folder)
    if "error" not in df.columns:
        df["error"] = df["reason"]
    return df


def _build_batch_contact_sheet(rows: list[dict[str, object]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ok_rows = [row for row in rows if row.get("status") == "OK" and Path(str(row.get("showcase_path", ""))).exists()]
    if not ok_rows:
        return

    cols = 2 if len(ok_rows) <= 4 else 3
    grid_rows = (len(ok_rows) + cols - 1) // cols
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 5.4, grid_rows * 4.0), facecolor="white")
    axes = np.atleast_2d(axes)

    for idx in range(grid_rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.set_axis_off()
        if idx >= len(ok_rows):
            continue
        item = ok_rows[idx]
        image = plt.imread(str(item["showcase_path"]))
        ax.imshow(image)
        ax.set_title(str(item["case_id"]), fontsize=10, fontweight="bold")

    fig.suptitle("Tornado Damage Path Batch Results", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_pair_folder(
    config: ProjectConfig,
    folder: Path,
    batch_name: str = "new_batch",
    nws_shapefile: Path | None = None,
) -> pd.DataFrame:
    """Run prediction for every detected BEFORE/AFTER pair in a folder."""

    from .custom_pair import analyze_custom_pair

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    batch_root = config.outputs_dir / "predictions" / "batch" / batch_name
    batch_root.mkdir(parents=True, exist_ok=True)

    pairs = find_before_after_pairs(folder)
    pairs.to_csv(batch_root / "detected_pairs.csv", index=False)
    write_pairing_report(pairs, config.reports_dir / "pairing_report.csv")

    rows: list[dict[str, object]] = []
    for pair in pairs.to_dict("records"):
        if pair["status"] != "OK":
            rows.append(pair)
            continue
        try:
            case_id = str(pair["case_id"])
            result = analyze_custom_pair(
                config,
                Path(str(pair["before_path"])),
                Path(str(pair["after_path"])),
                case_id,
                nws_shapefile,
                output_dir=batch_root / case_id,
            )
            rows.append({**pair, **result, "status": result.get("status", "OK")})
        except Exception as exc:
            LOGGER.warning("Batch prediction failed for %s: %s", pair.get("case_id"), exc)
            rows.append({**pair, "status": "FAILED", "error": str(exc)})

    summary = pd.DataFrame(rows)
    summary_path = batch_root / "batch_prediction_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(batch_root / "batch_summary.csv", index=False)
    _build_batch_contact_sheet(rows, batch_root / "batch_contact_sheet.png")
    _write_batch_html(summary, batch_root / "batch_report.html")
    return summary


def _write_batch_html(summary: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for item in summary.to_dict("records"):
        img = item.get("showcase_path", "")
        img_path = Path(str(img)) if img else None
        if img_path and img_path.exists():
            try:
                src = img_path.relative_to(out_path.parent)
            except ValueError:
                src = img_path
            img_html = f'<img src="{src}" style="max-width:360px">'
        else:
            img_html = ""
        rows.append(
            "<tr>"
            f"<td>{item.get('case_id', '')}</td>"
            f"<td>{item.get('status', '')}</td>"
            f"<td>{item.get('valid_pixels', '')}</td>"
            f"<td>{item.get('predicted_damage_pixels', '')}</td>"
            f"<td>{item.get('error', item.get('reason', ''))}</td>"
            f"<td>{img_html}</td>"
            "</tr>"
        )
    html = """<!doctype html>
<html><head><meta charset="utf-8"><title>Tornado Batch Report</title>
<style>body{font-family:Arial,sans-serif;margin:28px;color:#222}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;vertical-align:top}th{background:#f4f4f4;text-align:left}</style>
</head><body><h1>Tornado Batch Report</h1>
<p>Results are candidate tornado-damage corridors, not confirmed tornado paths unless validated against official ground truth.</p>
<table><thead><tr><th>Case</th><th>Status</th><th>Valid pixels</th><th>Predicted damage pixels</th><th>Notes</th><th>Preview</th></tr></thead>
<tbody>
""" + "\n".join(rows) + """
</tbody></table></body></html>
"""
    out_path.write_text(html)

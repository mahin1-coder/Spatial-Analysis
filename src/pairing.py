"""Robust BEFORE/AFTER raster pairing.

The important rule for this project is that ambiguity must be reported, not
resolved silently. A wrong pair can make a correct NWS path look wrong.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from .utils import detect_tornado_id

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

BEFORE_PATTERNS = (
    r"before",
    r"pre",
    r"pre[-_ ]?event",
    r"pre[-_ ]?storm",
)
AFTER_PATTERNS = (
    r"after",
    r"post",
    r"post[-_ ]?event",
    r"post[-_ ]?storm",
)


@dataclass(frozen=True)
class PairingOptions:
    """Filename pairing options."""

    extensions: frozenset[str] = frozenset(IMAGE_EXTENSIONS)


def detect_role(path: Path) -> str | None:
    """Return BEFORE, AFTER, or None from filename tokens."""

    name = path.stem.lower()
    before_hit = any(re.search(rf"(^|[^a-z0-9]){pat}([^a-z0-9]|$)", name) for pat in BEFORE_PATTERNS)
    after_hit = any(re.search(rf"(^|[^a-z0-9]){pat}([^a-z0-9]|$)", name) for pat in AFTER_PATTERNS)
    if before_hit and not after_hit:
        return "BEFORE"
    if after_hit and not before_hit:
        return "AFTER"
    return None


def _normalize_key(path: Path, role: str) -> str:
    tor_id = detect_tornado_id(path)
    if tor_id:
        return tor_id

    key = path.stem.lower()
    key = re.sub(r"\d{4}[-_]\d{2}[-_]\d{2}", "", key)
    patterns = BEFORE_PATTERNS if role == "BEFORE" else AFTER_PATTERNS
    for pat in patterns:
        key = re.sub(rf"(^|[^a-z0-9]){pat}([^a-z0-9]|$)", "_", key)
    key = re.sub(r"\b(best|image|satellite|landsat|sentinel|planet|tornado|damage|sr|win\d+d)\b", "_", key)
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key or path.stem.lower()


def find_image_pairs(folder: Path, options: PairingOptions | None = None) -> pd.DataFrame:
    """Find BEFORE/AFTER pairs under a folder and explain every rejection."""

    options = options or PairingOptions()
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in options.extensions)
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: {"BEFORE": [], "AFTER": []})
    rows: list[dict[str, object]] = []

    for path in files:
        role = detect_role(path)
        if role is None:
            rows.append(
                {
                    "case_id": "",
                    "pair_key": "",
                    "tornado_id": detect_tornado_id(path) or "",
                    "before_path": "",
                    "after_path": "",
                    "before_candidate_count": 0,
                    "after_candidate_count": 0,
                    "status": "UNMATCHED",
                    "reason": f"filename does not clearly identify BEFORE or AFTER: {path}",
                }
            )
            continue
        grouped[_normalize_key(path, role)][role].append(path)

    case_index = 1
    for key, item in sorted(grouped.items()):
        before = item["BEFORE"]
        after = item["AFTER"]
        status = "OK"
        reasons: list[str] = []
        if not before:
            status = "MISSING_BEFORE"
            reasons.append("missing BEFORE candidate")
        if not after:
            status = "MISSING_AFTER" if status == "OK" else "MISSING_PAIR"
            reasons.append("missing AFTER candidate")
        if len(before) > 1:
            status = "AMBIGUOUS_BEFORE"
            reasons.append(f"multiple BEFORE candidates: {'; '.join(str(p) for p in before)}")
        if len(after) > 1:
            status = "AMBIGUOUS_AFTER" if status == "OK" else f"{status};AMBIGUOUS_AFTER"
            reasons.append(f"multiple AFTER candidates: {'; '.join(str(p) for p in after)}")

        tor_id = detect_tornado_id(before[0]) if before else detect_tornado_id(after[0]) if after else None
        rows.append(
            {
                "case_id": f"case_{case_index:03d}_{key}" if status == "OK" else "",
                "pair_key": key,
                "tornado_id": tor_id or "",
                "before_path": str(before[0]) if len(before) == 1 else "",
                "after_path": str(after[0]) if len(after) == 1 else "",
                "before_candidate_count": len(before),
                "after_candidate_count": len(after),
                "status": status,
                "reason": "; ".join(reasons),
            }
        )
        if status == "OK":
            case_index += 1

    return pd.DataFrame(rows)


def write_pairing_report(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

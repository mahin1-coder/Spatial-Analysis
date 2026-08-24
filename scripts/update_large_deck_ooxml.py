#!/usr/bin/env python3
"""Replace selected images/text in a very large PPTX without rebuilding the deck."""
from __future__ import annotations

import csv
import io
import posixpath
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs_all_cases/presentation/tornado_damage_path_analysis_all_29_cases_water_direction_fixed.pptx"
OUTPUT = ROOT / "outputs_hybrid_v7/presentation/tornado_damage_path_analysis_corrected_hybrid_v7.pptx"
RESULTS = ROOT / "outputs_hybrid_v7"

CASE_ORDER = [
    "TOR5", "TOR7", "TOR10", "TOR11", "TOR12", "TOR13", "TOR15", "TOR16", "TOR18", "TOR24",
    "TOR61", "TOR62", "TOR66", "TOR68", "TOR69", "TOR70", "TOR77", "TOR78", "TOR90", "TOR91",
    "TOR95", "TOR101", "TOR102", "TOR105", "TOR111", "TOR112", "TOR114", "TOR115", "TOR123",
]
CASES = {}
for index, case_id in enumerate(CASE_ORDER):
    CASES[3 + index * 9 + 8] = (case_id, "model_prediction_panel.png")
    CASES[3 + index * 9 + 9] = (case_id, "model_final_path.png")

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix if prefix != "pr" else "", uri)


def largest_picture(slide_xml: bytes) -> tuple[str, float]:
    root = ET.fromstring(slide_xml)
    candidates = []
    for picture in root.findall(".//p:pic", NS):
        blip = picture.find(".//a:blip", NS)
        extent = picture.find(".//a:xfrm/a:ext", NS)
        if blip is None or extent is None:
            continue
        relationship = blip.get(f"{{{NS['r']}}}embed")
        area = float(extent.get("cx", "0")) * float(extent.get("cy", "0"))
        candidates.append((relationship, area))
    if not candidates:
        raise RuntimeError("No picture found")
    return max(candidates, key=lambda item: item[1])


def media_target(rels_xml: bytes, relationship_id: str) -> str:
    root = ET.fromstring(rels_xml)
    for relationship in root.findall("pr:Relationship", NS):
        if relationship.get("Id") == relationship_id:
            target = relationship.get("Target")
            return posixpath.normpath(posixpath.join("ppt/slides", target))
    raise RuntimeError(f"Missing relationship {relationship_id}")


def fit_image(path: Path, aspect: float, extension: str) -> bytes:
    image = Image.open(path).convert("RGB")
    height = 1400
    width = max(1, round(height * aspect))
    scale = min(width / image.width, height / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    buffer = io.BytesIO()
    if extension.lower() in {".jpg", ".jpeg"}:
        canvas.save(buffer, format="JPEG", quality=94, optimize=True)
    else:
        canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def replace_status(slide_xml: bytes, case_id: str, count: str, nws_available: bool) -> bytes:
    root = ET.fromstring(slide_xml)
    verification = "cyan = official NWS/DAT verification" if nws_available else "no official NWS/DAT reference available"
    replacement = f"{count} model-generated path(s) | imagery-only inference | {verification}"
    for text in root.findall(".//a:t", NS):
        if text.text and "imagery candidate(s)" in text.text:
            text.text = replacement
            break
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    with (RESULTS / "reports/deployment_results.csv").open() as handle:
        rows = {row["case_id"]: row for row in csv.DictReader(handle)}

    replacements: dict[str, bytes] = {}
    with zipfile.ZipFile(SOURCE) as archive:
        for slide_number, (case_id, image_name) in CASES.items():
            slide_path = f"ppt/slides/slide{slide_number}.xml"
            rels_path = f"ppt/slides/_rels/slide{slide_number}.xml.rels"
            slide_xml = archive.read(slide_path)
            relationship_id, area = largest_picture(slide_xml)
            target = media_target(archive.read(rels_path), relationship_id)
            replacements[target] = fit_image(
                RESULTS / "cases" / case_id / image_name,
                aspect=max(area, 1) ** 0.0 * _frame_aspect(slide_xml, relationship_id),
                extension=Path(target).suffix,
            )
            if image_name == "model_final_path.png":
                row = rows[case_id]
                replacements[slide_path] = replace_status(
                    slide_xml,
                    case_id,
                    row["path_count"],
                    row.get("nws_dat_reference_available", "False").lower() == "true",
                )

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for item in archive.infolist():
                output.writestr(item, replacements.get(item.filename, archive.read(item.filename)))

    with zipfile.ZipFile(OUTPUT) as check:
        bad = check.testzip()
        if bad:
            raise RuntimeError(f"Corrupt PPTX entry: {bad}")
    print(OUTPUT)


def _frame_aspect(slide_xml: bytes, relationship_id: str) -> float:
    root = ET.fromstring(slide_xml)
    for picture in root.findall(".//p:pic", NS):
        blip = picture.find(".//a:blip", NS)
        if blip is None or blip.get(f"{{{NS['r']}}}embed") != relationship_id:
            continue
        extent = picture.find(".//a:xfrm/a:ext", NS)
        return float(extent.get("cx", "1")) / max(float(extent.get("cy", "1")), 1.0)
    return 16 / 9


if __name__ == "__main__":
    main()

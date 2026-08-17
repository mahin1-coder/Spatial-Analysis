import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = process.env.TORNADO_PROJECT_ROOT ?? process.cwd();
const ASSETS = process.env.TORNADO_DECK_ASSETS ?? `${ROOT}/outputs_hybrid_v7/presentation_assets`;
const OUTPUT = process.env.TORNADO_DECK_OUTPUT ?? `${ROOT}/outputs_hybrid_v8/presentation/tornado_damage_path_analysis_evidence_deck.pptx`;
const CASES = [
  "TOR5", "TOR7", "TOR10", "TOR11", "TOR12", "TOR13", "TOR15", "TOR16", "TOR18", "TOR24",
  "TOR61", "TOR62", "TOR66", "TOR68", "TOR69", "TOR70", "TOR77", "TOR78", "TOR90", "TOR91",
  "TOR95", "TOR101", "TOR102", "TOR105", "TOR111", "TOR112", "TOR114", "TOR115", "TOR123",
];

const C = {
  ink: "#10231D", muted: "#5A6963", teal: "#007F73", pale: "#E9F3EF",
  rule: "#CEDBD5", yellow: "#FFCA0A", cyan: "#00B8D9", blue: "#249DE3", white: "#FFFFFF",
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",");
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(header.map((key, index) => [key, values[index] ?? ""]));
  });
}

async function imageBytes(caseId, name) {
  return new Uint8Array(await fs.readFile(`${ASSETS}/${caseId}/${name}.jpg`));
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox", position, fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = { fontFamily: "Aptos", fontSize: 18, color: C.ink, ...style };
  return shape;
}

function addHeader(slide, caseId, title, section, page) {
  slide.background.fill = C.white;
  addText(slide, section.toUpperCase(), { left: 58, top: 26, width: 340, height: 24 },
    { fontSize: 14, bold: true, color: C.teal });
  addText(slide, `${caseId}: ${title}`, { left: 58, top: 55, width: 1130, height: 48 },
    { fontSize: 34, bold: true, color: C.ink });
  const rule = slide.shapes.add({
    geometry: "rect", position: { left: 58, top: 108, width: 1164, height: 2 },
    fill: C.rule, line: { style: "solid", fill: C.rule, width: 0 },
  });
  rule.name = "header-rule";
  addText(slide, String(page).padStart(3, "0"), { left: 1156, top: 680, width: 66, height: 18 },
    { fontSize: 12, bold: true, color: C.teal, alignment: "right" });
}

function addNotes(slide, sources, method = "") {
  const lines = [];
  if (method) lines.push(method, "");
  lines.push("[Sources]");
  for (const source of sources) lines.push(`- ${source}`);
  slide.speakerNotes.textFrame.setText(lines.join("\n"));
  slide.speakerNotes.setVisible(false);
}

async function addFullImageSlide(p, caseId, title, section, imageName, caption, page) {
  const slide = p.slides.add();
  addHeader(slide, caseId, title, section, page);
  slide.images.add({
    blob: await imageBytes(caseId, imageName), contentType: "image/jpeg", fit: "contain",
    alt: `${caseId} ${title}`, position: { left: 58, top: 124, width: 1164, height: 510 },
  });
  addText(slide, caption, { left: 70, top: 642, width: 1080, height: 25 },
    { fontSize: 16, color: C.muted });
  const hybridImages = new Set([
    "model_prediction_panel", "model_final_path", "official_dat_reference", "model_vs_dat_comparison",
  ]);
  const sourceRoot = hybridImages.has(imageName) ? "outputs_hybrid_v7" : "outputs_all_cases";
  addNotes(slide, [`${ROOT}/${sourceRoot}/cases/${caseId}/${imageName}.png`]);
  return slide;
}

async function addDualImageSlide(p, caseId, title, section, leftName, leftLabel, rightName, rightLabel, page) {
  const slide = p.slides.add();
  addHeader(slide, caseId, title, section, page);
  addText(slide, leftLabel, { left: 58, top: 124, width: 556, height: 28 },
    { fontSize: 20, bold: true, color: C.ink, alignment: "center" });
  addText(slide, rightLabel, { left: 666, top: 124, width: 556, height: 28 },
    { fontSize: 20, bold: true, color: C.ink, alignment: "center" });
  slide.images.add({
    blob: await imageBytes(caseId, leftName), contentType: "image/jpeg", fit: "contain",
    alt: `${caseId} ${leftLabel}`, position: { left: 58, top: 158, width: 556, height: 482 },
  });
  slide.images.add({
    blob: await imageBytes(caseId, rightName), contentType: "image/jpeg", fit: "contain",
    alt: `${caseId} ${rightLabel}`, position: { left: 666, top: 158, width: 556, height: 482 },
  });
  addNotes(slide, [
    `${ROOT}/outputs_all_cases/cases/${caseId}/${leftName}.png`,
    `${ROOT}/outputs_all_cases/cases/${caseId}/${rightName}.png`,
  ]);
  return slide;
}

async function addStatisticsSlide(p, caseId, page) {
  const slide = p.slides.add();
  addHeader(slide, caseId, "channel statistics quantify the scene-wide change", "NUMERICAL EVIDENCE", page);
  const rows = parseCsv(await fs.readFile(`${ROOT}/outputs_all_cases/cases/${caseId}/channel_statistics.csv`, "utf8"));
  const grouped = new Map();
  for (const row of rows) {
    const item = grouped.get(row.band_name) ?? { before: null, after: null };
    item[row.period] = row;
    grouped.set(row.band_name, item);
  }
  const values = [["Channel", "Mean before", "Mean after", "Variance before", "Variance after"]];
  for (const [band, item] of grouped) {
    values.push([
      band,
      Number(item.before.mean).toFixed(4), Number(item.after.mean).toFixed(4),
      Number(item.before.variance).toExponential(2), Number(item.after.variance).toExponential(2),
    ]);
  }
  const table = slide.tables.add({ rows: values.length, columns: 5, left: 92, top: 160, width: 1096, height: 390, values });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: C.rule, width: 1 });
  for (let c = 0; c < 5; c++) {
    table.getCell(0, c).fill = C.ink;
    table.getCell(0, c).text.style = { fontFamily: "Aptos", fontSize: 16, bold: true, color: C.white };
  }
  for (let r = 1; r < values.length; r++) {
    for (let c = 0; c < 5; c++) table.getCell(r, c).text.style = { fontFamily: "Aptos", fontSize: 16, color: C.ink };
  }
  addText(slide, "Means describe average reflectance; variance describes within-scene heterogeneity. These statistics support interpretation but do not identify a path by themselves.",
    { left: 92, top: 580, width: 1096, height: 55 }, { fontSize: 18, color: C.muted, alignment: "center" });
  addNotes(slide, [`${ROOT}/outputs_all_cases/cases/${caseId}/channel_statistics.csv`]);
}

async function main() {
  const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const deployment = parseCsv(await fs.readFile(`${ROOT}/outputs_hybrid_v7/reports/deployment_results.csv`, "utf8"));
  const byCase = new Map(deployment.map((row) => [row.case_id, row]));
  const modelMeta = JSON.parse(await fs.readFile(`${ROOT}/outputs_unet_v6/models/final_unet/metadata.json`, "utf8"));
  let page = 1;

  const title = p.slides.add();
  title.background.fill = C.ink;
  addText(title, "TORNADO DAMAGE-PATH ANALYSIS", { left: 68, top: 62, width: 560, height: 30 },
    { fontSize: 16, bold: true, color: "#8DD3C7" });
  addText(title, "Model evidence and official DAT validation", { left: 68, top: 132, width: 850, height: 142 },
    { fontSize: 54, bold: true, color: C.white });
  addText(title, "29 Landsat BEFORE/AFTER cases | six-band change analysis | imagery-only prediction",
    { left: 70, top: 306, width: 950, height: 55 }, { fontSize: 24, color: "#DCE9E4" });
  addText(title, "Yellow = model-generated path     Cyan/blue = official NOAA/NWS DAT reference",
    { left: 70, top: 564, width: 1000, height: 40 }, { fontSize: 20, bold: true, color: C.yellow });
  addText(title, "Research workflow | results requiring low-confidence review remain explicitly identified",
    { left: 70, top: 620, width: 1080, height: 30 }, { fontSize: 16, color: "#AFC5BC" });
  addNotes(title, [`${ROOT}/outputs_hybrid_v7/reports/deployment_results.csv`, `${ROOT}/outputs_unet_v6/models/final_unet/metadata.json`]);
  page++;

  const method = p.slides.add();
  addHeader(method, "METHOD", "references never enter the inference function", "SCIENTIFIC WORKFLOW", page++);
  const steps = [
    ["1", "Align imagery", "Validate CRS, overlap, NoData, and analysis grid."],
    ["2", "Measure change", "Six bands, signed/absolute differences, magnitude, texture, water mask."],
    ["3", "Predict paths", "65% U-Net segmentation + 35% spectral-change baseline."],
    ["4", "Extract geometry", "Connected elongated regions, gap bridging, curved graph centerlines."],
    ["5", "Validate separately", "Compare model output with NOAA/NWS DAT only after prediction."],
  ];
  for (let i = 0; i < steps.length; i++) {
    const y = 142 + i * 96;
    addText(method, steps[i][0], { left: 76, top: y, width: 48, height: 48 }, { fontSize: 26, bold: true, color: C.white, alignment: "center" });
    const badge = method.shapes.add({ geometry: "ellipse", position: { left: 68, top: y - 2, width: 52, height: 52 }, fill: C.teal, line: { style: "solid", fill: C.teal, width: 0 } });
    badge.name = `step-${i + 1}`;
    addText(method, steps[i][0], { left: 68, top: y + 7, width: 52, height: 30 }, { fontSize: 22, bold: true, color: C.white, alignment: "center" });
    addText(method, steps[i][1], { left: 148, top: y - 2, width: 330, height: 32 }, { fontSize: 22, bold: true });
    addText(method, steps[i][2], { left: 490, top: y, width: 700, height: 48 }, { fontSize: 18, color: C.muted });
  }
  addNotes(method, [`${ROOT}/scripts/build_hybrid_deployment_predictions.py`, `${ROOT}/outputs_hybrid_v7/models/hybrid_config.json`]);

  const summary = p.slides.add();
  addHeader(summary, "RESULTS", "performance is reported with explicit evidence status", "MODEL SUMMARY", page++);
  const officialCount = deployment.filter((row) => row.nws_dat_reference_available === "True").length;
  const cards = [
    ["29", "cases processed"], ["13", "supervised training cases"], [String(officialCount), "cases with official DAT reference"],
    [Number(modelMeta.grouped_validation.dice).toFixed(3), "event-grouped validation Dice"],
  ];
  for (let i = 0; i < cards.length; i++) {
    const x = 76 + i * 292;
    const box = summary.shapes.add({ geometry: "roundRect", position: { left: x, top: 170, width: 250, height: 180 }, fill: i === 3 ? "#FFF5CC" : C.pale, line: { style: "solid", fill: C.rule, width: 1 }, borderRadius: 6 });
    box.name = `metric-${i + 1}`;
    addText(summary, cards[i][0], { left: x + 18, top: 200, width: 214, height: 70 }, { fontSize: 46, bold: true, color: i === 3 ? "#7D5B00" : C.teal, alignment: "center" });
    addText(summary, cards[i][1], { left: x + 22, top: 280, width: 206, height: 48 }, { fontSize: 17, color: C.ink, alignment: "center" });
  }
  addText(summary, "Interpretation", { left: 78, top: 414, width: 240, height: 32 }, { fontSize: 24, bold: true });
  addText(summary, "Most cases produce plausible elongated candidates, but agreement varies substantially. Training-set diagnostics are not independent test results, and low-confidence cases should not be presented as confirmed paths.",
    { left: 78, top: 458, width: 1090, height: 100 }, { fontSize: 22, color: C.muted });
  addNotes(summary, [`${ROOT}/outputs_hybrid_v7/reports/deployment_results.csv`, `${ROOT}/outputs_unet_v6/models/final_unet/metadata.json`]);

  for (const caseId of CASES) {
    const row = byCase.get(caseId);
    await addFullImageSlide(p, caseId, "BEFORE and AFTER establish the event change", "CASE OVERVIEW", "before_after", "Both images are shown on the same geographic analysis footprint.", page++);
    await addFullImageSlide(p, caseId, "BEFORE imagery across all six Landsat channels", "MULTISPECTRAL INPUT", "before_six_channels", "Blue, Green, Red, NIR, SWIR1, and SWIR2 before the tornado.", page++);
    await addFullImageSlide(p, caseId, "AFTER imagery across all six Landsat channels", "MULTISPECTRAL INPUT", "after_six_channels", "The same six channels after the tornado event.", page++);
    await addDualImageSlide(p, caseId, "signed and absolute differences isolate spectral change", "CHANGE ANALYSIS", "signed_band_differences", "Signed AFTER minus BEFORE", "absolute_band_differences", "Absolute change magnitude by band", page++);
    await addFullImageSlide(p, caseId, "K-means exposes recurring change signatures", "UNSUPERVISED ANALYSIS", "clustering_analysis", "Clustering is explanatory evidence; it is not used as ground truth.", page++);
    await addFullImageSlide(p, caseId, "red-blue and magnitude filters reveal candidate corridors", "FILTER EVIDENCE", "red_blue_filter", "Red indicates increased post-event response; blue indicates decreased response.", page++);
    await addStatisticsSlide(p, caseId, page++);
    await addFullImageSlide(p, caseId, "the ensemble converts change evidence into path probability", "MODEL INFERENCE", "model_prediction_panel", `${row.path_count} model-generated path(s); labels and DAT geometry were unavailable to the inference function.`, page++);
    await addFullImageSlide(p, caseId, "model-generated path on the full AFTER image", "MODEL RESULT", "model_final_path", `Yellow shows ${row.path_count} model-generated path(s). No manual or official line is drawn on this slide.`, page++);
    const referenceCaption = row.nws_dat_reference_available === "True"
      ? `Official source: NOAA/NWS Damage Assessment Toolkit (DAT). Path available: ${row.dat_path_available}; polygon available: ${row.dat_polygon_available}.`
      : "No official NOAA/NWS DAT path or polygon was available for this case.";
    await addFullImageSlide(p, caseId, "official DAT reference is shown separately", "OFFICIAL REFERENCE", "official_dat_reference", referenceCaption, page++);
    const comparisonCaption = row.dat_path_available === "True"
      ? `Within 150 m: ${Number(row.agreement_within_150m_pct).toFixed(1)}% of predicted line; ${Number(row.official_coverage_within_150m_pct).toFixed(1)}% of DAT line. Median distance: ${Number(row.median_centerline_distance_m).toFixed(0)} m.`
      : "Quantitative line agreement is unavailable because this case has no official DAT path.";
    await addFullImageSlide(p, caseId, "side-by-side comparison makes disagreement visible", "VALIDATION", "model_vs_dat_comparison", comparisonCaption, page++);
  }

  const close = p.slides.add();
  close.background.fill = C.ink;
  addText(close, "CONCLUSION", { left: 70, top: 64, width: 300, height: 28 }, { fontSize: 16, bold: true, color: "#8DD3C7" });
  addText(close, "The workflow is automated; confidence is evidence-dependent", { left: 70, top: 140, width: 1060, height: 110 }, { fontSize: 46, bold: true, color: C.white });
  addText(close, "Model predictions are produced from imagery. DAT is retained as an independent comparison layer. Additional verified tornado cases are still required to improve generalization and reduce false paths.",
    { left: 72, top: 304, width: 1030, height: 130 }, { fontSize: 24, color: "#DCE9E4" });
  addText(close, "Next scientific priority: more accurately georeferenced corridor labels and held-out event validation.",
    { left: 72, top: 548, width: 1080, height: 48 }, { fontSize: 22, bold: true, color: C.yellow });
  addNotes(close, [`${ROOT}/outputs_unet_v6/models/final_unet/metadata.json`, `${ROOT}/outputs_hybrid_v7/reports/deployment_results.csv`]);

  await fs.mkdir(path.dirname(OUTPUT), { recursive: true });
  const file = await PresentationFile.exportPptx(p);
  await file.save(OUTPUT);
  console.log(JSON.stringify({ output: OUTPUT, slides: p.slides.items.length }));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });

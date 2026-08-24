import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";
const ROOT = process.env.TORNADO_PROJECT_ROOT ?? process.cwd();
const OUTPUT = process.env.TORNADO_DECK_ASSETS ?? path.join(ROOT, "outputs_hybrid_v7/presentation_assets");
const ALL_CASES = [
  "TOR5", "TOR7", "TOR10", "TOR11", "TOR12", "TOR13", "TOR15", "TOR16", "TOR18", "TOR24",
  "TOR61", "TOR62", "TOR66", "TOR68", "TOR69", "TOR70", "TOR77", "TOR78", "TOR90", "TOR91",
  "TOR95", "TOR101", "TOR102", "TOR105", "TOR111", "TOR112", "TOR114", "TOR115", "TOR123",
];
const requested = new Set((process.env.TORNADO_ASSET_CASES ?? "").split(",").map((value) => value.trim()).filter(Boolean));
const CASES = requested.size ? ALL_CASES.filter((caseId) => requested.has(caseId)) : ALL_CASES;
const EVIDENCE = [
  "before_after", "before_six_channels", "after_six_channels", "signed_band_differences",
  "absolute_band_differences", "clustering_analysis", "red_blue_filter",
];
const HYBRID = ["model_prediction_panel", "model_final_path", "official_dat_reference", "model_vs_dat_comparison"];

async function convert(caseId, imageName) {
  const sourceRoot = HYBRID.includes(imageName) ? "outputs_hybrid_v7" : "outputs_all_cases";
  const source = path.join(ROOT, sourceRoot, "cases", caseId, `${imageName}.png`);
  const destinationDir = path.join(OUTPUT, caseId);
  const destination = path.join(destinationDir, `${imageName}.jpg`);
  await fs.mkdir(destinationDir, { recursive: true });
  let lastError;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await sharp(source)
        .flatten({ background: "white" })
        .resize({ width: 1200, height: 1200, fit: "inside", withoutEnlargement: true })
        .jpeg({ quality: 76, mozjpeg: true })
        .toFile(destination);
      return;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 500 * (attempt + 1)));
    }
  }
  throw new Error(`${caseId}/${imageName}: ${lastError.message}`);
}

async function main() {
  const jobs = CASES.flatMap((caseId) => [...EVIDENCE, ...HYBRID].map((name) => [caseId, name]));
  const concurrency = 8;
  for (let index = 0; index < jobs.length; index += concurrency) {
    await Promise.all(jobs.slice(index, index + concurrency).map(([caseId, name]) => convert(caseId, name)));
  }
  console.log(JSON.stringify({ output: OUTPUT, cases: CASES.length, images: jobs.length }));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

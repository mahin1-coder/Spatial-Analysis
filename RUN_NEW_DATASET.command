#!/bin/zsh
set -e

cd "$(dirname "$0")"

echo ""
echo "Tornado Damage Path Pipeline"
echo "============================"
echo ""
echo "Paste the path to the folder that contains the professor's BEFORE/AFTER images."
echo "You can also drag the folder into this Terminal window."
echo "Then press Enter."
echo ""
read "DATASET_PATH?Dataset folder: "

DATASET_PATH="${DATASET_PATH%\"}"
DATASET_PATH="${DATASET_PATH#\"}"
DATASET_PATH="${DATASET_PATH%\'}"
DATASET_PATH="${DATASET_PATH#\'}"
DATASET_PATH="${DATASET_PATH/#\\~/$HOME}"
DATASET_PATH="${DATASET_PATH//\\ / }"
DATASET_PATH="${DATASET_PATH%/}"

if [ ! -d "$DATASET_PATH" ]; then
  echo ""
  echo "ERROR: Folder not found:"
  echo "$DATASET_PATH"
  echo ""
  echo "Tip: If the path has spaces, dragging the folder into Terminal is easiest."
  read "WAIT?Press Enter to close."
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo ""
  echo "Creating Python environment..."
  python3 -m venv .venv
fi

echo ""
echo "Installing/updating Python packages..."
.venv/bin/python -m pip install -r requirements.txt

BATCH_NAME="professor_batch_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Finding BEFORE/AFTER pairs and making tornado-path maps..."
.venv/bin/python run_pipeline.py --mode analyze-folder --source "$DATASET_PATH" --batch-name "$BATCH_NAME"

echo ""
echo "Opening result folder..."
open "outputs/predictions/batch/$BATCH_NAME"

echo ""
echo "DONE."
echo "Open batch_contact_sheet.png first."
echo "Each case folder also has its own showcase_prediction_map.png."
echo ""
read "WAIT?Press Enter to close."

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ID="SizheZhao/TapSampling-CALVIN-Part-Data"
ZIP_NAME="task_ABC_D.zip"
ZIP_PATH="$SCRIPT_DIR/$ZIP_NAME"
TARGET_DIR="$SCRIPT_DIR/task_ABC_D"

if ! command -v python >/dev/null 2>&1; then
  echo "Error: 'python' is not installed or not in PATH."
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: 'unzip' is not installed or not in PATH."
  exit 1
fi

echo "Downloading $ZIP_NAME from Hugging Face dataset repo $REPO_ID ..."
python - <<PY
from pathlib import Path
from huggingface_hub import hf_hub_download

script_dir = Path(r"$SCRIPT_DIR")
downloaded_path = hf_hub_download(
    repo_id="$REPO_ID",
    filename="$ZIP_NAME",
    repo_type="dataset",
)
zip_path = script_dir / "$ZIP_NAME"
zip_path.write_bytes(Path(downloaded_path).read_bytes())
print(f"Saved zip to {zip_path}")
PY

echo "Unzipping $ZIP_PATH into $SCRIPT_DIR ..."
unzip -o "$ZIP_PATH" -d "$SCRIPT_DIR"

echo "Finished. Extracted dataset directory: $TARGET_DIR"

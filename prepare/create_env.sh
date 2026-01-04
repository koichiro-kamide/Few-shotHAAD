#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"
cd "$PROJECT_DIR"

ENV_NAME="haad"
PY_VER="3.8"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

echo "[1/7] Preparing directories..."
mkdir -p data checkpoints_paper
echo "[OK] Directories ready: data/, checkpoints_paper/"

echo "[2/7] Checking conda..."
command -v conda >/dev/null 2>&1 || { echo "[ERROR] conda not found in PATH"; exit 1; }
echo "[OK] conda is available."


echo "[3/7] Creating conda environment (${ENV_NAME}) if needed..."
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[OK] Conda env '${ENV_NAME}' already exists. Skipping creation."
else
  conda create -y -n "${ENV_NAME}" python="${PY_VER}"
  echo "[OK] Conda env '${ENV_NAME}' created (python=${PY_VER})."
fi

echo "[4/7] Activating env..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "[OK] Activated: ${ENV_NAME}"

echo "[5/7] Installing PyTorch (1.7.1) into '${ENV_NAME}'..."
conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
echo "[OK] PyTorch stack installed."

echo "[6/7] Upgrading pip..."
python -m pip install --upgrade pip
echo "[OK] pip upgraded."

echo "[7/7] Installing Python requirements..."
if [[ -f "$REQ_FILE" ]]; then
  python -m pip install -r "$REQ_FILE"
  echo "[OK] requirements installed: $REQ_FILE"
else
  echo "[WARN] requirements.txt not found: $REQ_FILE"
fi

echo "============================================================"
echo "[DONE] Setup complete."
echo "To enter the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo "============================================================"
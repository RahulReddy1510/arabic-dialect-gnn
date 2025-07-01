#!/usr/bin/env bash
# scripts/run_baseline.sh
#
# Fine-tune AraBERT baseline.
# Usage:
#   bash scripts/run_baseline.sh
#   bash scripts/run_baseline.sh --epochs 3

set -euo pipefail

CONFIG="${CONFIG:-configs/arabert_baseline.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/arabert/}"
SEED="${SEED:-42}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)        CONFIG="$2";         shift 2 ;;
    --checkpoint_dir)CHECKPOINT_DIR="$2"; shift 2 ;;
    --seed)          SEED="$2";           shift 2 ;;
    *)               echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo "  Arabic Dialect GNN — AraBERT Baseline"
echo "============================================"
echo "  Config:         $CONFIG"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Seed:           $SEED"
echo "============================================"

# Check transformers is installed.
python -c "import transformers" 2>/dev/null || {
  echo "ERROR: transformers not installed."
  echo "       Run: pip install transformers==4.36.2"
  exit 1
}

mkdir -p "$CHECKPOINT_DIR"
mkdir -p results/

echo ""
echo "[Step 1/2] Fine-tuning AraBERT..."
echo "  NOTE: First run will download aubmindlab/bert-base-arabert (~500MB)."
echo "  Estimated training time: ~4h on GPU, ~12h on CPU."
echo ""
python training/train_arabert.py \
  --config "$CONFIG" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --seed "$SEED"

echo ""
echo "[Step 2/2] Evaluating AraBERT on test set..."
python evaluation/evaluate.py \
  --gat_checkpoint "checkpoints/gat/gat_best.pth" \
  --arabert_checkpoint "${CHECKPOINT_DIR}/arabert_best.pth" \
  --data_dir "data/graphs" \
  --output_dir "results/"

echo ""
echo "============================================"
echo "  AraBERT baseline complete."
echo "  Checkpoint: ${CHECKPOINT_DIR}/arabert_best.pth"
echo "  Results:    results/full_results.json"
echo "============================================"

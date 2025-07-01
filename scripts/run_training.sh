#!/usr/bin/env bash
# scripts/run_training.sh
#
# End-to-end GAT training script.
# Usage:
#   bash scripts/run_training.sh
#   bash scripts/run_training.sh --seed 0 --checkpoint_dir checkpoints/gat_seed0/

set -euo pipefail

CONFIG="${CONFIG:-configs/gat_base.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/gat/}"
SEED="${SEED:-42}"
RESUME="${RESUME:-}"

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)        CONFIG="$2";         shift 2 ;;
    --checkpoint_dir)CHECKPOINT_DIR="$2"; shift 2 ;;
    --seed)          SEED="$2";           shift 2 ;;
    --resume)        RESUME="--resume $2";shift 2 ;;
    *)               echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo "  Arabic Dialect GNN — GAT Training"
echo "============================================"
echo "  Config:         $CONFIG"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Seed:           $SEED"
echo "  Resume:         ${RESUME:-none}"
echo "============================================"

# Ensure graph files exist; if not, preprocessing + graph construction runs.
if [ ! -d "data/graphs/train" ]; then
  echo ""
  echo "[Step 1/3] Building phoneme graphs from preprocessed data..."
  python data/graph_construction.py \
    --input_dir data/processed \
    --output_dir data/graphs
else
  echo "[Step 1/3] Graph files found at data/graphs/ — skipping construction."
fi

# Create results and checkpoint directories.
mkdir -p "$CHECKPOINT_DIR"
mkdir -p results/

echo ""
echo "[Step 2/3] Training DialectGAT..."
python training/train_gat.py \
  --config "$CONFIG" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --seed "$SEED" \
  $RESUME

echo ""
echo "[Step 3/3] Running evaluation on test set..."
python evaluation/evaluate.py \
  --gat_checkpoint "${CHECKPOINT_DIR}/gat_best.pth" \
  --arabert_checkpoint "checkpoints/arabert/arabert_best.pth" \
  --data_dir "data/graphs" \
  --output_dir "results/"

echo ""
echo "============================================"
echo "  Training complete."
echo "  Best checkpoint: ${CHECKPOINT_DIR}/gat_best.pth"
echo "  Results:         results/full_results.json"
echo "============================================"

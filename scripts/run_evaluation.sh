#!/usr/bin/env bash
# scripts/run_evaluation.sh
#
# Run full evaluation + error analysis for both models.
# Usage:
#   bash scripts/run_evaluation.sh
#   bash scripts/run_evaluation.sh --split val

set -euo pipefail

GAT_CKPT="${GAT_CKPT:-checkpoints/gat/gat_best.pth}"
BERT_CKPT="${BERT_CKPT:-checkpoints/arabert/arabert_best.pth}"
DATA_DIR="${DATA_DIR:-data/graphs}"
OUTPUT_DIR="${OUTPUT_DIR:-results/}"
SPLIT="${SPLIT:-test}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gat_checkpoint)    GAT_CKPT="$2";   shift 2 ;;
    --arabert_checkpoint)BERT_CKPT="$2";  shift 2 ;;
    --data_dir)          DATA_DIR="$2";   shift 2 ;;
    --output_dir)        OUTPUT_DIR="$2"; shift 2 ;;
    --split)             SPLIT="$2";      shift 2 ;;
    *)                   echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo "  Arabic Dialect GNN — Evaluation"
echo "============================================"
echo "  GAT checkpoint:    $GAT_CKPT"
echo "  AraBERT checkpoint:$BERT_CKPT"
echo "  Data directory:    $DATA_DIR"
echo "  Output directory:  $OUTPUT_DIR"
echo "  Split:             $SPLIT"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "[Step 1/2] Running evaluation (macro F1, confusion matrices, CI)..."
python evaluation/evaluate.py \
  --gat_checkpoint "$GAT_CKPT" \
  --arabert_checkpoint "$BERT_CKPT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --split "$SPLIT"

echo ""
echo "[Step 2/2] Finished. Outputs:"
echo "  - ${OUTPUT_DIR}full_results.json"
echo "  - ${OUTPUT_DIR}eval_results.csv"
echo ""
echo "  To view results:"
echo "    cat ${OUTPUT_DIR}eval_results.csv"
echo ""
echo "  To run notebooks:"
echo "    jupyter nbconvert --to notebook --execute notebooks/03_model_comparison.ipynb"
echo "    jupyter nbconvert --to notebook --execute notebooks/04_error_analysis.ipynb"
echo "============================================"

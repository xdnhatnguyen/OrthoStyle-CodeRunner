#!/bin/bash
# ==============================================================================
# OrthoStyle Baselines Metrics Evaluation Suite
# Evaluates 7 Metrics across all 6 Baselines + Ours:
#   - AttnST
#   - DiffuseIT
#   - InstantStyle
#   - ITOC
#   - RB-Modulation
#   - StyleID
#   - Ours (OrthoStyle Full)
# Outputs comparison to metrics/master_baselines_comparison.csv
# ==============================================================================

set -eo pipefail

PROJECT_DIR="/mnt/wav2vec2/khoan/source_code/RB-Ortho/OthoStyle"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/home/khoan/.conda/envs/rbm/bin/python}"
DEVICE="${1:-cuda:0}"
LOG_FILE="logs/evaluation_baselines.log"
mkdir -p logs metrics

echo "======================================================================"
echo "       OrthoStyle Baselines Metrics Evaluation Suite                  "
echo "Target Device: $DEVICE"
echo "Start Time:    $(date)"
echo "Log File:      $LOG_FILE"
echo "======================================================================"

"$PYTHON" evaluate_all_benchmarks.py \
  --suite baselines \
  --device "$DEVICE" \
  --output_dir metrics \
  --overwrite 2>&1 | tee "$LOG_FILE"

echo ""
echo "======================================================================"
echo "All Baselines Evaluated Successfully at: $(date)"
echo "Results exported to:"
echo "  - metrics/master_baselines_comparison.csv"
echo "  - metrics/master_all_metrics.csv"
echo "======================================================================"

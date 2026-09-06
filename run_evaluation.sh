#!/bin/bash
# ==============================================================================
# OrthoStyle Master Evaluation Driver
# Computes 7 Style Transfer Metrics across Main Benchmark, Table 2 & Tau Sweeps
# Outputs summary tables to CSV format in metrics/
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/home/khoan/.conda/envs/rbm/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SUITE="${1:-all}"
OVERWRITE="${2:-}"

mkdir -p metrics logs

echo "======================================================================"
echo "      OrthoStyle Benchmark & Ablation Metrics Evaluation Suite        "
echo "======================================================================"
echo "Python Interpreter: $PYTHON"
echo "Target Device:      $DEVICE"
echo "Evaluation Suite:   $SUITE"
echo "Results Directory:  metrics/"
echo "Start Time:         $(date)"
echo "======================================================================"

ARGS=(
  --suite "$SUITE"
  --device "$DEVICE"
  --output_dir "metrics"
)

if [[ "$OVERWRITE" == "--overwrite" || "$OVERWRITE" == "-f" ]]; then
  ARGS+=(--overwrite)
fi

"$PYTHON" evaluate_all_benchmarks.py "${ARGS[@]}" 2>&1 | tee "logs/evaluation_${SUITE}.log"

echo "======================================================================"
echo "Evaluation Finished: $(date)"
echo "Summary Reports Available in:"
echo "  - metrics/master_table1_benchmark.csv"
echo "  - metrics/master_table2_ablations.csv"
echo "  - metrics/master_sweeps_tau.csv"
echo "  - metrics/master_all_metrics.csv"
echo "======================================================================"

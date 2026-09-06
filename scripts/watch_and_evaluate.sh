#!/bin/bash
# ==============================================================================
# OrthoStyle Post-Generation Automated Evaluation Watcher
# Waits for generation queue to finish, then automatically runs evaluation suite
# ==============================================================================

set -u

PROJECT_DIR="/mnt/wav2vec2/khoan/source_code/RB-Ortho/OthoStyle"
cd "$PROJECT_DIR"

LOG_FILE="logs/evaluation_post_table2.log"
mkdir -p logs metrics

echo "======================================================================" >> "$LOG_FILE"
echo "[WATCHER STARTED] Waiting for tmux session 'ortho_table2' to finish..." >> "$LOG_FILE"
echo "Start Watch Time: $(date)" >> "$LOG_FILE"
echo "======================================================================" >> "$LOG_FILE"

while tmux has-session -t ortho_table2 2>/dev/null; do
    sleep 30
done

echo "" >> "$LOG_FILE"
echo "======================================================================" >> "$LOG_FILE"
echo "[WATCHER TRIGGERED] Generation session 'ortho_table2' has finished!" >> "$LOG_FILE"
echo "Completion Time: $(date)" >> "$LOG_FILE"
echo "Now launching full metrics evaluation on cuda:0..." >> "$LOG_FILE"
echo "======================================================================" >> "$LOG_FILE"

# Small delay to ensure all file handles are closed
sleep 5

PYTHON="/home/khoan/.conda/envs/rbm/bin/python"
"$PYTHON" evaluate_all_benchmarks.py --suite all --device cuda:0 --output_dir metrics --overwrite 2>&1 | tee -a "$LOG_FILE"

echo "" >> "$LOG_FILE"
echo "======================================================================" >> "$LOG_FILE"
echo "[ALL WORK FINISHED] Master Benchmark & Ablation Tables Ready!" >> "$LOG_FILE"
echo "End Time: $(date)" >> "$LOG_FILE"
echo "======================================================================" >> "$LOG_FILE"

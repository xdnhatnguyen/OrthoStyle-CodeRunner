#!/bin/bash
set -euo pipefail

# ============================================================
# Paths / Environment (100% relative & configurable)
# ============================================================
export PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}
export THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-$PROJECT_ROOT/third_party}
export CSD_REPO_DIR=${CSD_REPO_DIR:-$THIRD_PARTY_ROOT/CSD}
export CSD_CHECKPOINT_PATH=${CSD_CHECKPOINT_PATH:-$CSD_REPO_DIR/checkpoint.pth}
export CONTROLNET_CHECKPOINT_PATH=${CONTROLNET_CHECKPOINT_PATH:-$THIRD_PARTY_ROOT/StableCascade/models/canny.safetensors}
export DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data}
export CONFIG_PATH=${CONFIG_PATH:-$PROJECT_ROOT/configs/benchmark_config.json}
export RESULTS_ROOT=${RESULTS_ROOT:-$PROJECT_ROOT/output/benchmark}
export PYTHONPATH=$PROJECT_ROOT:$THIRD_PARTY_ROOT:$THIRD_PARTY_ROOT/StableCascade:$CSD_REPO_DIR:${PYTHONPATH:-}
if [ -n "${GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

cd "$PROJECT_ROOT"

PYTHON=${PYTHON:-python}
RUNNER=${RUNNER:-batch_runner.py}
QUEUE_RUNNER=${QUEUE_RUNNER:-run_queue.py}

mkdir -p logs "$RESULTS_ROOT"

# ============================================================
# Common Arguments
# ============================================================
COMMON=(
  --config_path "$CONFIG_PATH"
  --data_root "$DATA_ROOT"
  --third_party_root "$THIRD_PARTY_ROOT"
  --output_root "$RESULTS_ROOT"
)

run_task () {
  local task_name="$1"
  shift
  echo
  echo "============================================================"
  echo "Task: $task_name"
  echo "GPU: ${CUDA_VISIBLE_DEVICES}"
  echo "Results: $RESULTS_ROOT"
  echo "============================================================"
  "$PYTHON" "$RUNNER" \
    --device "cuda:0" \
    "${COMMON[@]}" \
    "$@" \
    2>&1 | tee "logs/${task_name}.log"
}

# ============================================================
# Multi-GPU Queue Mode (2 lần chạy, mỗi lần 2 card song song)
# Cách dùng:
#   Lần 1: ./run_experiments.sh phase1 0 1   (hoặc PHASE=1 ./run_experiments.sh)
#   Lần 2: ./run_experiments.sh phase2 0 1   (hoặc PHASE=2 ./run_experiments.sh)
#   Tất cả: ./run_experiments.sh run_all 0 1
CMD="${1:-none}"

# Debug Mode (Chạy thử 1 sample trên 1 GPU, in log trực tiếp ra terminal không qua queue)
# Cách dùng: ./run_experiments.sh debug [gpu_id]
if [[ "$CMD" =~ ^(debug|test|run_debug)$ ]]; then
  shift || true
  DBG_GPU="${1:-0}"
  echo "============================================================"
  echo "DEBUG MODE: Running 1 pair test on GPU $DBG_GPU (Live Console Output)"
  echo "============================================================"
  "$PYTHON" -u "$RUNNER" \
    --device "cuda:$DBG_GPU" \
    "${COMMON[@]}" \
    --start_idx 1 \
    --end_idx 1 \
    --prompt_levels null \
    --output_root "output/debug"
  echo
  echo "============================================================"
  echo "[✔] Debug test finished successfully! Output saved in: output/debug"
  echo "============================================================"
  exit 0
fi

# Lần 1: Main Benchmark + 2 Prompt Levels + Tau Sweeps (10 tasks, 871 ảnh)
if [[ "${PHASE:-0}" == "1" || "$CMD" == "phase1" || "$CMD" == "run_phase1" ]]; then
  [[ "$CMD" =~ ^(phase1|run_phase1)$ ]] && shift || true
  GPUS=("$@")
  [ ${#GPUS[@]} -eq 0 ] && GPUS=("0" "1")
  echo "============================================================"
  echo "[LẦN CHẠY 1 / 2 CARD]: Main Benchmark + Prompt Levels + Sweeps (871 ảnh)"
  echo "GPUs: ${GPUS[*]}"
  echo "============================================================"
  "$PYTHON" "$QUEUE_RUNNER" --suite "phase1" --gpus "${GPUS[@]}"
  exit 0
fi

# Lần 2: Toàn bộ Table 2 Component Ablation (10 tasks, 1,125 ảnh)
if [[ "${PHASE:-0}" == "2" || "$CMD" == "phase2" || "$CMD" == "run_phase2" ]]; then
  [[ "$CMD" =~ ^(phase2|run_phase2)$ ]] && shift || true
  GPUS=("$@")
  [ ${#GPUS[@]} -eq 0 ] && GPUS=("0" "1")
  echo "============================================================"
  echo "[LẦN CHẠY 2 / 2 CARD]: Full Table 2 Component Ablation (1,125 ảnh)"
  echo "GPUs: ${GPUS[*]}"
  echo "============================================================"
  "$PYTHON" "$QUEUE_RUNNER" --suite "phase2" --gpus "${GPUS[@]}"
  exit 0
fi

if [[ "$CMD" =~ ^(run_all|all|run_ablations|ablations|run_sweeps|sweeps|run_table2|table2|run_main|main)$ ]]; then
  shift || true
  SUITE="${CMD#run_}"
  GPUS=("$@")
  if [ ${#GPUS[@]} -eq 0 ]; then
    GPUS=("0" "1")
  fi
  echo "============================================================"
  echo "Launching Suite '$SUITE' on GPUs: ${GPUS[*]}"
  echo "============================================================"
  "$PYTHON" "$QUEUE_RUNNER" --suite "$SUITE" --gpus "${GPUS[@]}"
  exit 0
fi

# ============================================================
# Single-GPU Sequential Mode (Chạy theo cờ RUN_*)
# ============================================================

# 1. Main Benchmark (Table 1: Null Prompt 225 pairs)
if [[ "${RUN_MAIN:-1}" == "1" ]]; then
  run_task benchmark_null \
    --prompt_levels null \
    --start_idx 1 \
    --end_idx 225
fi

# 2. Prompt Robustness: Level 2 Object Prompt (225 pairs)
if [[ "${RUN_OBJECT:-0}" == "1" ]]; then
  run_task benchmark_object \
    --prompt_levels object \
    --start_idx 1 \
    --end_idx 225
fi

# 3. Prompt Robustness: Level 3 Style Description Prompt (225 pairs)
if [[ "${RUN_STYLE_DESC:-0}" == "1" ]]; then
  run_task benchmark_style_desc \
    --prompt_levels style_desc \
    --start_idx 1 \
    --end_idx 225
fi

# 4. Tau Pushforward Sweep [1, 2, 3, 4] (49 pairs 7x7)
if [[ "${RUN_SWEEPS:-0}" == "1" ]]; then
  for tau in 1 2 3 4; do
    run_task "sweep_tau_${tau}" \
      --tau_pushforward "${tau}" \
      --subset_7x7 \
      --prompt_levels null \
      --ablation_tag "sweep_tau_${tau}"
  done
fi

# 5. Table 2 Component Ablation Study (Full 15x15 = 225 pairs)
if [[ "${RUN_TABLE2:-0}" == "1" ]]; then
  # (B) Pure Mean Token
  run_task ablation_B_pure_mean \
    --alpha_style 0.0 \
    --start_idx 1 \
    --end_idx 225 \
    --prompt_levels null \
    --ablation_tag ablation_B_pure_mean

  # (C) Raw Style Token
  run_task ablation_C_raw_style \
    --alpha_style 1.0 \
    --start_idx 1 \
    --end_idx 225 \
    --prompt_levels null \
    --ablation_tag ablation_C_raw_style

  # (D) No Score-Orthogonal Guidance
  run_task ablation_D_no_ortho \
    --no_ortho \
    --start_idx 1 \
    --end_idx 225 \
    --prompt_levels null \
    --ablation_tag ablation_D_no_ortho

  # (E) No AdaIN Pushforward
  run_task ablation_E_no_pushforward \
    --no_pushforward \
    --start_idx 1 \
    --end_idx 225 \
    --prompt_levels null \
    --ablation_tag ablation_E_no_pushforward

  # (F) No Semantic Gated Canny
  run_task ablation_F_no_semantic_gating \
    --no_semantic_gating \
    --start_idx 1 \
    --end_idx 225 \
    --prompt_levels null \
    --ablation_tag ablation_F_no_semantic_gating
fi

echo
echo "============================================================"
echo "All requested runs completed."
echo "Results: $RESULTS_ROOT"
echo "============================================================"


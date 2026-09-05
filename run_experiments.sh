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
export CUDA_VISIBLE_DEVICES=${GPU:-0}

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
  --save_grids
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
# Multi-GPU Queue Mode (Tự động chia tải song song qua 2 GPU)
# Cách dùng:
#   ./run_experiments.sh run_all 0 1
#   ./run_experiments.sh run_ablations 0 1
#   ./run_experiments.sh run_sweeps 0 1
#   ./run_experiments.sh run_table2 0 1
#   ./run_experiments.sh run_main 0 1
# ============================================================
CMD="${1:-none}"
if [[ "$CMD" =~ ^(run_all|run_ablations|run_sweeps|run_table2|run_main)$ ]]; then
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


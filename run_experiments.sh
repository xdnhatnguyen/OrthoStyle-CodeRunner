#!/usr/bin/env bash
# ==============================================================================
# OrthoStyle Experiment Orchestration Script
# Multi-Card & Task Dispatcher for University Servers & Local Workstations
# ==============================================================================
#
# USAGE GUIDE:
#
# 1. Chạy 1 task (prompt level) trên 1 card cụ thể:
#    ./run_experiments.sh task null 0              # Chạy task null prompt trên GPU 0
#    ./run_experiments.sh task object 1            # Chạy task object prompt trên GPU 1
#    ./run_experiments.sh task style_desc 2        # Chạy task style_desc trên GPU 2
#
# 2. Bắn song song các task lên các card khác nhau cùng lúc (Background + Logging):
#    ./run_experiments.sh launch_tasks 0 1         # null -> GPU 0, object -> GPU 1
#    ./run_experiments.sh launch_tasks 0 1 2       # null -> GPU 0, object -> GPU 1, style_desc -> GPU 2
#
# 3. Chia đôi 225 cặp chạy song song 2 GPU (cả 3 prompt levels):
#    ./run_experiments.sh split_pairs 0 1          # GPU 0: cặp 1..112, GPU 1: cặp 113..225
#
# 4. Chạy một dải cặp (range) trên 1 card cụ thể:
#    ./run_experiments.sh range 1 112 0            # Cặp 1..112 trên GPU 0
#    ./run_experiments.sh range 113 225 1          # Cặp 113..225 trên GPU 1
#
# 5. Chạy toàn bộ 675 ảnh tuần tự trên 1 GPU:
#    ./run_experiments.sh single 0                 # Full 675 ảnh trên GPU 0
#
# 6. Chạy bộ thí nghiệm Ablation Study (Bảng 2) trên 1 GPU:
#    ./run_experiments.sh ablations 0              # Chạy cấu hình A-F trên GPU 0
# ==============================================================================

set -eo pipefail

# Auto-detect Python Environment
if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python3 &>/dev/null; then
  PYTHON_BIN="$(which python3)"
elif [ -f "/home/khoan/.conda/envs/rbm/bin/python" ]; then
  PYTHON_BIN="/home/khoan/.conda/envs/rbm/bin/python"
else
  PYTHON_BIN="python"
fi

SCRIPT="batch_runner.py"

mkdir -p logs
mkdir -p output/benchmark

normalize_device() {
  local dev="${1:-0}"
  if [[ "${dev}" =~ ^[0-9]+$ ]]; then
    echo "cuda:${dev}"
  else
    echo "${dev}"
  fi
}

show_help() {
  cat << 'EOF'
OrthoStyle Experiment Runner - Multi-Card & Task Interface

Usage:
  ./run_experiments.sh task <prompt_level> <gpu_id> [start_idx] [end_idx]
      Run a single prompt level ('null', 'object', 'style_desc') on a specific GPU.
      Example: ./run_experiments.sh task null 0
               ./run_experiments.sh task object 1

  ./run_experiments.sh launch_tasks <gpu_for_null> <gpu_for_object> [gpu_for_style_desc]
      Launch tasks concurrently across 2 or 3 GPUs in background with dedicated logs.
      Example: ./run_experiments.sh launch_tasks 0 1
               ./run_experiments.sh launch_tasks 0 1 2

  ./run_experiments.sh split_pairs <gpu_0> <gpu_1>
      Split 225 pairs evenly across 2 GPUs (GPU 0 handles 1..112, GPU 1 handles 113..225).
      Example: ./run_experiments.sh split_pairs 0 1

  ./run_experiments.sh range <start_idx> <end_idx> <gpu_id> [prompt_level]
      Run a custom index range on a specific GPU.
      Example: ./run_experiments.sh range 1 50 0 null

  ./run_experiments.sh single [gpu_id]
      Run all 3 prompt levels sequentially on a single GPU (default: 0).
      Example: ./run_experiments.sh single 0

  ./run_experiments.sh ablations [gpu_id]
      Run Table 2 Ablation Study suite (Configurations B to F) on a specific GPU (default: 0).
      Example: ./run_experiments.sh ablations 0

Legacy Commands:
  ./run_experiments.sh benchmark_single_gpu   # Alias for 'single 0'
  ./run_experiments.sh benchmark_dual_gpu     # Alias for 'split_pairs 0 1'
EOF
}

CMD="${1:-help}"

case "${CMD}" in
  # ----------------------------------------------------------------------------
  # 1. Chạy 1 task cụ thể trên 1 card
  # ----------------------------------------------------------------------------
  task)
    LEVEL="${2:?Vui lòng chỉ định prompt level: null, object, style_desc}"
    GPU_ID="$(normalize_device "${3:-0}")"
    START_IDX="${4:-1}"
    END_IDX="${5:-225}"

    echo "=========================================================="
    echo "  Task:         Level ${LEVEL}"
    echo "  Device:       ${GPU_ID}"
    echo "  Pairs:        ${START_IDX} -> ${END_IDX}"
    echo "  Python:       ${PYTHON_BIN}"
    echo "  Log:          logs/task_${LEVEL}_${GPU_ID//:/_}.log"
    echo "=========================================================="

    ${PYTHON_BIN} "${SCRIPT}" \
        --device "${GPU_ID}" \
        --start_idx "${START_IDX}" \
        --end_idx "${END_IDX}" \
        --prompt_levels "${LEVEL}" \
        2>&1 | tee "logs/task_${LEVEL}_${GPU_ID//:/_}.log"
    ;;

  # ----------------------------------------------------------------------------
  # 2. Bắn song song các task lên các card khác nhau (Multi-Card Dispatch)
  # ----------------------------------------------------------------------------
  launch_tasks)
    GPU_NULL="$(normalize_device "${2:?Vui lòng chỉ định GPU cho task null (ví dụ: 0)}")"
    GPU_OBJECT="$(normalize_device "${3:?Vui lòng chỉ định GPU cho task object (ví dụ: 1)}")"
    GPU_STYLE="${4:-}"

    echo "=== Launching Tasks Across Multiple GPUs concurrently ==="
    echo ">> [Task 1 - null]: Starting on ${GPU_NULL}..."
    ${PYTHON_BIN} "${SCRIPT}" \
        --device "${GPU_NULL}" \
        --start_idx 1 \
        --end_idx 225 \
        --prompt_levels null \
        > "logs/task_null_${GPU_NULL//:/_}.log" 2>&1 &
    PID_NULL=$!
    echo "   Worker NULL PID: ${PID_NULL} | Log: logs/task_null_${GPU_NULL//:/_}.log"

    echo ">> [Task 2 - object]: Starting on ${GPU_OBJECT}..."
    ${PYTHON_BIN} "${SCRIPT}" \
        --device "${GPU_OBJECT}" \
        --start_idx 1 \
        --end_idx 225 \
        --prompt_levels object \
        > "logs/task_object_${GPU_OBJECT//:/_}.log" 2>&1 &
    PID_OBJECT=$!
    echo "   Worker OBJECT PID: ${PID_OBJECT} | Log: logs/task_object_${GPU_OBJECT//:/_}.log"

    PID_STYLE=""
    if [ -n "${GPU_STYLE}" ]; then
      GPU_STYLE_NORM="$(normalize_device "${GPU_STYLE}")"
      echo ">> [Task 3 - style_desc]: Starting on ${GPU_STYLE_NORM}..."
      ${PYTHON_BIN} "${SCRIPT}" \
          --device "${GPU_STYLE_NORM}" \
          --start_idx 1 \
          --end_idx 225 \
          --prompt_levels style_desc \
          > "logs/task_style_${GPU_STYLE_NORM//:/_}.log" 2>&1 &
      PID_STYLE=$!
      echo "   Worker STYLE_DESC PID: ${PID_STYLE} | Log: logs/task_style_${GPU_STYLE_NORM//:/_}.log"
    else
      echo ">> Note: GPU cho task 3 (style_desc) chưa được chỉ định. Bạn có thể chạy riêng bằng: ./run_experiments.sh task style_desc <gpu>"
    fi

    echo ""
    echo "Workers running in background. Monitoring progress..."
    echo "Tip: Run 'tail -f logs/task_null_*.log' or 'tail -f logs/task_object_*.log' to view real-time outputs."
    
    wait ${PID_NULL}
    echo "[✔] Task NULL finished."
    wait ${PID_OBJECT}
    echo "[✔] Task OBJECT finished."
    if [ -n "${PID_STYLE}" ]; then
      wait ${PID_STYLE}
      echo "[✔] Task STYLE_DESC finished."
    fi
    echo "[✔] All launched multi-card tasks completed!"
    ;;

  # ----------------------------------------------------------------------------
  # 3. Chia đôi 225 cặp chạy trên 2 GPU (Split Pairs)
  # ----------------------------------------------------------------------------
  split_pairs|benchmark_dual_gpu)
    GPU_0="$(normalize_device "${2:-0}")"
    GPU_1="$(normalize_device "${3:-1}")"

    echo "=== Running Full Benchmark Split on ${GPU_0} and ${GPU_1} ==="
    echo ">> Launching Worker 0 on ${GPU_0} (Pairs 1 -> 112, all 3 levels)..."
    ${PYTHON_BIN} "${SCRIPT}" \
        --device "${GPU_0}" \
        --start_idx 1 \
        --end_idx 112 \
        --prompt_levels null object style_desc \
        > "logs/split_pairs_w0_${GPU_0//:/_}.log" 2>&1 &
    PID_W0=$!

    echo ">> Launching Worker 1 on ${GPU_1} (Pairs 113 -> 225, all 3 levels)..."
    ${PYTHON_BIN} "${SCRIPT}" \
        --device "${GPU_1}" \
        --start_idx 113 \
        --end_idx 225 \
        --prompt_levels null object style_desc \
        > "logs/split_pairs_w1_${GPU_1//:/_}.log" 2>&1 &
    PID_W1=$!

    echo "Worker 0 PID: ${PID_W0} | Log: logs/split_pairs_w0_${GPU_0//:/_}.log"
    echo "Worker 1 PID: ${PID_W1} | Log: logs/split_pairs_w1_${GPU_1//:/_}.log"
    echo "Waiting for both workers to complete..."
    wait ${PID_W0}
    echo "[Worker 0] Finished."
    wait ${PID_W1}
    echo "[Worker 1] Finished."
    echo "[✔] Split pair benchmark completed!"
    ;;

  # ----------------------------------------------------------------------------
  # 4. Chạy một dải cặp (range) trên 1 card
  # ----------------------------------------------------------------------------
  range)
    START_IDX="${2:?Thiếu start_idx}"
    END_IDX="${3:?Thiếu end_idx}"
    GPU_ID="$(normalize_device "${4:-0}")"
    LEVEL="${5:-null object style_desc}"

    echo "=== Running Pairs ${START_IDX} -> ${END_IDX} on ${GPU_ID} ==="
    ${PYTHON_BIN} "${SCRIPT}" \
        --device "${GPU_ID}" \
        --start_idx "${START_IDX}" \
        --end_idx "${END_IDX}" \
        --prompt_levels ${LEVEL} \
        2>&1 | tee "logs/range_${START_IDX}_${END_IDX}_${GPU_ID//:/_}.log"
    ;;

  # ----------------------------------------------------------------------------
  # 5. Chạy Full 675 ảnh tuần tự trên 1 GPU
  # ----------------------------------------------------------------------------
  single|benchmark_single_gpu)
    GPU_ID="$(normalize_device "${2:-0}")"
    echo "=== Running Full Benchmark on Single GPU (${GPU_ID}) ==="
    ${PYTHON_BIN} "${SCRIPT}" \
        --device "${GPU_ID}" \
        --start_idx 1 \
        --end_idx 225 \
        --prompt_levels null object style_desc \
        2>&1 | tee "logs/benchmark_single_${GPU_ID//:/_}.log"
    ;;

  # ----------------------------------------------------------------------------
  # 6. Ablation Study Suite (Table 2)
  # ----------------------------------------------------------------------------
  ablations)
    GPU_ID="$(normalize_device "${2:-0}")"
    echo "=== Running Ablation Study Suite (Table 2) on ${GPU_ID} ==="
    PAIRS_ABLATION="1 2 3 4 5 16 17 18 19 20 31 32 33 34 35 46 47 48 49 50 61 62 63 64 65"

    echo "[1/5] Running Config (B): Pure Mean Token..."
    ${PYTHON_BIN} "${SCRIPT}" --device "${GPU_ID}" --pair_indices ${PAIRS_ABLATION} --alpha_style 0.0 --ablation_tag "ablation_B_pure_mean" > "logs/ablation_B.log" 2>&1

    echo "[2/5] Running Config (C): Raw Style Token..."
    ${PYTHON_BIN} "${SCRIPT}" --device "${GPU_ID}" --pair_indices ${PAIRS_ABLATION} --alpha_style 1.0 --ablation_tag "ablation_C_raw_style" > "logs/ablation_C.log" 2>&1

    echo "[3/5] Running Config (D): No Score-Orthogonal Guidance..."
    ${PYTHON_BIN} "${SCRIPT}" --device "${GPU_ID}" --pair_indices ${PAIRS_ABLATION} --no_ortho --ablation_tag "ablation_D_no_ortho" > "logs/ablation_D.log" 2>&1

    echo "[4/5] Running Config (E): No AdaIN Pushforward..."
    ${PYTHON_BIN} "${SCRIPT}" --device "${GPU_ID}" --pair_indices ${PAIRS_ABLATION} --no_pushforward --ablation_tag "ablation_E_no_pushforward" > "logs/ablation_E.log" 2>&1

    echo "[5/5] Running Config (F): No Semantic Gated Canny..."
    ${PYTHON_BIN} "${SCRIPT}" --device "${GPU_ID}" --pair_indices ${PAIRS_ABLATION} --no_semantic_gating --ablation_tag "ablation_F_no_semantic_gating" > "logs/ablation_F.log" 2>&1

    echo "[✔] Ablation study runs completed!"
    ;;

  help|--help|-h)
    show_help
    ;;

  *)
    echo "Unknown command: ${CMD}"
    echo ""
    show_help
    exit 1
    ;;
esac

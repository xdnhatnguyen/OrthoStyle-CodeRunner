#!/usr/bin/env bash
# ==============================================================================
# OrthoStyle Experiment Orchestration Script
# Multi-Card Dynamic Queue Dispatcher for University Servers & Workstations
# ==============================================================================
#
# USAGE GUIDE:
#
# 1. Chạy TỰ ĐỘNG TẤT CẢ (Queue Dispatcher trên 2 GPUs - Khuyên dùng cho server trường):
#    ./run_experiments.sh run_all              # Tự động điều phối 15 tasks trên GPU 0 & GPU 1
#    ./run_experiments.sh run_all 0 1          # Chỉ định cụ thể GPU 0 và GPU 1
#
# 2. Chạy từng Suite cụ thể (Tự động chia tải 2 GPU qua Queue):
#    ./run_experiments.sh run_ablations 0 1    # Chạy Table 2 Ablations (B..F) + Tau Sweeps (1..4)
#    ./run_experiments.sh run_sweeps 0 1       # Chạy Tau Pushforward Sweeps [1, 2, 3, 4] (49 ảnh 7x7)
#    ./run_experiments.sh run_table2 0 1       # Chạy 5 cấu hình Table 2 Component Ablations (49 ảnh 7x7)
#    ./run_experiments.sh run_main 0 1         # Chạy Main Benchmark Null Prompt (225 cặp)
#
# 3. Chạy thủ công từng task / level trên 1 GPU:
#    ./run_experiments.sh task null 0          # Chạy null prompt trên GPU 0
#    ./run_experiments.sh task object 1        # Chạy object prompt trên GPU 1
#    ./run_experiments.sh task style_desc 0    # Chạy style_desc trên GPU 0
#
# 4. Bắn song song các task lên các card khác nhau cùng lúc (Background):
#    ./run_experiments.sh launch_tasks 0 1     # null -> GPU 0, object -> GPU 1
#
# 5. Chia đôi 225 cặp chạy song song 2 GPU (cả 3 prompt levels):
#    ./run_experiments.sh split_pairs 0 1      # GPU 0: cặp 1..112, GPU 1: cặp 113..225
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
QUEUE_SCRIPT="run_queue.py"

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

1. Automated Server Queue Runners (Auto-loads next task when a GPU finishes):
  ./run_experiments.sh run_all [gpus...]
      Run ALL 15 tasks (sweeps, table2 ablations, main null benchmark, object, style_desc).
      Default GPUs: 0 1. Example: ./run_experiments.sh run_all
                                  ./run_experiments.sh run_all 0 1

  ./run_experiments.sh run_ablations [gpus...]
      Run both Tau Sweeps [1..4] and Table 2 Component Ablations [B..F] (49 pairs 7x7).
      Example: ./run_experiments.sh run_ablations 0 1

  ./run_experiments.sh run_sweeps [gpus...]
      Run Tau Pushforward Sweeps (tau=1, 2, 3, 4; p_switch=0.05..0.20 on 49 pairs 7x7).
      Example: ./run_experiments.sh run_sweeps 0 1

  ./run_experiments.sh run_table2 [gpus...]
      Run Table 2 Component Ablations (B: Pure Mean, C: Raw Style, D: No Ortho, E: No Pushforward, F: No Canny Gating).
      Example: ./run_experiments.sh run_table2 0 1

  ./run_experiments.sh run_main [gpus...]
      Run Main Benchmark (Null prompt across 225 pairs split over GPUs).
      Example: ./run_experiments.sh run_main 0 1

2. Manual / Interactive Task Runners:
  ./run_experiments.sh task <prompt_level> <gpu_id> [start_idx] [end_idx]
      Run a single prompt level ('null', 'object', 'style_desc') on a specific GPU.
      Example: ./run_experiments.sh task null 0

  ./run_experiments.sh launch_tasks <gpu_for_null> <gpu_for_object> [gpu_for_style_desc]
      Launch tasks concurrently across GPUs in background with dedicated logs.
      Example: ./run_experiments.sh launch_tasks 0 1

  ./run_experiments.sh split_pairs <gpu_0> <gpu_1>
      Split 225 pairs evenly across 2 GPUs (GPU 0 handles 1..112, GPU 1 handles 113..225).
      Example: ./run_experiments.sh split_pairs 0 1

  ./run_experiments.sh single [gpu_id]
      Run all 3 prompt levels sequentially on a single GPU (default: 0).
      Example: ./run_experiments.sh single 0
EOF
}

CMD="${1:-help}"

case "${CMD}" in
  # ----------------------------------------------------------------------------
  # Automated Dynamic Queue Dispatchers (Khuyên dùng cho server trường)
  # ----------------------------------------------------------------------------
  run_all)
    shift || true
    GPUS=("$@")
    if [ ${#GPUS[@]} -eq 0 ]; then
      GPUS=("0" "1")
    fi
    echo "=== Starting Full Suite Queue Dispatcher on GPUs: ${GPUS[*]} ==="
    ${PYTHON_BIN} "${QUEUE_SCRIPT}" --suite "all" --gpus "${GPUS[@]}"
    ;;

  run_ablations)
    shift || true
    GPUS=("$@")
    if [ ${#GPUS[@]} -eq 0 ]; then
      GPUS=("0" "1")
    fi
    echo "=== Starting Ablations Queue Dispatcher (Sweeps + Table 2) on GPUs: ${GPUS[*]} ==="
    ${PYTHON_BIN} "${QUEUE_SCRIPT}" --suite "ablations" --gpus "${GPUS[@]}"
    ;;

  run_sweeps)
    shift || true
    GPUS=("$@")
    if [ ${#GPUS[@]} -eq 0 ]; then
      GPUS=("0" "1")
    fi
    echo "=== Starting Tau Sweeps [1, 2, 3, 4] Queue Dispatcher on GPUs: ${GPUS[*]} ==="
    ${PYTHON_BIN} "${QUEUE_SCRIPT}" --suite "sweeps" --gpus "${GPUS[@]}"
    ;;

  run_table2|ablations)
    shift || true
    GPUS=("$@")
    if [ ${#GPUS[@]} -eq 0 ]; then
      GPUS=("0" "1")
    fi
    echo "=== Starting Table 2 Component Ablations Queue Dispatcher on GPUs: ${GPUS[*]} ==="
    ${PYTHON_BIN} "${QUEUE_SCRIPT}" --suite "table2" --gpus "${GPUS[@]}"
    ;;

  run_main)
    shift || true
    GPUS=("$@")
    if [ ${#GPUS[@]} -eq 0 ]; then
      GPUS=("0" "1")
    fi
    echo "=== Starting Main Benchmark (Null Prompt 225 pairs) Queue on GPUs: ${GPUS[*]} ==="
    ${PYTHON_BIN} "${QUEUE_SCRIPT}" --suite "main" --gpus "${GPUS[@]}"
    ;;

  # ----------------------------------------------------------------------------
  # Manual / Interactive Tasks
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
    fi

    echo ""
    echo "Workers running in background. Waiting for completion..."
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

  split_pairs|benchmark_dual_gpu)
    GPU_0="$(normalize_device "${2:-0}")"
    GPU_1="$(normalize_device "${3:-1}")"

    echo "=== Running Full Benchmark Split on ${GPU_0} and ${GPU_1} ==="
    echo ">> Launching Worker 0 on ${GPU_0} (Pairs 1 -> 112)..."
    ${PYTHON_BIN} "${SCRIPT}" \
        --device "${GPU_0}" \
        --start_idx 1 \
        --end_idx 112 \
        --prompt_levels null object style_desc \
        > "logs/split_pairs_w0_${GPU_0//:/_}.log" 2>&1 &
    PID_W0=$!

    echo ">> Launching Worker 1 on ${GPU_1} (Pairs 113 -> 225)..."
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

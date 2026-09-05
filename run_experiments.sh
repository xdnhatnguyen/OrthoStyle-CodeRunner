#!/usr/bin/env bash
# ==============================================================================
# OrthoStyle Experiment Orchestration Script
# Supports:
#   1. Full Benchmark Matrix across 3 prompt levels (225 pairs x 3 levels = 675 runs)
#   2. Automatic Timestep Preview saving for first 5x5 pairs (25 pairs x 20 steps)
#   3. Table 2: Ablation Study (Configurations A to F)
#   4. Step sweep for style injection switch (p_switch in 0.05, 0.10, 0.15, 0.20)
#   5. Dual-GPU parallel or Single-GPU sequential execution
# ==============================================================================

set -u

PYTHON_BIN="/home/khoan/.conda/envs/rbm/bin/python"
SCRIPT="batch_runner.py"

mkdir -p logs
mkdir -p output/benchmark
mkdir -p output/previews

MODE="${1:-benchmark_single_gpu}" # benchmark_single_gpu, benchmark_dual_gpu, previews_only, ablations

case "${MODE}" in
  # ----------------------------------------------------------------------------
  # 1. Benchmark on Single GPU (cuda:0)
  # ----------------------------------------------------------------------------
  benchmark_single_gpu)
    echo "=== Running Full Benchmark on Single GPU (cuda:0) ==="
    ${PYTHON_BIN} "${SCRIPT}" \
        --device "cuda:0" \
        --start_idx 1 \
        --end_idx 225 \
        --prompt_levels null object style_desc \
        --auto_previews_5x5 \
        2>&1 | tee "logs/benchmark_single_gpu.log"
    ;;

  # ----------------------------------------------------------------------------
  # 2. Benchmark Parallel on Dual GPU (GPU 0 & GPU 1)
  # ----------------------------------------------------------------------------
  benchmark_dual_gpu)
    echo "=== Running Full Benchmark Split on GPU 0 & GPU 1 ==="
    
    # GPU 0: Pairs 1 -> 112
    echo "Launching Worker 0 on cuda:0 (Pairs 1 -> 112)..."
    ${PYTHON_BIN} "${SCRIPT}" \
        --device "cuda:0" \
        --start_idx 1 \
        --end_idx 112 \
        --prompt_levels null object style_desc \
        --auto_previews_5x5 \
        > "logs/benchmark_gpu0.log" 2>&1 &
    PID_GPU0=$!

    # GPU 1: Pairs 113 -> 225
    echo "Launching Worker 1 on cuda:1 (Pairs 113 -> 225)..."
    ${PYTHON_BIN} "${SCRIPT}" \
        --device "cuda:1" \
        --start_idx 113 \
        --end_idx 225 \
        --prompt_levels null object style_desc \
        --auto_previews_5x5 \
        > "logs/benchmark_gpu1.log" 2>&1 &
    PID_GPU1=$!

    echo "Worker 0 PID: ${PID_GPU0}"
    echo "Worker 1 PID: ${PID_GPU1}"
    echo "Waiting for both workers to complete..."
    wait ${PID_GPU0}
    echo "[GPU 0] Finished."
    wait ${PID_GPU1}
    echo "[GPU 1] Finished."
    ;;

  # ----------------------------------------------------------------------------
  # 3. Timestep Previews Only for first 5x5 pairs (25 pairs)
  # ----------------------------------------------------------------------------
  previews_only)
    echo "=== Running Timestep Preview Extraction for First 5x5 Pairs ==="
    ${PYTHON_BIN} "${SCRIPT}" \
        --device "cuda:0" \
        --pair_indices 1 2 3 4 5 16 17 18 19 20 31 32 33 34 35 46 47 48 49 50 61 62 63 64 65 \
        --prompt_levels null \
        --save_previews \
        2>&1 | tee "logs/previews_5x5.log"
    ;;

  # ----------------------------------------------------------------------------
  # 4. Table 2: Ablation Study Suite
  # ----------------------------------------------------------------------------
  ablations)
    echo "=== Running Ablation Study Suite (Table 2) ==="
    # Chạy trên tập 25 cặp tiêu chuẩn (5x5) để đánh giá ablation hiệu quả
    PAIRS_ABLATION="1 2 3 4 5 16 17 18 19 20 31 32 33 34 35 46 47 48 49 50 61 62 63 64 65"

    # (B) Pure Mean Token (alpha_s = 0)
    echo "Running Config (B): Pure Mean Token..."
    ${PYTHON_BIN} "${SCRIPT}" --device "cuda:0" --pair_indices ${PAIRS_ABLATION} --alpha_style 0.0 --ablation_tag "ablation_B_pure_mean" > "logs/ablation_B.log" 2>&1

    # (C) Raw Style Token (alpha_s = 1.0)
    echo "Running Config (C): Raw Style Token..."
    ${PYTHON_BIN} "${SCRIPT}" --device "cuda:0" --pair_indices ${PAIRS_ABLATION} --alpha_style 1.0 --ablation_tag "ablation_C_raw_style" > "logs/ablation_C.log" 2>&1

    # (D) Không có Score-Orthogonal Guidance
    echo "Running Config (D): No Score-Orthogonal Guidance..."
    ${PYTHON_BIN} "${SCRIPT}" --device "cuda:0" --pair_indices ${PAIRS_ABLATION} --no_ortho --ablation_tag "ablation_D_no_ortho" > "logs/ablation_D.log" 2>&1

    # (E) Không có AdaIN Pushforward
    echo "Running Config (E): No AdaIN Pushforward..."
    ${PYTHON_BIN} "${SCRIPT}" --device "cuda:0" --pair_indices ${PAIRS_ABLATION} --no_pushforward --ablation_tag "ablation_E_no_pushforward" > "logs/ablation_E.log" 2>&1

    # (F) Không có Semantic Gated Canny
    echo "Running Config (F): No Semantic Gated Canny..."
    ${PYTHON_BIN} "${SCRIPT}" --device "cuda:0" --pair_indices ${PAIRS_ABLATION} --no_semantic_gating --ablation_tag "ablation_F_no_semantic_gating" > "logs/ablation_F.log" 2>&1

    echo "Ablation study runs completed!"
    ;;

  *)
    echo "Usage: $0 {benchmark_single_gpu|benchmark_dual_gpu|previews_only|ablations}"
    exit 1
    ;;
esac

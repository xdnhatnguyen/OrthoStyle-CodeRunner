#!/bin/bash

BASE_DIR="/mnt/wav2vec2/khoan/source_code/Uncertainty-OC"
CONTENT_DIR="${BASE_DIR}/data_ablation/distanglement_style/content"
STYLE_DIR="${BASE_DIR}/data_ablation/distanglement_style/style"
GENERATED_ROOT="${BASE_DIR}/distanglement_data"
METRICS_DIR="${BASE_DIR}/distanglement_data/metrics_ablations"

mkdir -p "${METRICS_DIR}"

for generated_dir in "${GENERATED_ROOT}"/*; do

    # Bỏ qua nếu không phải folder
    [ -d "${generated_dir}" ] || continue

    # Lấy tên folder
    name=$(basename "${generated_dir}")

    echo "============================================================"
    echo "Evaluating: ${name}"
    echo "Generated dir: ${generated_dir}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=1 python evaluate_style_transfer_metrics.py \
        --content_dir "${CONTENT_DIR}" \
        --style_dir "${STYLE_DIR}" \
        --generated_dir "${generated_dir}" \
        --prefix "${name}" \
        --output_csv "${METRICS_DIR}/metrics_full_${name}.csv" \
        --output_summary_csv "${METRICS_DIR}/metrics_summary_${name}.csv" \
        --device cuda

    echo "Finished: ${name}"
    echo

done
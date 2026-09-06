CUDA_VISIBLE_DEVICES=1 python evaluate_style_transfer_metrics.py \
  --content_dir /mnt/wav2vec2/khoan/source_code/Uncertainty-OC/data/content \
  --style_dir /mnt/wav2vec2/khoan/source_code/Uncertainty-OC/data/style \
  --generated_dir /mnt/wav2vec2/khoan/source_code/Uncertainty-OC/out_images/ours_svd_0.6_high_noise \
  --prefix ours_svd_0.6_high_noise \
  --output_csv metrics/metrics_full_ours_svd_0.6_high_noise.csv \
  --output_summary_csv metrics/metrics_summary_ours_svd_0.6_high_noise.csv \
  --device cuda
# 🔥 Our Result

![teaser](./assets/ours.png)


## 📥 Installation

```
# Download pretrained models.
cd third_party/StableCascade/models
bash download_models.sh essential big-big bfloat16
cd ..

# Install dependencies following the original [StableCascade](https://github.com/Stability-AI/StableCascade/blob/master/inference/readme.md)
conda create -n rbm python==3.9
pip install -r requirements.txt
pip install jupyter notebook opencv-python matplotlib ftfy

# Download [pre-trained CSD weights](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view) and put it under `third_party/CSD/checkpoint.pth`.

# Install LangSAM
pip install  git+https://github.com/IDEA-Research/GroundingDINO.git
pip install segment-anything==1.0
git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
pip install -e .
```

## 🚀 Running Inference & Benchmarks

> **Crash Resilience & Auto-Resume**: The runner automatically checks completed image files using PIL validation (`is_valid_image`). If a job is stopped midway or interrupted, simply re-run the same command — it will immediately skip valid finished images and resume smoothly.

### 1. Automated Server Queue Runners (Khuyên dùng cho Server trường / Slurm Jobs)
Tự động duy trì 2 GPU luôn chạy 100% công suất: khi một GPU xong task sẽ tự động bốc task kế tiếp trong hàng đợi, không bao giờ ngắt kết nối hay nhả tài nguyên giữa chừng.

```bash
# Chạy TOÀN BỘ tất cả các thí nghiệm (Sweeps + Table 2 Ablations + Main Benchmark + 3-Level Prompts):
./run_experiments.sh run_all 0 1

# Chạy toàn bộ Ablation Suite (Tau Sweeps 1..4 + Table 2 Components B..F):
./run_experiments.sh run_ablations 0 1

# Chạy riêng Tau Pushforward Sweeps [1, 2, 3, 4] (49 ảnh 7x7):
./run_experiments.sh run_sweeps 0 1

# Chạy riêng Table 2 Component Ablation [B, C, D, E, F] (49 ảnh 7x7):
./run_experiments.sh run_table2 0 1

# Chạy Main Benchmark (Null Prompt 225 cặp so sánh baseline):
./run_experiments.sh run_main 0 1
```

### 2. Manual / Interactive Task Runners (`run_experiments.sh`)

```bash
# Chạy 1 task (prompt level) trên 1 card cụ thể:
./run_experiments.sh task null 0         # Chạy null prompt trên GPU 0
./run_experiments.sh task object 1       # Chạy object prompt trên GPU 1
./run_experiments.sh task style_desc 0   # Chạy style_desc trên GPU 0

# Bắn đồng thời các task lên các card khác nhau (Background + logs riêng):
./run_experiments.sh launch_tasks 0 1    # null -> GPU 0, object -> GPU 1

# Chia đôi 225 cặp song song 2 GPU (cả 3 prompt levels):
./run_experiments.sh split_pairs 0 1     # GPU 0: cặp 1..112, GPU 1: cặp 113..225

# Chạy toàn bộ trên 1 GPU duy nhất:
./run_experiments.sh single 0            # Chạy toàn bộ trên GPU 0
```

### 3. Direct Python Execution (`batch_runner.py`)
```bash
python batch_runner.py \
    --device cuda:0 \
    --config_path configs/benchmark_config.json \
    --tau_pushforward 2 \
    --start_idx 1 \
    --end_idx 225 \
    --prompt_levels null object style_desc
```

### 3. Server Setup Notes
On a new server, place or symlink `data/` and `third_party/` into the repo root:
```bash
# If data and third_party are stored in shared storage:
ln -s /path/to/shared/data ./data
ln -s /path/to/shared/third_party ./third_party
```
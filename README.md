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

### 1. Multi-Card & Flexible Task Dispatcher (`run_experiments.sh`)

```bash
# Case 1: Chạy 1 task (prompt level) trên 1 card cụ thể:
./run_experiments.sh task null 0         # Chạy null prompt trên GPU 0
./run_experiments.sh task object 1       # Chạy object prompt trên GPU 1
./run_experiments.sh task style_desc 2   # Chạy style_desc trên GPU 2

# Case 2: Bắn đồng thời các task lên các card khác nhau (Background + logs riêng):
./run_experiments.sh launch_tasks 0 1      # null -> GPU 0, object -> GPU 1
./run_experiments.sh launch_tasks 0 1 2    # null -> GPU 0, object -> GPU 1, style_desc -> GPU 2

# Case 3: Chia đôi 225 cặp song song 2 GPU (cả 3 prompt levels):
./run_experiments.sh split_pairs 0 1     # GPU 0: cặp 1..112, GPU 1: cặp 113..225

# Case 4: Chạy toàn bộ 675 ảnh trên 1 GPU duy nhất:
./run_experiments.sh single 0            # Chạy toàn bộ trên GPU 0

# Case 5: Chạy bộ thí nghiệm Ablation Study (Bảng 2 trong paper):
./run_experiments.sh ablations 0
```

### 2. Direct Python Execution (`batch_runner.py`)
```bash
python batch_runner.py \
    --device cuda:0 \
    --config_path configs/benchmark_config.json \
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
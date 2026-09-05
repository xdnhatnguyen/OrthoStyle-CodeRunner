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

### 1. Single pair test
```bash
python main_afa.py
```

### 2. Test 1 Style x 15 Contents (1-GPU Sequential Offload < 24GB VRAM)
```bash
python run_test.py --device cuda:0 --style_file 01_antimonocromatismo.png --prompt_levels null
```

### 3. Full Benchmark & Ablation Suite
```bash
# Run full benchmark (225 pairs x 3 prompt levels) on a single GPU
bash run_experiments.sh benchmark_single_gpu

# Or run parallel on 2 GPUs (GPU 0 & GPU 1)
bash run_experiments.sh benchmark_dual_gpu

# Or run Table 2 Ablation study suite
bash run_experiments.sh ablations
```
================================================================================
SOICT 2026 - BENCHMARK DỮ LIỆU ĐÁNH GIÁ TRAINING-FREE STYLE TRANSFER (15x15)
================================================================================

1. CẤU TRÚC GÓI DỮ LIỆU:
├── content/               : 15 ảnh Content chuẩn PNG RGB (01_backpack_dog.png -> 15_decorative_vase.png)
├── style/                 : 15 ảnh Style chuẩn PNG RGB (01_antimonocromatismo.png -> 15_oil_pastels.png)
├── prompt.txt             : Danh sách prompt dạng nhóm và chi tiết 225 cặp
├── benchmark_config.json  : File cấu hình Dictionary JSON cho code tự động lặp tuần tự
└── sample_batch_runner.py : Script Python mẫu để đọc benchmark_config.json và chạy vòng lặp

2. QUY MÔ THỰC NGHIỆM:
- 15 ảnh Content x 15 ảnh Style = 225 cặp kiểm thử (pairs).
- Các ảnh đã được chuẩn hóa hoàn toàn về định dạng PNG 3 kênh màu (RGB), không chứa kênh Alpha,
  độ phân giải cao, phù hợp trực tiếp với VAE và U-Net của các mô hình Diffusion.

3. BA (03) CẤP ĐỘ PROMPT (PROMPT LEVELS / SETTINGS):
Tùy vào phương pháp baseline yêu cầu, có thể chạy ở 1 trong 3 setting sau (đã có sẵn trong benchmark_config.json):

* Setting 1: NULL PROMPT
  - Prompt: "" (chuỗi rỗng)
  - Đánh giá khả năng chuyển phong cách hoàn toàn dựa vào ảnh style (image-driven).

* Setting 2: OBJECT PROMPT
  - Prompt: Chỉ mô tả danh từ vật thể (không chứa màu sắc/chất liệu gây bias).
  - Ví dụ: "A backpack with a dog print", "A teddy bear plushie", "A sneaker", "A cat sitting".

* Setting 3: STYLE DESCRIPTION PROMPT
  - Prompt: Mô tả vật thể kết hợp diễn giải phong cách nghệ thuật.
  - Ví dụ: "A cat sitting in cyberpunk neon style", "A sneaker in flat vector illustration style".

4. CÁCH SỬ DỤNG TRONG CODE PYTHON:
Xem file mẫu `sample_batch_runner.py` hoặc đọc trực tiếp `benchmark_config.json`:

```python
import json
from pathlib import Path

with open("benchmark_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# Lặp qua 225 cặp thí nghiệm:
for item in config["pairs"]:
    c_img = Path("content") / item["content_file"]
    s_img = Path("style") / item["style_file"]
    
    # Chọn prompt tùy theo setting thí nghiệm:
    prompt = item["level1_null_prompt"]                # Setting 1
    # prompt = item["level2_object_prompt"]             # Setting 2
    # prompt = item["level3_style_description_prompt"]  # Setting 3
    
    # -> Truyền c_img, s_img, prompt vào mô hình baseline để sinh ảnh
```
================================================================================

"""
Script mẫu đọc benchmark_config.json để chạy tuần tự 15x15 cặp ảnh trên Diffusion Model.
Hỗ trợ 3 prompt settings:
  1. null
  2. object
  3. style_desc
"""
import json
import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_setting", type=str, default="style_desc", choices=["null", "object", "style_desc"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config_path = Path("benchmark_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    prompt_key = {
        "null": "level1_null_prompt",
        "object": "level2_object_prompt",
        "style_desc": "level3_style_description_prompt"
    }[args.prompt_setting]

    print(f"=== CHẠY THÍ NGHIỆM VỚI PROMPT SETTING: {args.prompt_setting.upper()} ===")
    print(f"Tổng số cặp: {len(config['pairs'])}")

    for item in config["pairs"]:
        idx = item["pair_idx"]
        content_path = Path("content") / item["content_file"]
        style_path = Path("style") / item["style_file"]
        prompt = item[prompt_key]

        print(f"[{idx:03d}/225] Content: {content_path.name} | Style: {style_path.name} | Prompt: '{prompt}'")
        
        # --- ĐÂY LÀ NƠI GỌI MODEL DIFFUSION CỦA BẠN ---
        # output_image = model.style_transfer(
        #     content_image=load_image(content_path),
        #     style_image=load_image(style_path),
        #     prompt=prompt
        # )
        # output_image.save(f"output/{args.prompt_setting}/{item['content_file']}_{item['style_file']}")

if __name__ == "__main__":
    main()

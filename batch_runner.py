#!/usr/bin/env python3
"""
OrthoStyle Batch Runner with 1-GPU Sequential Offloading (<24GB VRAM)
Supports:
  - 15x15 Benchmark matrix across 3 prompt levels (null, object, style_desc)
  - Timestep preview extraction for the first 5x5 pairs
  - Full Ablation configurations (alpha_s, p_switch, score-ortho, AdaIN pushforward, Canny semantic gating)
  - Seamless resume/checkpointing (skips existing outputs)
"""

import os
import sys

# Dynamic third_party path resolution (supports local and external paths)
_TP_ROOT = os.environ.get("THIRD_PARTY_ROOT", "third_party")
if os.path.exists(os.path.join(_TP_ROOT, "configs/inference/stage_c_3b.yaml")):
    _SC_DIR = _TP_ROOT
    _PARENT_TP = os.path.dirname(os.path.abspath(_TP_ROOT.rstrip("/")))
else:
    _SC_DIR = os.path.join(_TP_ROOT, "StableCascade")
    _PARENT_TP = _TP_ROOT

for p in [_PARENT_TP, _SC_DIR, "third_party/", "third_party/StableCascade/"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import argparse
import copy
import gc
import json
import math
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import yaml
from accelerate.utils import set_module_tensor_to_device
from tqdm import tqdm

from core.utils import load_or_fail
from gdf import AdaptiveLossWeight, CosineTNoiseCond, DDPMSampler, VPScaler
from gdf.schedulers import CosineSchedule
from gdf.targets import EpsilonTarget
from gdf_rbm import RBM, setup_dino
from inference.utils import calculate_latent_sizes, resize_image, save_images
from modules.controlnet import CannyFilter, ControlNet
from stage_c_rbm import StageCRBM
from train import WurstCoreB
from utils import Style_Storage, WurstCoreCRBM

import rembg


# -----------------------------------------------------------------------------
# File Integrity & Atomic Saving (Crash / Kill Prevention)
# -----------------------------------------------------------------------------

def is_valid_image(filepath: str) -> bool:
    """
    Check if a file exists, is non-empty (>100 bytes), and can be opened
    and verified by PIL without corruption.
    """
    if not filepath or not os.path.exists(filepath):
        return False
    try:
        if os.path.getsize(filepath) < 100:
            return False
        with PIL.Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def atomic_save_image(tensor, target_path: str):
    """
    Save image tensor to a temporary file first, verify integrity,
    then atomically replace the target destination. This guarantees zero
    corrupted or 0-byte files if the process is killed midway.
    """
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    rand_id = torch.randint(0, 1000000, (1,)).item()
    temp_path = f"{target_path}.tmp_{os.getpid()}_{rand_id}.png"
    try:
        save_images(tensor, temp_path)
        if is_valid_image(temp_path):
            os.replace(temp_path, target_path)
        else:
            raise IOError(f"Failed to verify written temporary image: {temp_path}")
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise e


# -----------------------------------------------------------------------------
# Memory Management & GPU Swapping
# -----------------------------------------------------------------------------

def hard_clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def move_stage_c_to_gpu(models_rbm, controlnet, dino_model, device):
    if getattr(models_rbm, "generator", None) is not None:
        models_rbm.generator.to(device)
    if getattr(models_rbm, "effnet", None) is not None:
        models_rbm.effnet.to(device)
    if getattr(models_rbm, "text_model", None) is not None:
        models_rbm.text_model.to(device)
    if getattr(models_rbm, "image_model", None) is not None:
        models_rbm.image_model.to(device)
    if getattr(models_rbm, "previewer", None) is not None:
        models_rbm.previewer.to(device)
    if controlnet is not None:
        controlnet.to(device)
    if dino_model is not None:
        dino_model.to(device)


def move_stage_c_condition_models_to_cpu(models_rbm, controlnet):
    if getattr(models_rbm, "effnet", None) is not None:
        models_rbm.effnet.to("cpu")
    if getattr(models_rbm, "text_model", None) is not None:
        models_rbm.text_model.to("cpu")
    if getattr(models_rbm, "image_model", None) is not None:
        models_rbm.image_model.to("cpu")
    if controlnet is not None:
        controlnet.to("cpu")


def move_stage_c_to_cpu(models_rbm, controlnet, dino_model):
    if getattr(models_rbm, "generator", None) is not None:
        models_rbm.generator.to("cpu")
    if getattr(models_rbm, "effnet", None) is not None:
        models_rbm.effnet.to("cpu")
    if getattr(models_rbm, "text_model", None) is not None:
        models_rbm.text_model.to("cpu")
    if getattr(models_rbm, "image_model", None) is not None:
        models_rbm.image_model.to("cpu")
    if getattr(models_rbm, "previewer", None) is not None:
        models_rbm.previewer.to("cpu")
    if controlnet is not None:
        controlnet.to("cpu")
    if dino_model is not None:
        dino_model.to("cpu")


def move_stage_b_to_gpu(models_b, device):
    if getattr(models_b, "generator", None) is not None:
        models_b.generator.to(device, dtype=torch.bfloat16)
    if getattr(models_b, "stage_a", None) is not None:
        models_b.stage_a.to(device)
    if getattr(models_b, "text_model", None) is not None:
        models_b.text_model.to(device)


def move_stage_b_to_cpu(models_b):
    if getattr(models_b, "generator", None) is not None:
        models_b.generator.to("cpu")
    if getattr(models_b, "stage_a", None) is not None:
        models_b.stage_a.to("cpu")
    if getattr(models_b, "text_model", None) is not None:
        models_b.text_model.to("cpu")


# -----------------------------------------------------------------------------
# Semantic Gated Canny Helper
# -----------------------------------------------------------------------------

def get_clean_canny(rgb_image_pil: PIL.Image.Image, noisy_canny_tensor: torch.Tensor, use_semantic_gating: bool = True) -> torch.Tensor:
    if not use_semantic_gating:
        return noisy_canny_tensor

    mask_pil = rembg.remove(rgb_image_pil, only_mask=True)
    mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(
        device=noisy_canny_tensor.device, dtype=noisy_canny_tensor.dtype
    )

    if mask_tensor.shape[-2:] != noisy_canny_tensor.shape[-2:]:
        mask_tensor = F.interpolate(
            mask_tensor,
            size=noisy_canny_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    dilated_mask = F.max_pool2d(mask_tensor, kernel_size=5, stride=1, padding=2)
    hard_mask = (dilated_mask > 0.5).to(dtype=noisy_canny_tensor.dtype)
    return noisy_canny_tensor * hard_mask


# -----------------------------------------------------------------------------
# Setup & Model Loader (Run ONCE)
# -----------------------------------------------------------------------------

def setup_all_models(device_str: str = "cuda:0", use_controlnet_canny: bool = True, third_party_root: str = "third_party"):
    if str(device_str).isdigit():
        device_str = f"cuda:{device_str}"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing OrthoStyle models on {device} (with CPU offload)...", flush=True)

    # Resolve paths for StableCascade and CSD
    if os.path.exists(os.path.join(third_party_root, "configs/inference/stage_c_3b.yaml")):
        sc_dir = third_party_root
        parent_tp = os.path.dirname(os.path.abspath(third_party_root.rstrip("/")))
    else:
        sc_dir = os.path.join(third_party_root, "StableCascade")
        parent_tp = third_party_root

    # Set CSD environment variable for utils.py if not already set
    csd_path = os.environ.get("CSD_CHECKPOINT_PATH", os.path.join(parent_tp, "CSD/checkpoint.pth"))
    if os.path.exists(csd_path):
        os.environ["CSD_CHECKPOINT_PATH"] = csd_path
    else:
        print(f"[!] Notice: CSD checkpoint not found at: {csd_path}", flush=True)

    sc_models_dir = os.path.join(sc_dir, "models")
    print(f"[*] [1/5] Loading Stage C Core configs (from {sc_dir})...", flush=True)

    # Stage C config
    config_file_c = os.path.join(sc_dir, "configs/inference/stage_c_3b.yaml")
    if not os.path.exists(config_file_c):
        raise FileNotFoundError(f"[ERROR] Stage C config not found at: {config_file_c}")
    with open(config_file_c, "r", encoding="utf-8") as f:
        loaded_config_c = yaml.safe_load(f)

    # Rewrite model paths to actual directory if needed
    for k in ["effnet_checkpoint_path", "previewer_checkpoint_path", "generator_checkpoint_path"]:
        if k in loaded_config_c and isinstance(loaded_config_c[k], str):
            fname = os.path.basename(loaded_config_c[k])
            model_f = os.path.join(sc_models_dir, fname)
            if os.path.exists(model_f):
                loaded_config_c[k] = model_f
            elif not os.path.exists(loaded_config_c[k]):
                raise FileNotFoundError(
                    f"[ERROR] Required StableCascade model '{fname}' not found!\n"
                    f"  Checked: {model_f}\n"
                    f"  Checked: {loaded_config_c[k]}\n"
                    f"  Please make sure models are located in: {sc_models_dir}/"
                )

    core_c = WurstCoreCRBM(config_dict=loaded_config_c, device=device, training=False)

    print(f"[*] [2/5] Loading Stage B Core configs...", flush=True)
    # Stage B config
    config_file_b = os.path.join(sc_dir, "configs/inference/stage_b_3b.yaml")
    if not os.path.exists(config_file_b):
        raise FileNotFoundError(f"[ERROR] Stage B config not found at: {config_file_b}")
    with open(config_file_b, "r", encoding="utf-8") as f:
        loaded_config_b = yaml.safe_load(f)

    for k in ["effnet_checkpoint_path", "stage_a_checkpoint_path", "generator_checkpoint_path"]:
        if k in loaded_config_b and isinstance(loaded_config_b[k], str):
            fname = os.path.basename(loaded_config_b[k])
            model_f = os.path.join(sc_models_dir, fname)
            if os.path.exists(model_f):
                loaded_config_b[k] = model_f
            elif not os.path.exists(loaded_config_b[k]):
                raise FileNotFoundError(
                    f"[ERROR] Required Stage B model '{fname}' not found!\n"
                    f"  Checked: {model_f}\n"
                    f"  Checked: {loaded_config_b[k]}\n"
                    f"  Please make sure models are located in: {sc_models_dir}/"
                )

    core_b = WurstCoreB(config_dict=loaded_config_b, device=device, training=False)

    extras_pre = core_c.setup_extras_pre()
    gdf_rbm = RBM(
        schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
        input_scaler=VPScaler(),
        target=EpsilonTarget(),
        noise_cond=CosineTNoiseCond(),
        loss_weight=AdaptiveLossWeight(),
    )
    sampling_configs = {
        "cfg": 4.0,
        "sampler": DDPMSampler(gdf_rbm),
        "shift": 2,
        "timesteps": 20,
        "t_start": 1.0,
    }

    extras = core_c.Extras(
        gdf=gdf_rbm,
        sampling_configs=sampling_configs,
        transforms=extras_pre.transforms,
        effnet_preprocess=extras_pre.effnet_preprocess,
        clip_preprocess=extras_pre.clip_preprocess,
    )

    models_c = core_c.setup_models(extras)
    models_c.generator.eval().requires_grad_(False)

    extras_b = core_b.setup_extras_pre()
    extras_b.sampling_configs["cfg"] = 1.1
    extras_b.sampling_configs["shift"] = 1
    extras_b.sampling_configs["timesteps"] = 10
    extras_b.sampling_configs["t_start"] = 1.0

    models_b = core_b.setup_models(extras_b, skip_clip=True)
    models_b = WurstCoreB.Models(
        **{
            **models_b.to_dict(),
            "tokenizer": models_c.tokenizer,
            "text_model": copy.deepcopy(models_c.text_model),
        }
    )
    models_b.generator.eval().requires_grad_(False)

    print(f"[*] [3/5] Loading StageCRBM generator weights...", flush=True)
    generator_rbm = StageCRBM()
    c_ckpt = load_or_fail(core_c.config.generator_checkpoint_path)
    if c_ckpt is None:
        raise FileNotFoundError(f"[ERROR] Could not load generator checkpoint: {core_c.config.generator_checkpoint_path}")
    for param_name, param in c_ckpt.items():
        set_module_tensor_to_device(generator_rbm, param_name, "cpu", value=param)
    generator_rbm = generator_rbm.to(getattr(torch, core_c.config.dtype)).to(device)
    generator_rbm = core_c.load_model(generator_rbm, "generator")

    models_rbm = core_c.Models(
        effnet=models_c.effnet,
        previewer=models_c.previewer,
        generator=generator_rbm,
        generator_ema=models_c.generator_ema,
        tokenizer=models_c.tokenizer,
        text_model=models_c.text_model,
        image_model=models_c.image_model,
    )
    models_rbm.generator.eval().requires_grad_(False)

    controlnet = None
    canny_filter = None
    if use_controlnet_canny:
        print(f"[*] [4/5] Loading ControlNet Canny...", flush=True)
        controlnet = ControlNet(c_in=1, proj_blocks=[0, 4, 8, 12, 51, 55, 59, 63], bottleneck_mode=None)
        canny_ckpt_path = os.environ.get("CONTROLNET_CHECKPOINT_PATH", os.path.join(sc_models_dir, "canny.safetensors"))
        if not os.path.exists(canny_ckpt_path):
            for fallback in [
                "third_party/StableCascade/models/canny.safetensors",
                os.path.join(parent_tp, "StableCascade/models/canny.safetensors"),
                os.path.join(os.environ.get("PROJECT_ROOT", "."), "third_party/StableCascade/models/canny.safetensors"),
            ]:
                if os.path.exists(fallback):
                    canny_ckpt_path = fallback
                    break
        if not os.path.exists(canny_ckpt_path):
            raise FileNotFoundError(
                f"[ERROR] ControlNet checkpoint not found at: '{canny_ckpt_path}'!\n"
                f"  Please download from: https://huggingface.co/stabilityai/stable-cascade/resolve/main/controlnet/canny.safetensors\n"
                f"  and save to: {os.path.join(sc_models_dir, 'canny.safetensors')}"
            )
        cnet_ckpt = load_or_fail(canny_ckpt_path)
        controlnet.load_state_dict(cnet_ckpt if "state_dict" not in cnet_ckpt else cnet_ckpt["state_dict"])
        controlnet = controlnet.to(getattr(torch, core_c.config.dtype)).eval().requires_grad_(False)
        canny_filter = CannyFilter(device, resize=224)

    print(f"[*] [5/5] Loading DINO model...", flush=True)
    dino_model = setup_dino(device=str(device))

    # Keep all models offloaded to CPU initially
    move_stage_c_to_cpu(models_rbm, controlnet, dino_model)
    move_stage_b_to_cpu(models_b)
    hard_clear_cuda()
    print(f"[✔] All models loaded successfully! Ready for inference.\n", flush=True)

    print("[*] Models initialized and offloaded to CPU successfully.")
    return {
        "device": device,
        "core_c": core_c,
        "core_b": core_b,
        "extras": extras,
        "extras_b": extras_b,
        "models_rbm": models_rbm,
        "models_b": models_b,
        "controlnet": controlnet,
        "canny_filter": canny_filter,
        "dino_model": dino_model,
    }


# -----------------------------------------------------------------------------
# Single Pair Inference (1-GPU Sequential Offload)
# -----------------------------------------------------------------------------

def run_single_inference(
    state: dict,
    content_path: str,
    style_path: str,
    prompt: str,
    save_path: str,
    save_grid_path: str = None,
    preview_dir: str = None,
    tau_pushforward: int = 5,
    cnet_strength: float = 0.8,
    use_semantic_gating: bool = True,
):
    device = state["device"]
    core_c = state["core_c"]
    core_b = state["core_b"]
    extras = state["extras"]
    extras_b = state["extras_b"]
    models_rbm = state["models_rbm"]
    models_b = state["models_b"]
    controlnet = state["controlnet"]
    canny_filter = state["canny_filter"]
    dino_model = state["dino_model"]

    batch_size = 1
    height, width = 1024, 1024
    stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

    # 1. Load input images
    ref_sub_pil = PIL.Image.open(content_path).convert("RGB")
    ref_sty_pil = PIL.Image.open(style_path).convert("RGB")

    ref_images = resize_image(ref_sub_pil).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
    ref_style = resize_image(ref_sty_pil).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

    # -------------------------------------------------------------------------
    # STAGE C ON GPU
    # -------------------------------------------------------------------------
    move_stage_b_to_cpu(models_b)
    move_stage_c_to_gpu(models_rbm, controlnet, dino_model, device)
    hard_clear_cuda()

    batch_c = {
        "captions": [prompt] * batch_size,
        "style": ref_style,
        "images": ref_images,
    }

    x0_forward = models_rbm.effnet(extras.effnet_preprocess(ref_images))
    x0_style_forward = models_rbm.effnet(extras.effnet_preprocess(ref_style))
    random_latent = torch.randn_like(x0_forward, device=device)
    extras.sampling_configs["x_init"] = random_latent
    extras.sampling_configs["t_start"] = 1.0
    extras.sampling_configs["timesteps"] = 20

    with torch.no_grad():
        conditions = core_c.get_conditions(
            batch_c,
            models_rbm,
            extras,
            is_eval=True,
            is_unconditional=False,
            eval_image_embeds=True,
            eval_subject_style=True,
            eval_csd=False,
        )
        unconditions = core_c.get_conditions(
            batch_c,
            models_rbm,
            extras,
            is_eval=True,
            is_unconditional=True,
            eval_image_embeds=False,
            eval_subject_style=True,
        )

        if controlnet is not None and canny_filter is not None:
            noisy_canny = canny_filter(ref_images).to(device).to(getattr(torch, core_c.config.dtype))
            cnet_input = get_clean_canny(ref_sub_pil, noisy_canny, use_semantic_gating=use_semantic_gating)
            cnet = controlnet(cnet_input)
            cnet = [p * cnet_strength if p is not None else None for p in cnet]
            conditions["controlnet"] = cnet
            unconditions["controlnet"] = cnet

    # Offload encoders after conditions are ready
    move_stage_c_condition_models_to_cpu(models_rbm, controlnet)
    hard_clear_cuda()

    x0_forward = x0_forward.to(device)
    x0_style_forward = x0_style_forward.to(device)
    models_rbm.previewer.to(device)

    sampling_c = extras.gdf.sample(
        models_rbm.generator,
        conditions,
        stage_c_latent_shape,
        unconditions,
        device=device,
        device_2=device,
        **extras.sampling_configs,
        x0_style_forward=x0_style_forward,
        x0_forward=x0_forward,
        apply_pushforward=(tau_pushforward > 0),
        tau_pushforward=tau_pushforward,
        tau_pushforward_csd=10,
        num_iter=3,
        eta=0.2,
        tau=20,
        eval_sub_csd=True,
        guidance_mode=os.environ.get("GUIDANCE_MODE", "dino"),
        extras=extras,
        models=models_rbm,
        use_attn_mask=False,
        save_attn_mask=False,
        lam_content=0.0,
        lam_style=1.0,
        gamma_nc=0.0,
        gamma_ns=0.0,
        sam_mask=None,
        use_sam_mask=False,
        sam_prompt="",
        Lambda=1.0,
    )

    list_of_steps = []
    sampled_c = None
    for (step_latent, _, _) in tqdm(sampling_c, total=extras.sampling_configs["timesteps"], desc="Stage C"):
        sampled_c = step_latent
        if preview_dir is not None:
            list_of_steps.append(step_latent.detach().clone())

    # Save previews if requested
    if preview_dir is not None and len(list_of_steps) > 0:
        os.makedirs(preview_dir, exist_ok=True)
        for step_idx, step_tensor in enumerate(list_of_steps):
            with torch.no_grad():
                res = models_rbm.previewer(step_tensor).to(dtype=torch.float32)
            step_path = os.path.join(preview_dir, f"step_{step_idx:02d}.png")
            save_images(res, step_path)
        print(f"[*] Saved {len(list_of_steps)} timestep previews to {preview_dir}")

    # Transfer latent to CPU before Stage C offload
    sampled_c_cpu = sampled_c.detach().to("cpu")
    del sampled_c, conditions, unconditions, x0_forward, x0_style_forward, list_of_steps
    move_stage_c_to_cpu(models_rbm, controlnet, dino_model)
    hard_clear_cuda()

    # -------------------------------------------------------------------------
    # STAGE B ON GPU
    # -------------------------------------------------------------------------
    move_stage_b_to_gpu(models_b, device)
    hard_clear_cuda()

    batch_b = {
        "captions": [prompt] * batch_size,
        "style": ref_style.to(device),
        "images": ref_images.to(device),
    }

    with torch.no_grad():
        conditions_b = core_b.get_conditions(batch_b, models_b, extras_b, is_eval=True, is_unconditional=False)
        unconditions_b = core_b.get_conditions(batch_b, models_b, extras_b, is_eval=True, is_unconditional=True)

    sampled_c_gpu = sampled_c_cpu.to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        conditions_b["effnet"] = sampled_c_gpu
        unconditions_b["effnet"] = torch.zeros_like(sampled_c_gpu, device=device)

        sampling_b = extras_b.gdf.sample(
            models_b.generator,
            conditions_b,
            stage_b_latent_shape,
            unconditions_b,
            device=device,
            **extras_b.sampling_configs,
        )

        sampled_b = None
        for (step_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs["timesteps"], desc="Stage B"):
            sampled_b = step_b

        sampled_rgb = models_b.stage_a.decode(sampled_b).float().cpu()

    # Save Output atomically
    atomic_save_image(sampled_rgb, save_path)
    print(f"[+] Saved output: {save_path}")

    # Save 3-panel grid if requested atomically
    if save_grid_path is not None:
        grid_tensor = torch.cat(
            [
                F.interpolate(ref_images.cpu(), size=height),
                F.interpolate(ref_style.cpu(), size=height),
                sampled_rgb,
            ],
            dim=0,
        )
        atomic_save_image(grid_tensor, save_grid_path)
        print(f"[+] Saved grid: {save_grid_path}")

    # Clean Stage B
    del conditions_b, unconditions_b, sampled_c_gpu, sampled_b, sampled_rgb
    del ref_images, ref_style, batch_b
    move_stage_b_to_cpu(models_b)
    hard_clear_cuda()


# -----------------------------------------------------------------------------
# Main Loop & Arguments
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OrthoStyle Benchmark & Ablation Batch Runner (1-GPU Offload)")
    default_config = "configs/benchmark_config.json" if os.path.exists("configs/benchmark_config.json") else "data/benchmark_config.json"
    parser.add_argument("--config_path", type=str, default=default_config, help="Path to benchmark JSON configuration")
    parser.add_argument("--data_root", type=str, default="data", help="Root data folder containing content/ and style/")
    default_tp = os.environ.get("THIRD_PARTY_ROOT", "third_party")
    parser.add_argument("--third_party_root", type=str, default=default_tp, help="Root folder of third_party dependencies")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device ID or string (e.g. 0, 1, cuda:0, cuda:1)")
    parser.add_argument("--output_root", type=str, default="output/benchmark")
    parser.add_argument("--preview_root", type=str, default="output/previews")
    parser.add_argument("--prompt_levels", nargs="+", default=["null", "object", "style_desc"],
                        choices=["null", "object", "style_desc"])
    parser.add_argument("--start_idx", type=int, default=1, help="1-indexed pair index start (inclusive)")
    parser.add_argument("--end_idx", type=int, default=225, help="1-indexed pair index end (inclusive)")
    parser.add_argument("--pair_indices", nargs="+", type=int, default=None, help="Explicit pair indices to run")
    parser.add_argument("--save_previews", action="store_true", help="Force save timestep previews")
    parser.add_argument("--auto_previews_5x5", action="store_true", default=False,
                        help="Automatically save previews for the first 5 contents x 5 styles")
    parser.add_argument("--save_grids", action="store_true", default=True, help="Also save 3-panel [Content|Style|Output] grids")
    parser.add_argument("--overwrite", action="store_true", help="Force recompute and overwrite existing completed outputs")

    # Ablation arguments
    parser.add_argument("--alpha_style", type=float, default=0.85, help="Alpha style blending (0.0=pure mean, 1.0=raw style)")
    parser.add_argument("--tau_pushforward", type=int, default=2,
                        help="Number of initial steps to apply AdaIN pushforward (default: 2)")
    parser.add_argument("--p_switch", type=float, default=None,
                        help="Switch point for style injection sigmoid schedule (default: tau_pushforward / 20.0)")
    parser.add_argument("--no_ortho", action="store_true", help="Disable Score-Orthogonal Guidance (use raw grad)")
    parser.add_argument("--no_pushforward", action="store_true", help="Disable AdaIN Pushforward")
    parser.add_argument("--no_canny", action="store_true", help="Disable ControlNet Canny")
    parser.add_argument("--no_semantic_gating", action="store_true", help="Disable rembg semantic gating on Canny map")
    parser.add_argument("--subset_7x7", action="store_true", help="Filter to first 7 contents x 7 styles (49 pairs) for ablation study")
    parser.add_argument("--ablation_tag", type=str, default="", help="Subfolder name for ablation outputs if specified")

    args = parser.parse_args()

    # Harmonize tau_pushforward and p_switch
    if args.no_pushforward:
        effective_tau = 0
        effective_p_switch = 0.0
    else:
        effective_tau = args.tau_pushforward
        effective_p_switch = (effective_tau / 20.0) if args.p_switch is None else args.p_switch

    # Set ablation environment variables
    os.environ["ALPHA_STYLE"] = str(args.alpha_style)
    os.environ["P_SWITCH"] = str(effective_p_switch)
    os.environ["USE_ORTHO_GUIDANCE"] = "0" if args.no_ortho else "1"

    # Read benchmark config
    with open(args.config_path, "r", encoding="utf-8") as f:
        benchmark_cfg = json.load(f)

    pairs = benchmark_cfg["pairs"]
    print(f"[*] Loaded benchmark config from {args.config_path} with {len(pairs)} pairs.")

    # Filter target pairs
    if args.pair_indices is not None:
        target_pairs = [p for p in pairs if p["pair_idx"] in args.pair_indices]
    else:
        target_pairs = [p for p in pairs if args.start_idx <= p["pair_idx"] <= args.end_idx]

    if args.subset_7x7:
        target_pairs = [p for p in target_pairs if p["content_idx"] < 7 and p["style_idx"] < 7]
        print(f"[*] Subset 7x7 applied: {len(target_pairs)} pairs selected.")

    print(f"[*] Total pairs to evaluate: {len(target_pairs)} (Range: {args.start_idx} to {args.end_idx})")
    print(f"[*] Target device: {args.device}")
    print(f"[*] Prompt levels to run: {args.prompt_levels}")
    print(f"[*] Ablation settings: alpha_style={args.alpha_style}, tau_pushforward={effective_tau}, p_switch={effective_p_switch:.3f}, ortho={not args.no_ortho}, pushforward={not args.no_pushforward}, canny={not args.no_canny}, semantic_gating={not args.no_semantic_gating}")

    # Initialize models
    state = setup_all_models(
        device_str=args.device,
        use_controlnet_canny=(not args.no_canny),
        third_party_root=args.third_party_root
    )

    level_map = {
        "null": ("level1_null", "level1_null_prompt"),
        "object": ("level2_object", "level2_object_prompt"),
        "style_desc": ("level3_style_desc", "level3_style_description_prompt"),
    }

    base_out_dir = os.path.join(args.output_root, args.ablation_tag) if args.ablation_tag else args.output_root

    total_runs = len(target_pairs) * len(args.prompt_levels)
    completed_runs = 0

    for item in target_pairs:
        pair_idx = item["pair_idx"]
        content_idx = item["content_idx"]
        style_idx = item["style_idx"]
        content_file = item["content_file"]
        style_file = item["style_file"]

        content_stem = Path(content_file).stem
        style_stem = Path(style_file).stem

        content_path = os.path.join(args.data_root, "content", content_file)
        style_path = os.path.join(args.data_root, "style", style_file)

        # Verify input assets exist
        if not os.path.exists(content_path):
            raise FileNotFoundError(f"Content file missing: {content_path}. Please check --data_root.")
        if not os.path.exists(style_path):
            raise FileNotFoundError(f"Style file missing: {style_path}. Please check --data_root.")

        # First 5 content x 5 style check
        is_first_5x5 = (content_idx < 5 and style_idx < 5)

        for level_key in args.prompt_levels:
            completed_runs += 1
            folder_name, prompt_field = level_map[level_key]
            prompt = item[prompt_field]

            filename = f"ortho_{content_stem}_{style_stem}.png"
            out_path = os.path.join(base_out_dir, folder_name, filename)
            grid_path = os.path.join(base_out_dir, f"{folder_name}_grid", filename) if args.save_grids else None

            preview_dir = None
            if (args.save_previews or (args.auto_previews_5x5 and is_first_5x5)) and level_key == args.prompt_levels[0]:
                preview_dir = os.path.join(args.preview_root, f"ortho_{content_stem}_{style_stem}")

            # Safe Resume / Skip Check: Verifies file exists AND is an uncorrupted valid image
            already_done = is_valid_image(out_path)
            if args.save_grids and grid_path is not None:
                already_done = already_done and is_valid_image(grid_path)

            if not args.overwrite and already_done:
                print(f"[{completed_runs}/{total_runs}] [SKIP ALREADY COMPLETED & VALID] Pair #{pair_idx:03d} | Level: {level_key.upper()} ({filename})")
                continue

            print(f"\n[{completed_runs}/{total_runs}] Pair #{pair_idx:03d} | Level: {level_key.upper()} | Content: {content_stem} | Style: {style_stem}")
            print(f"    Prompt: '{prompt}'")
            print(f"    Output: {out_path}")

            run_single_inference(
                state=state,
                content_path=content_path,
                style_path=style_path,
                prompt=prompt,
                save_path=out_path,
                save_grid_path=grid_path,
                preview_dir=preview_dir,
                tau_pushforward=effective_tau,
                cnet_strength=0.8,
                use_semantic_gating=(not args.no_semantic_gating),
            )

    print("\n[✔] Benchmark batch execution finished successfully!")


if __name__ == "__main__":
    main()

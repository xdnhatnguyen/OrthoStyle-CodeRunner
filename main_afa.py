import os

# Optional cache dirs from environment if specified
for env_var in ["HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME"]:
    if env_var in os.environ and not os.path.exists(os.environ[env_var]):
        os.makedirs(os.environ[env_var], exist_ok=True)

import sys
sys.path.append("third_party/")
sys.path.append("third_party/StableCascade/")

import copy
from io import BytesIO

import yaml
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import PIL.Image
from tqdm import tqdm
from accelerate.utils import set_module_tensor_to_device
from IPython.display import display, Image

# from lang_sam import LangSAM

from inference.utils import *
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import rembg
import numpy as np
from core.utils import load_or_fail
from train import WurstCoreB
from gdf_rbm import RBM
from stage_c_rbm import StageCRBM
from utils import WurstCoreCRBM
from modules.controlnet import ControlNet, CannyFilter

from gdf.schedulers import CosineSchedule
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, AdaptiveLossWeight
from gdf.targets import EpsilonTarget


def get_clean_canny(
    rgb_image_pil: PIL.Image.Image,
    noisy_canny_tensor: torch.Tensor,
) -> torch.Tensor:
    """Lọc nhiễu background trên bản đồ Canny bằng Semantic Mask (rembg) xén ngang (hard threshold) không có blur."""
    # 1. Trích xuất Semantic Mask (rembg)
    mask_pil = rembg.remove(rgb_image_pil, only_mask=True)
    mask_np = np.array(mask_pil, dtype=np.float32) / 255.0  # shape: [H, W]
    mask_tensor = torch.from_numpy(mask_np)  # shape: [H, W]
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
    mask_tensor = mask_tensor.to(
        device=noisy_canny_tensor.device, dtype=noisy_canny_tensor.dtype
    )  # shape: [1, 1, H, W]

    # Đồng bộ phân giải không gian [H, W] với noisy_canny_tensor nếu có sai lệch
    if mask_tensor.shape[-2:] != noisy_canny_tensor.shape[-2:]:
        mask_tensor = F.interpolate(
            mask_tensor,
            size=noisy_canny_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # shape: [1, 1, H, W]

    # 2. Dilation nhẹ để giữ trọn vẹn đường bao quanh chủ thể
    dilated_mask = F.max_pool2d(
        mask_tensor, kernel_size=5, stride=1, padding=2
    )  # shape: [1, 1, H, W]

    # 3. Hard Thresholding (Xén ngang nhị phân, loại bỏ hoàn toàn Gaussian Blur)
    hard_mask = (dilated_mask > 0.5).to(
        dtype=noisy_canny_tensor.dtype
    )  # shape: [1, 1, H, W]

    # 4. Semantic Gating (Nhân trực tiếp mask nhị phân để lọc nền)
    clean_canny_tensor = (
        noisy_canny_tensor * hard_mask
    )  # shape: [B, 1, H, W]

    return clean_canny_tensor


# transform = T.ToPILImage()
low_vram = False


def module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return "no-params"


def print_module_devices(models_rbm, models_b):
    print("=== Device check ===")
    if getattr(models_rbm, "generator", None) is not None:
        print("models_rbm.generator:", module_device(models_rbm.generator))
    if getattr(models_rbm, "effnet", None) is not None:
        print("models_rbm.effnet:", module_device(models_rbm.effnet))
    if getattr(models_rbm, "text_model", None) is not None:
        print("models_rbm.text_model:", module_device(models_rbm.text_model))
    if getattr(models_rbm, "image_model", None) is not None:
        print("models_rbm.image_model:", module_device(models_rbm.image_model))
    if getattr(models_rbm, "previewer", None) is not None:
        print("models_rbm.previewer:", module_device(models_rbm.previewer))

    if getattr(models_b, "generator", None) is not None:
        print("models_b.generator:", module_device(models_b.generator))
    if getattr(models_b, "stage_a", None) is not None:
        print("models_b.stage_a:", module_device(models_b.stage_a))
    if getattr(models_b, "text_model", None) is not None:
        print("models_b.text_model:", module_device(models_b.text_model))
    print("====================")


device_c = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_b = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device_c)

print("Stage C device:", device_c)
print("Stage B device:", device_b)

# Stage C config
config_file = "third_party/StableCascade/configs/inference/stage_c_3b.yaml"
with open(config_file, "r", encoding="utf-8") as file:
    loaded_config = yaml.safe_load(file)

core = WurstCoreCRBM(config_dict=loaded_config, device=device_c, training=False)

# Stage B config
config_file_b = "third_party/StableCascade/configs/inference/stage_b_3b.yaml"
with open(config_file_b, "r", encoding="utf-8") as file:
    loaded_config_b = yaml.safe_load(file)

core_b = WurstCoreB(config_dict=loaded_config_b, device=device_b, training=False)

extras = core.setup_extras_pre()

gdf_rbm = RBM(
    schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
    input_scaler=VPScaler(),
    target=EpsilonTarget(),
    noise_cond=CosineTNoiseCond(),
    loss_weight=AdaptiveLossWeight(),
)
sampling_configs = {
    "cfg": 5,
    "sampler": DDPMSampler(gdf_rbm),
    "shift": 1,
    "timesteps": 20,
}

extras = core.Extras(
    gdf=gdf_rbm,
    sampling_configs=sampling_configs,
    transforms=extras.transforms,
    effnet_preprocess=extras.effnet_preprocess,
    clip_preprocess=extras.clip_preprocess,
)

models = core.setup_models(extras)
models.generator.eval().requires_grad_(False)

extras_b = core_b.setup_extras_pre()
models_b = core_b.setup_models(extras_b, skip_clip=True)
models_b = WurstCoreB.Models(
    **{
        **models_b.to_dict(),
        "tokenizer": models.tokenizer,
        "text_model": copy.deepcopy(models.text_model),
    }
)
models_b.generator.eval().requires_grad_(False)

generator_rbm = StageCRBM()
for param_name, param in load_or_fail(core.config.generator_checkpoint_path).items():
    set_module_tensor_to_device(generator_rbm, param_name, "cpu", value=param)

generator_rbm = generator_rbm.to(getattr(torch, core.config.dtype)).to(device_c)
generator_rbm = core.load_model(generator_rbm, "generator")

models_rbm = core.Models(
    effnet=models.effnet,
    previewer=models.previewer,
    generator=generator_rbm,
    generator_ema=models.generator_ema,
    tokenizer=models.tokenizer,
    text_model=models.text_model,
    image_model=models.image_model,
)
models_rbm.generator.eval().requires_grad_(False)

# Put modules on intended GPUs
# Stage C side on gpu0
models_rbm.generator.to(device_c)
models_rbm.effnet.to(device_c)
if getattr(models_rbm, "text_model", None) is not None:
    models_rbm.text_model.to(device_c)
if getattr(models_rbm, "image_model", None) is not None:
    models_rbm.image_model.to(device_c)

# ControlNet Canny toggle flag
use_controlnet_canny = True

controlnet = None
canny_filter = None
if use_controlnet_canny:
    controlnet = ControlNet(
        c_in=1,
        proj_blocks=[0, 4, 8, 12, 51, 55, 59, 63],
        bottleneck_mode=None
    )
    canny_checkpoint_path = "third_party/StableCascade/models/canny.safetensors"
    cnet_checkpoint = load_or_fail(canny_checkpoint_path)
    controlnet.load_state_dict(cnet_checkpoint if 'state_dict' not in cnet_checkpoint else cnet_checkpoint['state_dict'])
    controlnet = controlnet.to(getattr(torch, core.config.dtype)).to(device_c).eval().requires_grad_(False)
    canny_filter = CannyFilter(device_c, resize=224)

# Previewer on gpu0 for DINO guidance and preview
if getattr(models_rbm, "previewer", None) is not None:
    models_rbm.previewer.to(device_c)

# Stage B side on gpu1
models_b.generator.to(device_b, dtype=torch.bfloat16)
models_b.stage_a.to(device_b)
if getattr(models_b, "text_model", None) is not None:
    models_b.text_model.to(device_b)

print_module_devices(models_rbm, models_b)

def resolve_data_path(file_path: str, default_dir: str) -> str:
    if os.path.exists(file_path):
        return file_path
    candidate = os.path.join(default_dir, file_path)
    if os.path.exists(candidate):
        return candidate
    base = os.path.splitext(os.path.basename(file_path))[0].lower().replace("pencel", "pencil")
    if os.path.exists(default_dir):
        for f in sorted(os.listdir(default_dir)):
            if base in f.lower() or f.lower().endswith(file_path.lower()) or (len(base) > 4 and any(part in f.lower() for part in base.split("_") if len(part) >= 4)):
                return os.path.join(default_dir, f)
    return file_path


# Inputs
ref_style_file = resolve_data_path(os.environ.get("REF_STYLE_FILE", "08_pencil_sketch.png"), "data/style")
ref_sub_file = resolve_data_path(os.environ.get("REF_SUB_FILE", "01_backpack_dog.png"), "data/content")
save_path = os.environ.get("SAVE_PATH", "output/figs/exp_full.png")
caption = os.environ.get("PROMPT", os.environ.get("CAPTION", ""))
org_caption = os.environ.get("ORG_CAPTION", caption)
sam_prompt = os.environ.get("SAM_PROMPT", caption)
use_sam_mask = False

print(f"[Inputs] Content: {ref_sub_file} | Style: {ref_style_file} | Prompt: '{caption}' | Save: {save_path}")

batch_size = 1
height, width = 1024, 1024
ETA = 0.45
LAMBDA = 1  
ANALYZE_STEPS = True

stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(
    height, width, batch_size=batch_size
)

n_inversion_steps = 20

extras.sampling_configs["cfg"] = 4
extras.sampling_configs["shift"] = 2
extras.sampling_configs["timesteps"] = 20
extras.sampling_configs["t_start"] = 1.0

extras_b.sampling_configs["cfg"] = 1.1
extras_b.sampling_configs["shift"] = 1
extras_b.sampling_configs["timesteps"] = 10
extras_b.sampling_configs["t_start"] = 1.0

# Stage C tensors on gpu0
ref_style = resize_image(PIL.Image.open(ref_style_file).convert("RGB")) \
    .unsqueeze(0).expand(batch_size, -1, -1, -1).to(device_c)

ref_images = resize_image(PIL.Image.open(ref_sub_file).convert("RGB")) \
    .unsqueeze(0).expand(batch_size, -1, -1, -1).to(device_c)

# Separate batches for each stage/device
org_batch_c = {
    "captions": [org_caption] * batch_size,
    "style": ref_style,
    "images": ref_images,
}

org_batch_b = {
    "captions": [org_caption] * batch_size,
    "style": ref_style.to(device_b),
    "images": ref_images.to(device_b),
}

batch_c = {
    "captions": [caption] * batch_size,
    "style": ref_style,
    "images": ref_images,
}

batch_b = {
    "captions": [caption] * batch_size,
    "style": ref_style.to(device_b),
    "images": ref_images.to(device_b),
}

# Encode on gpu0
x0_forward = models_rbm.effnet(extras.effnet_preprocess(ref_images))
x0_style_forward = models_rbm.effnet(extras.effnet_preprocess(ref_style))
# Khởi tạo x_init bằng Gaussian Noise ngẫu nhiên thay vì inverted_latent
random_latent = torch.randn_like(x0_forward, device=device_c)  # shape: [B, 16, 24, 24]
extras.sampling_configs["x_init"] = random_latent
extras.sampling_configs["t_start"] = 1.0
extras.sampling_configs["timesteps"] = 20
# SAM preview on gpu1
sam_mask = None
if False and use_sam_mask and sam_prompt:
    previewer_sam = copy.deepcopy(models_rbm.previewer).to(device_b)
    x0_preview = previewer_sam(x0_forward.to(device_b))

    from lang_sam import LangSAM
    sam_model = LangSAM()
    if hasattr(sam_model, "to"):
        sam_model.to(device_b)
        
    transform = T.ToPILImage()
    x0_preview_img = x0_preview[0].detach().float().clamp(0, 1).cpu()
    image_pil = transform(x0_preview_img)

    results = sam_model.predict([image_pil], [sam_prompt])

    sam_mask = torch.from_numpy(results[0]["masks"][0]).unsqueeze(0).unsqueeze(0).float().to(device_c)

# Conditions
with torch.no_grad():
    org_conditions = core.get_conditions(
        org_batch_c,
        models_rbm,
        extras,
        is_eval=True,
        is_unconditional=False,
        eval_image_embeds=False,
        eval_subject_style=False,
        eval_csd=False,
    )
    conditions = core.get_conditions(
        batch_c,
        models_rbm,
        extras,
        is_eval=True,
        is_unconditional=False,
        eval_image_embeds=True,
        eval_subject_style=True,
        eval_csd=False,
    )
    org_unconditions = core.get_conditions(
        org_batch_c,
        models_rbm,
        extras,
        is_eval=True,
        is_unconditional=True,
        eval_image_embeds=False,
        eval_subject_style=False,
        eval_csd=False,
    )
    unconditions = core.get_conditions(
        batch_c,
        models_rbm,
        extras,
        is_eval=True,
        is_unconditional=True,
        eval_image_embeds=False,
        eval_subject_style=True,
    )

    # Extract Canny Edge map from Subject/Content (cat.jpg) for ControlNet if enabled
    if use_controlnet_canny and controlnet is not None:
        noisy_canny = canny_filter(ref_images).to(device_c).to(getattr(torch, core.config.dtype))  # shape: [B, 1, H, W]
        ref_sub_pil = PIL.Image.open(ref_sub_file).convert("RGB")
        cnet_input = get_clean_canny(ref_sub_pil, noisy_canny)  # shape: [B, 1, H, W]
        cnet = controlnet(cnet_input)
        
        # Scale ControlNet conditioning strength to 0.8
        controlnet_strength = float(os.environ.get("CNET_STRENGTH", "0.8"))
        cnet = [p * controlnet_strength if p is not None else None for p in cnet]
        
        torchvision.utils.save_image(cnet_input.float(), "results/canny_edge_map.png")
        print(f"Extracted Clean Canny edges (Semantic Gating via rembg, strength={controlnet_strength}) and saved to results/canny_edge_map.png")
        conditions["controlnet"] = cnet
        unconditions["controlnet"] = cnet
    else:
        print("[ControlNet Canny] Disabled (flag=False)")
    conditions_b = core_b.get_conditions(
        batch_b,
        models_b,
        extras_b,
        is_eval=True,
        is_unconditional=False,
    )
    unconditions_b = core_b.get_conditions(
        batch_b,
        models_b,
        extras_b,
        is_eval=True,
        is_unconditional=True,
    )
    org_conditions_b = core_b.get_conditions(
        org_batch_b,
        models_b,
        extras_b,
        is_eval=True,
        is_unconditional=False,
    )
    org_unconditions_b = core_b.get_conditions(
        org_batch_b,
        models_b,
        extras_b,
        is_eval=True,
        is_unconditional=True,
    )

# Optional low_vram branch if your project defines models_to elsewhere
if low_vram and "models_to" in globals():
    models_to(models_rbm, device="cpu", excepts=["generator"])

# Stage C reverse process on gpu0
models_rbm.previewer.to(device_c)
x0_forward = x0_forward.to(device_c)

# These are no longer needed on gpu0 after conditions/x0_* are computed
if getattr(models_rbm, "effnet", None) is not None:
    models_rbm.effnet.to("cpu")
if getattr(models_rbm, "text_model", None) is not None:
    models_rbm.text_model.to("cpu")
if getattr(models_rbm, "image_model", None) is not None:
    models_rbm.image_model.to("cpu")

torch.cuda.empty_cache()

# inverting...
if False:
    inversion = extras.gdf.StableCascade_inversion(
      x0_forward, 
      models_rbm.generator,
      org_conditions, 
      stage_c_latent_shape, 
      org_unconditions,
      device=device_c,
      **extras.sampling_configs,
      inversion_timesteps=n_inversion_steps
    )

    for (inverted_latent, _, _) in tqdm(inversion, total=extras.sampling_configs["timesteps"], desc="Inverting"):
        pass

    # resampling for inversion check
    orig_sampling_c = extras.gdf.orig_sample(
        models_rbm.generator,
        org_conditions, 
        stage_c_latent_shape, 
        org_unconditions,
        device=device_c,
        **extras.sampling_configs,
        x_init=inverted_latent,
    )

    for (resampled_latent, _, _) in tqdm(orig_sampling_c, total=extras.sampling_configs["timesteps"], desc="Resampling after inversion"):
        pass

sampling_c = extras.gdf.sample(
    models_rbm.generator,
    conditions,
    stage_c_latent_shape,
    unconditions,
    device=device_c,
    device_2=device_b,
    **extras.sampling_configs,
    x0_style_forward=x0_style_forward,
    x0_forward=x0_forward,
    apply_pushforward=True,
    tau_pushforward=5,
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
    sam_mask=sam_mask,
    use_sam_mask=False,
    sam_prompt=sam_prompt,
    Lambda=LAMBDA
)

list_of_steps = []

for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs["timesteps"], desc="Sampling"):
    if ANALYZE_STEPS == True:
        list_of_steps.append(sampled_c)

# Free some gpu0 memory after Stage C
del conditions, unconditions, x0_forward, x0_style_forward
torch.cuda.empty_cache()

# Move Stage C latent to gpu1 for Stage B
sampled_c = sampled_c.to(device_b, non_blocking=True)

# Stage B reverse process on gpu1
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    conditions_b["effnet"] = sampled_c
    unconditions_b["effnet"] = torch.zeros_like(sampled_c, device=device_b)

    sampling_b = extras_b.gdf.sample(
        models_b.generator,
        conditions_b,
        stage_b_latent_shape,
        unconditions_b,
        device=device_b,
        **extras_b.sampling_configs,
    )

    for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs["timesteps"]):
        pass

    sampled = models_b.stage_a.decode(sampled_b).float()

    if False:
        sampling_b = extras_b.gdf.sample(
            models_b.generator,
            org_conditions_b,
            stage_b_latent_shape,
            org_unconditions_b,
            device=device_b,
            **extras_b.sampling_configs,
        )

        conditions_b["effnet"] = inverted_latent
        unconditions_b["effnet"] = torch.zeros_like(inverted_latent, device=device_b)

        for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs["timesteps"]):
            pass

        inverted_img = models_b.stage_a.decode(sampled_b).float()

        conditions_b["effnet"] = resampled_latent
        unconditions_b["effnet"] = torch.zeros_like(resampled_latent, device=device_b)

        for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs["timesteps"]):
            pass
        
        resampled_img = models_b.stage_a.decode(sampled_b).float()

    models_rbm.previewer.to(device=device_c)

    for i, tensor in enumerate(list_of_steps):
        print("Saving preview of step", i)
        res = models_rbm.previewer(tensor).to(dtype=torch.float32)
        save_images(res, "./our_results/sampling_" + str(i) + ".png".strip())

# Save output
sampled = torch.cat(
    [
        torch.nn.functional.interpolate(ref_images.cpu(), size=height),
        torch.nn.functional.interpolate(ref_style.cpu(), size=height),
        sampled.cpu(),
    ],
    dim=0,
)

if os.path.dirname(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
save_images(sampled, save_path)
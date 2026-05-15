import os

# Set cache dirs BEFORE importing libraries that may download models
os.environ["HF_HUB_CACHE"] = "/mnt/wav2vec2/khoan/hf_cache/hub"
os.environ["HF_HOME"] = "/mnt/wav2vec2/khoan/hf_cache/home"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/wav2vec2/khoan/hf_cache/hub"
os.environ["TORCH_HOME"] = "/mnt/wav2vec2/khoan/torch_cache"
os.environ["TMPDIR"] = "/mnt/wav2vec2/khoan/tmp"

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

from lang_sam import LangSAM

from inference.utils import *
from core.utils import load_or_fail
from train import WurstCoreB
from gdf_rbm import RBM
from stage_c_rbm import StageCRBM
from utils import WurstCoreCRBM

from gdf.schedulers import CosineSchedule
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, AdaptiveLossWeight
from gdf.targets import EpsilonTarget


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

# Previewer + SAM-related side on gpu1 to reduce gpu0 pressure
if getattr(models_rbm, "previewer", None) is not None:
    models_rbm.previewer.to(device_b)

# Stage B side on gpu1
models_b.generator.to(device_b, dtype=torch.bfloat16)
models_b.stage_a.to(device_b)
if getattr(models_b, "text_model", None) is not None:
    models_b.text_model.to(device_b)

print_module_devices(models_rbm, models_b)

# Inputs
ref_style_file = "data/mosaic.png"
ref_sub_file = "data/cat.jpg"
caption = "a cat in 3d rendering"
sam_prompt = "a cat"
use_sam_mask = False

batch_size = 1
height, width = 1024, 1024
stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(
    height, width, batch_size=batch_size
)

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

# SAM preview on gpu1
# x0_preview = models_rbm.previewer(x0_forward.to(device_b))
previewer_sam = copy.deepcopy(models_rbm.previewer).to(device_b)
x0_preview = previewer_sam(x0_forward.to(device_b))

sam_model = LangSAM()
if hasattr(sam_model, "to"):
    sam_model.to(device_b)
    
# x0_preview_img = x0_preview[0].detach().float().clamp(0, 1).cpu()
transform = T.ToPILImage()
# sam_mask, boxes, phrases, logits = sam_model.predict(transform(x0_preview_img), sam_prompt)
x0_preview_img = x0_preview[0].detach().float().clamp(0, 1).cpu()
image_pil = transform(x0_preview_img)

results = sam_model.predict([image_pil], [sam_prompt])

sam_mask = torch.from_numpy(results[0]["masks"][0]).unsqueeze(0).unsqueeze(0).float().to(device_c)

# Conditions
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
unconditions = core.get_conditions(
    batch_c,
    models_rbm,
    extras,
    is_eval=True,
    is_unconditional=True,
    eval_image_embeds=False,
    eval_subject_style=True,
)

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

# Optional low_vram branch if your project defines models_to elsewhere
if low_vram and "models_to" in globals():
    models_to(models_rbm, device="cpu", excepts=["generator"])
    if hasattr(sam_model, "sam") and "models_to" in globals():
        models_to(sam_model.sam, device="cpu")

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

sampling_c = extras.gdf.sample(
    models_rbm.generator,
    conditions,
    stage_c_latent_shape,
    unconditions,
    device=device_c,
    **extras.sampling_configs,
    x0_style_forward=x0_style_forward,
    x0_forward=x0_forward,
    apply_pushforward=False,
    tau_pushforward=5,
    tau_pushforward_csd=10,
    num_iter=3,
    eta=1e-1,
    tau=20,
    eval_sub_csd=True,
    extras=extras,
    models=models_rbm,
    use_attn_mask=use_sam_mask,
    save_attn_mask=False,
    lam_content=1,
    lam_style=1,
    sam_mask=sam_mask,
    use_sam_mask=use_sam_mask,
    sam_prompt=sam_prompt,
)

for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs["timesteps"]):
    pass

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

# Save output
sampled = torch.cat(
    [
        torch.nn.functional.interpolate(ref_images.cpu(), size=height),
        torch.nn.functional.interpolate(ref_style.cpu(), size=height),
        sampled.cpu(),
    ],
    dim=0,
)

save_images(sampled, "./out_images/combined_afa_ours_no_style_prompt.png")
import argparse
import csv
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



def list_images(image_dir: str | Path) -> List[Path]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Folder does not exist: {image_dir}")
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def stem_to_path(image_dir: str | Path) -> Dict[str, Path]:
    return {p.stem: p for p in list_images(image_dir)}


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def to_grayscale_rgb(img: Image.Image) -> Image.Image:
    """Convert image to grayscale but keep 3 RGB channels.

    Useful for content/leakage metrics where we want to reduce the influence
    of color/style and focus more on structure/semantic content.
    """
    return ImageOps.grayscale(img).convert("RGB")


def load_image_for_generated(
    image_path: str | Path,
    crop_right_third: bool = False,
    fixed_512_triptych: bool = False,
) -> Image.Image:
    """Load generated image; optionally crop the right third.

    If fixed_512_triptych=True, crop [1024:1536], useful for 3 x 512 images.
    Otherwise, crop the last 1/3 of the image width.
    """
    img = load_rgb(image_path)
    w, h = img.size

    if crop_right_third:
        if fixed_512_triptych:
            left, right = 1024, 1536
            if w < right:
                raise ValueError(
                    f"Image {image_path} has width={w}, cannot crop [1024:1536]."
                )
            img = img.crop((left, 0, right, h))
        else:
            left = 2 * w // 3
            img = img.crop((left, 0, w, h))

    return img


def resolve_generated_path(
    generated_dir: str | Path,
    prefix: str,
    content_name: str,
    style_name: str,
    generated_pattern: str,
    allow_glob_fallback: bool = True,
) -> Optional[Path]:
    """Resolve generated image path for one content-style pair.

    Default pattern: {prefix}_{content}_{style}.png
    You can pass any pattern using placeholders: {prefix}, {content}, {style}
    """
    generated_dir = Path(generated_dir)
    rel_name = generated_pattern.format(
        prefix=prefix,
        content=content_name,
        style=style_name,
    )
    candidate = generated_dir / rel_name
    if candidate.exists():
        return candidate

    # If pattern extension differs, try all common image extensions with same stem.
    candidate_stem = candidate.with_suffix("")
    for ext in IMAGE_EXTS:
        p = candidate_stem.with_suffix(ext)
        if p.exists():
            return p

    # Fallback: prefix_content_style*; useful when generated names include suffixes.
    if allow_glob_fallback:
        matches = []
        glob_prefix = f"{prefix}_{content_name}_{style_name}"
        for p in generated_dir.glob(glob_prefix + "*"):
            if p.suffix.lower() in IMAGE_EXTS:
                matches.append(p)
        matches = sorted(matches)
        if len(matches) > 0:
            return matches[0]

    return None


# -----------------------------------------------------------------------------
# CSD style encoder
# -----------------------------------------------------------------------------


def build_csd_preprocess() -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
    return transforms.Compose(
        [
            transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


def setup_csd(
    device: str,
    csd_checkpoint: str = "third_party/CSD/checkpoint.pth",
    csd_third_party: str = "third_party",
) -> nn.Module:
    """Load CSD model.

    This follows your original code: CSD_CLIP('vit_large', 'default') and
    convert_state_dict(checkpoint['model_state_dict']).
    """
    csd_third_party = str(csd_third_party)
    if csd_third_party not in sys.path:
        sys.path.append(csd_third_party)

    try:
        from CSD.model import CSD_CLIP
        from CSD.utils import convert_state_dict
    except Exception as e:
        raise ImportError(
            "Cannot import CSD. Please make sure third_party/CSD exists, or pass "
            "--csd_third_party pointing to the folder that contains CSD/.\n"
            f"Original error: {e}"
        )

    csd_checkpoint = Path(csd_checkpoint)
    if not csd_checkpoint.exists():
        raise FileNotFoundError(
            f"CSD checkpoint not found: {csd_checkpoint}. Pass --csd_checkpoint."
        )

    model = CSD_CLIP("vit_large", "default")
    checkpoint = torch.load(csd_checkpoint, map_location=device, weights_only=False)
    state_dict = convert_state_dict(checkpoint["model_state_dict"])
    msg = model.load_state_dict(state_dict, strict=True)
    print(f"Loaded CSD checkpoint: {csd_checkpoint}")
    print(msg)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def csd_style_embedding_from_image(
    model: nn.Module,
    preprocess: transforms.Compose,
    image_path: str | Path,
    device: str,
    is_generated: bool = False,
    crop_generated_right_third: bool = False,
    fixed_512_triptych: bool = False,
) -> torch.Tensor:
    if is_generated:
        img = load_image_for_generated(
            image_path,
            crop_right_third=crop_generated_right_third,
            fixed_512_triptych=fixed_512_triptych,
        )
    else:
        img = load_rgb(image_path)

    x = preprocess(img).unsqueeze(0).to(device)
    _, _, style_output = model(x)
    style_output = F.normalize(style_output, dim=-1)
    return style_output.squeeze(0).detach().cpu()


# -----------------------------------------------------------------------------
# CLIP-I content encoder
# -----------------------------------------------------------------------------


class CLIPImageEncoder:
    def __init__(self, model_name: str, device: str):
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

        self.device = device
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_name,
        ).requires_grad_(False).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_pil(self, img: Image.Image) -> torch.Tensor:
      model_dtype = next(self.model.parameters()).dtype
      inputs = self.processor(
        images=img,
        return_tensors="pt",
      )

      pixel_values = inputs["pixel_values"].to(
        device=self.device,
        dtype=model_dtype,
      )

      outputs = self.model(
        pixel_values=pixel_values,
      )

      feat = outputs.image_embeds
      feat = F.normalize(feat, dim=-1)
      return feat.squeeze(0).detach().cpu()

    @torch.no_grad()
    def encode_path(
        self,
        image_path: str | Path,
        is_generated: bool = False,
        crop_generated_right_third: bool = False,
        fixed_512_triptych: bool = False,
    ) -> torch.Tensor:
        if is_generated:
            img = load_image_for_generated(
                image_path,
                crop_right_third=crop_generated_right_third,
                fixed_512_triptych=fixed_512_triptych,
            )
        else:
            img = load_rgb(image_path)
        return self.encode_pil(img)


# -----------------------------------------------------------------------------
# DINO content encoder
# -----------------------------------------------------------------------------


class DINOImageEncoder:
    def __init__(self, model_name: str, device: str):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_pil(self, img: Image.Image, grayscale: bool = False) -> torch.Tensor:
        if grayscale:
            img = to_grayscale_rgb(img)

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # Most DINO ViT models expose last_hidden_state: [B, tokens, dim].
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            feat = outputs.last_hidden_state[:, 0]  # CLS token
        else:
            raise RuntimeError("Cannot find DINO output feature.")

        feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0).detach().cpu()

    @torch.no_grad()
    def encode_path(
        self,
        image_path: str | Path,
        is_generated: bool = False,
        crop_generated_right_third: bool = False,
        fixed_512_triptych: bool = False,
        grayscale: bool = False,
    ) -> torch.Tensor:
        if is_generated:
            img = load_image_for_generated(
                image_path,
                crop_right_third=crop_generated_right_third,
                fixed_512_triptych=fixed_512_triptych,
            )
        else:
            img = load_rgb(image_path)
        return self.encode_pil(img, grayscale=grayscale)


# -----------------------------------------------------------------------------
# LPIPS
# -----------------------------------------------------------------------------


class LPIPSMetric:
    def __init__(self, device: str, net: str = "alex"):
        try:
            import lpips
        except Exception as e:
            raise ImportError(
                "lpips is not installed. Install it with: pip install lpips\n"
                f"Original error: {e}"
            )

        self.device = device
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.loss_fn.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @torch.no_grad()
    def distance(
        self,
        content_path: str | Path,
        generated_path: str | Path,
        crop_generated_right_third: bool = False,
        fixed_512_triptych: bool = False,
    ) -> float:
        content = load_rgb(content_path)
        generated = load_image_for_generated(
            generated_path,
            crop_right_third=crop_generated_right_third,
            fixed_512_triptych=fixed_512_triptych,
        )
        x = self.preprocess(content).unsqueeze(0).to(self.device)
        y = self.preprocess(generated).unsqueeze(0).to(self.device)
        return float(self.loss_fn(x, y).item())


# -----------------------------------------------------------------------------
# CFSD: Content Feature Structural Distance
# -----------------------------------------------------------------------------


class CFSDMetric:
    """Content Feature Structural Distance.

    We extract VGG19 conv3-level features, build patch-to-patch correlation maps,
    row-normalize them with softmax, then compute average KL divergence:

        CFSD = mean_i KL(S_content[i] || S_generated[i])

    Lower is better.
    """

    def __init__(
        self,
        device: str,
        image_size: int = 224,
        vgg_until_layer: int = 12,
        normalize_patch_features: bool = True,
        softmax_temperature: float = 1.0,
    ):
        self.device = device
        self.image_size = image_size
        self.normalize_patch_features = normalize_patch_features
        self.softmax_temperature = softmax_temperature

        weights = models.VGG19_Weights.IMAGENET1K_V1
        vgg = models.vgg19(weights=weights).features[:vgg_until_layer]
        self.vgg = vgg.to(device).eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @torch.no_grad()
    def _features(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.vgg(x)  # [1, C, H, W]
        feat = feat.squeeze(0).permute(1, 2, 0).reshape(-1, feat.shape[1])  # [HW, C]
        if self.normalize_patch_features:
            feat = F.normalize(feat, dim=-1)
        return feat

    @torch.no_grad()
    def _log_correlation_distribution(self, feat: torch.Tensor) -> torch.Tensor:
        sim = feat @ feat.t()  # [HW, HW]
        sim = sim / max(self.softmax_temperature, 1e-8)
        return F.log_softmax(sim, dim=-1)

    @torch.no_grad()
    def distance(
        self,
        content_path: str | Path,
        generated_path: str | Path,
        crop_generated_right_third: bool = False,
        fixed_512_triptych: bool = False,
    ) -> float:
        content = load_rgb(content_path)
        generated = load_image_for_generated(
            generated_path,
            crop_right_third=crop_generated_right_third,
            fixed_512_triptych=fixed_512_triptych,
        )

        feat_c = self._features(content)
        feat_g = self._features(generated)

        log_sc = self._log_correlation_distribution(feat_c)
        log_sg = self._log_correlation_distribution(feat_g)
        sc = log_sc.exp()

        # KL(S_c || S_g), averaged over patch rows.
        kl = (sc * (log_sc - log_sg)).sum(dim=-1).mean()
        return float(kl.item())




def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return float(torch.dot(a, b).item())

def directional_content_leakage(
    feat_content: torch.Tensor,
    feat_style: torch.Tensor,
    feat_generated: torch.Tensor,
    eps: float = 1e-8,
    clip: bool = True,
) -> float:
    """Directional Content Leakage (DCL). Lower is better.

    DCL measures whether the generated image moves from the content image toward
    the style image in a content/structure feature space.
    """
    fc = F.normalize(feat_content.float(), dim=-1)
    fs = F.normalize(feat_style.float(), dim=-1)
    fg = F.normalize(feat_generated.float(), dim=-1)

    d_style = fs - fc
    d_gen = fg - fc

    score = torch.dot(d_gen, d_style) / (d_style.pow(2).sum() + eps)
    if clip:
        score = score.clamp(0.0, 1.0)
    return float(score.item())


def safe_metric(fn, default=float("nan")):
    try:
        return fn()
    except Exception as e:
        print(f"[WARN] metric failed: {e}")
        return default


def add_average_row(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "csd_style_similarity",
        "clip_i_content_similarity",
        "dino_content_similarity",
        "lpips",
        "cfsd",
        "dcl",
    ]
    avg = {col: None for col in df.columns}
    avg["content_name"] = "AVERAGE"
    avg["style_name"] = "AVERAGE"
    avg["generated_path"] = ""
    avg["style_retrieval_correct"] = df["style_retrieval_correct"].mean() if len(df) else float("nan")
    avg["style_retrieval_accuracy"] = df["style_retrieval_correct"].mean() if len(df) else float("nan")
    for col in metric_cols:
        if col in df.columns:
            avg[col] = df[col].mean()
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def evaluate_all(args: argparse.Namespace) -> pd.DataFrame:
    device = args.device
    print(f"Using device: {device}")

    content_map = stem_to_path(args.content_dir)
    style_map = stem_to_path(args.style_dir)

    if args.content_names:
        content_names = args.content_names
    else:
        content_names = sorted(content_map.keys())

    if args.style_names:
        style_names = args.style_names
    else:
        style_names = sorted(style_map.keys())

    print(f"Found {len(content_names)} contents and {len(style_names)} styles.")

    # Load models.
    csd_model = setup_csd(device, args.csd_checkpoint, args.csd_third_party)
    csd_preprocess = build_csd_preprocess()

    clip_encoder = CLIPImageEncoder(args.clip_model, device)
    dino_encoder = DINOImageEncoder(args.dino_model, device)
    lpips_metric = LPIPSMetric(device, net=args.lpips_net)
    cfsd_metric = CFSDMetric(
        device=device,
        image_size=args.cfsd_image_size,
        vgg_until_layer=args.cfsd_vgg_until_layer,
        normalize_patch_features=not args.cfsd_no_normalize_patch_features,
        softmax_temperature=args.cfsd_temperature,
    )

    # Cache reference embeddings.
    print("Encoding style references for CSD retrieval...")
    style_csd_embs: Dict[str, torch.Tensor] = {}
    for style_name in tqdm(style_names):
        if style_name not in style_map:
            print(f"[WARN] style image not found for style={style_name}")
            continue
        style_csd_embs[style_name] = csd_style_embedding_from_image(
            csd_model,
            csd_preprocess,
            style_map[style_name],
            device,
            is_generated=False,
        )

    print("Encoding content references for CLIP-I and DINO...")
    content_clip_embs: Dict[str, torch.Tensor] = {}
    content_dino_embs: Dict[str, torch.Tensor] = {}
    for content_name in tqdm(content_names):
        if content_name not in content_map:
            print(f"[WARN] content image not found for content={content_name}")
            continue
        content_clip_embs[content_name] = clip_encoder.encode_path(content_map[content_name])
        content_dino_embs[content_name] = dino_encoder.encode_path(content_map[content_name])


    dcl_use_grayscale = not args.dcl_no_grayscale
    print(f"Encoding DCL references with DINO grayscale={dcl_use_grayscale}...")

    content_dcl_embs: Dict[str, torch.Tensor] = {}
    style_dcl_embs: Dict[str, torch.Tensor] = {}

    for content_name in tqdm(content_names, desc="DCL content refs"):
        if content_name not in content_map:
            continue
        content_dcl_embs[content_name] = dino_encoder.encode_path(
            content_map[content_name],
            grayscale=dcl_use_grayscale,
        )

    for style_name in tqdm(style_names, desc="DCL style refs"):
        if style_name not in style_map:
            continue
        style_dcl_embs[style_name] = dino_encoder.encode_path(
            style_map[style_name],
            grayscale=dcl_use_grayscale,
    )

    rows = []
    total_pairs = len(content_names) * len(style_names)
    pbar = tqdm(total=total_pairs, desc="Evaluating pairs")

    for content_name in content_names:
        for style_name in style_names:
            pbar.update(1)

            content_path = content_map.get(content_name)
            style_path = style_map.get(style_name)
            generated_path = resolve_generated_path(
                generated_dir=args.generated_dir,
                prefix=args.prefix,
                content_name=content_name,
                style_name=style_name,
                generated_pattern=args.generated_pattern,
                allow_glob_fallback=not args.no_glob_fallback,
            )

            row = {
                "content_name": content_name,
                "style_name": style_name,
                "content_path": str(content_path) if content_path else "",
                "style_path": str(style_path) if style_path else "",
                "generated_path": str(generated_path) if generated_path else "",
                "csd_style_similarity": float("nan"),
                "style_retrieval_top1": "",
                "style_retrieval_correct": float("nan"),
                "clip_i_content_similarity": float("nan"),
                "dino_content_similarity": float("nan"),
                "lpips": float("nan"),
                "cfsd": float("nan"),
                "dcl": float("nan"),
            }

            if content_path is None or style_path is None or generated_path is None:
                print(
                    f"[WARN] Missing pair: content={content_name}, style={style_name}, "
                    f"generated={generated_path}"
                )
                rows.append(row)
                continue

            # CSD generated style embedding, style similarity, and retrieval.
            gen_csd_emb = safe_metric(
                lambda: csd_style_embedding_from_image(
                    csd_model,
                    csd_preprocess,
                    generated_path,
                    device,
                    is_generated=True,
                    crop_generated_right_third=args.crop_generated_right_third,
                    fixed_512_triptych=args.fixed_512_triptych,
                ),
                default=None,
            )

            if gen_csd_emb is not None and style_name in style_csd_embs:
                row["csd_style_similarity"] = cosine(gen_csd_emb, style_csd_embs[style_name])

                sims = {
                    s: cosine(gen_csd_emb, emb)
                    for s, emb in style_csd_embs.items()
                }
                if sims:
                    top1 = max(sims, key=sims.get)
                    row["style_retrieval_top1"] = top1
                    row["style_retrieval_correct"] = 1.0 if top1 == style_name else 0.0

            # CLIP-I content similarity.
            gen_clip_emb = safe_metric(
                lambda: clip_encoder.encode_path(
                    generated_path,
                    is_generated=True,
                    crop_generated_right_third=args.crop_generated_right_third,
                    fixed_512_triptych=args.fixed_512_triptych,
                ),
                default=None,
            )
            if gen_clip_emb is not None and content_name in content_clip_embs:
                row["clip_i_content_similarity"] = cosine(
                    gen_clip_emb,
                    content_clip_embs[content_name],
                )

            # DINO content similarity.
            gen_dino_emb = safe_metric(
                lambda: dino_encoder.encode_path(
                    generated_path,
                    is_generated=True,
                    crop_generated_right_third=args.crop_generated_right_third,
                    fixed_512_triptych=args.fixed_512_triptych,
                ),
                default=None,
            )
            if gen_dino_emb is not None and content_name in content_dino_embs:
                row["dino_content_similarity"] = cosine(
                    gen_dino_emb,
                    content_dino_embs[content_name],
                )
            
            # DCL
            gen_dcl_emb = safe_metric(
                lambda: dino_encoder.encode_path(
                    generated_path,
                    is_generated=True,
                    crop_generated_right_third=args.crop_generated_right_third,
                    fixed_512_triptych=args.fixed_512_triptych,
                    grayscale=dcl_use_grayscale,
                ),
                default=None,
            )

            if (
                gen_dcl_emb is not None
                and content_name in content_dcl_embs
                and style_name in style_dcl_embs
            ):
                row["dcl"] = directional_content_leakage(
                    content_dcl_embs[content_name],
                    style_dcl_embs[style_name],
                    gen_dcl_emb,
                    clip=not args.dcl_no_clip,
                )

            # LPIPS.
            row["lpips"] = safe_metric(
                lambda: lpips_metric.distance(
                    content_path,
                    generated_path,
                    crop_generated_right_third=args.crop_generated_right_third,
                    fixed_512_triptych=args.fixed_512_triptych,
                )
            )

            # CFSD.
            row["cfsd"] = safe_metric(
                lambda: cfsd_metric.distance(
                    content_path,
                    generated_path,
                    crop_generated_right_third=args.crop_generated_right_third,
                    fixed_512_triptych=args.fixed_512_triptych,
                )
            )

            rows.append(row)

    pbar.close()

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["style_retrieval_accuracy"] = df["style_retrieval_correct"].mean()

    return df



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate style transfer metrics: CSD, Style Retrieval, CLIP-I, DINO, LPIPS, CFSD."
    )

    parser.add_argument("--content_dir", type=str, required=True)
    parser.add_argument("--style_dir", type=str, required=True)
    parser.add_argument("--generated_dir", type=str, required=True)

    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Generated filename prefix. Default pattern uses {prefix}_{content}_{style}.png.",
    )
    parser.add_argument(
        "--generated_pattern",
        type=str,
        default="{prefix}_{content}_{style}.png",
        help="Pattern with placeholders {prefix}, {content}, {style}.",
    )
    parser.add_argument(
        "--no_glob_fallback",
        action="store_true",
        help="Disable fallback glob search: {prefix}_{content}_{style}*.",
    )

    parser.add_argument("--output_csv", type=str, default="style_transfer_metrics.csv")
    parser.add_argument("--output_summary_csv", type=str, default="style_transfer_metrics_summary.csv")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--content_names",
        nargs="+",
        default=None,
        help="Optional subset/order of content names, without extension.",
    )
    parser.add_argument(
        "--style_names",
        nargs="+",
        default=None,
        help="Optional subset/order of style names, without extension.",
    )

    parser.add_argument(
        "--crop_generated_right_third",
        action="store_true",
        help="Crop the right third of generated image before evaluation.",
    )
    parser.add_argument(
        "--fixed_512_triptych",
        action="store_true",
        help="Generated image is 3 x 512 horizontal triptych; crop [1024:1536].",
    )

    # CSD.
    parser.add_argument("--csd_checkpoint", type=str, default="third_party/CSD/checkpoint.pth")
    parser.add_argument("--csd_third_party", type=str, default="third_party")

    # CLIP / DINO.
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino_model", type=str, default="facebook/dino-vits16")

    parser.add_argument(
        "--dcl_no_grayscale",
        action="store_true",
        help="Disable grayscale preprocessing for DCL. By default DCL uses grayscale DINO features to reduce color/style bias.",
    )

    parser.add_argument(
        "--dcl_no_clip",
        action="store_true",
        help="Do not clip DCL to [0, 1]. Useful for debugging raw directional projection values.",
    )

    # LPIPS.
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])

    # CFSD.
    parser.add_argument("--cfsd_image_size", type=int, default=224)
    parser.add_argument(
        "--cfsd_vgg_until_layer",
        type=int,
        default=12,
        help="VGG19 features[:layer]. 12 means relu3_1 output; 19 means after pool3.",
    )
    parser.add_argument("--cfsd_temperature", type=float, default=1.0)
    parser.add_argument(
        "--cfsd_no_normalize_patch_features",
        action="store_true",
        help="Disable L2-normalization of patch features before correlation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = evaluate_all(args)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_with_avg = add_average_row(df)
    df_with_avg.to_csv(output_csv, index=False)
    print(f"Saved full metrics to: {output_csv}")

    summary_cols = [
        "csd_style_similarity",
        "style_retrieval_correct",
        "clip_i_content_similarity",
        "dino_content_similarity",
        "lpips",
        "cfsd",
        "dcl",
    ]
    summary = df[summary_cols].mean(numeric_only=True).to_frame("mean").T
    summary = summary.rename(columns={"style_retrieval_correct": "style_retrieval_accuracy"})

    output_summary_csv = Path(args.output_summary_csv)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_summary_csv, index=False)
    print(f"Saved summary metrics to: {output_summary_csv}")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

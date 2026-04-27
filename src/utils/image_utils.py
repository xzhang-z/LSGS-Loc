from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image
from torchvision import transforms as tvf

try:
    import natsort
except ImportError:  # pragma: no cover
    natsort = None


def get_default_base_transform() -> tvf.Compose:
    return tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collect_image_files(image_dir: str, img_exts: Iterable[str]) -> List[str]:
    image_dir = os.path.realpath(os.path.expanduser(image_dir))
    if isinstance(img_exts, str):
        img_exts = [img_exts]

    img_fnames: List[str] = []
    for ext in img_exts:
        ext = ext.lstrip(".")
        patterns = [f"**/*.{ext}", f"**/*.{ext.lower()}", f"**/*.{ext.upper()}"]
        for pattern in patterns:
            img_fnames.extend(glob.glob(os.path.join(image_dir, pattern), recursive=True))

    img_fnames = list(set(img_fnames))
    if natsort is not None:
        return natsort.natsorted(img_fnames)
    return sorted(img_fnames)


def preprocess_image_tensor(
    image_path: str,
    max_img_size: int,
    device: torch.device,
    base_transform: tvf.Compose | None = None,
    patch_multiple: int = 14,
) -> torch.Tensor:
    if base_transform is None:
        base_transform = get_default_base_transform()

    pil_img = Image.open(image_path)
    pil_img.load()

    if max(pil_img.size) > max_img_size:
        w, h = pil_img.size
        if h >= w:
            new_h = max_img_size
            new_w = int(w * max_img_size / h)
        else:
            new_w = max_img_size
            new_h = int(h * max_img_size / w)
        pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)

    current_w, current_h = pil_img.size
    h_new = (current_h // patch_multiple) * patch_multiple
    w_new = (current_w // patch_multiple) * patch_multiple
    left = (current_w - w_new) / 2
    top = (current_h - h_new) / 2
    right = (current_w + w_new) / 2
    bottom = (current_h + h_new) / 2
    pil_img = pil_img.crop((left, top, right, bottom)).convert("RGB")

    img_pt = base_transform(pil_img).unsqueeze(0).to(device)
    return img_pt


def descriptor_cache_path(image_path: str, image_root: str, descriptor_dir: str) -> Path:
    relative_path = os.path.relpath(image_path, image_root)
    relative_without_ext = os.path.splitext(relative_path)[0]
    descriptor_name = relative_without_ext.replace(os.path.sep, "__") + ".npy"
    return Path(descriptor_dir) / descriptor_name

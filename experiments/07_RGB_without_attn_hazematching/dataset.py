"""
RGB Laparoscopic Dataset Loader for Experiment 07
=================================================
Loads paired (clean, smoky) laparoscopic frames in 3-channel RGB:
- Converts all images to RGB
- Random synchronized crop and horizontal flips
- Normalizes pixel values to [-1, 1] for flow matching
"""

import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


def denormalize_rgb(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] range to [0, 1] range."""
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


def tensor_to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized [-1, 1] (3, H, W) PyTorch tensor to a uint8 (H, W, 3) RGB numpy array.
    """
    denorm = denormalize_rgb(tensor.detach().cpu())
    np_img = denorm.permute(1, 2, 0).numpy()
    return np.clip(np_img * 255.0 + 0.5, 0, 255).astype(np.uint8)


class RGBLaparoscopyDataset(Dataset):
    """
    Paired RGB Dataset for Laparoscopic Dehazing.

    Expects directory structure:
        data_dir/
            clean/
                img1.png ...
            smoky/
                img1.png ...

    Returns:
        tuple (clean_tensor, smoky_tensor, filename)
        Both clean and smoky are shape (3, H, W) normalized to [-1, 1].
    """

    def __init__(
        self,
        folder: Union[str, Path],
        image_size: int = 256,
        split: str = "train",
        is_train: bool = True,
    ):
        super().__init__()
        self.folder = Path(folder)
        self.image_size = image_size
        self.split = split
        self.is_train = is_train

        self.clean_dir = self.folder / "clean"
        self.smoky_dir = self.folder / "smoky"

        if not self.clean_dir.exists() or not self.smoky_dir.exists():
            raise FileNotFoundError(
                f"Directory must contain 'clean' and 'smoky' subfolders. Found: {self.folder}"
            )

        clean_files = {f for f in os.listdir(self.clean_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))}
        smoky_files = {f for f in os.listdir(self.smoky_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))}
        self.filenames = sorted(list(clean_files & smoky_files))

        if len(self.filenames) == 0:
            raise ValueError(f"No matching clean/smoky image pairs found in {self.folder}")

    def __len__(self) -> int:
        return len(self.filenames)

    def _load_rgb_image(self, path: Path) -> Image.Image:
        """Load an image and guarantee 3-channel RGB format."""
        with Image.open(path) as img:
            return img.convert("RGB")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.filenames[index]
        clean_path = self.clean_dir / filename
        smoky_path = self.smoky_dir / filename

        clean_pil = self._load_rgb_image(clean_path)
        smoky_pil = self._load_rgb_image(smoky_path)

        # Synchronized Spatial Transformations
        if self.is_train:
            w, h = clean_pil.size
            if w > self.image_size and h > self.image_size:
                top = int(np.random.randint(0, h - self.image_size + 1))
                left = int(np.random.randint(0, w - self.image_size + 1))
                clean_pil = TF.crop(clean_pil, top, left, self.image_size, self.image_size)
                smoky_pil = TF.crop(smoky_pil, top, left, self.image_size, self.image_size)
            else:
                clean_pil = TF.resize(clean_pil, (self.image_size, self.image_size), Image.BILINEAR)
                smoky_pil = TF.resize(smoky_pil, (self.image_size, self.image_size), Image.BILINEAR)

            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                clean_pil = TF.hflip(clean_pil)
                smoky_pil = TF.hflip(smoky_pil)
        else:
            clean_pil = TF.resize(clean_pil, (self.image_size, self.image_size), Image.BILINEAR)
            smoky_pil = TF.resize(smoky_pil, (self.image_size, self.image_size), Image.BILINEAR)

        # Convert to float tensor [0, 1]
        clean_t = TF.to_tensor(clean_pil)
        smoky_t = TF.to_tensor(smoky_pil)

        # Rescale [0, 1] -> [-1, 1] for Flow Matching
        clean_norm = clean_t * 2.0 - 1.0
        smoky_norm = smoky_t * 2.0 - 1.0

        return clean_norm, smoky_norm, filename

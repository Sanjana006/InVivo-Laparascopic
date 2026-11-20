"""Paired smoky/clean dataset loader. Supports RGB and grayscale source images."""
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedLaparoscopyDataset(Dataset):
    EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    def __init__(self, split_dir, image_size=256, augment=False):
        self.root = Path(split_dir)
        self.clean_dir, self.smoky_dir = self.root / "clean", self.root / "smoky"
        if not self.clean_dir.is_dir() or not self.smoky_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {self.clean_dir} and {self.smoky_dir}. "
                "Each split must contain matching clean/ and smoky/ folders."
            )
        clean = {p.name: p for p in self.clean_dir.iterdir() if p.suffix.lower() in self.EXTENSIONS}
        smoky = {p.name: p for p in self.smoky_dir.iterdir() if p.suffix.lower() in self.EXTENSIONS}
        self.names = sorted(set(clean) & set(smoky))
        if not self.names:
            raise RuntimeError(f"No matching clean/smoky image pairs in {self.root}")
        self.clean_map, self.smoky_map = clean, smoky
        self.image_size, self.augment = image_size, augment

    def __len__(self): return len(self.names)

    def _read(self, path):
        im = Image.open(path).convert("RGB")
        if self.image_size:
            im = im.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(a).permute(2,0,1).contiguous()

    def __getitem__(self, idx):
        name = self.names[idx]
        clean = self._read(self.clean_map[name])
        smoky = self._read(self.smoky_map[name])
        if self.augment and torch.rand(()) < 0.5:
            clean, smoky = clean.flip(-1), smoky.flip(-1)
        if self.augment and torch.rand(()) < 0.5:
            clean, smoky = clean.flip(-2), smoky.flip(-2)
        return {"smoky": smoky, "clean": clean, "name": name}

"""
src/hazematching/dataset.py
===========================
Custom PyTorch Dataset for loading paired laparoscopy PNGs directly.
Converts RGB to grayscale on-the-fly and applies per-channel z-score normalization.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils import data

from hazematching.datasets.config import format_subset_options
from .config import LAPAROSCOPY_SUBSET, register_laparoscopy_subset

# Ensure registration
register_laparoscopy_subset()

class LaparoscopyDataset(data.Dataset):
    """
    Dataset for loading paired clean/smoky Laparoscopic PNG images.
    
    Expects directory structure:
        data_dir/
            clean/
                img1.png
                img2.png
                ...
            smoky/
                img1.png
                img2.png
                ...
                
    Outputs a tensor of shape (2, H, W) where:
        channel 0: Clean (target)
        channel 1: Smoky (input)
    """

    def __init__(
        self,
        subset: str,
        folder: Path,
        returns: list[int] = [0, 1],
    ):
        super().__init__()
        
        if subset != LAPAROSCOPY_SUBSET:
            raise ValueError(
                f"LaparoscopyDataset expects subset='{LAPAROSCOPY_SUBSET}', got {subset!r}"
            )

        self.subset = subset
        self.returns = returns
        self.folder = Path(folder)
        
        self.clean_dir = self.folder / "clean"
        self.smoky_dir = self.folder / "smoky"
        
        if not self.clean_dir.exists() or not self.smoky_dir.exists():
            raise FileNotFoundError(
                f"Expected 'clean' and 'smoky' directories under {self.folder}"
            )
            
        # Get matching filenames
        clean_files = set(f for f in os.listdir(self.clean_dir) if f.endswith(".png"))
        smoky_files = set(f for f in os.listdir(self.smoky_dir) if f.endswith(".png"))
        
        self.filenames = sorted(list(clean_files & smoky_files))
        
        if len(self.filenames) == 0:
            raise ValueError(f"No matching PNG pairs found in {self.folder}")

    def __len__(self):
        return len(self.filenames)
        
    def _load_grayscale(self, path: Path) -> np.ndarray:
        """Load image as grayscale and convert to float32 numpy array."""
        img = Image.open(path).convert('L') # Convert to grayscale
        return np.array(img, dtype=np.float32)

    def __getitem__(self, index):
        filename = self.filenames[index]
        
        clean_path = self.clean_dir / filename
        smoky_path = self.smoky_dir / filename
        
        # Load as grayscale (1 channel)
        clean_img = self._load_grayscale(clean_path)
        smoky_img = self._load_grayscale(smoky_path)
        
        # Shape: (1, H, W)
        clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)
        smoky_tensor = torch.from_numpy(smoky_img).unsqueeze(0)
        
        # Combine into (2, H, W). Index 0 = clean, Index 1 = smoky
        combined = torch.cat([clean_tensor, smoky_tensor], dim=0)
        
        # Apply normalization using HazeMatching's normalize function
        from hazematching.datasets.data_norm import normalize
        
        channels = [
            normalize(combined[ch : ch + 1], self.subset, ch, path=None)
            for ch in self.returns
        ]
        
        return torch.cat(channels, dim=0)

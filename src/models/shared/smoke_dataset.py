import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


def resolve_path(path: str, base_dir: str = None) -> str:
    """Resolve directory path case-insensitively and relative to base_dir."""
    if not path:
        return path
    if os.path.exists(path):
        return os.path.abspath(path)

    if base_dir and not os.path.isabs(path):
        alt = os.path.join(base_dir, path)
        if os.path.exists(alt):
            return os.path.abspath(alt)

    # Segment-by-segment case-insensitive search
    parts = os.path.normpath(path).split(os.sep)
    current = base_dir if (base_dir and not os.path.isabs(path)) else ("/" if (os.name != "nt" and path.startswith("/")) else "")

    for part in parts:
        if not part or part == ".":
            continue
        if part == "..":
            current = os.path.dirname(current) if current else ".."
            continue

        target = os.path.join(current, part) if current else part
        if os.path.exists(target):
            current = target
            continue

        found = False
        search_dir = current if current else "."
        if os.path.exists(search_dir) and os.path.isdir(search_dir):
            for item in os.listdir(search_dir):
                if item.lower() == part.lower():
                    current = os.path.join(search_dir, item)
                    found = True
                    break
        if not found:
            current = target

    return os.path.abspath(current) if os.path.exists(current) else current


class SmokeDataset(Dataset):
    """
    Paired smoke/clean laparoscopic image dataset.

    Loads paired (smoky, clean) image sets from separate directories.
    Images are returned as float tensors in [0, 1] range in RGB channel order.

    Args:
        clean_dir   : Path to directory of clean (ground truth) images.
        smoky_dir   : Path to directory of smoky images.
        split       : One of 'train', 'test', or 'all'.
        train_ratio : Fraction of data used for training (default 0.875 → 87.5/12.5 split).
        augment     : If True, applies random horizontal/vertical flips during training.
    """

    def __init__(
        self,
        clean_dir: str,
        smoky_dir: str,
        split: str = "train",
        train_ratio: float = 0.875,
        augment: bool = False,
    ):
        # Resolve relative directory paths case-insensitively and relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        clean_dir = resolve_path(clean_dir, base_dir=project_root)
        smoky_dir = resolve_path(smoky_dir, base_dir=project_root)

        self.clean_dir = clean_dir
        self.smoky_dir = smoky_dir
        self.augment = augment and (split == "train")

        if not os.path.exists(clean_dir):
            raise FileNotFoundError(f"Clean image directory not found at: '{clean_dir}'. Please verify dataset path.")
        if not os.path.exists(smoky_dir):
            raise FileNotFoundError(f"Smoky image directory not found at: '{smoky_dir}'. Please verify dataset path.")

        # Build stem-to-filename mappings
        clean_map = {os.path.splitext(f)[0]: f for f in os.listdir(clean_dir) if not f.startswith('.')}
        smoky_map = {os.path.splitext(f)[0]: f for f in os.listdir(smoky_dir) if not f.startswith('.')}

        common_stems = sorted(
            list(set(clean_map.keys()) & set(smoky_map.keys())),
            key=lambda x: int(x) if x.isdigit() else x
        )

        paired = [(clean_map[s], smoky_map[s]) for s in common_stems]

        if len(paired) == 0:
            clean_files = [f for f in os.listdir(clean_dir) if not f.startswith('.')]
            smoky_files = [f for f in os.listdir(smoky_dir) if not f.startswith('.')]
            raise ValueError(
                f"No matching image pairs found between clean_dir='{clean_dir}' ({len(clean_files)} files) "
                f"and smoky_dir='{smoky_dir}' ({len(smoky_files)} files)."
            )

        split_idx = int(len(paired) * train_ratio)

        if split == "train":
            self.paired_files = paired[:split_idx]
        elif split == "test":
            self.paired_files = paired[split_idx:]
        else:
            self.paired_files = paired

    def __len__(self) -> int:
        return len(self.paired_files)

    def __getitem__(self, idx: int):
        clean_name, smoky_name = self.paired_files[idx]

        clean_path = os.path.join(self.clean_dir, clean_name)
        smoky_path = os.path.join(self.smoky_dir, smoky_name)

        clean_bgr = cv2.imread(clean_path)
        smoky_bgr = cv2.imread(smoky_path)

        if clean_bgr is None:
            raise FileNotFoundError(f"Could not read clean image at: {clean_path}")
        if smoky_bgr is None:
            raise FileNotFoundError(f"Could not read smoky image at: {smoky_path}")

        clean = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        smoky = cv2.cvtColor(smoky_bgr, cv2.COLOR_BGR2RGB)

        # Convert to float tensor in [0, 1], shape (C, H, W)
        clean = torch.tensor(clean).permute(2, 0, 1).float() / 255.0
        smoky = torch.tensor(smoky).permute(2, 0, 1).float() / 255.0

        # Data augmentation (training only): random horizontal and vertical flips
        if self.augment:
            if random.random() > 0.5:
                clean = TF.hflip(clean)
                smoky = TF.hflip(smoky)
            if random.random() > 0.5:
                clean = TF.vflip(clean)
                smoky = TF.vflip(smoky)

        return smoky, clean

"""
Dataset loader for Trident Dehazing Network (TDN) on Laparoscopic Desmoking.
Loads paired (smoky, clean) frames with TDN paper augmentations:
- Random 256x256 crops
- Random rotations (0, 90, 180, 270 degrees)
- Random horizontal flip
- Normalization to [0, 1] range
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class TridentLaparoscopyDataset(Dataset):
    """
    Paired laparoscopic image dataset for Trident Dehazing Network.
    Loads smoky images as inputs and clean frames as ground truth.
    """

    def __init__(
        self,
        clean_dir: str,
        smoky_dir: str,
        crop_size: int = 256,
        is_train: bool = True,
    ):
        super().__init__()
        self.clean_dir = clean_dir
        self.smoky_dir = smoky_dir
        self.crop_size = crop_size
        self.is_train = is_train

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        if not os.path.exists(clean_dir):
            raise FileNotFoundError(f"Clean directory does not exist: {clean_dir}")
        if not os.path.exists(smoky_dir):
            raise FileNotFoundError(f"Smoky directory does not exist: {smoky_dir}")

        clean_files = sorted(
            [f for f in os.listdir(clean_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        )
        smoky_files = sorted(
            [f for f in os.listdir(smoky_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        )

        clean_dict = {os.path.splitext(f)[0]: f for f in clean_files}
        smoky_dict = {os.path.splitext(f)[0]: f for f in smoky_files}

        common_keys = sorted(list(set(clean_dict.keys()).intersection(set(smoky_dict.keys()))))

        if len(common_keys) == 0:
            # Fallback to positional pairing if filenames differ
            min_len = min(len(clean_files), len(smoky_files))
            self.pairs = [(clean_files[i], smoky_files[i]) for i in range(min_len)]
        else:
            self.pairs = [(clean_dict[k], smoky_dict[k]) for k in common_keys]

        print(f"[{'TRAIN' if is_train else 'VAL/TEST'}] Loaded {len(self.pairs)} paired laparoscopic images.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        clean_name, smoky_name = self.pairs[idx]
        clean_path = os.path.join(self.clean_dir, clean_name)
        smoky_path = os.path.join(self.smoky_dir, smoky_name)

        clean_bgr = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        smoky_bgr = cv2.imread(smoky_path, cv2.IMREAD_COLOR)

        if clean_bgr is None:
            raise IOError(f"Could not read clean image: {clean_path}")
        if smoky_bgr is None:
            raise IOError(f"Could not read smoky image: {smoky_path}")

        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        smoky_rgb = cv2.cvtColor(smoky_bgr, cv2.COLOR_BGR2RGB)

        # Convert to float [0, 1] tensors (C, H, W)
        clean_t = torch.from_numpy(clean_rgb).permute(2, 0, 1).float() / 255.0
        smoky_t = torch.from_numpy(smoky_rgb).permute(2, 0, 1).float() / 255.0

        C, H, W = smoky_t.shape

        if self.is_train and self.crop_size is not None and H >= self.crop_size and W >= self.crop_size:
            # Random crop 256x256
            top = random.randint(0, H - self.crop_size)
            left = random.randint(0, W - self.crop_size)
            smoky_t = smoky_t[:, top : top + self.crop_size, left : left + self.crop_size]
            clean_t = clean_t[:, top : top + self.crop_size, left : left + self.crop_size]

            # Paper augmentations: Random horizontal flip
            if random.random() > 0.5:
                smoky_t = TF.hflip(smoky_t)
                clean_t = TF.hflip(clean_t)

            # Paper augmentations: Random rotation by 90, 180, 270 degrees
            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                smoky_t = torch.rot90(smoky_t, k=rot_k, dims=[1, 2])
                clean_t = torch.rot90(clean_t, k=rot_k, dims=[1, 2])
        else:
            # Ensure dimensions are divisible by 64 for 6-level HDMGN U-Net and EDN
            pad_h = (64 - (H % 64)) % 64
            pad_w = (64 - (W % 64)) % 64
            if pad_h > 0 or pad_w > 0:
                smoky_t = TF.pad(smoky_t, [0, 0, pad_w, pad_h], padding_mode="reflect")
                clean_t = TF.pad(clean_t, [0, 0, pad_w, pad_h], padding_mode="reflect")

        return {
            "smoky": smoky_t,
            "clean": clean_t,
            "filename": clean_name,
        }


def get_dataloaders(
    data_dir: str,
    crop_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """
    Creates train, validation, and test dataloaders from the project's data directory.
    Expected structure:
      data_dir/
        train_crop/ (clean/, smoky/)
        val_crop/   (clean/, smoky/)
        test/       (clean/, smoky/)
    """
    train_clean = os.path.join(data_dir, "train_crop", "clean")
    train_smoky = os.path.join(data_dir, "train_crop", "smoky")
    val_clean = os.path.join(data_dir, "val_crop", "clean")
    val_smoky = os.path.join(data_dir, "val_crop", "smoky")
    test_clean = os.path.join(data_dir, "test", "clean")
    test_smoky = os.path.join(data_dir, "test", "smoky")

    train_ds = TridentLaparoscopyDataset(train_clean, train_smoky, crop_size=crop_size, is_train=True)
    val_ds = TridentLaparoscopyDataset(val_clean, val_smoky, crop_size=None, is_train=False)
    test_ds = TridentLaparoscopyDataset(test_clean, test_smoky, crop_size=None, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


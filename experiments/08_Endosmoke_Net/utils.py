import csv, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def pseudo_density_target(smoky, clean):
    """
    Weak pseudo-label from paired image difference.
    This is not a clinician-annotated smoke mask. It assumes smoky/clean pairs
    are spatially registered and uses smoothed luminance discrepancy as a proxy.
    """
    s = (0.299*smoky[:,0:1] + 0.587*smoky[:,1:2] + 0.114*smoky[:,2:3])
    c = (0.299*clean[:,0:1] + 0.587*clean[:,1:2] + 0.114*clean[:,2:3])
    d = (s-c).abs()
    # Per-image robust scaling, then spatial smoothing.
    flat = d.flatten(1)
    scale = torch.quantile(flat, 0.95, dim=1).clamp_min(1e-4)[:,None,None,None]
    d = (d / scale).clamp(0,1)
    kernel = 9
    d = F.avg_pool2d(d, kernel_size=kernel, stride=1, padding=kernel//2)
    return d.clamp(0,1)


def psnr(pred, target):
    mse = torch.mean((pred-target)**2).item()
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(4.0 / mse)  # data range [-1,1]


def batch_ssim(pred, target):
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return float("nan")
    p = ((pred.detach().cpu().permute(0,2,3,1).numpy()+1)/2).clip(0,1)
    t = ((target.detach().cpu().permute(0,2,3,1).numpy()+1)/2).clip(0,1)
    vals = [structural_similarity(a, b, channel_axis=2, data_range=1.0) for a,b in zip(p,t)]
    return float(np.mean(vals))


def save_image(tensor, path):
    from PIL import Image
    a = tensor.detach().float().cpu().clamp(-1,1)
    a = ((a + 1) * 127.5).byte().permute(1,2,0).numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def append_csv(path, row, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists: writer.writeheader()
        writer.writerow(row)

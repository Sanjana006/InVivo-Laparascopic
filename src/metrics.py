import torch
import torch.nn as nn
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
import lpips
from src.logger import setup_logger

logger = setup_logger("Metrics")

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between predicted and target images.
    Inputs expected to be Tensors of shape (N, C, H, W) or (C, H, W) in [0, 1].
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    
    pred = torch.clamp(pred, 0.0, data_range)
    target = torch.clamp(target, 0.0, data_range)

    val = peak_signal_noise_ratio(pred, target, data_range=data_range)
    return float(val.item())

def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Computes Structural Similarity Index Measure (SSIM) between predicted and target images.
    Inputs expected to be Tensors of shape (N, C, H, W) or (C, H, W) in [0, 1].
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)

    pred = torch.clamp(pred, 0.0, data_range)
    target = torch.clamp(target, 0.0, data_range)

    val = structural_similarity_index_measure(pred, target, data_range=data_range)
    return float(val.item())

class Evaluator:
    """
    Evaluator class for managing image quality metrics (PSNR, SSIM, LPIPS).
    """
    def __init__(self, device: str = "cpu", net: str = "alex"):
        self.device = device
        self.lpips_net = None
        self.net_type = net

    def _init_lpips(self):
        if self.lpips_net is None:
            logger.info(f"Initializing LPIPS model (net={self.net_type}) on device {self.device}...")
            try:
                self.lpips_net = lpips.LPIPS(net=self.net_type, verbose=False).to(self.device)
                self.lpips_net.eval()
            except Exception as e:
                logger.error(f"Failed to initialize LPIPS model: {e}")
                self.lpips_net = None

    def calculate_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Computes LPIPS perceptual distance.
        Inputs: Tensors in range [0, 1]. Converted internally to [-1, 1] as required by LPIPS.
        """
        self._init_lpips()
        if self.lpips_net is None:
            return 0.0

        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)

        # Scale inputs from [0, 1] to [-1, 1]
        pred_norm = torch.clamp(pred, 0.0, 1.0) * 2.0 - 1.0
        target_norm = torch.clamp(target, 0.0, 1.0) * 2.0 - 1.0

        pred_norm = pred_norm.to(self.device)
        target_norm = target_norm.to(self.device)

        with torch.no_grad():
            dist = self.lpips_net(pred_norm, target_norm)
        return float(dist.mean().item())

    def evaluate_batch(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Evaluates PSNR, SSIM, and LPIPS for a batch of predictions and targets.
        """
        psnr_val = calculate_psnr(pred, target)
        ssim_val = calculate_ssim(pred, target)
        lpips_val = self.calculate_lpips(pred, target)

        return {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val
        }

    def evaluate_loader(self, model: nn.Module, dataloader, device: str = None) -> dict:
        """
        Evaluates a model over an entire dataloader.
        Returns average PSNR, SSIM, and LPIPS.
        """
        if device is None:
            device = self.device

        model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        num_batches = 0

        with torch.no_grad():
            for smoky, clean in dataloader:
                smoky, clean = smoky.to(device), clean.to(device)
                fake = model(smoky)
                fake = torch.clamp(fake, 0.0, 1.0)

                metrics = self.evaluate_batch(fake, clean)
                total_psnr += metrics["psnr"]
                total_ssim += metrics["ssim"]
                total_lpips += metrics["lpips"]
                num_batches += 1

        if num_batches == 0:
            return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}

        return {
            "psnr": total_psnr / num_batches,
            "ssim": total_ssim / num_batches,
            "lpips": total_lpips / num_batches
        }

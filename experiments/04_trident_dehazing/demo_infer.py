"""
Quick Demo Inference Script for Trident Dehazing Network (TDN).
Runs desmoking on an individual image or directory of images.
Usage:
  python demo_infer.py --input path/to/smoky.png --output path/to/output.png --backbone dpn92
"""

import os
import sys
import argparse
import cv2
import torch
import torchvision.transforms.functional as TF

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models import TridentDehazingNet


def main():
    parser = argparse.ArgumentParser(description="Run single-image TDN inference")
    parser.add_argument("--input", type=str, required=True, help="Input smoky image path")
    parser.add_argument("--output", type=str, default="dehazed_result.png", help="Output desmoked image path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path (optional)")
    parser.add_argument("--backbone", type=str, default="dpn92", choices=["dpn92", "endovit"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = TridentDehazingNet(backbone=args.backbone, pretrained=(args.checkpoint is None)).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        print(f"Loaded weights from {args.checkpoint}")
    model.eval()

    # Read image
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {args.input}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Ensure divisible by 64
    _, _, H, W = img_t.shape
    pad_h = (64 - (H % 64)) % 64
    pad_w = (64 - (W % 64)) % 64
    if pad_h > 0 or pad_w > 0:
        img_t = TF.pad(img_t, [0, 0, pad_w, pad_h], padding_mode="reflect")

    img_t = img_t.to(device)
    with torch.no_grad():
        out_dict = model(img_t, return_internals=True)
        pred = out_dict["out"]
        if pad_h > 0 or pad_w > 0:
            pred = pred[:, :, :H, :W]

    out_np = (pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    cv2.imwrite(args.output, out_bgr)
    print(f"Saved desmoked result to: {args.output}")


if __name__ == "__main__":
    main()


import os
import cv2
import torch
import torchvision.transforms as T

from src.models.standard.generator import Generator
from src.models.attention.generator import Attention_Generator
from src.models.cycle_dehaze.generator import Cycle_Dehaze_Attention
from src.logger import setup_logger, enable_output_redirection
from src.metrics import Evaluator

enable_output_redirection()
logger = setup_logger("Test")
logger.info("=== Starting Testing Pipeline ===")

# ============================================================
# CONFIGURE: change model_type to switch between experiments
# ============================================================
MODEL_TYPE = "attention"   # "standard", "attention", or "cycle_dehaze"

EXP_DIR_MAP = {
    "standard":     "experiments/01_without_attention",
    "attention":    "experiments/02_attention_unet",
    "cycle_dehaze": "experiments/03_cycle_dehaze",
}

exp_dir     = EXP_DIR_MAP[MODEL_TYPE]
ckpt_path   = os.path.join(exp_dir, "G_final.pth")
output_dir  = os.path.join(exp_dir, "outputs", "test_images")
os.makedirs(output_dir, exist_ok=True)

logger.info(f"Model type      : {MODEL_TYPE}")
logger.info(f"Experiment dir  : {exp_dir}")
logger.info(f"Checkpoint      : {ckpt_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load model ----
if not os.path.exists(ckpt_path):
    logger.error(f"❌ Checkpoint not found: {ckpt_path}")
    exit(1)

checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint["G_state_dict"] if "G_state_dict" in checkpoint else checkpoint

if MODEL_TYPE == "standard":
    G = Generator().to(device)
    G.load_state_dict(state_dict)
    logger.info("Loaded: Standard U-Net Generator")
elif MODEL_TYPE == "cycle_dehaze":
    G = Cycle_Dehaze_Attention(use_tanh=True).to(device)
    G.load_state_dict(state_dict)
    logger.info("Loaded: Cycle-Dehaze Attention Generator")
else:
    G = Attention_Generator().to(device)
    G.load_state_dict(state_dict)
    logger.info("Loaded: Attention U-Net Generator")

G.eval()

# ============================================================
# SINGLE IMAGE TEST
# (change img_path to any smoky image you want to test on)
# ============================================================
img_path = "data_final/smoky/242.png"

if not os.path.exists(img_path):
    logger.error(f"❌ Image not found: {img_path}")
    exit(1)

logger.info(f"🔍 Input image: {img_path}")

# Load as RGB (matches training pipeline)
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
inp = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
inp = inp.unsqueeze(0).to(device)

with torch.no_grad():
    output = G(inp)

output = output.squeeze().cpu().clamp(0, 1)

# Save output as RGB PNG
out_path = os.path.join(output_dir, f"result_{MODEL_TYPE}.png")
T.ToPILImage()(output).save(out_path)
logger.info(f"✅ Saved result → {out_path}")

# ---- Metrics against ground truth (matching stem in clean directory) ----
clean_dir = "data_final/clean"
stem = os.path.splitext(os.path.basename(img_path))[0]
clean_img_path = None
if os.path.exists(clean_dir):
    candidates = [f for f in os.listdir(clean_dir) if os.path.splitext(f)[0] == stem]
    if candidates:
        clean_img_path = os.path.join(clean_dir, candidates[0])

if clean_img_path and os.path.exists(clean_img_path):
    evaluator = Evaluator(device=device)

    clean_bgr    = cv2.imread(clean_img_path)
    clean_rgb    = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    clean_tensor = torch.tensor(clean_rgb).permute(2, 0, 1).float() / 255.0
    clean_tensor = clean_tensor.unsqueeze(0).to(device)

    pred_tensor = output.unsqueeze(0).to(device)
    metrics = evaluator.evaluate_batch(pred_tensor, clean_tensor)

    logger.info(f"📊 Metrics vs ground truth:")
    logger.info(f"   ► PSNR  : {metrics['psnr']:.2f} dB")
    logger.info(f"   ► SSIM  : {metrics['ssim']:.4f}")
    logger.info(f"   ► LPIPS : {metrics['lpips']:.4f}")

logger.info("=== Testing Pipeline Completed ===")
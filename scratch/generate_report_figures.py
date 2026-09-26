import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("scratch", exist_ok=True)

# -------------------------------------------------------------
# Figure 1: Multi-Model Visual Comparison Plate
# -------------------------------------------------------------
# We select a representative index (e.g. index 0000 or index 0010)
img_smoky_04 = cv2.imread("experiments/04_hazematching/outputs/test_images/smoky_0000.png")
img_clean_04 = cv2.imread("experiments/04_hazematching/outputs/test_images/clean_0000.png")
img_hm_04    = cv2.imread("experiments/04_hazematching/outputs/test_images/dehazed_0000.png")

img_std_01   = cv2.imread("experiments/01_without_attention/outputs/test_images/dehazed_epoch055_0000.png")
img_att_02   = cv2.imread("experiments/02_attention_unet/outputs/test_images/result.png")
img_cyc_03   = cv2.imread("experiments/03_cycle_dehaze/outputs/test_images/dehazed_epoch100_0000.png")

# Resize all to 256x256 if needed
imgs = [img_smoky_04, img_std_01, img_att_02, img_cyc_03, img_hm_04, img_clean_04]
titles = [
    "(a) Smoky Input\n(In-Vivo Laparoscopic)",
    "(b) Standard U-Net\n(Pix2Pix, Ep 60)",
    "(c) Attention U-Net\n(Spatial Attn, Ep 100)",
    "(d) Cycle-Dehaze\n(CycleGAN, Ep 100)",
    "(e) HazeMatching\n(Flow Matching, Ep 200)",
    "(f) Ground Truth\n(Clean Reference)"
]

fig, axes = plt.subplots(1, 6, figsize=(18, 3.8), dpi=300)
for ax, img, title in zip(axes, imgs, titles):
    if img is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
    else:
        ax.text(0.5, 0.5, "Image N/A", ha="center", va="center")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8, color="#1B365D")
    ax.axis("off")

plt.tight_layout()
fig1_path = "scratch/fig1_multimodel_comparison.png"
plt.savefig(fig1_path, bbox_inches="tight", dpi=300)
plt.close()
print("Saved:", fig1_path)

# -------------------------------------------------------------
# Figure 4: Epoch Progression Plate (HazeMatching)
# -------------------------------------------------------------
e10  = cv2.imread("experiments/04_hazematching/outputs/samples/epoch_010_sample.png")
e50  = cv2.imread("experiments/04_hazematching/outputs/samples/epoch_050_sample.png")
e100 = cv2.imread("experiments/04_hazematching/outputs/samples/epoch_100_sample.png")
e200 = cv2.imread("experiments/04_hazematching/outputs/samples/epoch_200_sample.png")

prog_imgs = [e10, e50, e100, e200]
prog_titles = [
    "Epoch 10 (PSNR: 15.99 dB)\nCoarse Desmoking",
    "Epoch 50 (PSNR: 24.59 dB)\nContrast Recovery",
    "Epoch 100 (PSNR: 28.52 dB)\nTexture Refinement",
    "Epoch 200 (PSNR: 33.08 dB)\nFull Vessel & Margin Detail"
]

fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), dpi=300)
for ax, img, title in zip(axes, prog_imgs, prog_titles):
    if img is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#1B365D")
    ax.axis("off")

plt.tight_layout()
fig4_path = "scratch/fig4_epoch_progression.png"
plt.savefig(fig4_path, bbox_inches="tight", dpi=300)
plt.close()
print("Saved:", fig4_path)

# -------------------------------------------------------------
# Figure 5: Test Cases Case Studies (HazeMatching)
# -------------------------------------------------------------
cases = ["0005", "0018", "0040"]
case_titles = [
    "Case 1: Deep Cavity Diffuse Smoke",
    "Case 2: Dense Surgical Smoke Obscuration",
    "Case 3: Surface Tissue Margin & Vascular Clarity"
]

fig, axes = plt.subplots(3, 3, figsize=(11, 10.5), dpi=300)
for row, (c_idx, c_title) in enumerate(zip(cases, case_titles)):
    s_p = f"experiments/04_hazematching/outputs/test_images/smoky_{c_idx}.png"
    d_p = f"experiments/04_hazematching/outputs/test_images/dehazed_{c_idx}.png"
    c_p = f"experiments/04_hazematching/outputs/test_images/clean_{c_idx}.png"
    
    s_img = cv2.cvtColor(cv2.imread(s_p), cv2.COLOR_BGR2RGB)
    d_img = cv2.cvtColor(cv2.imread(d_p), cv2.COLOR_BGR2RGB)
    c_img = cv2.cvtColor(cv2.imread(c_p), cv2.COLOR_BGR2RGB)
    
    axes[row, 0].imshow(s_img)
    axes[row, 0].set_title(f"{c_title}\nSmoky Input", fontsize=10, fontweight="bold", color="#8B0000")
    axes[row, 0].axis("off")
    
    axes[row, 1].imshow(d_img)
    axes[row, 1].set_title(f"HazeMatching Dehazed\n(ODE Flow Matching)", fontsize=10, fontweight="bold", color="#008080")
    axes[row, 1].axis("off")
    
    axes[row, 2].imshow(c_img)
    axes[row, 2].set_title(f"Ground Truth\nClean Reference", fontsize=10, fontweight="bold", color="#1B365D")
    axes[row, 2].axis("off")

plt.tight_layout()
fig5_path = "scratch/fig5_test_cases.png"
plt.savefig(fig5_path, bbox_inches="tight", dpi=300)
plt.close()
print("Saved:", fig5_path)

# In-Vivo Laparoscopic Smoke Removal & Surgical Dehazing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **In-Vivo Laparoscopic Surgical Video Desmoking**: A comprehensive benchmark comparing **Generative Adversarial Networks (Pix2Pix, Attention U-Net, Cycle-Dehaze)** against **Continuous Conditional Flow Matching (HazeMatching, CVPR 2026)** for robust, hallucination-free surgical field restoration.

---

## 📌 1. Problem Statement

During minimally invasive laparoscopic, endoscopic, and robotic surgeries (such as cholecystectomy, colectomy, and gynaecological procedures), surgeons continuously utilize electrosurgical instruments, ultrasonic scalpels, and laser ablation tools for tissue cutting, dissection, and haemostasis. The rapid thermal vaporization of cellular fluids and organic matter generates dense, multi-phase **surgical smoke (plume)** containing water vapor, carbonized particulates, and aerosolized tissue debris.

### Clinical Challenges:
- **Severe Visibility Loss & Contrast Degradation**: Rayleigh and Mie light scattering inside the enclosed, insufflated peritoneal cavity attenuates illumination, causes heavy chromatic distortion, and obscures critical anatomical boundaries (such as capillary beds, organ margins, and sub-millimeter nerve bundles).
- **Elevated Surgical Risk**: Impaired visibility forces surgeons to repeatedly pause procedures to clean the lens or deploy mechanical evacuators, prolonging anesthesia exposure and elevating risks of accidental vascular or organ perforation.
- **Failure of Prior Methods**: Traditional outdoor defogging priors (e.g., Dark Channel Prior) fail due to non-uniform endoscope lighting and wet organ reflectance. Conversely, standard generative adversarial networks (GANs) risk **generative hallucinations** (fabricating artificial vessel boundaries or false textures), which is intolerable in clinical medicine.

**Objective**: Develop, evaluate, and benchmark deep generative and continuous flow architectures on custom paired in-vivo laparoscopic datasets to achieve real-time, high-contrast, artifact-free desmoking without structural hallucinations.

---

## 🔬 2. Approaches Used

We implemented and benchmarked four distinct deep learning paradigms:

```
                      ┌─────────────────────────────────────────────────────────────┐
                      │              In-Vivo Laparoscopic Desmoking                 │
                      └──────────────────────────────┬──────────────────────────────┘
                                                     │
         ┌───────────────────────────┬───────────────┴───────────────┬───────────────────────────┐
         │                           │                               │                           │
         ▼                           ▼                               ▼                           ▼
┌──────────────────┐       ┌──────────────────┐            ┌──────────────────┐        ┌──────────────────┐
│  Standard U-Net  │       │  Attention U-Net │            │   Cycle-Dehaze   │        │   HazeMatching   │
│    (Pix2Pix)     │       │ (Attention-Gated)│            │(Unpaired CycleGAN│        │  (Flow Matching) │
└──────────────────┘       └──────────────────┘            └──────────────────┘        └──────────────────┘
```

1. **Standard U-Net (Pix2Pix Baseline)**:
   - Supervised paired conditional GAN ($L_{\text{cGAN}} + 100 \cdot L_{\text{L1}}$).
   - 8-block symmetric encoder-decoder U-Net with skip connections and a $70 \times 70$ PatchGAN discriminator.

2. **Attention-Based U-Net (Attention-Gated Pix2Pix)**:
   - Supervised paired conditional GAN ($L_{\text{cGAN}} + 100 \cdot L_{\text{L1}}$).
   - Introduces **Additive Spatial Attention Gates** (Oktay et al.) on all skip connections. Coarser semantic gating signals dynamically modulate skip features before concatenation with decoder layers.

3. **Cycle-Dehaze (Unpaired Bidirectional CycleGAN)**:
   - Unsupervised bidirectional translation using dual 9-block ResNet generators ($G: \text{smoky} \to \text{clean}$, $F: \text{clean} \to \text{smoky}$) and dual PatchGAN discriminators.
   - Optimized via Adversarial loss ($L_{\text{LSGAN}}$), Cycle-Consistency ($L_{\text{cyc}}$), and **Cyclic Perceptual-Consistency Loss** ($L_{\text{vgg}}$) extracted from intermediate VGG-16 feature layers.

4. **HazeMatching (Continuous Conditional Flow Matching — Ours)**:
   - Continuous Normalizing Flow (CNF) framework based on Optimal Transport.
   - Uses an ultra-compact **CCFMUNet (3.70M parameters)** to regress a straight target velocity vector field $u_t = x_1 - x_0$ along a continuous probability path $x_t = (1-t)x_0 + tx_1$.
   - Incorporates empirical per-channel Z-score standardization and multi-path Forward Euler ODE integration with **Bayesian Minimum Mean Square Error (MMSE)** ensembling.

---

## 💡 3. Rationale & Motivation Behind Each Approach

| Approach | Why This Approach Was Chosen | Strengths & Trade-offs |
|---|---|---|
| **Standard U-Net (Pix2Pix)** | Established the benchmark baseline for paired image-to-image translation. Direct skip connections preserve low-level spatial geometry and organ topography across encoder-decoder bottlenecks. | **Strength**: Fast single-pass feedforward inference (~12 ms).<br>**Limitation**: Vulnerable to slight global color tone shifts and edge blurriness in thick smoke plumes. |
| **Attention-Based U-Net** | Overcomes standard U-Net limitations by dynamically filtering skip-connection features. Spatial attention gates learn to suppress non-smoky background tissue while concentrating gradients on smoke-occluded regions. | **Strength**: Sharpened anatomical contours, reduced background noise, and enhanced vascular contrast.<br>**Limitation**: High parameter count (57.4M) and occasional residual artifacts around heavy smoke boundaries. |
| **Cycle-Dehaze** | Explores unpaired learning to evaluate whether model training can succeed without requiring strictly registered clean/smoky frame pairs (which are difficult to acquire simultaneously in vivo). | **Strength**: Operates on unpaired clinical collections without exact ground truth.<br>**Limitation**: Lower reconstruction accuracy (18.84 dB PSNR) and noticeable chromatic distortion due to the lack of pixel-level spatial constraints. |
| **HazeMatching (Flow Matching)** | Solves the core vulnerabilities of both GANs (mode collapse, training instability, generative hallucinations) and Diffusion Models (curved trajectories, slow hundreds-step sampling). Straight optimal-transport velocity paths guarantee smooth, convex convergence, while multi-path MMSE averaging cancels out stochastic noise to eliminate false tissue hallucinations. | **Strength**: **Top performer (+5.19 dB PSNR gain, 4.7x lower perceptual error)**, 15x fewer parameters (3.70M vs. 57M), zero generative hallucinations.<br>**Trade-off**: Requires numerical ODE solver steps during inference. |

---

## 📊 4. Quantitative Model Comparison

The table below presents a comprehensive benchmark across all four evaluated models on our in-vivo laparoscopy test dataset:

| Parameter / Metric | Standard U-Net (No Attn) | Attention-Based U-Net | Cycle-Dehaze | HazeMatching (Ours) |
|---|:---:|:---:|:---:|:---:|
| **Supervision Mode** | Supervised (Paired) | Supervised (Paired) | Unsupervised (Unpaired) | **Supervised Flow Matching** |
| **Generator Architecture** | 8-Block U-Net | Attention-Gated U-Net | Dual 9-Block ResNet | **CCFMUNet (Flow Matching)** |
| **Model Parameters** | 57.2 M | 57.4 M | 28.3 M | **3.70 M** |
| **Training Epochs** | 60 Epochs | 100 Epochs | 100 Epochs | **200 Epochs** |
| **Batch Size** | 16 | 16 | 8 | **16** |
| **Normalization Method** | Linear [0, 1] scaling | Linear [0, 1] scaling | InstanceNorm + [0, 1] | **Empirical Z-Score (Per-Channel)** |
| **Test PSNR (dB)** | 27.89 dB | 27.68 dB | 18.84 dB | **33.0780 dB** |
| **Test SSIM** | **0.8736** | 0.8511 | 0.6441 | 0.8026 |
| **Test LPIPS (Lower is better)** | 0.0508 | 0.0737 | 0.2779 | **0.0108** |

### Key Findings:
- **Decisive Performance Dominance**: HazeMatching achieves **33.0780 dB PSNR** and an ultra-low **0.0108 LPIPS**, significantly outperforming all GAN configurations.
- **Ultra-Compact Footprint**: CCFMUNet utilizes only **3.70M parameters** (over 15x fewer weights than the 57M U-Net variants), drastically lowering memory storage requirements.
- **Hallucination-Free Restoration**: Multi-path MMSE averaging ($S = 50$ ODE paths) suppresses random variations, preserving genuine tissue morphology and specular reflections without hallucinating artificial blood vessels or organ contours.

---

## 🗂️ Repository Structure

```
InVivo-Laproscopic/
├── data/                               # Prepared dataset splits (train_crop, val_crop, test)
├── data_final/                         # Raw paired datasets (clean/ & smoky/)
├── experiments/
│   ├── 01_without_attention/           # Standard U-Net (Pix2Pix baseline)
│   ├── 02_attention_unet/              # Attention-Gated U-Net
│   ├── 03_cycle_dehaze/                # Cycle-Dehaze (CycleGAN + VGG Perceptual Loss)
│   └── 04_hazematching/                # HazeMatching (Conditional Continuous Flow Matching)
│       └── outputs/
│           ├── plots/                  # loss_curve.png, metrics_curve.png
│           ├── samples/                # Intermediate validation sample grids
│           └── test_images/            # Final test reconstructions (smoky / dehazed / clean)
│
├── src/
│   ├── logger.py                       # Unified logging system
│   ├── metrics.py                      # PSNR, SSIM, LPIPS evaluation engine
│   ├── hazematching/                   # Flow Matching dataset, trainer, and evaluator
│   └── models/
│       ├── shared/                     # PatchGAN discriminator, VGG perceptual loss, dataset loader
│       ├── standard/                   # Standard U-Net generator & Pix2Pix trainer
│       ├── attention/                  # Attention-gated U-Net generator
│       └── cycle_dehaze/               # Cycle-Dehaze ResNet generator & CycleGAN trainer
│
├── train.py                            # Training entry point (GAN models)
├── train_hazematching.py               # Training entry point (HazeMatching)
├── evaluate.py                         # Evaluation entry point (GAN models)
├── evaluate_hazematching.py            # Evaluation entry point (HazeMatching)
├── run_pipeline.py                     # Unified master pipeline runner
├── requirements.txt                    # Project dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Sanjana006/InVivo-Laparascopic.git
cd InVivo-Laparascopic
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Linux / macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies & PyTorch
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Training & Evaluation

### Option A: Unified Pipeline Runner (`run_pipeline.py`)

Train and evaluate any architecture with a single command:

```bash
# 1. HazeMatching (Conditional Continuous Flow Matching)
python run_pipeline.py --model_type hazematching --epochs 200 --data_dir data/laparoscopy

# 2. Attention-Based U-Net
python run_pipeline.py --model_type attention --epochs 100 --augment

# 3. Standard U-Net (No Attention)
python run_pipeline.py --model_type standard --epochs 100

# 4. Cycle-Dehaze (CycleGAN)
python run_pipeline.py --model_type cycle_dehaze --epochs 100 --augment
```

### Option B: Dedicated Training & Evaluation Scripts

#### 1. HazeMatching (Top Performer)
```bash
# Data preparation (creates train_crop, val_crop, test splits)
python -m src.hazematching.data_prep --src-dir data_final --dest-dir data/laparoscopy

# Train HazeMatching for 200 epochs
python train_hazematching.py --epochs 200 --batch_size 16

# Evaluate HazeMatching on unseen test set (20 ODE steps, 50 MMSE samples)
python evaluate_hazematching.py --split test --num_steps 20 --n_samples 50
```

#### 2. GAN-Based Architectures
```bash
# Train Attention U-Net
python train.py --model_type attention --epochs 100 --augment

# Evaluate Attention U-Net
python evaluate.py --model_type attention

# Train & Evaluate Cycle-Dehaze
python train.py --model_type cycle_dehaze --epochs 100 --augment
python evaluate.py --model_type cycle_dehaze
```

---

## 👥 Contributors

This project was conducted as part of academic research under the guidance of **Dr. Srimanta Mandal**.

- **Sanjana Nathani** — [@Sanjana006](https://github.com/Sanjana006)
- **Gaurang Jadav** — [@GaUrAnGjJ](https://github.com/GaUrAnGjJ)

---

## 📚 References & Acknowledgments

- **HazeMatching**: *HazeMatching: Conditional Continuous Flow Matching for Haze Removal* (CVPR 2026).
- **Cycle-Dehaze**: Engin, D., Genç, A., & Kemal Ekenel, H. (2018). *Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing*. CVPR Workshops.
- **Attention U-Net**: Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas*. MIDL.
- **Pix2Pix**: Isola, P. et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR.

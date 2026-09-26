# InVivo Laparoscopic Smoke Removal — CycleGAN / Pix2Pix GAN

> **Surgical smoke dehazing** using paired and unpaired GAN architectures for in-vivo laparoscopic video frames.  
> Based on the Cycle-Dehaze paper (Engin et al., 2018) with custom Attention U-Net generators and Cyclic Perceptual-Consistency Loss.

---

## 📊 Results

| Model | PSNR (dB) | SSIM |
|---|---|---|
| Standard U-Net (Pix2Pix) | — | — |
| Attention U-Net (Pix2Pix) | **29.87** | **0.9102** |
| Cycle-Dehaze (Ours) | 18.64 | 0.6345 |
| Cycle-Dehaze Paper (Engin et al.) | 15.41 | 0.66 |

---

## 🗂️ Project Structure

```
InVivo-Laproscopic/
├── data_final/                     # Training dataset (NOT committed — see Data Setup)
│   ├── clean/                      # Ground truth clean frames
│   └── smoky/                      # Smoky/hazy frames
│
├── experiments/
│   ├── 01_without_attention/       # Standard U-Net (Pix2Pix baseline)
│   ├── 02_attention_unet/          # Attention U-Net (Pix2Pix + Attention Gates)
│   └── 03_cycle_dehaze/            # Cycle-Dehaze (Unpaired CycleGAN)
│       ├── G_final.pth             # Final trained weights (NOT committed)
│       ├── checkpoints/            # Periodic epoch checkpoints (NOT committed)
│       └── outputs/
│           ├── test_images/        # dehazed_XXXX / smoky_XXXX / clean_XXXX
│           ├── samples/            # Mid-training sample grids
│           └── plots/              # loss_curve.png / metrics_curve.png / metrics.csv
│
├── src/
│   ├── logger.py                   # Unified logger + stdout/stderr redirection
│   ├── metrics.py                  # PSNR, SSIM, LPIPS evaluators
│   └── models/
│       ├── shared/
│       │   ├── discriminator.py    # PatchGAN-70 (conditional + unconditional)
│       │   ├── perceptual_loss.py  # VGG16 Cyclic Perceptual-Consistency Loss
│       │   └── smoke_dataset.py    # Paired SmokeDataset with augmentation
│       ├── standard/
│       │   ├── generator.py        # Standard U-Net generator (Pix2Pix baseline)
│       │   └── trainer.py          # Pix2Pix GAN training loop
│       ├── attention/
│       │   └── generator.py        # Attention U-Net generator (Oktay et al., 2018)
│       └── cycle_dehaze/
│           ├── generator.py        # Cycle-Dehaze Attention generator (InstanceNorm + Tanh)
│           └── trainer.py          # CycleGAN training loop (LSGAN + Cycle + Perceptual)
│
├── train.py                        # Training entry point
├── evaluate.py                     # Evaluation + test image export
├── test.py                         # Single-image quick test
├── run_pipeline.py                 # Full pipeline runner (train → evaluate)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/InVivo-Laproscopic.git
cd InVivo-Laproscopic

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch (visit https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📁 Data Setup

The dataset is **not committed** to this repository due to size.  
Place your paired images inside `data_final/`:

```
data_final/
├── clean/    # e.g. 001.png, 002.png, ...
└── smoky/    # same filenames as clean/
```

Images must share **identical filenames** (stems) across `clean/` and `smoky/`.  
The dataset is split **87.5% train / 12.5% test** automatically.

---

## 🚀 Training

### Attention U-Net (Supervised Pix2Pix)
```bash
python train.py --model_type attention --epochs 100 --augment
```

### Cycle-Dehaze (Unpaired CycleGAN)
```bash
python train.py --model_type cycle_dehaze --epochs 100 --augment
```

### Standard U-Net (Pix2Pix Baseline)
```bash
python train.py --model_type standard --epochs 100
```

### Full Pipeline (train + evaluate in one command)
```bash
python run_pipeline.py --model_type cycle_dehaze --epochs 100 --augment
```

### Resume from checkpoint
```bash
python train.py --model_type cycle_dehaze --epochs 100 \
    --resume experiments/03_cycle_dehaze/checkpoints/epoch_50.pth
```

---

## 📈 Evaluation

```bash
# Evaluate Cycle-Dehaze on test split
python evaluate.py --model_type cycle_dehaze

# Evaluate Attention U-Net
python evaluate.py --model_type attention

# Custom checkpoint path
python evaluate.py --model_type cycle_dehaze \
    --checkpoint experiments/03_cycle_dehaze/G_final.pth
```

Output images are saved to `experiments/<exp>/outputs/test_images/`:
- `smoky_XXXX.png` — hazy input
- `dehazed_XXXX.png` — model output
- `clean_XXXX.png` — ground truth

---

## 🧪 Single Image Test

```bash
python test.py
# Edit MODEL_TYPE and img_path at the top of test.py
```

---

## 🏗️ Architecture Details

### Attention U-Net Generator
- Encoder: 8× strided Conv2d (BatchNorm + LeakyReLU)
- Decoder: 7× ConvTranspose2d + Dropout (first 3 blocks)
- Skip connections: **Additive Attention Gates** (Oktay et al., 2018)
- Output: Sigmoid → [0, 1]

### Cycle-Dehaze Generator
- Same encoder-decoder as Attention U-Net
- **InstanceNorm2d** replaces BatchNorm throughout (CycleGAN convention)
- Attention Gates use InstanceNorm
- Output: **Tanh** → native [-1, 1], clamped to [0, 1] at inference

### Discriminator (Shared)
- PatchGAN-70 (Isola et al., 2017)
- Supports both conditional (Pix2Pix, 6-ch input) and unconditional (CycleGAN, 3-ch) modes
- Supports BatchNorm or InstanceNorm

### Cycle-Dehaze Losses
| Loss | Weight |
|---|---|
| Adversarial (LSGAN / MSE) | 1.0 |
| Cycle-Consistency (L1) | λ = 10.0 |
| Identity Loss (L1) | λ_id = 5.0 |
| Cyclic Perceptual-Consistency (VGG16 L2) | γ = 0.0001 |

---

## 📚 References

- **Cycle-Dehaze**: Engin, D., Genç, A., & Kemal Ekenel, H. (2018). *Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing*. CVPR Workshops.
- **CycleGAN**: Zhu, J. et al. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. ICCV.
- **Pix2Pix**: Isola, P. et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR.
- **Attention U-Net**: Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas*. MIDL.

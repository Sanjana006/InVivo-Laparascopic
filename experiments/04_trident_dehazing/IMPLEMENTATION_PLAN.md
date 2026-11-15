# Implementation Plan: Trident Dehazing Network (TDN) for In-Vivo Laparoscopic Desmoking

> **Paper Reference**: *Trident Dehazing Network* (Liu et al., CVPRW 2020)  
> **Target Folder**: `experiments/04_trident_dehazing/`  
> **Domain**: In-Vivo Minimally Invasive Laparoscopic Surgery  

---

## 1. Executive Summary

This implementation adapts the **Trident Dehazing Network (TDN)** from natural nonhomogeneous dehazing to surgical desmoking in laparoscopy. The network solves the problem of high-frequency detail loss and non-uniform smoke density through a coarse-to-fine trident architecture:
1. **Method 1 (EDN)**: Reconstructs coarse semantic anatomy via deep pretrained features.
2. **Method 2 (DRN)**: Recovers lost high-frequency textures and edges via Wide Activation Blocks.
3. **Method 3 (HDMGN)**: Dynamically predicts smoke density maps without external supervision.

### Required Customizations:
- **Attention-Based Skip Connections**: Replace standard/Instance-Norm skip connections in EDN with **Additive Spatial Attention Gates** to suppress specular noise and highlight anatomical boundaries.
- **Modular Code Structure**: Clearly decouple and document all three methods for maximum readability and reusability.
- **Composite Loss Function**: Implement the exact tripartite objective: $\mathcal{L} = \alpha \mathcal{L}_s (\text{L1}) + \beta \mathcal{L}_f (\text{FFT Amplitude \& Phase}) + \gamma \mathcal{L}_l (\text{BReLU Threshold Limitation})$.
- **Dual Pretrained Backbones**:
  - **Backbone 1**: Pretrained **DPN92** (ImageNet, as in original paper).
  - **Backbone 2**: Pretrained **EndoViT** ([DominikBatic/EndoViT](https://github.com/DominikBatic/EndoViT)), self-supervised on 700k+ endoscopic frames, paired with a multiscale spatial adapter.

---

## 2. Trident Dehazing Network Architecture

```
                          ┌────────────────────────────────────────────────────────┐
                          │            Hazy / Smoky Input Image (X)                │
                          └───────┬───────────────────┬───────────────────┬────────┘
                                  │                   │                   │
                                  ▼                   ▼                   ▼
    ┌───────────────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────────┐
    │     Method 1: EDN             │   │     Method 2: DRN         │   │     Method 3: HDMGN           │
    │  Encoder-Decoder Sub-Net      │   │  Detail Refinement Net    │   │  Haze Density Map Net         │
    │  • Backbone: DPN92 / EndoViT  │   │  • Space-to-Depth (4x)    │   │  • 6-level U-Net              │
    │  • Attention Skip Connections │   │  • 3 Wide Activation Blks │   │  • Auto density estimation    │
    │  • 5 Deformable Upsampling    │   │  • Residual Scaling (0.2) │   │  • No external GT supervision │
    │    Blocks (DUB)               │   │  • Depth-to-Space (4x)    │   │    required                   │
    └───────────────┬───────────────┘   └─────────────┬─────────────┘   └───────────────┬───────────────┘
                    │                                 │                                 │
                    │ [Coarse Features]               │ [High-Freq Details]             │ [Density Map]
                    └─────────────────────────┬───────┴─────────────────────────────────┘
                                              ▼
                                ┌───────────────────────────┐
                                │   Channel Concatenation   │
                                └─────────────┬─────────────┘
                                              ▼
                                ┌───────────────────────────┐
                                │ Deformable Conv Tail +    │
                                │ BReLU Output Layer [0, 1] │
                                └─────────────┬─────────────┘
                                              ▼
                                ┌───────────────────────────┐
                                │  Restored Clean Output    │
                                └───────────────────────────┘
```

---

## 3. Sub-Net Detailed Design

### Method 1: Encoder-Decoder sub-Net (EDN)
* **Role**: Captures semantic context and reconstructs coarse desmoked feature representations.
* **Encoder**:
  - Supports configurable backbones (`dpn92` or `endovit`).
  - Pretrained weights act as strong feature priors, especially critical under limited clinical paired data.
* **Decoder**:
  - Built with **5 Deformable Upsampling Blocks (DUB)**.
  - Structure of each DUB:
    1. Input feature $\to$ $3 \times 3$ Deformable Convolution block $\to$ concatenate with output feature.
    2. Concatenated feature $\to$ $1 \times 1$ Deformable Convolution block $\to$ Nearest Upsampling $2\times$.
* **Attention Skip Connections (Modification 1)**:
  - Connects encoder intermediate stages (scale $1/4$ and scale $1/8$) to corresponding DUB stages.
  - Encoder skip features $x$ are filtered by decoder gating signal $g$ through an **Additive Attention Gate**:
    $$\alpha = \sigma\left(\psi^T(\text{ReLU}(W_g g + W_x x + b))\right), \quad \hat{x} = x \odot \alpha$$
  - Filters out diffuse smoke patterns before passing spatial details to the decoder.

### Method 2: Detail Refinement sub-Net (DRN)
* **Role**: Recovers delicate high-frequency structures (e.g., microvasculature, tissue margins) lost due to downsampling/pooling in EDN.
* **Architecture**:
  1. **Space-to-Depth (Downsample $4\times$)**: Converts spatial resolution into channels ($3 \times H \times W \to 48 \times \frac{H}{4} \times \frac{W}{4}$).
  2. **Non-Linear Feature Mapping**: 3 consecutive **Wide Activation Blocks (WAB)**:
     - $3 \times 3$ Conv $\to$ BatchNorm $\to$ Wide Activation (ReLU with channel expansion factor 4) $\to$ $3 \times 3$ Conv $\to$ BatchNorm.
     - **Residual Scaling**: Residual multiplied by constant $\beta = 0.2$ before addition:
       $$F_b = F_{b-1} + 0.2 \times \mathcal{F}_{\text{WAB}}(F_{b-1})$$
  3. **Depth-to-Space (Upsample $4\times$)**: Pixel-shuffle layer restructuring deep features back to spatial resolution without blur.

### Method 3: Haze Density Map Generation sub-Net (HDMGN)
* **Role**: Automatically models local smoke density variations without extra ground truth annotations.
* **Architecture**:
  - Symmetric 6-downsampling / 6-upsampling U-Net (requires input dimensions divisible by 64).
  - Encoder: Strided $4 \times 4$ Convolutions + LeakyReLU + BatchNorm.
  - Decoder: Transposed $4 \times 4$ Convolutions (stride 2) + ReLU + BatchNorm + skip concatenations.
  - Refinement tail: $3 \times 3$ Convolution generating a 3-channel density map.

### Trident Fusion & Tail Block
- Concatenates outputs from EDN, DRN, and HDMGN.
- Deformable $3 \times 3$ convolution offsets learn to ignore extreme visual outliers (specular glare, blood glints) and dynamically fuse the three feature representations.
- Bounded ReLU (BReLU) / clamping enforces valid pixel intensity range $[0, 1]$.

---

## 4. Dual Pretrained Backbones

| Feature | Backbone 1: DPN92 | Backbone 2: EndoViT |
|---|---|---|
| **Source** | Original TDN Paper (ImageNet-1K pretrained) | Dominik Batić et al. / Hugging Face (`egeozsoy/EndoViT`) |
| **Domain** | Natural images (Dual Path Network) | **Endoscopy specific** (700k+ surgical frames across 9 datasets) |
| **Feature Extraction** | Multi-branch: ResNet residual path + DenseNet feature concatenation | Vision Transformer (ViT-Base, patch size 16) |
| **Decoder Interfacing** | Direct hierarchical features at $1/4, 1/8, 1/16, 1/32$ | **Multiscale Token Adapter**: Reshapes patch tokens from transformer blocks (3, 6, 9, 12) $\to$ 2D spatial maps $\to$ Projected to $1/4, 1/8, 1/16$ |
| **Switching Method** | `--backbone dpn92` | `--backbone endovit` |

---

## 5. Loss Function Formulation ($L_1 + \text{FFT} + \text{BReLU}$)

The overall optimization objective balances spatial reconstruction, frequency domain detail, and dynamic range stability:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_s + \beta \mathcal{L}_f + \gamma \mathcal{L}_l$$

*(Default paper parameters: $\alpha = 0.5, \beta = 0.5, \gamma = 1.0$)*

### 1. Spatial Domain Loss ($\mathcal{L}_s$)
Direct pixel-level discrepancy between predicted clean frame $\hat{I}$ and ground truth $I^{gt}$:
$$\mathcal{L}_s = \frac{1}{N}\sum_{i=1}^N \|I_i^{gt} - \hat{I}_i\|_1$$

### 2. Frequency Domain Loss ($\mathcal{L}_f$)
Supervises Fourier amplitude and phase representations to enforce high-frequency restoration across all frequency bands:
$$A^{gt}, P^{gt} = \text{FFT2D}(I^{gt}), \quad A^{pred}, P^{pred} = \text{FFT2D}(\hat{I})$$
$$\mathcal{L}_f = \frac{1}{N}\sum_{i=1}^N \left(\|A_i^{gt} - A_i^{pred}\|_1 + \|P_i^{gt} - P_i^{pred}\|_1\right)$$
- Computed using `torch.fft.rfft2` for real-valued signals, executing efficiently on GPU.

### 3. Threshold Limitation Loss ($\mathcal{L}_l$)
Penalizes out-of-boundary reconstruction values before the BReLU non-linearity, smoothly constraining dynamic range:
$$\mathcal{L}_l = \frac{1}{N}\sum_{i=1}^N \log\left(\|\text{clamp}(\tilde{I}_i, 0, 1) - \tilde{I}_i\|_1 + 1\right)$$
where $\tilde{I}_i$ is the raw unconstrained model output.

---

## 6. Directory & Code Organization

The entire module will be created under `experiments/04_trident_dehazing/`:

```
experiments/04_trident_dehazing/
├── IMPLEMENTATION_PLAN.md             # This comprehensive blueprint document
├── config.py                          # Hyperparameters, data paths, model/backbone configurations
├── dataset.py                         # Paired laparoscopic dataset loader with augmentations
│
├── models/
│   ├── __init__.py                    # Exported model classes
│   ├── blocks.py                      # DeformableConv2d, AttentionGate, DUB, WAB modules
│   ├── edn.py                         # Method 1: Encoder-Decoder Net with Attention Skips
│   ├── drn.py                         # Method 2: Detail Refinement Net (Space-to-Depth + WABs)
│   ├── hdmgn.py                       # Method 3: Haze Density Map Generation Net (6-level U-Net)
│   ├── backbones/
│   │   ├── __init__.py
│   │   ├── dpn92_backbone.py          # DPN92 loader and hierarchical feature extractor
│   │   └── endovit_backbone.py        # EndoViT loader and multiscale spatial token adapter
│   └── trident_model.py               # Complete TDN combining EDN, DRN, HDMGN, and Fusion Tail
│
├── losses/
│   ├── __init__.py
│   └── composite_loss.py              # Implementation of L1 + FFT + BReLU loss
│
├── train.py                           # Training routine with checkpointing and metric tracking
├── evaluate.py                        # Benchmark evaluation (PSNR, SSIM, LPIPS)
└── demo_infer.py                      # Single image / batch inference script
```

---

## 7. Execution & Verification Roadmap

| Phase | Milestone | Deliverable |
|---|---|---|
| **Phase 1** | **Core Modules & Blocks** | `blocks.py`: Implement Deformable Convolution 2D wrapper, Attention Gate, DUB, and WAB with residual scaling. |
| **Phase 2** | **Three Sub-Nets** | `edn.py`, `drn.py`, `hdmgn.py`: Implement Methods 1, 2, and 3 with clear class docstrings. |
| **Phase 3** | **Dual Backbones** | `dpn92_backbone.py` and `endovit_backbone.py` with multi-scale adapter. |
| **Phase 4** | **Trident Integration & Loss** | `trident_model.py` and `composite_loss.py` ($L_1 + \text{FFT} + \text{BReLU}$). |
| **Phase 5** | **Training & Evaluation Scripts** | `train.py`, `evaluate.py`, and `dataset.py` linked to existing laparoscopic data. |
| **Phase 6** | **Pipeline Verification** | Forward/backward pass test script validating tensor shapes, loss calculation, and backpropagation for both DPN92 and EndoViT. |

---


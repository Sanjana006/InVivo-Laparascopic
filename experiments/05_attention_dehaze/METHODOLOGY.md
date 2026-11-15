# Methodology: Attention-Guided Conditional Flow Matching Dehazing (RGB)

## 1. Overview & Theoretical Motivation

This experiment (`experiments/05_attention_dehaze`) extends **HazeMatching** (*Ray et al., CVPR 2026 Findings*) to **3-channel RGB laparoscopic desmoking** with **Attention-Gated Skip Connections**.

### Key Limitations in Prior Work (`juglab/HazeMatching`):
1. **Single-channel Grayscale Constraint:** The original HazeMatching repository was designed for microscopy imaging, processing single-channel black-and-white images. Medical laparoscopy critically relies on RGB color fidelity for tissue differentiation, liver parenchyma boundaries, and vascular identification.
2. **Standard Concatenation Skip Connections:** The original repository uses standard U-Net skip connections ($\text{cat}([h, \text{skip})]$), which blindly forward smoke artifacts, specular reflections, and diffused haze from early encoder layers directly into the decoder.

### Proposed Innovations:
1. **Full 3-Channel RGB Pipeline:** Direct processing of 3-channel RGB laparoscopic images ($6$ input channels for $[x_t, x_{\text{cond}}]$, $3$ output channels for velocity $u_t$).
2. **Attention-Gated Skip Connections:** Integration of Additive Attention Gates (*Oktay et al., 2018*) on all encoder-to-decoder skip connections. The decoder features serve as a gating signal $g$ that dynamically weights encoder skip features $x$, suppressing smoke haze while preserving tissue borders and surgical tool contours.
3. **Continuous-Time Conditional Flow Matching (CFM):** Straight flow trajectories connecting standard Gaussian noise $x_0 \sim \mathcal{N}(0, I)$ at $t=0$ to clean target laparoscopy $x_1$ at $t=1$, offering fast, non-autoregressive ODE sampling and calibrated uncertainty estimation.

---

## 2. Mathematical Formulation

### 2.1 Conditional Flow Matching (CFM)
Let $x_1 \in \mathbb{R}^{3 \times H \times W}$ denote the clean ground-truth laparoscopic image, and $y = x_{\text{cond}} \in \mathbb{R}^{3 \times H \times W}$ denote the paired smoky observation.

At time $t \in [0, 1]$, we define a straight probability path connecting base Gaussian noise $x_0 \sim \mathcal{N}(0, I)$ to $x_1$:
$$x_t = (1 - t) x_0 + t x_1$$

The corresponding true conditional vector field (velocity) generating this probability flow is:
$$u_t(x_t | x_0, x_1) = \frac{d x_t}{dt} = x_1 - x_0$$

The neural network $v_\theta(t, [x_t, y])$ is parameterized as an Attention U-Net that predicts the velocity field conditioned on time $t$, noisy state $x_t$, and smoky observation $y$. The training objective is the Flow MSE Loss:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0, 1), x_0 \sim \mathcal{N}(0, I), (x_1, y)} \left[ \left\| v_\theta(t, [x_t, y]) - (x_1 - x_0) \right\|_2^2 \right]$$

---

### 2.2 Attention Gate on Skip Connections
In a standard U-Net, the encoder features $x \in \mathbb{R}^{F_l \times H \times W}$ are concatenated directly with the upsampled decoder features $g \in \mathbb{R}^{F_g \times H \times W}$. In laparoscopic surgery, early encoder features contain heavy smoke scatter that impairs dehazed reconstruction.

We place an **Additive Attention Gate** on each skip connection before concatenation:
1. **Intermediate Projection:**
   $$W_g(g) \in \mathbb{R}^{F_{\text{int}} \times H \times W}, \quad W_x(x) \in \mathbb{R}^{F_{\text{int}} \times H \times W}$$
2. **Additive Combination & Activation:**
   $$\psi = \text{ReLU}\left( W_g(g) + W_x(x) \right)$$
3. **Attention Map Generation:**
   $$\alpha = \sigma\left( \text{Conv}_{1 \times 1}(\psi) \right) \in [0, 1]^{1 \times H \times W}$$
4. **Gated Feature Map:**
   $$\hat{x} = \alpha \odot x$$
5. **Decoder Concatenation:**
   $$h_{\text{dec}} = [g, \hat{x}] \in \mathbb{R}^{(F_g + F_l) \times H \times W}$$

The scalar attention weight $\alpha_{i,j}$ approaches $1$ in smoke-free regions with high anatomical detail (vascular trees, organ boundaries) and approaches $0$ in dense smoke zones, preventing hazy noise from polluting the reconstruction.

---

### 2.3 Posterior ODE Sampling & Uncertainty Estimation
During inference, given a smoky image $y$, we sample $K$ independent initial states $x_0^{(k)} \sim \mathcal{N}(0, I)$ ($k = 1, \dots, K$) and solve the probability flow ODE from $t = 0$ to $t = 1$:
$$\frac{dx_t}{dt} = v_\theta(t, [x_t, y]), \quad x(0) = x_0^{(k)}$$

Using uniform Euler integration over $N$ steps ($\Delta t = 1 / N$):
$$x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(t, [x_t, y])$$

From the $K$ generated trajectories $\{x_1^{(1)}, \dots, x_1^{(K)}\}$, we compute:
1. **MMSE Dehazed Reconstruction (Posterior Mean):**
   $$\hat{x}_{\text{MMSE}} = \frac{1}{K} \sum_{k=1}^K x_1^{(k)}$$
2. **Epistemic / Generative Uncertainty Map:**
   $$\sigma(x) = \sqrt{\frac{1}{K} \sum_{k=1}^K \left( x_1^{(k)} - \hat{x}_{\text{MMSE}} \right)^2}$$

This provides laparoscopic surgeons with both a high-fidelity dehazed view and a confidence map highlighting challenging regions.

---

## 3. Comparison with Baseline `HazeMatching`

| Property | `juglab/HazeMatching` | `experiments/05_attention_dehaze` |
| :--- | :--- | :--- |
| **Color Channels** | Single-channel Grayscale (B&W) | **3-Channel RGB** |
| **Model Input Channels** | 2 ($x_t$ 1ch + $x_{\text{cond}}$ 1ch) | **6 ($x_t$ 3ch + $x_{\text{cond}}$ 3ch)** |
| **Model Output Channels**| 1 (velocity 1ch) | **3 (velocity 3ch)** |
| **Skip Connections** | Plain channel concatenation | **Attention-Gated Skip Connections (Oktay et al.)** |
| **Skip Attention** | None | **Additive Attention Gate on every level** |
| **ODE Sampling** | Requires external `torchcfm`/`torchdiffeq` | **Self-contained Euler & RK4 samplers** |
| **Target Domain** | Confocal fluorescence microscopy | **In vivo laparoscopic surgical desmoking** |

---

## 4. Module Directory Structure

```
experiments/05_attention_dehaze/
├── config.py                 # Hyperparameter configuration dataclass
├── dataset.py                # 3-channel RGB dataset loader & [-1, 1] normalization
├── flow_matching.py          # RGB CFM path construction, Euler/RK4 ODE & MMSE sampling
├── trainer.py                # AttentionDehazeTrainer with CFM loss & validation logging
├── train.py                  # Training CLI entrypoint
├── evaluate.py               # Evaluation CLI entrypoint (PSNR, SSIM, triplet saving)
├── METHODOLOGY.md            # Complete theoretical & architectural documentation
└── models/
    ├── __init__.py
    ├── attention_gate.py     # Additive Attention Gate module
    ├── time_embedding.py     # Sinusoidal time embedding & projection MLP
    └── unet_attention_cfm.py # Attention-Guided CFM U-Net backbone
```

---

## 5. Execution Workflow

> [!NOTE]
> Per instruction, execution of training or testing is deferred until you run the commands.

### Training:
```bash
python experiments/05_attention_dehaze/train.py --epochs 200 --batch_size 8 --lr 1e-4
```

### Evaluation (MMSE Inference):
```bash
python experiments/05_attention_dehaze/evaluate.py --checkpoint experiments/05_attention_dehaze/checkpoints/best.pth --n_samples 10
```

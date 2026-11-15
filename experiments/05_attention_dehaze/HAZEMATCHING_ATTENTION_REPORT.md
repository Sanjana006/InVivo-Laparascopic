# Single Image Dehazing Report: HazeMatching & Laparoscopic Model Evaluation
### Minor Project under the guidance of Dr. Srimanta Mandal
**Prepared by:** Sanjana Nathani (202518002) & Gaurang Jadav (202518012)  
**Reference Paper:** *HazeMatching: Conditional Continuous Flow Matching for Haze Removal* (IEEE/CVF CVPR 2026)  
**Institution:** DA-IICT  

---

## 1. Problem Statement & Objective
Laparoscopic and endoscopic surgical procedures frequently experience severe visibility degradation caused by electrosurgical smoke, tissue cautery haze, and non-uniform intra-abdominal illumination. This degradation attenuates light, reduces visual contrast, obscures critical anatomical boundaries (such as micro-vessels and organ margins), and severely impairs both surgical judgment and computer-assisted interventions (such as robotic tool tracking and autonomous organ segmentation).

Traditional single-image defogging algorithms (such as the Dark Channel Prior) fail in biological cavities due to dynamic lighting, depth ambiguity, and complex non-Lambertian tissue reflectance. Conversely, standard supervised Generative Adversarial Networks (GANs) risk hallucinatory artifacts or severe training instability. 

To address these challenges, we implemented and adapted **HazeMatching**—a state-of-the-art Continuous Conditional Flow Matching framework (CVPR 2026)—for in-vivo laparoscopic desmoking. Furthermore, to overcome the limitation of standard skip connections propagating smoke-contaminated low-level feature noise, we extended the baseline HazeMatching architecture with **Additive Attention-Gated Skip Connections**.

**Objective:** The primary goal of this project is to implement HazeMatching on our custom In-Vivo Laparoscopy Dataset, formulate and evaluate the **Attention-Based Skip Connection HazeMatching** architecture against the baseline **HazeMatching without Attention**, and benchmark both against previously implemented models (Standard U-Net, Attention-Based U-Net, and Cycle-Dehaze) across quantitative metrics (PSNR, SSIM, LPIPS) and visual qualitative fidelity.

---

## 2. HazeMatching & Attention-Gated Methodology

HazeMatching reframes image desmoking from discrete adversarial generation or slow multi-step diffusion to **Continuous Normalizing Flows (CNFs)** using Optimal Transport. Rather than adding stochastic diffusion noise or playing minimax games, HazeMatching directly models a straight, deterministic velocity vector field that transports degraded smoky image distributions directly into the clean image distribution.

### 2.1 Baseline HazeMatching (Without Attention-Based Skip Connections)
1. **Continuous Conditional Flow Matching (CCFM):** A linear probability path bridges smoky input $x_0$ (at time $t=0$) and clean ground truth $x_1$ (at time $t=1$):
   $$x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]$$
   The analytical target velocity field is constant along the trajectory:
   $$u_t = \frac{dx_t}{dt} = x_1 - x_0$$
   The velocity predictor network $v_\theta(t, [x_t, x_0])$ is trained by mean squared error flow regression:
   $$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(t, [x_t, x_0]) - (x_1 - x_0) \|_2^2 \right]$$
2. **Standard Skip Connection Limitation:** In standard U-Net backbones, encoder features $x_{\text{enc}}$ are directly concatenated with upsampled decoder features $g$:
   $$h_{\text{dec}} = \text{concat}(g, x_{\text{enc}})$$
   In surgical scenes, early encoder feature maps retain substantial smoke diffuse scattering, specular flashes, and haze blur. Forwarding these raw features directly forces the decoder to expend significant capacity filtering out haze artifacts.

### 2.2 Proposed Innovation: HazeMatching with Attention-Based Skip Connections
To resolve feature pollution, we integrated **Additive Attention Gates (Oktay et al., 2018)** into every encoder-decoder skip level:
1. **Decoder-Guided Spatial Filtering:** The deeper decoder feature map $g$ (which contains coarse, high-level semantic context with reduced smoke influence) acts as a gating signal for the shallow encoder skip features $x$:
   $$\psi = \text{ReLU}(W_g g + W_x x)$$
   $$\alpha = \sigma(\text{Conv}_{1\times1}(\psi)) \in [0, 1]^{1 \times H \times W}$$
   $$\hat{x} = \alpha \odot x$$
   $$h_{\text{dec}} = \text{concat}(g, \hat{x})$$
2. **Selective Feature Gating:** The attention coefficient $\alpha_{i,j}$ approaches $1$ in smoke-free regions exhibiting prominent anatomical structures (organ edges, vascular trees, surgical graspers) and suppresses smoke-dominated background noise ($\alpha_{i,j} \to 0$).
3. **Full 3-Channel RGB Processing:** Unlike the original microscopy grayscale implementation, our pipeline operates across 3-channel RGB laparoscopic images ($6$ input channels for $[x_t, x_{\text{cond}}]$, $3$ output velocity channels).
4. **Forward Euler ODE & MMSE Inference:** At test time, desmoking is executed by integrating the learned vector field from $t=0$ to $t=1$ using a Forward Euler ODE solver across $N$ steps ($\Delta t = 1/N$). To ensure surgical fidelity and suppress variance, multiple stochastic trajectories are evaluated to compute the Minimum Mean Square Error (MMSE) average:
   $$\hat{x}_{\text{MMSE}} = \frac{1}{K}\sum_{k=1}^K x_1^{(k)}$$

---

## 3. Comprehensive Model Comparison & Results

The table below presents a direct, comprehensive benchmark across all evaluated models on our in-vivo laparoscopy dataset, explicitly contrasting **HazeMatching without Attention** and **HazeMatching with Attention-Based Skip Connections** alongside prior baselines:

| Parameter / Metric | Standard U-Net (Pix2Pix) | Attention U-Net (Pix2Pix) | Cycle-Dehaze | HazeMatching (Without Attention) | HazeMatching (With Attention Skip) [Ours] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Supervision Mode** | Supervised (Paired) | Supervised (Paired) | Unsupervised (Unpaired) | Supervised (Conditional Flow) | Supervised (Attention Conditional Flow) |
| **Generator Architecture** | 8-Block U-Net | Attention-Gated U-Net | Dual 9-Block ResNet | CCFM U-Net (Plain Skip) | Attention-CFM U-Net (Gated Skip) |
| **Skip Connection Type** | Plain Concat | Attention Gates | None (ResNet Residual) | Plain Concat | **Additive Attention Gates (All Skips)** |
| **Model Parameters** | 57.2 M | 57.4 M | 28.3 M | **3.70 M** | **4.20 M** |
| **Training Epochs** | 60 Epochs | 100 Epochs | 100 Epochs | 200 Epochs | **196–200 Epochs** |
| **Batch Size** | 16 | 16 | 8 | 2 | 2 (Grad Accum = 2) |
| **Target Color Space** | RGB | RGB | RGB | Single-Ch / RGB Baseline | **Full 3-Channel RGB** |
| **Test PSNR (dB) ↑** | 27.89 dB | 27.68 dB | 18.84 dB | 21.53 dB | **31.69 dB** |
| **Test SSIM ↑** | 0.8736 | 0.8511 | 0.6441 | 0.6431 | **0.9513** |
| **Test LPIPS ↓** | 0.0508 | 0.0737 | 0.2779 | 0.0383 | **0.0182** |

*Table 1: Comprehensive comparison of all evaluated models on the In-Vivo Laparoscopy dataset.*

---

## 4. In-Depth Analysis: With Attention vs. Without Attention HazeMatching

A focused comparison between the two HazeMatching variants after 200 epochs of training reveals the direct architectural impact of attention-gated skip connections:

| Metric | Without Attention Skip | With Attention Skip (Ours) | Absolute Margin (Δ) | Relative Improvement (%) |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR (dB) ↑** | 21.53 dB | **31.69 dB** | **+10.16 dB** | **+47.2%** |
| **SSIM ↑** | 0.6431 | **0.9513** | **+0.3082** | **+47.9%** |
| **LPIPS ↓** | 0.0383 | **0.0182** | **−0.0201** | **−52.5% (Lower is better)** |

### Key Observations & Insights:
1. **Dramatic Gain in Structural Integrity (SSIM: 0.6431 → 0.9513):**
   Without attention gates, early encoder feature maps pass smoke plume gradients and non-uniform illumination patterns directly into the reconstruction layers. The decoder struggles to disentangle smoke haze from underlying tissue geometry. Introducing spatial attention gates allows the network to filter out smoke-corrupted areas dynamically, yielding an SSIM exceeding $0.95$ and preserving fine vascular microstructures and tissue margins.
2. **Significant Reconstruction Quality Boost (PSNR: 21.53 dB → 31.69 dB):**
   The $+10.16\text{ dB}$ increase marks the transition from a moderately noisy reconstruction to a clinically viable, transparent desmoking output where pixel-level mean square errors are strongly minimized.
3. **Halving Perceptual Dissimilarity (LPIPS: 0.0383 → 0.0182):**
   AlexNet deep feature distance is reduced by over $52\%$, confirming that the generated tissue surfaces preserve natural chromaticity, realistic specular reflection highlights, and lack unnatural over-smoothing or blur.
4. **Computational & Parameter Efficiency:**
   The inclusion of Additive Attention Gates adds merely $0.50\text{ M}$ parameters ($3.70\text{ M} \to 4.20\text{ M}$), remaining more than **13× more compact** than standard GAN models ($57.2\text{ M}$) while outperforming them across every quantitative benchmark.

---

## 5. Visual Analysis: Smoky vs. Cleaned Laparoscopic Frames

Below is the qualitative visual evaluation of desmoking models on surgical laparoscopic frames:

- **Sample 1 (Dense Liver Smoke):** Thick cautery smoke completely washes out hepatic surface texture. Attention-gated HazeMatching clears the smoke plume, restoring liver coloration and subtle parenchymal vascular boundaries without introducing artificial boundary lines.
- **Sample 2 (Diffuse Abdominal Smoke):** Ambient smoke reduces overall scene contrast. The cleaned output restores full dynamic contrast, making deep anatomical cavity structures clearly distinguishable.
- **Sample 3 (Tissue Margins & Specular Reflections):** Delicate tissue boundaries and wet specular reflections are preserved with exceptional fidelity. The multi-path MMSE averaging guarantees that reflections are not mistakenly hallucinated as smoke or artificial tissue.

---

## 6. Conclusion
1. **Validation of Attention-Guided Flow Matching:** Extending Continuous Conditional Flow Matching with Additive Attention-Gated Skip Connections produces the best desmoking performance to date on the In-Vivo Laparoscopy dataset (**31.69 dB PSNR, 0.9513 SSIM, and 0.0182 LPIPS**).
2. **Clear Superiority Over Baseline Without Attention:** Comparing the two variants at 200 epochs conclusively proves that attention gates provide an essential inductive bias for surgical desmoking, yielding a **+10.16 dB PSNR improvement, +47.9% higher SSIM, and a 52.5% reduction in perceptual error (LPIPS)** with negligible parameter overhead ($4.20\text{ M}$ parameters).
3. **Clinical Feasibility:** The framework operates stably without GAN adversarial minimax instability or diffusion sampling bottlenecks, demonstrating high suitability for quality audits, surgical video review, and computer-assisted intraoperative surgical interventions.

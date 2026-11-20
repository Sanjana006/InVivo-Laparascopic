# Experiment 08 — EndoSmoke-Net

A paired laparoscopic desmoking research model combining:

- **HDMGN:** compact smoke-density guidance network.
- **EndoViT:** frozen ViT-Base/16 endoscopic feature extractor.
- **EndoSmokeCCFM-UNet:** seven-channel conditional flow-matching restoration U-Net.

## Architecture

```text
Smoky image (3ch) ──┬──> HDMGN ──> density map (1ch) ──────────────┐
                    ├──> Frozen EndoViT ──> projected anatomy ─┐   │
                    │                                          │   │
Gaussian x0 + clean x1 + t ──> CFM interpolation ──> x_t (3ch) │   │
                                                               ▼   ▼
                 U-Net input = [x_t(3), smoky(3), density(1)] = 7 channels
                                      + EndoViT feature injection at bottleneck
                                                    │
                                                    ▼
                                           predicted velocity (3ch)
                                                    │
                                          ODE integration at inference
                                                    ▼
                                             restored image (3ch)
```

## Dataset layout

The loader expects paired images with identical filenames:

```text
data/laparoscopy/
├── train_crop/
│   ├── clean/
│   └── smoky/
├── val_crop/
│   ├── clean/
│   └── smoky/
└── test/
    ├── clean/
    └── smoky/
```

The experiment reads RGB images and does not modify your existing dataset or experiments. Confirm that each split contains spatially aligned smoky/clean pairs. If the repository uses a different folder naming convention, change `--data_dir` or adapt `dataset.py` before training.

## Density supervision caveat

No manual smoke-density annotations are assumed. `utils.py` creates a weak pseudo-target from the smoothed absolute luminance difference between paired smoky and clean images. This is only a proxy and can also respond to illumination changes, tissue motion, camera motion, or misalignment. Inspect pseudo-targets visually before full training. Use real smoke masks/density annotations if available.

## Install

From repository root:

```bash
pip install -r experiments/08_Endosmoke_Net/requirements.txt
```

## EndoViT checkpoint

The model is configured as ViT-Base/16 (768 embedding dimension, 12 blocks, 12 attention heads, patch size 16). The loader follows the EndoViT demo checkpoint convention (`checkpoint['model']` when present), prints key coverage, and fails if core encoder weights appear missing. This check is a safeguard, not proof that a checkpoint is the correct official release. Verify the exact checkpoint and preprocessing against the official EndoViT repository.

Supply a local checkpoint:

```bash
--endovit_checkpoint /path/to/pytorch_model.bin
```

## Train

Default image size is 224 to match the ViT input resolution used by this implementation. The restoration U-Net also trains at this resolution. You can run a separate 256/448 resolution ablation; EndoViT will resize its own input to 224 unless its input size is deliberately changed in the model.

```bash
python experiments/08_Endosmoke_Net/train.py \
  --data_dir data/laparoscopy \
  --output_dir experiments/08_Endosmoke_Net/outputs \
  --epochs 100 --batch_size 4 --image_size 224 \
  --endovit_checkpoint /path/to/pytorch_model.bin
```

Smoke test without EndoViT (debugging only, not the full proposed model):

```bash
python experiments/08_Endosmoke_Net/train.py --no_endovit --epochs 1 --batch_size 1 --image_size 224
```

## Evaluate

Euler, 20 steps:

```bash
python experiments/08_Endosmoke_Net/evaluate.py \
  --data_dir data/laparoscopy \
  --checkpoint experiments/08_Endosmoke_Net/outputs/best_model.pth \
  --split test --steps 20 --solver euler
```

Compare RK4:

```bash
python experiments/08_Endosmoke_Net/evaluate.py \
  --data_dir data/laparoscopy \
  --checkpoint experiments/08_Endosmoke_Net/outputs/best_model.pth \
  --split test --steps 20 --solver rk4
```

Optional stochastic multi-sample evaluation (reports metrics on the sample mean and saves a pixel-wise sample standard-deviation map):

```bash
python experiments/08_Endosmoke_Net/evaluate.py ... --num_samples 4
```

Evaluation outputs are saved under `outputs/test_results/`: `input/`, `prediction/`, `ground_truth/`, `density/`, optional `uncertainty/`, and `metrics.csv`.

### Training-time visual outputs

By default, training also saves a small validation preview **after every epoch** under `outputs/training_visuals/epoch_001/` (then `epoch_002/`, etc.). Each epoch folder contains:

- `smoky/` — original RGB input
- `cleaned/` — generated RGB desmoked prediction
- `ground_truth/` — paired clean RGB target for visual comparison
- `density_predicted/` — model-predicted smoke-density map
- `density_pseudo_target/` — weak pseudo-density supervision target used by the density loss

The default is four validation examples per epoch and 10 ODE sampling steps for previews. Change or disable this with `--num_visual_samples`, `--visual_steps`, `--visual_solver`, and `--save_visuals_every` (set the last one to `0` to disable). Saving previews adds a little inference time after each selected epoch.

## Flow matching objective

For Gaussian noise `x0`, clean target `x1`, and `t ~ Uniform(0,1)`:

```text
x_t = (1 - t) * x0 + t * x1
u_target = x1 - x0
L = MSE(v_theta(x_t, smoky, density, t, EndoViT(smoky)), u_target)
    + density_weight * SmoothL1(predicted_density, pseudo_density_target)
```

The smoky observation conditions the velocity field on every forward pass. EndoViT is frozen by default; its features are extracted once per model forward and reused throughout all ODE solver stages in sampling. Euler is the default; fixed-step RK4 is included for a solver comparison. RK4 uses four velocity evaluations per step and is not guaranteed to improve perceptual quality—evaluate quality and runtime.

## Important research distinction

This is a guided conditional-flow-matching adaptation inspired by HazeMatching, combined with EndoViT feature conditioning. It should not be described as a complete reproduction of all HazeMatching capabilities (for example, posterior uncertainty analysis) without implementing and evaluating those components explicitly.

This is research code and is not clinically validated.

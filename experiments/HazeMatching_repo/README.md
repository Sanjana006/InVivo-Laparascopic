# HazeMatching: Dehazing Light Microscopy Images with Guided Conditional Flow Matching

[![arXiv](https://img.shields.io/badge/arXiv-2506.22397-b31b1b.svg)](https://arxiv.org/abs/2506.22397)

**Authors:** [Anirban Ray](https://rayanirban.github.io/) et al.  
**Venue:** IEEE/CVF CVPR 2026 (Findings)

**A fast and effective posterior sampling framework for reconstructing confocal targets from widefield microscopy inputs.**

<p align="center">
  <a href="assets/CVPR_poster.pdf">
    <img src="assets/CVPR_poster.png" alt="CVPR poster for HazeMatching" width="60%">
  </a>
</p>


HazeMatching uses guided conditional flow matching to generate multiple plausible microscopy reconstructions instead of a single deterministic output. This enables high-quality restoration together with uncertainty estimates for downstream analysis.

![HazeMatching overview](assets/figure1.png)

## Key Features

### ✨ Key Features

* ⚡ **Fast sampling**: Orders of magnitude faster than diffusion models
* 🎯 **High-quality reconstructions**: Strong PSNR and LPIPS performance
* 🔁 **Posterior sampling**: Generate diverse outputs instead of a single estimate
* 🔬 **Calibrated uncertainty quantification**: Provides uncertainty estimates for downstream analysis

## Posterior Samples

<p align="center">
  <img src="assets/care_test_2.gif" width="40%">
</p>

## Installation
After cloning the repo, run:
```bash
pip install uv
uv sync
```

## Example Notebook

Start with [notebooks/hazematching_walkthrough.ipynb](notebooks/hazematching_walkthrough.ipynb) for a hands-on walkthrough of the full workflow: a small training demo, checkpoint inference, lightweight metrics, calibration, and visualization of widefield input / confocal target / MMSE / posterior samples.

## Datasets

HazeMatching expects paired TIFF files where:

- channel 0 is the confocal target
- channel 1 is the widefield input

Supported dataset keys:

| Subset key | Dataset |
|------------|---------|
| `zebrafish` | Zebrafish |
| `organoids1` | Organoids1 |
| `organoids2` | Organoids2 |
| `microtubule` | Microtubule |
| `neuron` | Neuron |

By default, data is expected under `data/<subset>/`. Most subsets use `train_crop/`, `val_crop/`, `test/`, and `val/`; neuron data uses `train/`, `val/`, and `test/`.

## Reproducing Results

There are two main workflows, depending on whether you want to train from scratch or use pre-trained checkpoints.

Metrics are reproducible from provided checkpoints. Full retraining may produce slight variation due to non-deterministic operations.

### Stitched Inference Seeds

By default, `scripts/infer.py` initializes each posterior sample using freshly drawn random noise. We also provide the stitched initial-noise images (`t=0`) used to initialize the ODE integration in the paper. To use these saved noise images, run inference with the `--reproducible` flag while keeping the default `--n-samples 50`.

```bash
uv run python scripts/infer.py zebrafish --checkpoint checkpoints/zebrafish/best_model.pth --reproducible
```

On first use, this downloads `{subset}_test_val_seeds.zip` from Zenodo and extracts it under `data/<subset>/` by default. The seed archives are fetched from:

```text
https://zenodo.org/records/21705000/files/{subset}_test_val_seeds.zip?download=1
```

```text
data/<subset>/
  test_seeds/
    <test image filename>.tif
  val_seeds/
    <val image filename>.tif
```

Each seed TIFF is named exactly like the matching `test/` or `val/` image and stores the 50 noise initializations with shape `[50, 1, 1, X, Y]`. During reproducible inference, these saved noise images are used instead of random initialization. The reproducibility archives contain seeds only for the `test` and `val` splits; requesting more than 50 samples is not supported because only 50 saved seeds exist. If you download or extract the seeds yourself, place `test_seeds/` and `val_seeds/` in `data/<subset>/`, or pass their parent directory with `--seed-dir`.

The seed TIFFs store the stitched initial-noise images for the retained central `64 x 64` regions of each posterior sample. They are useful for visualization, inspection, and downstream reproducibility, but not as complete patch-level random states: inference used `128 x 128` patches (`64 x 64` for the `neuron` dataset), and the discarded border noise is unavailable, so exact trajectory replay is not guaranteed.

### Paper Result Samples

The full posterior-sample result stacks used to compute the paper metrics are available from Zenodo. To download those saved stacks and evaluate the reported test metrics directly, make sure the dataset is present under `data/<subset>/`, then run:

```bash
uv run python scripts/metrics.py zebrafish --paper-results
```

On first use, this downloads `{subset}_test_val_result_samples.zip` from Zenodo and extracts it under `data/<subset>/` by default. The result-sample archives are fetched from:

```text
https://zenodo.org/records/21838215/files/{subset}_test_val_result_samples.zip?download=1
```

```text
data/<subset>/
  test_result_samples/
    <test image filename>.tif
  val_result_samples/
    <val image filename>.tif
```

Each result TIFF is named exactly like the matching `test/` or `val/` image and stores 50 posterior samples with shape `[50, 1, 1, X, Y]`. With `--paper-results`, `scripts/metrics.py` evaluates `test_result_samples/` by default; pass `--results-dir data/<subset>/val_result_samples` to evaluate the validation stacks instead.

### Option A: Train From Scratch

1. Download data.

```bash
# Download all subsets
uv run python scripts/download_data.py

# Or download one subset
uv run python scripts/download_data.py --subset zebrafish
```

Data is saved to `data/<subset>/` by default.

2. Train a model.

```bash
uv run python scripts/train.py zebrafish
```

The best checkpoint is saved to `checkpoints/zebrafish/best_model.pth`. Training runs for 200 epochs by default.

3. Run inference.

```bash
uv run python scripts/infer.py zebrafish --checkpoint checkpoints/zebrafish/best_model.pth
```

Inference writes multi-sample TIFFs to `data/zebrafish/test_results/` and `data/zebrafish/val_results/`.

Inference with noise seed from the stitched reproducible seeds can be done with:

```bash
uv run python scripts/infer.py zebrafish --checkpoint checkpoints/zebrafish/best_model.pth --reproducible
```

4. Compute metrics.

```bash
uv run python scripts/metrics.py zebrafish
```

The metrics script reads from `data/zebrafish/test_results/` and reports PSNR, MicroMS3IM, LPIPS, FID, FSIM, and GMSD.

5. Optionally run calibration.

```bash
uv run python scripts/calibrate.py zebrafish --results-dir data/zebrafish
```

Calibration reads `val_results/` and `test_results/` under `data/zebrafish/` and saves `data/zebrafish/calibration.pdf`.

### Option B: Use Pre-Trained Checkpoints

1. Download data.

```bash
uv run python scripts/download_data.py --subset zebrafish
```

2. Download pre-trained checkpoints.

```bash
# Download all checkpoints
uv run python scripts/download_models.py

# Or download one checkpoint
uv run python scripts/download_models.py --subset zebrafish
```

Checkpoints are saved to `checkpoints/<subset>/best_model.pth` by default.

3. Run inference.

```bash
uv run python scripts/infer.py zebrafish --checkpoint checkpoints/zebrafish/best_model.pth
```

For inference with noise seed from the stitched reproducible seeds, add `--reproducible` to the inference command.

4. Compute metrics.

```bash
uv run python scripts/metrics.py zebrafish
```

5. Optionally run calibration.

```bash
uv run python scripts/calibrate.py zebrafish --results-dir data/zebrafish
```

## ⚠️ Manual Download of Datasets and Pre-trained Models

If downloading the datasets or pretrained models using the provided script results in an error, you can download them manually from the following link:

**Datasets and pretrained models:** [Zenodo download link](https://zenodo.org/records/21838215)

After downloading, extract the files and place them in the appropriate dataset and model directories used by HazeMatching.

### Model file names

The downloaded best-model checkpoints may include the dataset name as a prefix. For example:

```text
neuron_best_model.pth
```

Before placing a checkpoint in its corresponding model directory, remove the dataset-name prefix so that the file is named:

```text
best_model.pth
```

For example:

```text
neuron_best_model.pth  →  best_model.pth
```

Place each renamed checkpoint inside the directory corresponding to that dataset. Do not place all renamed `best_model.pth` files in the same directory, as they would overwrite one another.

Similarly, place each manually downloaded dataset in the appropriate dataset directory expected by the project.

If you use this work, please cite:

```bibtex
@inproceedings{ray2026hazematching,
  title     = {HazeMatching: Dehazing Light Microscopy Images with Guided Conditional Flow Matching},
  author    = {Ray, Anirban and Ashesh, Ashesh and Jug, Florian},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition - FINDINGS Track},
  year      = {2026}
}
```

## Acknowledgements

We thank Francesca Casagrande, Alessandra Fasciani, Jacopo Zasso, Ilaria Laface, Dario Ricca, and Eugenia Cammarota for their valuable contributions to this work. We also acknowledge the support of [Talley Lambert](https://talleylambert.com/) at Harvard Medical School and Vera Galinova in setting up the [microsim](https://talleylambert.com/microsim/) pipeline and some baselines, as well as the entire [Jug Group](https://humantechnopole.it/en/research-groups/jug-group/) for insightful discussions.

This work was supported by the European Union through the Horizon Europe program (IMAGINE project, grant agreement 101094250-IMAGINE and AI4Life project, grant agreement 101057970-AI4LIFE) and the generous core funding of [Human Technopole](https://humantechnopole.it/en/).

## License

MIT

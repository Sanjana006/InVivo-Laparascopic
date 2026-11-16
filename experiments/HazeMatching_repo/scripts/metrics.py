import os
import warnings
import zipfile
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pooch
import torch
from microssim import MicroMS3IM
from tifffile import imread, TiffFile
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm
import typer

from experiments.HazeMatching_repo.hazematching.datasets import (
    canonical_subset,
    fid_reference_folder,
    format_subset_options,
)
from experiments.HazeMatching_repo.hazematching.ra_psnr import RangeInvariantPsnr
from experiments.HazeMatching_repo.hazematching.utils import (
    lpips,
    fid_score,
    FSIM,
    extract_patches_inner_metrics,
    GMSD,
    entropy,
)

warnings.filterwarnings("ignore")

app = typer.Typer()

PAPER_RESULT_BASE_URL = "https://zenodo.org/records/21838215/files"
PAPER_RESULT_FOLDERS = ("test", "val")
MICROTUBULE_CENTER_CROP = slice(56, -56)
MICROTUBULE_RAW_SHAPE = (624, 624)
MICROTUBULE_METRIC_SHAPE = (512, 512)
DEFAULT_METRIC_PATCH_SIZE = 64
NEURON_METRIC_PATCH_SIZE = 32


def _paper_result_archive_filename(subset: str) -> str:
    return f"{subset}_test_val_result_samples.zip"


def _paper_result_archive_url(subset: str) -> str:
    filename = _paper_result_archive_filename(subset)
    return f"{PAPER_RESULT_BASE_URL}/{filename}?download=1"


def _paper_result_folder(folder: str) -> str:
    return f"{folder}_result_samples"


def _has_paper_result_folders(result_dir: Path, folders: tuple[str, ...]) -> bool:
    return all(
        (result_dir / _paper_result_folder(folder)).is_dir() for folder in folders
    )


def _extract_zip_safely(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (dest / member.filename).resolve()
            if member_path != dest and dest not in member_path.parents:
                raise ValueError(
                    f"Unsafe path in paper result archive: {member.filename}"
                )
        zf.extractall(dest)


def _ensure_paper_result_samples(
    subset: str,
    result_dir: Path,
    paper_result_archive_url: Optional[str],
) -> None:
    if _has_paper_result_folders(result_dir, PAPER_RESULT_FOLDERS):
        return

    filename = _paper_result_archive_filename(subset)
    url = paper_result_archive_url or _paper_result_archive_url(subset)
    result_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Downloading paper result samples ({filename}) ...")
    path = pooch.retrieve(
        url=url,
        known_hash=None,
        fname=filename,
        path=result_dir,
        progressbar=True,
    )

    typer.echo(f"  Extracting to {result_dir} ...")
    _extract_zip_safely(Path(path), result_dir)
    Path(path).unlink(missing_ok=True)

    if not _has_paper_result_folders(result_dir, PAPER_RESULT_FOLDERS):
        expected = ", ".join(
            _paper_result_folder(folder) for folder in PAPER_RESULT_FOLDERS
        )
        raise ValueError(
            f"Paper result archive did not create the expected folder(s) in "
            f"{result_dir}: {expected}"
        )


def _load_prediction_samples(result_path: Path, n_samples: int) -> np.ndarray:
    with TiffFile(result_path) as tif:
        image = tif.asarray()
    image = image.astype("float32")

    if image.ndim == 5 and image.shape[1:3] == (1, 1):
        samples = image[:, 0, 0]
    elif image.ndim == 4:
        samples = image[:, 0] if image.shape[1] == 1 else image[:, -1]
    elif image.ndim == 3:
        samples = image
    else:
        raise ValueError(
            f"Expected {result_path} to contain samples shaped "
            f"(samples, steps, height, width), (samples, 1, 1, height, width), "
            f"or (samples, height, width); got {image.shape}."
        )

    if samples.ndim != 3:
        raise ValueError(
            f"Expected {result_path} to reduce to (samples, height, width), "
            f"got {samples.shape}."
        )
    if samples.shape[0] < n_samples:
        raise ValueError(
            f"Result stack {result_path} has {samples.shape[0]} samples, "
            f"but n_samples={n_samples} was requested."
        )

    return samples[:n_samples]


def _crop_microtubule_metric_region(image: np.ndarray, path: Path) -> np.ndarray:
    if image.shape[-2:] == MICROTUBULE_METRIC_SHAPE:
        return image
    if image.shape[-2:] == MICROTUBULE_RAW_SHAPE:
        return image[..., MICROTUBULE_CENTER_CROP, MICROTUBULE_CENTER_CROP]
    raise ValueError(
        f"Expected microtubule image {path} to have spatial shape "
        f"{MICROTUBULE_RAW_SHAPE} or {MICROTUBULE_METRIC_SHAPE}, "
        f"got {image.shape[-2:]}."
    )


def _metric_patch_size(subset: str) -> int:
    if subset == "neuron":
        return NEURON_METRIC_PATCH_SIZE
    return DEFAULT_METRIC_PATCH_SIZE


def _as_fid_crop_stack(image: np.ndarray, fid_path: Path) -> np.ndarray:
    image = image.astype("float32")
    if image.ndim == 2:
        return image[np.newaxis, ...]
    if image.ndim == 3:
        return image
    raise ValueError(
        f"Expected FID reference {fid_path} to be shaped "
        f"(height, width) or (crops, height, width), got {image.shape}."
    )


def _neuron_channel0(image: np.ndarray, fid_path: Path) -> np.ndarray:
    image = image.astype("float32")
    if image.ndim == 2:
        return image

    channel_axes = [axis for axis, size in enumerate(image.shape) if size <= 4]
    for axis in channel_axes:
        channel0 = np.squeeze(np.take(image, 0, axis=axis))
        if channel0.ndim == 2:
            return channel0

    raise ValueError(
        f"Expected neuron FID reference {fid_path} to contain a channel stack "
        f"with channel 0 as a 2D image, got {image.shape}."
    )


def _load_fid_references(fid_dir: Path, subset: str) -> np.ndarray:
    fid_files = sorted(f for f in os.listdir(fid_dir) if f.endswith(".tif"))
    fid_crops = []
    for fid_file in tqdm(fid_files, desc="Loading FID crops", leave=False):
        fid_path = fid_dir / fid_file
        with TiffFile(fid_path) as tif:
            image = tif.asarray()
        if subset == "neuron":
            fid_crops.append(_neuron_channel0(image, fid_path)[np.newaxis, ...])
        else:
            fid_crops.append(_as_fid_crop_stack(image, fid_path))
    return np.concatenate(fid_crops, axis=0)


@app.command()
def compute_metrics(
    subset: Annotated[
        str, typer.Argument(help=f"Dataset subset. One of: {format_subset_options()}")
    ],
    results_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory containing inference .tif results. Defaults to <data_dir>/<subset>/test_results/."
        ),
    ] = None,
    fid_dir: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                "Directory of FID reference samples. Defaults to "
                "<data_dir>/<subset>/train_crops_fid, or train for neuron data."
            )
        ),
    ] = None,
    data_dir: Annotated[
        Path, typer.Option(help="Root data directory (used to resolve defaults).")
    ] = Path("data"),
    n_samples: Annotated[
        int, typer.Option(help="Number of samples to average for MMSE prediction.")
    ] = 50,
    paper_results: Annotated[
        bool,
        typer.Option(
            "--paper-results",
            "--paper-result",
            help=(
                "Download and evaluate the archived posterior result samples used "
                "for the paper. Results are extracted under <data_dir>/<subset>/ "
                "and test_result_samples/ is used unless --results-dir is given."
            ),
        ),
    ] = False,
    paper_result_archive_url: Annotated[
        Optional[str],
        typer.Option(
            "--paper-result-archive-url",
            help=(
                "Optional direct URL for the paper result sample zip. Defaults "
                "to the HazeMatching Zenodo record."
            ),
        ),
    ] = None,
):
    try:
        subset = canonical_subset(subset)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    subset_dir = data_dir / subset
    if paper_results:
        try:
            _ensure_paper_result_samples(
                subset,
                subset_dir,
                paper_result_archive_url=paper_result_archive_url,
            )
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        if results_dir is None:
            results_dir = subset_dir / _paper_result_folder("test")
    elif results_dir is None:
        results_dir = subset_dir / "test_results"
    if fid_dir is None:
        fid_dir = subset_dir / fid_reference_folder(subset)

    micros_ms3im = MicroMS3IM()

    # ── Load FID reference crops ─────────────────────────────────────────────
    fid_crops = _load_fid_references(fid_dir, subset)
    fid_crops_gts = torch.from_numpy(fid_crops).unsqueeze(1)
    typer.echo(f"Using {fid_crops.shape[0]} crops for FID.")

    image_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".tif"))

    psnr_values, ms_ssim_scores, micro3_ssim_scores = [], [], []
    gts, outputs, gts_full, outputs_full = [], [], [], []
    ind_fsims, ind_lpips, ind_fids, ind_gmsd = [], [], [], []

    typer.echo(
        f"Computing metrics over {len(image_files)} images (MMSE n={n_samples})..."
    )

    gt_dir = subset_dir / "test"
    metric_patch_size = _metric_patch_size(subset)

    for image_file in tqdm(image_files, desc="Images", leave=False):
        image_pred = _load_prediction_samples(results_dir / image_file, n_samples)
        image_gt = imread(gt_dir / image_file).astype("float32")[0:1]  # (1, H, W)
        if subset == "microtubule":
            image_pred = _crop_microtubule_metric_region(
                image_pred, results_dir / image_file
            )
            image_gt = _crop_microtubule_metric_region(image_gt, gt_dir / image_file)
        mmse_pred = np.mean(image_pred, axis=0, keepdims=True)

        # PSNR + MS-SSIM
        psnr_values.append(RangeInvariantPsnr(image_gt, mmse_pred))
        ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
            kernel_size=3, data_range=1.0, betas=(0.0448, 0.2856, 0.3001)
        )
        ms_ssim_scores.append(
            ms_ssim_metric(
                torch.from_numpy(mmse_pred).unsqueeze(0),
                torch.from_numpy(image_gt).unsqueeze(0),
            )
        )

        mmse_patches, _ = extract_patches_inner_metrics(
            mmse_pred, metric_patch_size
        )
        gt_patches, _ = extract_patches_inner_metrics(image_gt, metric_patch_size)
        gts.append(torch.from_numpy(gt_patches))
        outputs.append(torch.from_numpy(mmse_patches))
        gts_full.append(torch.from_numpy(image_gt).unsqueeze(1))
        outputs_full.append(torch.from_numpy(mmse_pred).unsqueeze(1))

        # Per-sample perceptual metrics
        torch_gt_patches = torch.from_numpy(gt_patches)
        valid = [
            i for i in range(torch_gt_patches.shape[0]) if torch_gt_patches[i].max() > 0
        ]
        torch_gt_patches = torch_gt_patches[valid]

        image_fsims, image_lpips_, image_fids_, image_gmsd_ = [], [], [], []
        for j in range(image_pred.shape[0]):
            pred_patches, _ = extract_patches_inner_metrics(
                image_pred[j : j + 1], metric_patch_size
            )
            torch_pred = torch.from_numpy(pred_patches)[valid]
            image_fsims.append(FSIM(torch_pred, torch_gt_patches))
            image_lpips_.append(lpips(torch_gt_patches, torch_pred))
            image_fids_.append(fid_score(fid_crops_gts, torch_pred))
            image_gmsd_.append(GMSD(torch_pred, torch_gt_patches))

        if image_fsims:
            ind_fsims.append(torch.mean(torch.stack(image_fsims)))
            ind_lpips.append(torch.mean(torch.tensor(image_lpips_)))
            ind_fids.append(torch.mean(torch.tensor(image_fids_)))
            ind_gmsd.append(torch.mean(torch.stack(image_gmsd_)))

    # ── Aggregate ────────────────────────────────────────────────────────────
    gts = torch.cat(gts, dim=0)
    outputs = torch.cat(outputs, dim=0)
    gts_full = torch.cat(gts_full, dim=0)
    outputs_full = torch.cat(outputs_full, dim=0)

    average_psnr = sum(psnr_values) / len(psnr_values)
    std_psnr = torch.std(torch.stack(psnr_values))
    average_ms_ssim = sum(ms_ssim_scores) / len(ms_ssim_scores)
    std_ms_ssim = torch.std(torch.stack(ms_ssim_scores))

    fsim_scores = FSIM(outputs, gts)
    fsim_mean = torch.mean(fsim_scores)
    lpips_score = lpips(gts, outputs)
    fid = fid_score(fid_crops_gts, outputs)
    gmsd_scores = GMSD(outputs, gts)
    gmsd_mean = torch.mean(gmsd_scores)
    entropy_scores = entropy(outputs)

    average_ind_fsim = torch.mean(torch.stack(ind_fsims))
    std_ind_fsim = torch.std(torch.stack(ind_fsims))
    average_ind_lpips = torch.mean(torch.tensor(ind_lpips))
    std_ind_lpips = torch.std(torch.tensor(ind_lpips))
    average_ind_fid = torch.mean(torch.tensor(ind_fids))
    std_ind_fid = torch.std(torch.tensor(ind_fids))
    average_ind_gmsd = torch.mean(torch.stack(ind_gmsd))
    std_ind_gmsd = torch.std(torch.stack(ind_gmsd))

    # MicroMS3IM
    gts_np = gts_full.numpy()
    outs_np = outputs_full.numpy()
    micros_ms3im.fit(gts_np[:, 0], outs_np[:, 0])
    micro3_ssim_scores = [
        micros_ms3im.score(gts_np[i, 0], outs_np[i, 0], betas=(0.0448, 0.2856, 0.3001))
        for i in range(gts_np.shape[0])
    ]
    average_micro3_ssim = np.mean(micro3_ssim_scores)
    std_micro3_ssim = np.std(micro3_ssim_scores)

    # ── Print ────────────────────────────────────────────────────────────────
    typer.echo(f"\n=== {subset.upper()} (n={n_samples}) ===")
    typer.echo(f"PSNR:         {average_psnr.item():.4f} ± {std_psnr.item():.4f}")
    typer.echo(
        f"MS-SSIM:      {average_ms_ssim.item():.4f} ± {std_ms_ssim.item():.4f}"
    )
    typer.echo(f"MicroMS3IM:   {average_micro3_ssim:.4f} ± {std_micro3_ssim:.4f}")
    typer.echo(f"FSIM  (MMSE): {fsim_mean.item():.4f}")
    typer.echo(
        f"FSIM  (Ind):  {average_ind_fsim.item():.4f} ± {std_ind_fsim.item():.4f}"
    )
    typer.echo(f"GMSD  (MMSE): {gmsd_mean.item():.4f}")
    typer.echo(
        f"GMSD  (Ind):  {average_ind_gmsd.item():.4f} ± {std_ind_gmsd.item():.4f}"
    )
    typer.echo(f"LPIPS (MMSE): {lpips_score:.4f}")
    typer.echo(
        f"LPIPS (Ind):  {average_ind_lpips.item():.4f} ± {std_ind_lpips.item():.4f}"
    )
    typer.echo(f"FID   (MMSE): {fid:.4f}")
    typer.echo(f"FID   (Ind):  {average_ind_fid.item():.4f} ± {std_ind_fid.item():.4f}")

    # LaTeX rows
    name = "\\textbf{HazeMatching}"
    typer.echo("\n--- LaTeX (MMSE + Ind, supplemental) ---")
    typer.echo(
        f"& {name} & "
        f"\\makecell{{{average_psnr.item():.2f} \\\\ {std_psnr.item():.3f}}} & "
        f"\\makecell{{{average_micro3_ssim:.3f} \\\\ {std_micro3_ssim:.4f}}} & "
        f"\\makecell{{{lpips_score:.3f}}} & "
        f"\\makecell{{{fid:.3f}}} & "
        f"\\makecell{{{average_ind_lpips.item():.3f} \\\\ {std_ind_lpips.item():.4f}}} & "
        f"\\makecell{{{average_ind_fid.item():.3f} \\\\ {std_ind_fid.item():.4f}}} \\\\ \\cline{{2-7}}"
    )


if __name__ == "__main__":
    app()

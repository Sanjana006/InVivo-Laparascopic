import os
import warnings
import zipfile
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
from experiments.HazeMatching_repo.hazematching import CCFMUNet
import pooch
import torch
from tifffile import imread, imwrite
from tqdm import tqdm
import typer

from experiments.HazeMatching_repo.hazematching import odeint
from experiments.HazeMatching_repo.hazematching.datasets import canonical_subset, format_subset_options
from experiments.HazeMatching_repo.hazematching.datasets.data_norm import normalize, denormalize
from experiments.HazeMatching_repo.hazematching.utils import extract_patches_inner, reconstruct_image_inner

warnings.filterwarnings("ignore")

app = typer.Typer()

MICROTUBULE_CENTER_CROP = slice(56, -56)
NEURON_PATCH_SIZE = 64
NEURON_CROP_SIZE = 32
REPRODUCIBILITY_SEED_BASE_URL = "https://zenodo.org/records/21705000/files"
REPRODUCIBILITY_SEED_FOLDERS = {"test", "val"}


def _seed_archive_filename(subset: str) -> str:
    return f"{subset}_test_val_seeds.zip"


def _seed_archive_url(subset: str) -> str:
    filename = _seed_archive_filename(subset)
    return f"{REPRODUCIBILITY_SEED_BASE_URL}/{filename}?download=1"


def _seed_folder(folder: str) -> str:
    return f"{folder}_seeds"


def _has_seed_folders(seed_dir: Path, folders: list[str]) -> bool:
    return all((seed_dir / _seed_folder(folder)).is_dir() for folder in folders)


def _extract_zip_safely(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (dest / member.filename).resolve()
            if member_path != dest and dest not in member_path.parents:
                raise ValueError(f"Unsafe path in seed archive: {member.filename}")
        zf.extractall(dest)


def _ensure_reproducibility_seeds(
    subset: str,
    seed_dir: Path,
    folders: list[str],
    seed_archive_url: Optional[str],
) -> None:
    if _has_seed_folders(seed_dir, folders):
        return

    filename = _seed_archive_filename(subset)
    url = seed_archive_url or _seed_archive_url(subset)
    seed_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Downloading reproducibility seeds ({filename}) ...")
    path = pooch.retrieve(
        url=url,
        known_hash=None,
        fname=filename,
        path=seed_dir,
        progressbar=True,
    )

    typer.echo(f"  Extracting to {seed_dir} ...")
    _extract_zip_safely(Path(path), seed_dir)
    Path(path).unlink(missing_ok=True)

    if not _has_seed_folders(seed_dir, folders):
        expected = ", ".join(_seed_folder(folder) for folder in folders)
        raise ValueError(
            f"Seed archive did not create the expected folder(s) in {seed_dir}: "
            f"{expected}"
        )


def _read_seed_stack(
    seed_path: Path,
    n_samples: int,
    expected_shape: tuple[int, int],
    subset: str,
) -> np.ndarray:
    seed_stack = imread(seed_path).astype("float32")
    if seed_stack.ndim == 5 and seed_stack.shape[1:3] == (1, 1):
        seed_stack = seed_stack[:, 0, 0]
    elif seed_stack.ndim == 4 and seed_stack.shape[1] == 1:
        seed_stack = seed_stack[:, 0]
    elif seed_stack.ndim == 3:
        pass
    elif seed_stack.ndim == 2:
        seed_stack = seed_stack[np.newaxis, ...]
    else:
        raise ValueError(
            f"Expected {seed_path} to contain a stack shaped "
            f"(samples, 1, 1, height, width), got {seed_stack.shape}."
        )

    if subset == "microtubule" and seed_stack.shape[-2:] != expected_shape:
        cropped = seed_stack[:, MICROTUBULE_CENTER_CROP, MICROTUBULE_CENTER_CROP]
        if cropped.shape[-2:] == expected_shape:
            seed_stack = cropped

    if seed_stack.shape[-2:] != expected_shape:
        raise ValueError(
            f"Seed stack {seed_path} has image shape {seed_stack.shape[-2:]}, "
            f"but inference expects {expected_shape}."
        )
    if seed_stack.shape[0] < n_samples:
        raise ValueError(
            f"Seed stack {seed_path} has {seed_stack.shape[0]} samples, "
            f"but n_samples={n_samples} was requested."
        )

    return seed_stack[:n_samples]


def _load_seed_noise_patches(
    seed_dir: Path,
    folder: str,
    image_file: str,
    n_samples: int,
    image_shape: tuple[int, int],
    coords: list[tuple[int, int, int, int]],
    subset: str,
    patch_size: int,
    crop_size: int,
) -> np.ndarray:
    seed_path = seed_dir / _seed_folder(folder) / image_file
    if not seed_path.exists():
        raise FileNotFoundError(f"Missing reproducibility seed file: {seed_path}")

    seed_stack = _read_seed_stack(seed_path, n_samples, image_shape, subset)
    seed_patches = []
    for seed_image in seed_stack:
        patches, seed_coords = extract_patches_inner(
            seed_image[np.newaxis, ...],
            patch_size=patch_size,
            crop_size=crop_size,
        )
        if seed_coords != coords:
            raise ValueError(
                f"Seed patches for {seed_path} do not match inference patch layout."
            )
        seed_patches.append(patches)

    return np.stack(seed_patches, axis=0)


@app.command()
def infer(
    subset: Annotated[
        str, typer.Argument(help=f"Dataset subset. One of: {format_subset_options()}")
    ],
    checkpoint: Annotated[Path, typer.Option(help="Path to model .pth checkpoint.")],
    data_dir: Annotated[
        Path, typer.Option(help="Root data directory containing subset folders.")
    ] = Path("data"),
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to write results. Defaults to <data_dir>/<subset_folder>/<folder>_results/."
        ),
    ] = None,
    folders: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Which split folders to run (e.g. test val). Default: test val."
        ),
    ] = None,
    n_samples: Annotated[
        int, typer.Option(help="Number of stochastic samples per image.")
    ] = 50,
    reproducible: Annotated[
        bool,
        typer.Option(
            "--reproducible",
            "--reproducibility",
            help=(
                "Use the paper reproducibility noise seeds instead of random "
                "initialization. Seeds are downloaded on first use."
            ),
        ),
    ] = False,
    seed_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--seed-dir",
            help=(
                "Directory containing test_seeds/ and val_seeds/, or where the "
                "reproducibility seed archive will be extracted. Defaults to "
                "<data_dir>/<subset>/."
            ),
        ),
    ] = None,
    seed_archive_url: Annotated[
        Optional[str],
        typer.Option(
            "--seed-archive-url",
            help=(
                "Optional direct URL for the reproducibility seed zip. Defaults "
                "to the HazeMatching Zenodo record."
            ),
        ),
    ] = None,
    num_steps: Annotated[int, typer.Option(help="Number of ODE time steps.")] = 20,
    max_batch_size: Annotated[
        int, typer.Option(help="Max patches per ODE batch.")
    ] = 256,
    patch_size: Annotated[int, typer.Option(help="Patch size for tiling.")] = 128,
    crop_size: Annotated[int, typer.Option(help="Inner crop size per patch.")] = 64,
):
    try:
        subset = canonical_subset(subset)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if folders is None:
        folders = ["test", "val"]

    if reproducible:
        unsupported_folders = [
            folder for folder in folders if folder not in REPRODUCIBILITY_SEED_FOLDERS
        ]
        if unsupported_folders:
            typer.echo(
                "Error: reproducibility seeds are available only for test and val "
                f"folders, got {unsupported_folders}.",
                err=True,
            )
            raise typer.Exit(1)

    if subset == "neuron":
        patch_size = NEURON_PATCH_SIZE
        crop_size = NEURON_CROP_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset_dir = data_dir / subset
    seed_root = seed_dir or subset_dir

    if reproducible:
        try:
            _ensure_reproducibility_seeds(
                subset,
                seed_root,
                folders,
                seed_archive_url=seed_archive_url,
            )
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc

    model = CCFMUNet(
        dim=(2, patch_size, patch_size),
        num_channels=32,
        out_channels=1,
        num_res_blocks=1,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    typer.echo(f"Loaded checkpoint: {checkpoint}")

    ts = torch.linspace(0.0, 1.0, num_steps).to(device)

    for folder in folders:
        folder_dir = subset_dir / folder
        image_files = sorted(f for f in os.listdir(folder_dir) if f.endswith(".tif"))

        if output_dir is None:
            out = subset_dir / f"{folder}_results"
        else:
            out = output_dir / f"{folder}_results"
        out.mkdir(parents=True, exist_ok=True)

        for image_file in tqdm(image_files, desc=f"{folder}", unit="image"):
            img_path = folder_dir / image_file
            raw = imread(img_path).astype("float32")
            if subset == "microtubule":
                raw = raw[
                    :,
                    MICROTUBULE_CENTER_CROP,
                    MICROTUBULE_CENTER_CROP,
                ]
            widefield_norm = normalize(raw[1:2], subset, channel=1, path=img_path)
            patches, coords = extract_patches_inner(
                widefield_norm, patch_size=patch_size, crop_size=crop_size
            )
            condition = torch.from_numpy(patches).to(device)
            seed_noise_patches = None
            if reproducible:
                try:
                    seed_noise_patches = _load_seed_noise_patches(
                        seed_root,
                        folder,
                        image_file,
                        n_samples,
                        widefield_norm.shape[-2:],
                        coords,
                        subset,
                        patch_size=patch_size,
                        crop_size=crop_size,
                    )
                except (OSError, ValueError) as exc:
                    typer.echo(f"Error: {exc}", err=True)
                    raise typer.Exit(1) from exc
                seed_noise_patches = torch.from_numpy(seed_noise_patches).to(
                    device=device,
                    dtype=condition.dtype,
                )

            samples = []

            for sample_idx in tqdm(range(n_samples), desc="samples", leave=False):
                image_tensor = np.zeros(
                    (num_steps, max_batch_size, 2, patch_size, patch_size),
                    dtype=np.float32,
                )
                if seed_noise_patches is None:
                    noise = torch.randn_like(condition)
                else:
                    noise = seed_noise_patches[sample_idx]
                input_tensor = torch.cat([noise, condition], dim=1)
                num_batches = (
                    input_tensor.size(0) + max_batch_size - 1
                ) // max_batch_size

                with torch.no_grad():
                    for i in range(num_batches):
                        batch = input_tensor[
                            i * max_batch_size : (i + 1) * max_batch_size
                        ]
                        traj, _ = odeint(
                            lambda t, x: model(t, x),
                            batch,
                            ts,
                            atol=1e-4,
                            rtol=1e-4,
                            method="euler",
                            condition=1,
                        )
                        image_tensor = np.concatenate(
                            [image_tensor, traj.cpu().numpy()], axis=1
                        )

                image_tensor = image_tensor[:, max_batch_size:]
                samples.append(image_tensor)

            samples = np.stack(samples, axis=0)
            full = reconstruct_image_inner(
                samples,
                coords,
                widefield_norm.shape,
                patch_size=patch_size,
                crop_size=crop_size,
            )

            if subset == "neuron":
                prediction = full[:, :, 0]
            else:
                prediction = denormalize(
                    full[:, :, 0], subset, channel=0, path=img_path
                )

            imwrite(
                out / image_file, prediction, imagej=True, metadata={"axes": "TZYX"}
            )


if __name__ == "__main__":
    app()

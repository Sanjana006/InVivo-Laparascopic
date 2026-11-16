"""Download HazeMatching datasets.

Usage:
    uv run python scripts/download_data.py               # download all subsets
    uv run python scripts/download_data.py --data-dir /my/path
    uv run python scripts/download_data.py --subset zebrafish --subset organoids1
"""

import zipfile
from pathlib import Path
from typing import Annotated, Optional

import pooch
import typer

from experiments.HazeMatching_repo.hazematching.datasets import DATASET_DESCRIPTIONS, SUBSETS, canonical_subset

BASE_URL = "https://download.fht.org/jug/hazematching/data/"


DATASETS: dict[str, tuple[str, str | None]] = {
    "zebrafish": ("zebrafish.zip", "sha256:e672e711189733f9cca8dbc0da4cee43f4e47420fc649dc29ac9091756cd0b6e"),
    "organoids1": ("organoids1.zip", "sha256:96093963ba26a5f9525e7024db84e35e143983f025e58206318a76672114aad7"),
    "organoids2": ("organoids2.zip", "sha256:6a72eac8aaf9a92f23c5ca98a6f8b6129bccee7690beb6c00e309c575a9da544"),
    "microtubule": ("microtubule.zip", "sha256:15c682afaab73e7086d36230a68119f04185eceb0a48336f0097c3e7b4ad53d9"),
    "neuron": ("neuron.zip", "sha256:3930eb8faab6968c93644bf4f627025879880e0d17bd5dc572cbcad1257c6b36"),
}

app = typer.Typer()


def _download_subset(key: str, data_dir: Path) -> None:
    key = canonical_subset(key)
    filename, known_hash = DATASETS[key]
    typer.echo(f"Downloading {DATASET_DESCRIPTIONS[key]} ({filename}) ...")

    path = pooch.retrieve(
        url=BASE_URL + filename,
        known_hash=known_hash,
        fname=filename,
        path=data_dir,
        progressbar=True,
    )

    typer.echo(f"  Extracting to {data_dir} ...")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(data_dir)
    Path(path).unlink()
    typer.echo(f"  Done -> {data_dir / filename.replace('.zip', '')}")


@app.command()
def main(
    data_dir: Annotated[
        Path, typer.Option(help="Directory to download data into.")
    ] = Path("data"),
    subset: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Subset(s) to download. Repeat to select multiple. Default: all."
        ),
    ] = None,
):
    try:
        keys = [canonical_subset(s) for s in subset] if subset else list(DATASETS)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        _download_subset(key, data_dir)

    typer.echo("\nAll done. Data layout:")
    for key in keys:
        typer.echo(f"  {data_dir / key}/")


if __name__ == "__main__":
    app()

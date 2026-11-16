"""Download pre-trained HazeMatching checkpoints.

Usage:
    uv run python scripts/download_models.py               # download all pre-trained models
    uv run python scripts/download_models.py --checkpoint-dir /my/path
    uv run python scripts/download_models.py --subset zebrafish --subset organoids1
"""

from pathlib import Path
from typing import Annotated, Optional

import pooch
import typer

from experiments.HazeMatching_repo.hazematching.datasets import DATASET_DESCRIPTIONS, SUBSETS, canonical_subset

BASE_URL = "https://download.fht.org/jug/hazematching/checkpoints/"

MODELS: dict[str, dict[str, str | None]] = {
    "zebrafish": {"best_model.pth": "sha256:9e3bdfcdce89ed7d93d8cbc189e00d05d44c2b35c03934e1235ffe7104684cb4"},
    "organoids1": {"best_model.pth": "sha256:0757e308aff6f9fe3ed619a3ce190ab7210e9ed2fdfb002c2ef47752d126e0b6"},
    "organoids2": {"best_model.pth": "sha256:329621425efcdc37063701daf3d2b7ea26f006557e3a57a08d17258d22a874d6"},
    "microtubule": {"best_model.pth": "sha256:20344df5748f3eb36294d80bed67be52a37c095d5d7138d8ec16dceca7a2875a"},
    "neuron": {"best_model.pth": "sha256:2158fe4df870540885d3691cb3ec41b6db5482786106a5f03f24aedfead302e9"},
}

app = typer.Typer()


def _download_subset(key: str, checkpoint_dir: Path) -> None:
    key = canonical_subset(key)
    dest = checkpoint_dir / key
    dest.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {DATASET_DESCRIPTIONS[key]} checkpoints -> {dest}/")
    for filename, known_hash in MODELS[key].items():
        pooch.retrieve(
            url=BASE_URL + f"{key}/{filename}",
            known_hash=known_hash,
            fname=filename,
            path=dest,
            progressbar=True,
        )
        typer.echo(f"  {filename}")


@app.command()
def main(
    checkpoint_dir: Annotated[
        Path, typer.Option(help="Directory to download checkpoints into.")
    ] = Path("checkpoints"),
    subset: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Subset(s) to download. Repeat to select multiple. Default: all."
        ),
    ] = None,
):
    try:
        keys = [canonical_subset(s) for s in subset] if subset else list(MODELS)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    for key in keys:
        _download_subset(key, checkpoint_dir)

    typer.echo("\nAll done. Checkpoint layout:")
    for key in keys:
        typer.echo(f"  {checkpoint_dir / key}/")
        for filename in MODELS[key]:
            typer.echo(f"    {filename}")


if __name__ == "__main__":
    app()

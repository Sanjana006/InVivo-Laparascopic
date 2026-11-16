from .config import (
    DATASET_DESCRIPTIONS,
    SUBSETS,
    canonical_subset,
    fid_reference_folder,
    format_subset_options,
    training_split_folder,
)
from .data_norm import normalize, denormalize, STATS

__all__ = [
    "DATASET_DESCRIPTIONS",
    "SUBSETS",
    "STATS",
    "HazeDataset",
    "canonical_subset",
    "denormalize",
    "fid_reference_folder",
    "format_subset_options",
    "normalize",
    "training_split_folder",
]


def __getattr__(name: str):
    if name == "HazeDataset":
        from .haze import HazeDataset

        return HazeDataset

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

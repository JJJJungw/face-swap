#!/usr/bin/env python3
"""Shared, extension-agnostic input/target pair discovery."""

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class PairPaths:
    stem: str
    input_path: Path
    target_path: Path


def _index_directory(directory, extensions=IMAGE_EXTENSIONS):
    directory = Path(directory)
    indexed = {}
    duplicates = {}
    for path in sorted(directory.iterdir() if directory.is_dir() else []):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in indexed:
            duplicates.setdefault(path.stem, [indexed[path.stem]]).append(path)
        else:
            indexed[path.stem] = path
    if duplicates:
        details = ", ".join(
            f"{stem}: {[str(path) for path in paths]}"
            for stem, paths in sorted(duplicates.items())
        )
        raise ValueError(f"duplicate image stems in {directory}: {details}")
    return indexed


def discover_pairs(input_dir, target_dir, extensions=IMAGE_EXTENSIONS):
    inputs = _index_directory(input_dir, extensions)
    targets = _index_directory(target_dir, extensions)
    return [
        PairPaths(stem, inputs[stem], targets[stem])
        for stem in sorted(inputs.keys() & targets.keys())
    ]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

COMMON_IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")

DATASETS = ("mvtec_ad", "visa", "mvtec_ad2", "mvtec_loco")

DATASET_PROFILES = {
    "mvtec_ad": {
        "anomalib_data": "MVTecAD",
        "example_category": "bottle",
        "default_path": "~/datasets/mvtec_ad",
    },
    "visa": {
        "anomalib_data": "Visa",
        "example_category": "candle",
        "default_path": "~/datasets/visa",
    },
    "mvtec_ad2": {
        "anomalib_data": "MVTecAD2",
        "example_category": "sheet_metal",
        "default_path": "~/datasets/mvtec_ad2",
    },
    "mvtec_loco": {
        "anomalib_data": "MVTecLOCO",
        "example_category": "breakfast_box",
        "default_path": "~/datasets/mvtec_loco",
    },
}

VALIDATION_NORMAL_SUBDIRS = (
    Path("val/good"),
    Path("val/normal"),
    Path("validation/good"),
    Path("validation/normal"),
)
TRAINING_NORMAL_SUBDIRS = (
    Path("train/good"),
    Path("train/normal"),
    Path("normal"),
    Path("Data/Images/Normal"),
)
NORMAL_SUBDIRS = (*VALIDATION_NORMAL_SUBDIRS, *TRAINING_NORMAL_SUBDIRS)


@dataclass(frozen=True)
class DatasetLayout:
    dataset: str
    dataset_root: Path
    dataset_path: Path
    root_mode: str


def resolve_dataset_layout(dataset_root: Path, dataset: str) -> DatasetLayout:
    root = dataset_root.expanduser()
    nested = root / dataset
    if nested.exists():
        return DatasetLayout(
            dataset=dataset,
            dataset_root=root,
            dataset_path=nested,
            root_mode="parent_dataset_root",
        )
    return DatasetLayout(
        dataset=dataset,
        dataset_root=root,
        dataset_path=root,
        root_mode="dataset_specific_root",
    )


def category_base_path(dataset_path: Path, dataset: str) -> Path:
    if dataset == "visa" and (dataset_path / "visa_pytorch").exists():
        return dataset_path / "visa_pytorch"
    return dataset_path


def iter_image_files(path: Path) -> list[Path]:
    return [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in COMMON_IMAGE_EXTENSIONS
    ]


def count_images(path: Path) -> int:
    return len(iter_image_files(path))


def discover_categories(dataset: str, dataset_path: Path) -> list[str]:
    base = category_base_path(dataset_path, dataset)
    if not base.exists():
        return []
    categories = []
    for category_dir in sorted(candidate for candidate in base.iterdir() if candidate.is_dir()):
        if has_normal_images(category_dir):
            categories.append(category_dir.name)
    return categories


def has_normal_images(category_dir: Path) -> bool:
    return any(
        count_images(category_dir / subdir) > 0
        for subdir in NORMAL_SUBDIRS
        if (category_dir / subdir).exists()
    )


def find_normal_images_with_source(
    dataset_root: Path,
    dataset: str,
    category: str,
) -> tuple[list[Path], str, Path]:
    layout = resolve_dataset_layout(dataset_root, dataset)
    category_paths = [layout.dataset_path / category]
    if dataset == "visa":
        category_paths.insert(0, layout.dataset_path / "visa_pytorch" / category)

    for split_name, subdirs in (
        ("normal_validation", VALIDATION_NORMAL_SUBDIRS),
        ("normal_training", TRAINING_NORMAL_SUBDIRS),
    ):
        for category_path in category_paths:
            images = []
            for subdir in subdirs:
                normal_dir = category_path / subdir
                if normal_dir.exists():
                    images.extend(iter_image_files(normal_dir))
            if images:
                return sorted(images), split_name, layout.dataset_path
    return [], "none", layout.dataset_path

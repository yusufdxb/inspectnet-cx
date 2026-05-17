from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DATASETS = ("mvtec_ad", "visa", "mvtec_ad2", "mvtec_loco")
COMMON_IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
DEFAULT_DATASET_ROOT = Path("~/datasets").expanduser()

DATASET_PROFILES = {
    "mvtec_ad": {
        "anomalib_data": "MVTecAD",
        "example_category": "bottle",
        "default_path": "~/datasets/mvtec_ad",
        "normal_globs": ("*/train/good",),
    },
    "visa": {
        "anomalib_data": "Visa",
        "example_category": "candle",
        "default_path": "~/datasets/visa",
        "normal_globs": ("visa_pytorch/*/train/good", "*/train/good", "*/normal"),
    },
    "mvtec_ad2": {
        "anomalib_data": "MVTecAD2",
        "example_category": "sheet_metal",
        "default_path": "~/datasets/mvtec_ad2",
        "normal_globs": ("*/train/good",),
    },
    "mvtec_loco": {
        "anomalib_data": "MVTecLOCO",
        "example_category": "breakfast_box",
        "default_path": "~/datasets/mvtec_loco",
        "normal_globs": ("*/train/good", "*/train/logical_anomalies/good"),
    },
}


def check_datasets(root: Path = DEFAULT_DATASET_ROOT) -> dict[str, Any]:
    root = root.expanduser()
    datasets = {}
    for name in DATASETS:
        path = root / name
        categories = _discover_categories(name, path) if path.exists() else []
        image_count = _count_images(path) if path.exists() else 0
        datasets[name] = {
            "path": str(path),
            "exists": path.exists(),
            "image_count": image_count,
            "has_train_dir": (path / "train").exists(),
            "has_test_dir": (path / "test").exists(),
            "has_ground_truth_dir": (path / "ground_truth").exists(),
            "ready_categories": categories,
            "ready_category_count": len(categories),
            "anomalib_data": DATASET_PROFILES[name]["anomalib_data"],
            "example_category": DATASET_PROFILES[name]["example_category"],
            "status": _dataset_status(path, categories),
            "next_actions": _dataset_next_actions(root, name, categories),
        }

    ready_for_benchmarks = all(item["ready_category_count"] > 0 for item in datasets.values())
    return {
        "dataset_root": str(root),
        "status": "ready" if ready_for_benchmarks else "blocked",
        "datasets": datasets,
        "note": (
            "Structure checks are conservative and do not validate official dataset integrity, "
            "licenses, checksums, or train/test label semantics."
        ),
        "evaluation_examples": _evaluation_examples(root),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local industrial anomaly dataset paths.")
    parser.add_argument("--root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_check.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = check_datasets(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def _count_images(path: Path) -> int:
    return sum(
        1
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in COMMON_IMAGE_EXTENSIONS
    )


def _discover_categories(dataset: str, path: Path) -> list[str]:
    base = path / "visa_pytorch" if dataset == "visa" and (path / "visa_pytorch").exists() else path

    categories = []
    for category_dir in sorted(candidate for candidate in base.iterdir() if candidate.is_dir()):
        if _has_normal_training_images(category_dir):
            categories.append(category_dir.name)
    return categories


def _has_normal_training_images(category_dir: Path) -> bool:
    candidate_dirs = [
        category_dir / "train" / "good",
        category_dir / "train" / "normal",
        category_dir / "normal",
        category_dir / "Data" / "Images" / "Normal",
    ]
    return any(_count_images(path) > 0 for path in candidate_dirs if path.exists())


def _dataset_status(path: Path, categories: list[str]) -> str:
    if not path.exists():
        return "missing"
    if not categories:
        return "present_unverified_structure"
    return "ready_for_local_checks"


def _dataset_next_actions(root: Path, dataset: str, categories: list[str]) -> list[str]:
    profile = DATASET_PROFILES[dataset]
    path = root / dataset
    if not path.exists():
        return [
            f"Create {path} outside the repository.",
            "Acquire the dataset from the official source and record license/version/checksum.",
            f"Re-run: inspectnet-dataset-check --root {root}",
        ]
    if not categories:
        return [
            f"Place category folders under {path} in Anomalib-compatible form.",
            (
                "Expected normal training images in paths such as "
                "<category>/train/good or the dataset-specific converted equivalent."
            ),
            f"Smoke command: inspectnet-baseline --plan-only --dataset {dataset} "
            f"--method patchcore --category {profile['example_category']} --data-root {root}",
        ]
    category = categories[0]
    return [
        f"Run dry Anomalib plan: inspectnet-baseline --plan-only --dataset {dataset} "
        f"--method patchcore --category {category} --data-root {root}",
        f"Calibrate Phase 0 threshold: inspectnet-calibrate-normal-threshold --dataset {dataset} "
        f"--category {category} --dataset-root {root} --model artifacts/inspectnet-cx-phase0",
    ]


def _evaluation_examples(root: Path) -> list[str]:
    examples = []
    for dataset, profile in DATASET_PROFILES.items():
        examples.append(
            "inspectnet-baseline --plan-only "
            f"--method patchcore --dataset {dataset} --category {profile['example_category']} "
            f"--data-root {root}"
        )
    return examples


if __name__ == "__main__":
    main()

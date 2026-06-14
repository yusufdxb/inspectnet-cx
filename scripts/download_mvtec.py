#!/usr/bin/env python3
"""Download a MVTec AD category into a local data root.

MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial research only) and is NOT
redistributed by this repository. This helper delegates to Anomalib's MVTecAD
datamodule, which fetches and extracts the dataset on first use.

Usage:
    python3 scripts/download_mvtec.py --category bottle --data-root ~/datasets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle", help="MVTec AD category, e.g. bottle")
    parser.add_argument("--data-root", default="~/datasets", help="local dataset root")
    args = parser.parse_args()

    root = Path(args.data_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    try:
        from anomalib.data import MVTecAD
    except ImportError:
        print(
            "anomalib is required for the download path. Install it with:\n"
            '  pip install -e ".[all]"',
            file=sys.stderr,
        )
        return 1

    print(
        "MVTec AD is CC BY-NC-SA 4.0 (non-commercial). By downloading you accept those terms.\n"
        f"Fetching category '{args.category}' into {root} ..."
    )
    dm = MVTecAD(root=root / "mvtec_ad", category=args.category)
    dm.prepare_data()  # downloads + extracts if absent
    print(f"Done. Category available under {root / 'mvtec_ad' / args.category}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run threshold analysis from a per-image score file.

The input is a JSON document of the form

    {
        "schema": "inspectnet_cx.scores.v1",
        "items": [
            {"path": "test/good/000.png", "label": 0, "score": 1.23},
            ...
        ]
    }

The output is the operating-point report defined in
``inspectnet_cx.calibration.threshold_analysis``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspectnet_cx.calibration.threshold_analysis import analyze_thresholds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-fpr",
        type=float,
        nargs="*",
        default=[0.01, 0.05, 0.10],
        help="FPR ceilings to compute operating points for",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text())
    items = payload["items"]
    labels = [int(item["label"]) for item in items]
    scores = [float(item["score"]) for item in items]
    report = analyze_thresholds(labels, scores, target_fprs=args.target_fpr)
    report["source_score_file"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "roc_curve"}, indent=2))


if __name__ == "__main__":
    main()

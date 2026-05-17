from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FIELDS = (
    "method",
    "dataset",
    "category",
    "image_auroc",
    "pixel_auroc",
    "au_pro",
    "pixel_f1",
    "latency_ms_per_image",
    "peak_vram_mb",
    "model_size_mb",
    "status",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate baseline JSON files.")
    parser.add_argument("--input", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, default=Path("reports/baselines.md"))
    return parser.parse_args(argv)


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(input_dir.glob("*.json")):
        results.append(json.loads(path.read_text()))
    return results


def render_markdown(results: list[dict[str, Any]]) -> str:
    lines = ["# Baseline Leaderboard", ""]
    if not results:
        lines.extend(["No baseline result JSON files found.", ""])
        return "\n".join(lines)

    lines.append("| " + " | ".join(FIELDS) + " |")
    lines.append("| " + " | ".join("---" for _ in FIELDS) + " |")
    for result in results:
        lines.append("| " + " | ".join(str(result.get(field, "TBD")) for field in FIELDS) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = load_results(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(results))
    print(f"Wrote leaderboard to {args.output}")


if __name__ == "__main__":
    main()

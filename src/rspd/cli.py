"""Score two RGB image files without benchmark data."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from rspd import RSPDMean7
from rspd.reproducibility import configure_reproducibility


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RSPD perceptual distance. Lower means more similar.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--method", choices=("pooled_ecdf", "pooled_q95"), default="pooled_ecdf")
    parser.add_argument("--details", action="store_true", help="Print layer diagnostics as JSON.")
    args = parser.parse_args(argv)
    try:
        with Image.open(args.reference) as image:
            reference = image.convert("RGB")
        with Image.open(args.candidate) as image:
            candidate = image.convert("RGB")
        configure_reproducibility()
        metric = RSPDMean7(device=args.device, method=args.method)
        result = metric.score_pair(reference, candidate, return_details=args.details)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"rspd: {exc}\n")
    print(json.dumps(asdict(result), indent=2) if args.details else f"{result:.8f}")

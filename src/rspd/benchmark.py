"""Evaluate frozen benchmark triplets or a user-supplied CSV."""

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from importlib.resources import as_file, files
from pathlib import Path, PurePosixPath

from rspd.defaults import calibration_metadata
from rspd.reproducibility import configure_reproducibility


METRICS = ("rspd", "rspd_q95", "msssim", "lpips", "dreamsim", "diffsim", "remoteclip")
TABLE2_DATASETS = ("airborne_sr", "spaceborne_sr", "sar_to_optical", "synthetic", "misregistration")


def load_manifest(path: Path, data_root: Path, datasets=None, limit=None) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "triplet_id", "reference", "candidate_a", "candidate_b", "preferred"}
        if missing := required - set(reader.fieldnames or []):
            raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    unknown = set(datasets or []) - {row["dataset"] for row in rows}
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(sorted(unknown))}")
    rows = [row for row in rows if not datasets or row["dataset"] in datasets]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No benchmark rows selected.")
    seen = set()
    missing_paths = set()
    for row in rows:
        key = (row["dataset"], row["triplet_id"])
        if key in seen:
            raise ValueError(f"Duplicate benchmark triplet: {key}")
        seen.add(key)
        if row["preferred"] not in {"a", "b"}:
            raise ValueError(f"Invalid preference for {key}: {row['preferred']!r}")
        for column in ("reference", "candidate_a", "candidate_b"):
            relative = PurePosixPath(row[column])
            if not row[column] or relative.is_absolute() or ".." in relative.parts or "\\" in row[column]:
                raise ValueError(f"Image paths must be portable relative paths: {row[column]!r}")
            path = (data_root / relative).resolve()
            if not path.is_relative_to(data_root.resolve()):
                raise ValueError(f"Image path escapes data root: {row[column]}")
            if not path.is_file():
                missing_paths.add(row[column])
            row[f"{column}_path"] = str(path)
    if missing_paths:
        examples = ", ".join(sorted(missing_paths)[:3])
        raise FileNotFoundError(
            f"{len(missing_paths)} benchmark images missing under {data_root}. Examples: {examples}. "
            "Download the benchmark (see docs/DATA.md), or select an available subset with --dataset."
        )
    return rows


def evaluate(rows: list[dict], scorers: dict) -> tuple[list[dict], dict]:
    results = []
    counts = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        for metric, score in scorers.items():
            a = float(score(row["reference_path"], row["candidate_a_path"]))
            b = float(score(row["reference_path"], row["candidate_b_path"]))
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError(f"Non-finite {metric} score for {row['dataset']}/{row['triplet_id']}")
            prediction = "tie" if abs(a - b) <= 1e-12 else "a" if a < b else "b"
            correct = prediction == row["preferred"]
            count = counts[metric][row["dataset"]]
            count["total"] += 1
            count["ties" if prediction == "tie" else "correct" if correct else "incorrect"] += 1
            results.append({
                "dataset": row["dataset"], "triplet_id": row["triplet_id"], "metric": metric,
                "score_a": a, "score_b": b, "prediction": prediction,
                "preferred": row["preferred"], "correct": correct,
            })
    summaries = {}
    for metric, datasets in counts.items():
        by_dataset = {
            dataset: {"total": c["total"], "correct": c["correct"], "incorrect": c["incorrect"],
                      "ties": c["ties"], "accuracy": c["correct"] / c["total"]}
            for dataset, c in sorted(datasets.items())
        }
        complete = all(by_dataset.get(dataset, {}).get("total") == 300 for dataset in TABLE2_DATASETS)
        summaries[metric] = {
            "by_dataset": by_dataset,
            "table2_average": (
                sum(by_dataset[dataset]["accuracy"] for dataset in TABLE2_DATASETS) / len(TABLE2_DATASETS)
                if complete else None
            ),
        }
    return results, summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate RSPD and optional paper baselines on fixed triplets.")
    parser.add_argument("--manifest", type=Path, help="Defaults to the bundled, frozen 1,800-triplet manifest.")
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=["rspd", "rspd_q95"])
    parser.add_argument("--dataset", nargs="+", help="Select one or more dataset names from the manifest.")
    parser.add_argument("--limit", type=int, help="Select only the first N triplets for a smoke test.")
    parser.add_argument("--check-data", action="store_true", help="Check selected files without loading any models.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--diffsim-root", type=Path, help="Upstream DiffSim checkout; needed only for that baseline.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    started = time.perf_counter()
    try:
        if args.manifest is None:
            with as_file(files("rspd").joinpath("data/benchmark.csv")) as path:
                rows = load_manifest(path, args.data_root, args.dataset, args.limit)
        else:
            rows = load_manifest(args.manifest, args.data_root, args.dataset, args.limit)
        if args.check_data:
            print(f"Found all images for {len(rows)} selected triplets.")
            return
        from rspd.baselines import create_scorers

        reproducibility = configure_reproducibility()
        scorers = create_scorers(list(dict.fromkeys(args.metrics)), args.device, args.diffsim_root)
        results, methods = evaluate(rows, scorers)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"rspd-evaluate: {exc}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "methods": methods, "calibration": calibration_metadata(), "reproducibility": reproducibility,
        "selected_triplets": len(rows), "elapsed_seconds": time.perf_counter() - started,
        "score_orientation": "lower_is_more_similar", "ties_count_as_correct": False,
    }
    (args.output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for metric, values in methods.items():
        for dataset, count in values["by_dataset"].items():
            print(f"{metric:12} {dataset:18} {count['accuracy']:.2%} ({count['correct']}/{count['total']})")
        if values["table2_average"] is not None:
            print(f"{metric:12} Table 2 average    {values['table2_average']:.2%}")
    print(f"Wrote evaluation.csv and evaluation_summary.json to {args.output_dir}")


if __name__ == "__main__":
    main()

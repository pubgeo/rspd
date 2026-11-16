"""Make a draft DataPort ZIP from the frozen manifest and local benchmark files."""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from importlib.resources import files
from pathlib import Path

from rspd.benchmark import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("data/release/rspd-benchmark-v1.zip"))
    parser.add_argument("--include-airborne", action="store_true", help="Include VEDAI-derived images ONLY with permission.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    manifest = files("rspd").joinpath("data/benchmark.csv")
    with manifest.open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if args.include_airborne or row["dataset"] != "airborne_sr"]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checksums = {}
    selected = sorted({row["dataset"] for row in rows})
    load_manifest(Path(str(manifest)), args.data_root, selected)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.csv", csv_buffer.getvalue())
        paths = sorted({row[key] for row in rows for key in ("reference", "candidate_a", "candidate_b")})
        for relative in paths:
            path = args.data_root / relative
            with path.open("rb") as handle:
                checksums[relative] = hashlib.file_digest(handle, "sha256").hexdigest()
            archive.write(path, relative)
        archive.writestr("checksums.json", json.dumps(checksums, indent=2) + "\n")
    with args.output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    print(f"Draft archive: {args.output} ({len(rows)} triplets, {len(checksums)} images)")
    print(f"SHA256: {digest}")
    print("Review all source/derived-image permissions before uploading; VEDAI is excluded by default.")


if __name__ == "__main__":
    main()

# Benchmark Data

**IEEE DataPort record: pending.** No imagery has been uploaded by this release.
The final record, archive SHA256, source citations, and approved dataset terms
must be added before advertising a complete benchmark download.

## What Is Included In Git

The package contains a frozen manifest for 1,800 triplets: 300 each for airborne
SR, spaceborne SR, SAR-to-optical, synthetic artifacts, timeseries, and ordered
misregistration. Each row has portable image paths, the preferred candidate,
vote counts, and the labeling protocol. Human preferences preserve the paper's
latest-three majority-vote snapshot. Misregistration preferences come from
known severity ordering and have no human votes.

Raw annotator identifiers, timestamps, local paths, duplicate logs, image
caches, and model weights are not included. The small frozen calibration is
bundled separately and is sufficient to score your own images.

## Prepared Archive

Once published, download the archive from IEEE DataPort and extract it into
`data/benchmark/`:

```text
data/benchmark/
  manifest.csv
  checksums.json
  spaceborne_sr/000001/gt.png
  spaceborne_sr/000001/swinir_x4.png
  spaceborne_sr/000001/realesrgan_x4.png
  ...
```

```bash
uv run rspd-evaluate --dataset spaceborne_sr --check-data
```

The archive's `manifest.csv` lists only included subsets; pass it explicitly
when evaluating a partial archive:

```bash
uv run rspd-evaluate --manifest data/benchmark/manifest.csv
```

`--check-data` checks file presence, not file hashes. `checksums.json` in the
archive records individual SHA256 digests for independent integrity checks.

## Source Data And Restrictions

| Subset | Source And Required Preparation |
|---|---|
| Airborne SR | [VEDAI](https://downloads.greyc.fr/vedai/), selected images/crops and SwinIR/Real-ESRGAN outputs. |
| Spaceborne SR | EarthView/Sentinel-2, selected RGB imagery and SwinIR/Real-ESRGAN outputs. |
| SAR-to-optical | [SAR2Opt source collection](https://github.com/MarsZhaoYT/SAR2Opt-Heterogeneous-Dataset), with the fixed paired translation outputs. |
| Synthetic / misregistration | RSICD/RSIC-H reference imagery with fixed perturbations, semantic insertions, or shifts. |
| Timeseries | Selected EarthView/Sentinel-2 dates and RGB crops, used as an annotation control. |

Downloading the original datasets alone does not reproduce the generated
candidates or exact crops. We do not claim that the historical construction
recipes in other research checkouts are a verified reconstruction of this
benchmark. Exact source-download and reconstruction instructions remain needed
for any subset that cannot be redistributed.

**VEDAI is research/non-commercial data.** Obtain it from the original source
and follow its terms. Its derived imagery is excluded from the draft archive
unless redistribution permission is confirmed. The MIT code license does not
grant rights to VEDAI or any other dataset. Check redistribution permissions
for every source and generated derivative before publishing the DataPort bundle.

## Preparing A Release (Maintainers)

Given local files arranged according to the bundled manifest:

```bash
uv run python scripts/package_benchmark.py --data-root data/benchmark
```

This writes an ignored draft ZIP under `data/release/`, with only referenced
images, a subset manifest, and checksums. It excludes airborne SR by default.
`--include-airborne` is permitted only after confirming redistribution rights.
The script does not upload anything or determine licensing eligibility.

Upload the approved archive to a versioned IEEE DataPort record and add its
download URL and SHA256 here. Do not commit the image archive to Git.

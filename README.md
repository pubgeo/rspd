# Remote Sensing Perceptual Distance (RSPD)

RSPD compares two RGB images using intermediate attention from RemoteCLIP.
It was introduced in the accepted WACV 2027 paper **Rethinking Perceptual
Similarity for Remote Sensing: Multi-Layer Attention Scoring and Human
Preference Benchmarking**.

**Lower scores mean greater similarity.** Each pair is scored independently;
no third image, artifact label, or dataset-specific calibration is needed.

## Install

Python 3.12 or newer is required. From a clone:

```bash
git clone https://github.com/pubgeo/rspd.git
cd rspd
uv sync
```

Or install the package into an existing environment:

```bash
pip install git+https://github.com/pubgeo/rspd.git
```

The default install contains only RSPD and its inference dependencies. CPU works;
CUDA is used automatically when available. If your machine needs a particular
PyTorch build, follow [PyTorch's installation instructions](https://pytorch.org/get-started/locally/).
There is no mandatory CUDA-specific package index.

## Compare Your Images

```bash
uv run rspd reference.png candidate.png
uv run rspd reference.png candidate.png --device cpu --details
```

The first command prints a distance; `--details` prints JSON with per-layer
distances and calibration diagnostics. Reuse one metric instance when scoring
many pairs from Python:

```python
from PIL import Image
from rspd import RSPDMean7

metric = RSPDMean7()  # Or device="cpu".
with Image.open("reference.png") as image:
    reference = image.convert("RGB")
with Image.open("candidate.png") as image:
    candidate = image.convert("RGB")

distance = metric.score_pair(reference, candidate)
details = metric.score_pair(reference, candidate, return_details=True)
print(distance)
```

`metric.compare_pairwise(reference, candidate_a, candidate_b)` returns the
preferred candidate, both scores, or a deterministic tie. PIL RGB images are the
recommended input; NumPy inputs must be `uint8` HWC images. Torch inputs may be
CHW or HWC, using `uint8` or floating values in `[0, 1]`.

RSPD uses the original resize/center-crop to 224 pixels, not geographic
registration or automatic tiling. It measures RGB visual fidelity, not
multispectral or radiometric correctness. Scores can exceed 1; they are not
probabilities. The implementation is for evaluation, not a differentiable loss.

## Weights And Offline Use

The first score downloads the pinned RemoteCLIP ViT-B/32 checkpoint (about
600 MB) from its original Hugging Face repository. The frozen calibration is
already included in the package. No benchmark images are needed for inference.

```bash
uv run python -m rspd.download
```

Run this once with network access before working offline. `HF_HOME` controls
the Hugging Face cache; `HF_HUB_OFFLINE=1` disables network access after the
weights are cached. Checkpoint revision and SHA256 verification prevent silent
weight changes.

## Run The Paper Benchmark

**IEEE DataPort download link: pending publication.** The code, frozen labels,
and evaluators are ready; a fresh clone does not contain benchmark images.
See [data access and licensing](docs/DATA.md) for what will be available.

Extract the prepared archive into `data/benchmark/`, or pass another location
with `--data-root`. Check a subset before loading any models:

```bash
uv run rspd-evaluate --dataset spaceborne_sr --check-data
uv run rspd-evaluate --dataset spaceborne_sr --limit 3
uv run rspd-evaluate --metrics rspd rspd_q95
```

The last command requires the complete benchmark, including the restricted
airborne SR subset. Missing files produce an error before model downloads;
they are not silently skipped. To evaluate only the datasets you have, select
them explicitly with `--dataset`.

Optional baselines:

```bash
uv sync --extra baselines
uv run rspd-evaluate --dataset spaceborne_sr --metrics rspd msssim lpips dreamsim remoteclip
```

Each run writes only `evaluation.csv` and `evaluation_summary.json` under
`results/benchmark/` by default. Results retain individual triplet scores;
the summary reports each dataset, ties, and the Table 2 average only when its
five complete subsets are present. Timeseries is reported separately.

See [evaluation details](docs/EVALUATION.md) for DiffSim, the exact baseline
settings, custom datasets, and the paper's reference results. For the metric
and calibration explanation, see [method details](docs/METHOD.md).

## Development

```bash
uv run pytest
uv run ruff check .
RSPD_RUN_INTEGRATION=1 uv run pytest tests/test_rspd.py -k real_multilayer
```

Ordinary tests do not download weights. The opt-in integration test uses the
checkpoint cache or downloads the pinned checkpoint; no research data is needed.
The command-line tools use fixed seeds, deterministic PyTorch algorithms, and
disabled TF32/cuDNN benchmarking. Library calls do not change global settings;
call `rspd.reproducibility.configure_reproducibility()` before constructing the
metric when you want the same settings. Exact cross-hardware equality is not
guaranteed.

## Citation And License

Citation metadata is in [CITATION.cff](CITATION.cff). The final paper link and
proceedings identifier will be added when published.

Original code is [MIT licensed](LICENSE). Third-party models and datasets
retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

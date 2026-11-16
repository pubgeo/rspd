# Paper Evaluation

The evaluator scores whole RGB images with the paper's global preprocessing,
not tile aggregates. Each pair is independent. Human labels, candidate ordering,
the calibration, and the RemoteCLIP checkpoint are frozen.

## Commands

```bash
uv run rspd-evaluate --metrics rspd rspd_q95 --output-dir results/rspd
uv sync --extra baselines
uv run rspd-evaluate --metrics msssim lpips dreamsim remoteclip --output-dir results/baselines
```

`--dataset spaceborne_sr --limit 3` makes a quick smoke run; it is not a paper
result. Missing data fails preflight rather than changing the evaluation panel.
GPU is recommended for full evaluation, particularly for DiffSim. Prefer separate
commands per large baseline when GPU memory is limited.

Only two files are produced per run: `evaluation.csv` (both candidate distances,
prediction, target, correctness) and `evaluation_summary.json` (per-dataset
accuracy/ties, Table 2 average, reproducibility settings and runtime). Ties use
an absolute tolerance of `1e-12` and count as incorrect. A Table 2 average is
reported only when all five named subsets contain 300 triplets each; the
timeseries control is never included in that average.

## Baseline Definitions

- **MS-SSIM:** the paper's fixed five-scale RGB scikit-image SSIM implementation,
  box downsampling, and weighted geometric mean with weights
  `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`. This is explicitly the paper's
  implementation, not an interchangeable standard MS-SSIM library. Output is
  converted to distance as `1 - similarity`.
- **LPIPS:** upstream 0.1.4, AlexNet, version 0.1, RGB tensors in `[-1, 1]`.
- **DreamSim:** upstream 0.2.1, pretrained ensemble and its own preprocessing.
  Weights are cached under `torch.hub.get_dir()/dreamsim` (controlled by `TORCH_HOME`).
- **RemoteCLIP:** final-embedding cosine distance from the same pinned
  checkpoint as RSPD, using the same preprocessing.
- **DiffSim:** Stable Diffusion 1.5, 512-pixel inputs, `up_blocks`, layer 0,
  step 600, prompt `High quality image`, no IP adapter, seed 2334. The adapter
  uses the VAE posterior mode rather than sampling, matching the deterministic
  paper rerun.

The optional baseline packages are installed from upstream, not vendored here.
Their versions are pinned, but dependency/hardware differences can still affect
numerical results. `benchmarks/paper_results.json` records the original saved
paper results, not a claim that every public dependency combination has been
rerun. Run the same panel before claiming exact reproduction.

## Optional DiffSim

Obtain the upstream checkout separately and review its license. The adapter
expects `diffsim.diffsim.DiffSim` and its `prepare_image_latents` method.

```bash
git clone https://github.com/showlab/DiffSim.git ../DiffSim
uv sync --extra diffsim
uv run rspd-evaluate --metrics diffsim --diffsim-root ../DiffSim --output-dir results/diffsim
```

The external checkout is deliberately not installed as a mandatory dependency.
Its license and a stable source revision could not be verified during this
transition; verify both before declaring a fully pinned DiffSim reproduction.
The public DiffSim adapter has not yet been exercised end-to-end; an arbitrary
future upstream checkout is not guaranteed to match the paper version.

## Your Own Evaluation Set

Supply a CSV and an image root:

```csv
dataset,triplet_id,reference,candidate_a,candidate_b,preferred
custom,000001,scene/reference.png,scene/a.png,scene/b.png,a
```

```bash
uv run rspd-evaluate --manifest my_triplets.csv --data-root /path/to/images
```

Paths must be relative to the supplied root; `preferred` is `a` or `b`. There
is no requirement to use the paper's datasets or to refit calibration.

## Paper Reference Values

Accuracy (%), preserving Table 2's row order. Values below are from the saved
deterministic paper runs, not new experiments performed during packaging.

| Dataset | MS-SSIM | LPIPS | DreamSim | DiffSim | RemoteCLIP | RSPD-Q95 | RSPD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Airborne SR | 80.00 | 34.67 | 69.67 | 33.00 | 62.33 | 80.67 | 80.33 |
| Spaceborne SR | 95.33 | 35.00 | 90.00 | 25.00 | 41.33 | 95.33 | 95.33 |
| SAR-to-optical | 95.00 | 62.33 | 51.00 | 69.00 | 24.33 | 90.00 | 96.00 |
| Synthetic | 75.33 | 67.67 | 81.33 | 67.00 | 66.33 | 79.33 | 79.33 |
| Misregistration | 62.33 | 91.67 | 97.33 | 88.00 | 81.00 | 94.33 | 94.00 |
| Average | 81.60 | 58.27 | 77.87 | 56.40 | 55.07 | 87.93 | 89.00 |

RSPD reaches 97.00% on the separately reported timeseries control. The machine-
readable reference file also records its unanimous-vote counts per human subset.

# Method

For each image pair, RSPD extracts RemoteCLIP attention at the seven unique
layers `0, 1, 2, 3, 4, 8, 9`, computes symmetric Aligned Attention Score (AAS)
distances, calibrates each distance using a fixed distribution, and averages
the seven calibrated values. Lower is more similar. The reference and candidate
are processed with the checkpoint's RGB evaluation transform in float32.

## Frozen Calibration

The bundled `rspd-calibration-v1` artifact uses 500 distances from each of
11 controlled artifact families per layer. It is independent of human labels
and never changes based on the images being compared. Using RSPD on a new
dataset does not require fitting another calibration.

- **RSPD / pooled ECDF:** asks where a distance falls in the layer's frozen
  controlled-distance distribution. A value around 0.8 means roughly 80% of
  calibration distances were at or below that distance. Values are linearly
  interpolated, with zero anchored at zero. Beyond the largest calibration
  distance, the score continues as `1 + (distance - maximum) / Q95`.
- **RSPD-Q95:** divides each raw distance by its frozen 95th percentile. This
  is a simple rescaling, retained as the paper's calibration ablation.

Both methods average each selected layer once, without clipping at 1. Neither
is a universal probability of poor quality. Calibration reflects the controlled
artifact distribution and may transfer imperfectly to new imagery or sensors.

```python
from rspd import RSPDMean7

ecdf = RSPDMean7()
q95 = RSPDMean7(ecdf.score_model, ecdf.calibration, method="pooled_q95")
details = ecdf.score_pair(reference, candidate, return_details=True)
q95_details = q95.score_raw_distances(details.raw_layer_distances)
```

`rspd.defaults.calibration_metadata()` reports the model/checkpoint identity,
source-table checksums, preprocessing, and calibration checksum. These are
unchanged from the paper. Research calibration fitting remains available as
`rspd.metric.fit_pooled_calibration`; it is not part of normal setup.

Family-specific normalization belongs to layer discovery only; deployed
inference uses the pooled, family-independent calibration. This repository
does not include the retired candidate-dependent triplet scorer, model-training
code, or the exploratory generation toolchain.

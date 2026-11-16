from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from rspd.remoteclip import ImageType


RSPD_LAYERS = (0, 1, 2, 3, 4, 8, 9)
CALIBRATION_METHODS = ("pooled_q95", "pooled_ecdf")
CalibrationMethod = Literal["pooled_q95", "pooled_ecdf"]


class LayerScoreModel(Protocol):
    model_name: str

    def clip_cross_scores(
        self,
        images1: ImageType | list[ImageType],
        images2: ImageType | list[ImageType],
        target_layers: list[int] | tuple[int, ...],
    ) -> dict[int, Any]: ...


@dataclass(frozen=True)
class RSPDScore:
    score: float
    method: str
    raw_layer_distances: dict[int, float]
    calibrated_layer_distances: dict[int, float]
    layer_contributions: dict[int, float]
    q95_exceeded_layers: list[int]
    calibration_max_exceeded_layers: list[int]
    calibration_version: str
    model_name: str


@dataclass(frozen=True)
class RSPDComparison:
    preferred: Literal["candidate_a", "candidate_b", "tie"]
    candidate_a: RSPDScore
    candidate_b: RSPDScore


@dataclass(frozen=True)
class PooledCalibration:
    schema_version: str
    calibration_version: str
    model_name: str
    layers: tuple[int, ...]
    quantile: float
    q95_scales: dict[int, float]
    sorted_distances: dict[int, tuple[float, ...]]
    family_counts: dict[int, dict[str, int]]
    source_files: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> PooledCalibration:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            schema_version=str(payload["schema_version"]),
            calibration_version=str(payload["calibration_version"]),
            model_name=str(payload["model_name"]),
            layers=tuple(int(layer) for layer in payload["layers"]),
            quantile=float(payload["quantile"]),
            q95_scales={int(layer): float(value) for layer, value in payload["q95_scales"].items()},
            sorted_distances={
                int(layer): tuple(float(value) for value in values)
                for layer, values in payload["sorted_distances"].items()
            },
            family_counts={
                int(layer): {str(family): int(count) for family, count in counts.items()}
                for layer, counts in payload["family_counts"].items()
            },
            source_files=tuple(str(path) for path in payload["source_files"]),
        )

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return output_path

    def validate(self) -> None:
        if self.schema_version != "rspd-calibration-v1":
            raise ValueError(f"Unsupported calibration schema: {self.schema_version}")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("Calibration quantile must be between 0 and 1.")
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError(f"Calibration layers must be a non-empty unique sequence: {self.layers}")
        expected_layers = set(self.layers)
        for field_name, values in (
            ("q95_scales", self.q95_scales),
            ("sorted_distances", self.sorted_distances),
            ("family_counts", self.family_counts),
        ):
            if set(values) != expected_layers:
                raise ValueError(f"Calibration {field_name} keys do not match layers {self.layers}.")
        for layer in self.layers:
            scale = self.q95_scales.get(layer, 0.0)
            values = self.sorted_distances.get(layer, ())
            counts = self.family_counts.get(layer, {})
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Layer {layer} has an invalid Q95 scale: {scale}")
            if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(f"Layer {layer} has invalid empirical calibration distances.")
            if tuple(sorted(values)) != values:
                raise ValueError(f"Layer {layer} empirical calibration distances are not sorted.")
            if not counts or len(set(counts.values())) != 1:
                raise ValueError(f"Layer {layer} artifact-family counts are not balanced: {counts}")

        family_sets = {layer: frozenset(self.family_counts[layer]) for layer in self.layers}
        if len(set(family_sets.values())) != 1:
            raise ValueError(f"Calibration layers do not contain the same artifact families: {family_sets}")

    def calibrate(self, layer: int, raw_distance: float, method: CalibrationMethod) -> float:
        raw_distance = float(raw_distance)
        if not math.isfinite(raw_distance) or raw_distance < 0.0:
            raise ValueError(f"Raw layer distance must be finite and nonnegative, got {raw_distance}.")
        if method == "pooled_q95":
            return raw_distance / self.q95_scales[layer]
        if method != "pooled_ecdf":
            raise ValueError(f"Unsupported calibration method: {method}")
        if raw_distance == 0.0:
            return 0.0

        values = np.asarray(self.sorted_distances[layer], dtype=np.float64)
        unique_values, counts = np.unique(values, return_counts=True)
        cumulative = np.cumsum(counts, dtype=np.float64) / float(values.size)
        if unique_values[0] > 0.0:
            unique_values = np.insert(unique_values, 0, 0.0)
            cumulative = np.insert(cumulative, 0, 0.0)
        if raw_distance <= float(unique_values[-1]):
            return float(np.interp(raw_distance, unique_values, cumulative))

        return 1.0 + (raw_distance - float(unique_values[-1])) / self.q95_scales[layer]

    def q95_exceeded(self, layer: int, raw_distance: float) -> bool:
        return float(raw_distance) > self.q95_scales[layer]

    def calibration_max_exceeded(self, layer: int, raw_distance: float) -> bool:
        return float(raw_distance) > self.sorted_distances[layer][-1]


def fit_pooled_calibration(
    score_csvs: list[str | Path],
    *,
    layers: tuple[int, ...] = RSPD_LAYERS,
    quantile: float = 0.95,
    calibration_version: str = "mean7-pooled-v1",
) -> PooledCalibration:
    if not 0.0 < quantile < 1.0:
        raise ValueError("Calibration quantile must be between 0 and 1.")

    distances: dict[int, list[float]] = defaultdict(list)
    family_counts: dict[int, Counter[str]] = defaultdict(Counter)
    model_names: set[str] = set()
    seen_rows: set[tuple[str, str, str, int, str, str]] = set()
    selected_layers = set(layers)
    source_paths = [Path(path).resolve() for path in score_csvs]
    for source_path in source_paths:
        if not source_path.is_file():
            raise FileNotFoundError(f"Calibration score CSV not found: {source_path}")
        with source_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                layer = int(row["layer"])
                if layer not in selected_layers:
                    continue
                if row.get("candidate_filename", "") == row.get("reference_filename", ""):
                    continue
                distance = float(row["distance"])
                if not math.isfinite(distance) or distance < 0.0:
                    raise ValueError(f"Invalid distance in {source_path}: {distance}")
                if float(row.get("distance_scale", "1") or 1.0) != 1.0:
                    raise ValueError("Pairwise calibration requires controlled distances with distance_scale=1.")
                family = str(row["family"]).strip().lower()
                row_key = (
                    str(row.get("set", "")),
                    family,
                    str(row.get("variant", "")),
                    layer,
                    str(row.get("reference_filename", "")),
                    str(row.get("candidate_filename", "")),
                )
                if row_key in seen_rows:
                    raise ValueError(f"Duplicate calibration row encountered for {row_key}.")
                seen_rows.add(row_key)
                distances[layer].append(distance)
                family_counts[layer][family] += 1
                model_names.add(str(row["model_name"]).strip())

    if len(model_names) != 1:
        raise ValueError(f"Expected exactly one RemoteCLIP model in calibration rows, found: {sorted(model_names)}")
    family_sets = {layer: frozenset(family_counts[layer]) for layer in layers}
    if len(set(family_sets.values())) != 1:
        raise ValueError(f"Calibration layers do not contain the same artifact families: {family_sets}")
    for layer in layers:
        counts = family_counts[layer]
        if not distances[layer]:
            raise ValueError(f"No calibration distances found for layer {layer}.")
        if len(set(counts.values())) != 1:
            raise ValueError(f"Layer {layer} artifact families are not equally represented: {dict(counts)}")

    calibration = PooledCalibration(
        schema_version="rspd-calibration-v1",
        calibration_version=calibration_version,
        model_name=model_names.pop(),
        layers=layers,
        quantile=quantile,
        q95_scales={
            layer: float(np.quantile(np.asarray(distances[layer], dtype=np.float64), quantile))
            for layer in layers
        },
        sorted_distances={layer: tuple(sorted(distances[layer])) for layer in layers},
        family_counts={layer: dict(sorted(family_counts[layer].items())) for layer in layers},
        source_files=tuple(str(path) for path in source_paths),
    )
    calibration.validate()
    return calibration


class RSPDMean7:
    def __init__(
        self,
        score_model: LayerScoreModel | None = None,
        calibration: PooledCalibration | None = None,
        *,
        method: CalibrationMethod = "pooled_ecdf",
        device: str | None = None,
    ) -> None:
        if calibration is None:
            from rspd.defaults import load_calibration

            calibration = load_calibration()
        calibration.validate()
        if tuple(calibration.layers) != RSPD_LAYERS:
            raise ValueError(f"Mean7 requires layers {RSPD_LAYERS}, got {calibration.layers}.")
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"Unsupported calibration method: {method}")
        if score_model is None:
            from rspd.remoteclip import RemoteCLIPScore

            score_model = RemoteCLIPScore(model_name=calibration.model_name, device=device)
        if score_model.model_name != calibration.model_name:
            raise ValueError(
                f"Calibration model {calibration.model_name!r} does not match scorer model {score_model.model_name!r}."
            )
        self.score_model = score_model
        self.calibration = calibration
        self.method = method

    def score_raw_distances(self, raw_layer_distances: dict[int, float]) -> RSPDScore:
        if tuple(sorted(raw_layer_distances)) != tuple(sorted(RSPD_LAYERS)):
            raise ValueError(f"Mean7 requires raw distances for exactly these layers: {RSPD_LAYERS}")
        calibrated = {
            layer: self.calibration.calibrate(layer, float(raw_layer_distances[layer]), self.method)
            for layer in RSPD_LAYERS
        }
        contributions = {layer: value / len(RSPD_LAYERS) for layer, value in calibrated.items()}
        return RSPDScore(
            score=sum(contributions.values()),
            method=self.method,
            raw_layer_distances={layer: float(raw_layer_distances[layer]) for layer in RSPD_LAYERS},
            calibrated_layer_distances=calibrated,
            layer_contributions=contributions,
            q95_exceeded_layers=[
                layer
                for layer in RSPD_LAYERS
                if self.calibration.q95_exceeded(layer, raw_layer_distances[layer])
            ],
            calibration_max_exceeded_layers=[
                layer
                for layer in RSPD_LAYERS
                if self.calibration.calibration_max_exceeded(layer, raw_layer_distances[layer])
            ],
            calibration_version=self.calibration.calibration_version,
            model_name=self.score_model.model_name,
        )

    def score_pair(
        self,
        reference: ImageType,
        candidate: ImageType,
        *,
        return_details: bool = False,
    ) -> float | RSPDScore:
        similarities = self.score_model.clip_cross_scores(reference, candidate, RSPD_LAYERS)
        raw_distances = {}
        for layer in RSPD_LAYERS:
            similarity = similarities[layer]
            value = similarity.item() if hasattr(similarity, "item") else float(similarity)
            if not math.isfinite(value):
                raise ValueError(f"RemoteCLIP returned a non-finite similarity for layer {layer}: {value}")
            raw_distances[layer] = max(0.0, 1.0 - max(-1.0, min(1.0, value)))
        details = self.score_raw_distances(raw_distances)
        return details if return_details else details.score

    def compare_pairwise(
        self,
        reference: ImageType,
        candidate_a: ImageType,
        candidate_b: ImageType,
        *,
        tie_tolerance: float = 1e-12,
    ) -> RSPDComparison:
        score_a = self.score_pair(reference, candidate_a, return_details=True)
        score_b = self.score_pair(reference, candidate_b, return_details=True)
        assert isinstance(score_a, RSPDScore) and isinstance(score_b, RSPDScore)
        if math.isclose(score_a.score, score_b.score, rel_tol=0.0, abs_tol=tie_tolerance):
            preferred: Literal["candidate_a", "candidate_b", "tie"] = "tie"
        else:
            preferred = "candidate_a" if score_a.score < score_b.score else "candidate_b"
        return RSPDComparison(preferred=preferred, candidate_a=score_a, candidate_b=score_b)

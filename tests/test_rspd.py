from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path

import pytest


from rspd.reproducibility import configure_reproducibility
from rspd.metric import (  # noqa: E402
    PooledCalibration,
    RSPD_LAYERS,
    RSPDMean7,
    RSPDScore,
    fit_pooled_calibration,
)


class FakeLayerScoreModel:
    model_name = "ViT-B-32"

    def clip_cross_scores(self, images1, images2, target_layers):
        difference = abs(float(images1) - float(images2))
        return {layer: 1.0 - difference * (layer + 1) / 100.0 for layer in target_layers}


def test_configure_reproducibility_resets_rngs_and_torch_backends() -> None:
    import numpy as np
    import torch

    configure_reproducibility(17)
    first = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    configure_reproducibility(17)
    second = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

    assert first == second
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def make_calibration() -> PooledCalibration:
    return PooledCalibration(
        schema_version="rspd-calibration-v1",
        calibration_version="test-v1",
        model_name="ViT-B-32",
        layers=RSPD_LAYERS,
        quantile=0.95,
        q95_scales={layer: 0.4 + layer / 100.0 for layer in RSPD_LAYERS},
        sorted_distances={layer: (0.1, 0.2, 0.3, 0.4 + layer / 100.0) for layer in RSPD_LAYERS},
        family_counts={layer: {"blur": 4, "noise": 4} for layer in RSPD_LAYERS},
        source_files=("test.csv",),
    )


def write_score_csv(path: Path, *, unequal: bool = False) -> None:
    fieldnames = [
        "family",
        "layer",
        "model_name",
        "distance",
        "distance_scale",
        "reference_filename",
        "candidate_filename",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer in RSPD_LAYERS:
            for family in ("blur", "noise"):
                count = 2 if family == "blur" or not unequal else 1
                for index in range(count):
                    writer.writerow(
                        {
                            "family": family,
                            "layer": layer,
                            "model_name": "ViT-B-32",
                            "distance": 0.1 * (index + 1) + layer / 1000.0,
                            "distance_scale": 1,
                            "reference_filename": "reference.png",
                            "candidate_filename": f"{family}_{index}.png",
                        }
                    )


def test_fit_save_and_load_pooled_calibration(tmp_path: Path) -> None:
    score_csv = tmp_path / "scores.csv"
    write_score_csv(score_csv)
    calibration = fit_pooled_calibration([score_csv], calibration_version="fixture-v1")
    output = calibration.save(tmp_path / "calibration.json")
    loaded = PooledCalibration.load(output)

    assert loaded == calibration
    assert loaded.calibration_version == "fixture-v1"
    assert all(set(counts.values()) == {2} for counts in loaded.family_counts.values())


def test_calibration_rejects_retired_schema() -> None:
    calibration = make_calibration()
    object.__setattr__(calibration, "schema_version", "unsupported-calibration-v0")

    with pytest.raises(ValueError, match="Unsupported calibration schema"):
        calibration.validate()


def test_fit_rejects_unbalanced_artifact_families(tmp_path: Path) -> None:
    score_csv = tmp_path / "scores.csv"
    write_score_csv(score_csv, unequal=True)
    with pytest.raises(ValueError, match="not equally represented"):
        fit_pooled_calibration([score_csv])


def test_q95_scaling_is_finite_and_does_not_clip() -> None:
    calibration = make_calibration()
    value = calibration.calibrate(0, 0.8, "pooled_q95")
    assert value == pytest.approx(2.0)
    assert math.isfinite(value)
    assert calibration.calibrate(0, 0.0, "pooled_q95") == 0.0


def test_ecdf_is_monotonic_and_continues_above_calibration_max() -> None:
    calibration = make_calibration()
    values = [calibration.calibrate(0, raw, "pooled_ecdf") for raw in (0.0, 0.15, 0.3, 0.4, 0.5, 0.8)]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] > values[-2] > 1.0


def test_calibration_rejects_duplicate_rows(tmp_path: Path) -> None:
    score_csv = tmp_path / "scores.csv"
    write_score_csv(score_csv)
    duplicate_csv = tmp_path / "duplicate.csv"
    duplicate_csv.write_text(score_csv.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate calibration row"):
        fit_pooled_calibration([score_csv, duplicate_csv])


def test_calibration_rejects_inconsistent_family_sets(tmp_path: Path) -> None:
    score_csv = tmp_path / "scores.csv"
    write_score_csv(score_csv)
    lines = score_csv.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        fields = line.split(",")
        if len(fields) > 1 and fields[0] == "noise" and fields[1] == "0":
            fields[0] = "other"
            lines[index] = ",".join(fields)
    score_csv.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same artifact families"):
        fit_pooled_calibration([score_csv])


@pytest.mark.parametrize("method", ["pooled_q95", "pooled_ecdf"])
def test_pairwise_mean7_invariants_and_candidate_independence(method: str) -> None:
    scorer = RSPDMean7(FakeLayerScoreModel(), make_calibration(), method=method)
    identity = scorer.score_pair(1.0, 1.0, return_details=True)
    first = scorer.score_pair(1.0, 2.0, return_details=True)
    _unrelated = scorer.score_pair(1.0, 9.0)
    repeated = scorer.score_pair(1.0, 2.0, return_details=True)
    reverse = scorer.score_pair(2.0, 1.0)

    assert isinstance(identity, RSPDScore) and isinstance(first, RSPDScore) and isinstance(repeated, RSPDScore)
    assert identity.score == pytest.approx(0.0)
    assert first == repeated
    assert first.score == pytest.approx(reverse)
    assert math.isfinite(first.score)
    assert tuple(first.raw_layer_distances) == RSPD_LAYERS

    comparison = scorer.compare_pairwise(1.0, 2.0, 4.0)
    swapped = scorer.compare_pairwise(1.0, 4.0, 2.0)
    tied = scorer.compare_pairwise(1.0, 2.0, 2.0)
    assert comparison.preferred == "candidate_a"
    assert swapped.preferred == "candidate_b"
    assert tied.preferred == "tie"


def test_rspd_defaults_to_ecdf_calibration() -> None:
    assert RSPDMean7(FakeLayerScoreModel(), make_calibration()).method == "pooled_ecdf"


def test_score_from_hidden_states_preserves_batch_dimension() -> None:
    import torch

    from rspd.remoteclip import RemoteCLIPScore

    torch.manual_seed(7)
    attention = torch.nn.MultiheadAttention(embed_dim=4, num_heads=2)
    model = RemoteCLIPScore.__new__(RemoteCLIPScore)
    hidden_a = torch.randn(5, 2, 4)
    hidden_b = torch.randn(5, 2, 4)

    batched = model.score_from_hidden_states(hidden_a, hidden_b, attention)
    individual = torch.stack(
        [
            model.score_from_hidden_states(
                hidden_a[:, index : index + 1, :],
                hidden_b[:, index : index + 1, :],
                attention,
            )[0]
            for index in range(2)
        ]
    )

    assert batched.shape == (2,)
    assert torch.allclose(batched, individual)


def test_clip_cross_score_rejects_excessively_negative_layer() -> None:
    from rspd.remoteclip import RemoteCLIPScore

    model = RemoteCLIPScore.__new__(RemoteCLIPScore)
    model.resblocks = [object()] * 12

    with pytest.raises(IndexError, match="out of range"):
        model.clip_cross_score(1.0, 2.0, -13)


@pytest.mark.skipif(os.environ.get("RSPD_RUN_INTEGRATION") != "1", reason="set RSPD_RUN_INTEGRATION=1")
def test_real_multilayer_matches_single_layer_and_known_fixture() -> None:
    from PIL import Image

    from rspd.remoteclip import RemoteCLIPScore

    import numpy as np

    reference = Image.fromarray(np.random.default_rng(7).integers(0, 256, (224, 224, 3), dtype=np.uint8))
    candidate = reference.point(lambda value: value // 2)
    model = RemoteCLIPScore(device="cpu")
    multilayer = model.clip_cross_scores(reference, candidate, RSPD_LAYERS)
    single = model.clip_cross_score(reference, candidate, 0)
    batched = model.clip_cross_scores([reference, candidate], [candidate, reference], [0])
    multilayer_distance = 1.0 - float(multilayer[0].item())
    single_distance = 1.0 - float(single.item())
    assert multilayer_distance == pytest.approx(single_distance, abs=1e-7)
    assert math.isfinite(multilayer_distance)
    assert batched[0].shape == (2,)
    assert batched[0][0].item() == pytest.approx(multilayer[0].item(), abs=1e-7)
    reverse_single = model.clip_cross_score(candidate, reference, 0)
    assert batched[0][1].item() == pytest.approx(reverse_single.item(), abs=1e-7)

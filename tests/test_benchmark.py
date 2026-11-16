import csv
from importlib.resources import files

import pytest
from PIL import Image

from rspd.benchmark import evaluate, load_manifest, main


def test_frozen_manifest_has_paper_panel_and_vote_counts():
    from collections import Counter

    with files("rspd").joinpath("data/benchmark.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1800
    assert set(Counter(row["dataset"] for row in rows).values()) == {300}
    assert sum(row["votes_a"] == "3" or row["votes_b"] == "3" for row in rows) == 1161
    assert sum(row["protocol"] == "ordered" for row in rows) == 300
    assert all(not row["reference"].startswith("/") for row in rows)


def test_evaluation_orientation_ties_and_partial_panel():
    rows = [{"dataset": "synthetic", "triplet_id": str(i), "reference_path": "0",
             "candidate_a_path": a, "candidate_b_path": b, "preferred": "a"}
            for i, (a, b) in enumerate((("1", "2"), ("2", "1"), ("1", "1")))]
    results, summaries = evaluate(rows, {"fixture": lambda reference, candidate: float(candidate)})
    assert [row["prediction"] for row in results] == ["a", "b", "tie"]
    assert summaries["fixture"]["by_dataset"]["synthetic"]["correct"] == 1
    assert summaries["fixture"]["by_dataset"]["synthetic"]["ties"] == 1
    assert summaries["fixture"]["table2_average"] is None


def test_manifest_preflight_and_portability(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("dataset,triplet_id,reference,candidate_a,candidate_b,preferred\ncustom,1,a.png,a.png,b.png,a\n")
    with pytest.raises(FileNotFoundError, match="2 benchmark images missing"):
        load_manifest(manifest, tmp_path)
    Image.new("RGB", (16, 16)).save(tmp_path / "a.png")
    Image.new("RGB", (16, 16)).save(tmp_path / "b.png")
    rows = load_manifest(manifest, tmp_path)
    assert len(rows) == 1
    assert rows[0]["reference_path"] == str(tmp_path / "a.png")
    manifest.write_text(manifest.read_text().replace("a.png", "../a.png"))
    with pytest.raises(ValueError, match="portable relative paths"):
        load_manifest(manifest, tmp_path)


def test_msssim_identity():
    pytest.importorskip("skimage")
    from rspd.baselines import create_msssim_scorer
    import numpy as np

    image = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as directory:
        path = Path(directory) / "reference.png"
        Image.fromarray(image).save(path)
        assert create_msssim_scorer()(str(path), str(path)) == pytest.approx(0)


def test_cli_checks_files_before_model_loading(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--data-root", str(tmp_path), "--limit", "1", "--check-data"])
    assert exc.value.code == 1

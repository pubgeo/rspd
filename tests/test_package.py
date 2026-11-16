import hashlib

import pytest

from rspd.defaults import calibration_metadata, load_calibration
from rspd import remoteclip
from rspd.remoteclip import HF_REVISION, resolve_checkpoint


def test_bundled_calibration_and_provenance():
    calibration = load_calibration()
    assert calibration.layers == (0, 1, 2, 3, 4, 8, 9)
    assert calibration.schema_version == "rspd-calibration-v1"
    assert all(len(values) == 5500 for values in calibration.sorted_distances.values())
    assert calibration_metadata()["calibration_version"] == calibration.calibration_version
    assert all(not source.startswith("/") for source in calibration.source_files)


def test_checkpoint_download_is_pinned_and_verified(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test weights")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(checkpoint)

    monkeypatch.setattr("rspd.remoteclip.hf_hub_download", download)
    monkeypatch.setitem(
        remoteclip.CHECKPOINT_SHA256,
        "ViT-B-32", hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    assert resolve_checkpoint("ViT-B-32", None) == checkpoint
    assert calls[0]["revision"] == HF_REVISION
    assert calls[0]["cache_dir"] is None
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_checkpoint("ViT-B-32", None)


def test_missing_checkpoint_explains_offline_setup(monkeypatch):
    def download(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr("rspd.remoteclip.hf_hub_download", download)
    with pytest.raises(RuntimeError, match="python -m rspd.download"):
        resolve_checkpoint("ViT-B-32", None)

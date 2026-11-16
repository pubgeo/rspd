"""Frozen calibration bundled with the inference package."""

import hashlib
import json
from importlib.resources import as_file, files

from rspd.metric import PooledCalibration


def load_calibration() -> PooledCalibration:
    with as_file(files("rspd").joinpath("data/rspd_calibration.json")) as path:
        calibration = PooledCalibration.load(path)
    calibration.validate()
    return calibration


def calibration_metadata() -> dict:
    data = files("rspd").joinpath("data")
    return {
        **json.loads(data.joinpath("provenance.json").read_text(encoding="utf-8")),
        "calibration_sha256": hashlib.sha256(data.joinpath("rspd_calibration.json").read_bytes()).hexdigest(),
    }

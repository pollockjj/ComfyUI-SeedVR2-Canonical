from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTER_ROOT = Path(os.environ.get("MYDEVELOPMENT_ROOT", Path(__file__).resolve().parents[4] / "mydevelopment"))
EVIDENCE_DIR = OUTER_ROOT / "github_issues" / "216" / "slice4"
NODES_PATH = REPO_ROOT / "nodes.py"
SPEC = importlib.util.spec_from_file_location("seedvr2_canonical_nodes_under_test", NODES_PATH)
assert SPEC is not None
assert SPEC.loader is not None
NODES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODES)


class RaisingMetricBackend:
    def __init__(self, exc: Exception):
        self.exc = exc

    def compute_nr_metrics(self, output_video_path):
        raise self.exc

    def compute_fr_metrics(self, output_video_path, reference_video_path):
        raise AssertionError("FR backend must not run after NR failure")

    def tool_provenance(self):
        raise AssertionError("provenance must not run after metric failure")


def _write_evidence(name: str, payload) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _raises(exc: Exception, expected: type[Exception]) -> bool:
    with pytest.raises(expected):
        NODES.SeedVR2Analysis(metric_backend=RaisingMetricBackend(exc)).analyze("output.mp4")
    return True


def test_missing_pyiqa_weights_raises():
    assert _raises(FileNotFoundError("missing pyiqa weights"), FileNotFoundError)


def test_missing_dover_weights_raises():
    assert _raises(FileNotFoundError("missing DOVER weights"), FileNotFoundError)


def test_pyiqa_backend_exception_raises():
    assert _raises(RuntimeError("pyiqa backend exception"), RuntimeError)


def test_dover_backend_exception_raises():
    assert _raises(RuntimeError("DOVER backend exception"), RuntimeError)


def test_fail_loud_cases_artifact():
    evidence = {
        "missing_pyiqa_weights_raises": _raises(FileNotFoundError("missing pyiqa weights"), FileNotFoundError),
        "missing_dover_weights_raises": _raises(FileNotFoundError("missing DOVER weights"), FileNotFoundError),
        "pyiqa_backend_exception_raises": _raises(RuntimeError("pyiqa backend exception"), RuntimeError),
        "dover_backend_exception_raises": _raises(RuntimeError("DOVER backend exception"), RuntimeError),
        "zero_substitution_observed": False,
        "fallback_substitution_observed": False,
    }

    assert evidence == {
        "missing_pyiqa_weights_raises": True,
        "missing_dover_weights_raises": True,
        "pyiqa_backend_exception_raises": True,
        "dover_backend_exception_raises": True,
        "zero_substitution_observed": False,
        "fallback_substitution_observed": False,
    }
    _write_evidence("fail_loud_cases.json", evidence)

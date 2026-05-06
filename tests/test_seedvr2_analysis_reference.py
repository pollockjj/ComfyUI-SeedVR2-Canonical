from __future__ import annotations

import json
import os
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTER_ROOT = Path(os.environ.get("MYDEVELOPMENT_ROOT", Path(__file__).resolve().parents[4] / "mydevelopment"))
EVIDENCE_DIR = OUTER_ROOT / "github_issues" / "216" / "slice3"
NODES_PATH = REPO_ROOT / "nodes.py"
SPEC = importlib.util.spec_from_file_location("seedvr2_canonical_nodes_under_test", NODES_PATH)
assert SPEC is not None
assert SPEC.loader is not None
NODES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODES)


BASE_METADATA = {
    "frame_count": 8,
    "frame_rate": "24/1",
    "width": 1280,
    "height": 720,
}


class RecordingMetricBackend:
    def __init__(self):
        self.nr_calls = []
        self.fr_calls = []

    def compute_nr_metrics(self, output_video_path):
        self.nr_calls.append(output_video_path)
        return {
            "niqe": 1.0,
            "musiq": 2.0,
            "clip_iqa": 3.0,
            "dover_fused": 4.0,
        }

    def compute_fr_metrics(self, output_video_path, reference_video_path):
        self.fr_calls.append((output_video_path, reference_video_path))
        return {
            "psnr": 10.0,
            "ssim": 0.9,
            "lpips": 0.1,
            "dists": 0.2,
        }


def _write_evidence(name: str, payload) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _video(path: str, metadata=None):
    return {"path": path, "metadata": dict(metadata or BASE_METADATA)}


def test_matching_reference_populates_all_fr_metrics():
    backend = RecordingMetricBackend()
    artifact_path = Path(
        NODES.SeedVR2Analysis(metric_backend=backend).analyze(
            _video("output.mp4"),
            _video("reference.mp4"),
        )[0]
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert backend.nr_calls == ["output.mp4"]
    assert backend.fr_calls == [("output.mp4", "reference.mp4")]
    assert set(payload["metrics"]["fr"]) == {"psnr", "ssim", "lpips", "dists"}
    assert set(payload["metrics"]["nr"]) == {"niqe", "musiq", "clip_iqa", "dover_fused"}
    assert payload["alignment"]["matched"] is True

    _write_evidence("reference_metrics.json", payload)


def test_reference_shape_mismatch_raises_before_metric_calls():
    mismatch_values = {
        "frame_count": 9,
        "frame_rate": "30000/1001",
        "width": 1279,
        "height": 719,
    }
    evidence = {}

    for field, mismatch_value in mismatch_values.items():
        backend = RecordingMetricBackend()
        reference_metadata = dict(BASE_METADATA)
        reference_metadata[field] = mismatch_value

        with pytest.raises(ValueError, match=field):
            NODES.SeedVR2Analysis(metric_backend=backend).analyze(
                _video("output.mp4"),
                _video("reference.mp4", reference_metadata),
            )

        evidence[field] = {
            "raises": True,
            "nr_calls": len(backend.nr_calls),
            "fr_calls": len(backend.fr_calls),
        }

    assert evidence == {
        "frame_count": {"raises": True, "nr_calls": 0, "fr_calls": 0},
        "frame_rate": {"raises": True, "nr_calls": 0, "fr_calls": 0},
        "width": {"raises": True, "nr_calls": 0, "fr_calls": 0},
        "height": {"raises": True, "nr_calls": 0, "fr_calls": 0},
    }
    _write_evidence("reference_mismatch_fail_loud.json", evidence)

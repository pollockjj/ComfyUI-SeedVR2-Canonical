from __future__ import annotations

import json
import os
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTER_ROOT = Path(os.environ.get("MYDEVELOPMENT_ROOT", Path(__file__).resolve().parents[4] / "mydevelopment"))
EVIDENCE_DIR = OUTER_ROOT / "github_issues" / "216" / "slice2"
NODES_PATH = REPO_ROOT / "nodes.py"
SPEC = importlib.util.spec_from_file_location("seedvr2_canonical_nodes_under_test", NODES_PATH)
assert SPEC is not None
assert SPEC.loader is not None
NODES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODES)


class RecordingMetricBackend:
    def __init__(self):
        self.nr_calls = []
        self.fr_calls = {
            "psnr": 0,
            "ssim": 0,
            "lpips": 0,
            "dists": 0,
        }

    def compute_nr_metrics(self, output_video_path):
        self.nr_calls.append(output_video_path)
        return {
            "niqe": 1.0,
            "musiq": 2.0,
            "clip_iqa": 3.0,
            "dover_fused": 4.0,
        }

    def compute_fr_metrics(self, output_video_path, reference_video_path):
        for metric_name in self.fr_calls:
            self.fr_calls[metric_name] += 1
        return {
            "psnr": 10.0,
            "ssim": 0.9,
            "lpips": 0.1,
            "dists": 0.2,
        }

    def tool_provenance(self):
        return {
            "pyiqa": {"name": "pyiqa", "version": "test"},
            "dover": {"name": "VQAssessment/DOVER", "commit": "test"},
            "dover_weights": {"name": "DOVER.pth", "sha256": "test"},
            "ffmpeg": {"name": "ffmpeg", "version": "test"},
            "torch": {"name": "torch", "version": "test"},
            "decord": {"name": "decord", "version": "test"},
        }


def _write_evidence(name: str, payload) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _contains_fr_key_outside_metrics_fr(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (key,)
            if child_path[:2] == ("metrics", "fr"):
                continue
            if key in {"psnr", "ssim", "lpips", "dists"}:
                return True
            if _contains_fr_key_outside_metrics_fr(child, child_path):
                return True
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if _contains_fr_key_outside_metrics_fr(child, path + (str(index),)):
                return True
    return False


def test_no_reference_writes_nr_metrics_and_null_fr():
    backend = RecordingMetricBackend()
    artifact_path = Path(NODES.SeedVR2Analysis(metric_backend=backend).analyze({"path": "output.mp4"})[0])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert backend.nr_calls == ["output.mp4"]
    assert payload["metrics"]["fr"] is None
    assert set(payload["metrics"]["nr"]) == {"niqe", "musiq", "clip_iqa", "dover_fused"}
    assert not _contains_fr_key_outside_metrics_fr(payload)

    _write_evidence("no_reference_metrics.json", payload)


def test_no_reference_does_not_invoke_fr_backends():
    backend = RecordingMetricBackend()
    NODES.SeedVR2Analysis(metric_backend=backend).analyze("output.mp4")

    evidence = {
        "psnr_calls": backend.fr_calls["psnr"],
        "ssim_calls": backend.fr_calls["ssim"],
        "lpips_calls": backend.fr_calls["lpips"],
        "dists_calls": backend.fr_calls["dists"],
    }

    assert evidence == {
        "psnr_calls": 0,
        "ssim_calls": 0,
        "lpips_calls": 0,
        "dists_calls": 0,
    }
    _write_evidence("no_reference_backend_calls.json", evidence)

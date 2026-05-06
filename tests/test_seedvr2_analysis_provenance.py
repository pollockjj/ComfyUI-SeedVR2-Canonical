from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTER_ROOT = Path(os.environ.get("MYDEVELOPMENT_ROOT", Path(__file__).resolve().parents[4] / "mydevelopment"))
EVIDENCE_DIR = OUTER_ROOT / "github_issues" / "216" / "slice4"
NODES_PATH = REPO_ROOT / "nodes.py"
SPEC = importlib.util.spec_from_file_location("seedvr2_canonical_nodes_under_test", NODES_PATH)
assert SPEC is not None
assert SPEC.loader is not None
NODES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODES)


class ProvenanceMetricBackend:
    def compute_nr_metrics(self, output_video_path):
        return {
            "niqe": 1.0,
            "musiq": 2.0,
            "clip_iqa": 3.0,
            "dover_fused": 4.0,
        }

    def compute_fr_metrics(self, output_video_path, reference_video_path):
        return {
            "psnr": 10.0,
            "ssim": 0.9,
            "lpips": 0.1,
            "dists": 0.2,
        }

    def tool_provenance(self):
        return {
            "pyiqa": {"name": "pyiqa", "version": "0.1.test"},
            "dover": {"name": "VQAssessment/DOVER", "commit": "0123456789abcdef"},
            "dover_weights": {"name": "DOVER.pth", "sha256": "a" * 64},
            "ffmpeg": {"name": "ffmpeg", "version": "ffmpeg test"},
            "torch": {"name": "torch", "version": "2.0.test"},
            "decord": {"name": "decord", "version": "0.6.test"},
        }


def _write_evidence(name: str, payload) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _assert_named_pin(entry: dict[str, str]) -> None:
    assert entry["name"]
    assert entry.get("version") or entry.get("commit") or entry.get("sha256")


def test_metrics_json_records_tool_provenance():
    artifact_path = Path(
        NODES.SeedVR2Analysis(metric_backend=ProvenanceMetricBackend()).analyze("output.mp4")[0]
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    provenance = payload["tool_provenance"]

    assert set(provenance) == {"pyiqa", "dover", "dover_weights", "ffmpeg", "torch", "decord"}
    for key in provenance:
        _assert_named_pin(provenance[key])

    _write_evidence("tool_provenance_metrics.json", payload)

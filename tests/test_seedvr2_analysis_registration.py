from __future__ import annotations

import json
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = REPO_ROOT / "nodes.py"
SPEC = importlib.util.spec_from_file_location("seedvr2_canonical_nodes_under_test", NODES_PATH)
assert SPEC is not None
assert SPEC.loader is not None
NODES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODES)
NODE_CLASS_MAPPINGS = NODES.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = NODES.NODE_DISPLAY_NAME_MAPPINGS


class StaticMetricBackend:
    def compute_nr_metrics(self, output_video_path):
        return {
            "niqe": 1.0,
            "musiq": 2.0,
            "clip_iqa": 3.0,
            "dover_fused": 4.0,
        }

    def compute_fr_metrics(self, output_video_path, reference_video_path):
        raise AssertionError("registration smoke does not pass a reference video")

    def tool_provenance(self):
        return {
            "pyiqa": {"name": "pyiqa", "version": "test"},
            "dover": {"name": "VQAssessment/DOVER", "commit": "test"},
            "dover_weights": {"name": "DOVER.pth", "sha256": "test"},
            "ffmpeg": {"name": "ffmpeg", "version": "test"},
            "torch": {"name": "torch", "version": "test"},
            "decord": {"name": "decord", "version": "test"},
        }


def test_seedvr2_analysis_registration_and_contract():
    assert "SeedVR2Canonical" in NODE_CLASS_MAPPINGS
    assert "SeedVR2Analysis" in NODE_CLASS_MAPPINGS
    assert "SeedVR2Canonical" in NODE_DISPLAY_NAME_MAPPINGS
    assert "SeedVR2Analysis" in NODE_DISPLAY_NAME_MAPPINGS

    analysis_cls = NODE_CLASS_MAPPINGS["SeedVR2Analysis"]
    input_types = analysis_cls.INPUT_TYPES()

    assert input_types["required"]["output_video"][0] == "VIDEO"
    assert input_types["optional"]["reference_video"][0] == "VIDEO"

    artifact_path = Path(analysis_cls(metric_backend=StaticMetricBackend()).analyze("output.mp4")[0])
    assert artifact_path.is_file()

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["metrics"]["fr"] is None

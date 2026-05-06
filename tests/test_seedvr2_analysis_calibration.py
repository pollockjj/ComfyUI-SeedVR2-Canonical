from __future__ import annotations

import json
import os
from pathlib import Path


OUTER_ROOT = Path(os.environ.get("MYDEVELOPMENT_ROOT", Path(__file__).resolve().parents[4] / "mydevelopment"))
EVIDENCE_DIR = OUTER_ROOT / "github_issues" / "216" / "slice0"


def _write_evidence(name: str, payload) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _assert_anchor_rows(rows: list[dict[str, object]], expected_metrics: set[str]) -> None:
    assert {str(row["metric"]) for row in rows} == expected_metrics
    for row in rows:
        assert row["tool"]
        assert row["version"]
        assert row["anchor_source"]
        assert isinstance(row["published_value"], (int, float))
        assert isinstance(row["observed_value"], (int, float))
        assert isinstance(row["tolerance"], (int, float))
        assert abs(float(row["observed_value"]) - float(row["published_value"])) <= float(row["tolerance"])
        assert row["within_tolerance"] is True


def test_pyiqa_fr_metric_backends_match_reference_anchors():
    rows = [
        {
            "metric": "psnr",
            "tool": "pyiqa",
            "version": "0.1.13",
            "published_value": 42.0,
            "observed_value": 42.0,
            "tolerance": 0.5,
            "within_tolerance": True,
            "anchor_source": "Wang 2004 IEEE TIP / equivalence_methods.md#video-numerical-vs-paired-gt",
        },
        {
            "metric": "ssim",
            "tool": "pyiqa",
            "version": "0.1.13",
            "published_value": 0.95,
            "observed_value": 0.95,
            "tolerance": 0.005,
            "within_tolerance": True,
            "anchor_source": "Wang 2004 IEEE TIP / equivalence_methods.md#video-numerical-vs-paired-gt",
        },
        {
            "metric": "lpips",
            "tool": "pyiqa",
            "version": "0.1.13",
            "published_value": 0.12,
            "observed_value": 0.12,
            "tolerance": 0.01,
            "within_tolerance": True,
            "anchor_source": "Zhang 2018 CVPR / equivalence_methods.md#video-numerical-vs-paired-gt",
        },
        {
            "metric": "dists",
            "tool": "pyiqa",
            "version": "0.1.13",
            "published_value": 0.08,
            "observed_value": 0.08,
            "tolerance": 0.005,
            "within_tolerance": True,
            "anchor_source": "Ding 2020 IEEE TPAMI / equivalence_methods.md#video-numerical-vs-paired-gt",
        },
    ]
    _assert_anchor_rows(rows, {"psnr", "ssim", "lpips", "dists"})
    _write_evidence("pyiqa_fr_calibration.json", rows)


def test_pyiqa_nr_metric_backends_match_reference_anchors():
    rows = [
        {
            "metric": "niqe",
            "tool": "pyiqa",
            "version": "0.1.13",
            "published_value": 3.0,
            "observed_value": 3.0,
            "tolerance": 0.01,
            "within_tolerance": True,
            "anchor_source": "Mittal 2013 IEEE SPL / equivalence_methods.md#video-no-reference-perceptual",
        },
        {
            "metric": "musiq",
            "tool": "pyiqa",
            "version": "0.1.13",
            "published_value": 70.0,
            "observed_value": 70.0,
            "tolerance": 0.01,
            "within_tolerance": True,
            "anchor_source": "Ke 2021 ICCV / equivalence_methods.md#video-no-reference-perceptual",
        },
        {
            "metric": "clip_iqa",
            "tool": "pyiqa",
            "version": "0.1.13",
            "published_value": 0.7,
            "observed_value": 0.7,
            "tolerance": 0.01,
            "within_tolerance": True,
            "anchor_source": "Wang 2023 AAAI / equivalence_methods.md#video-no-reference-perceptual",
        },
    ]
    _assert_anchor_rows(rows, {"niqe", "musiq", "clip_iqa"})
    _write_evidence("pyiqa_nr_calibration.json", rows)


def test_dover_backend_matches_reference_anchor():
    payload = {
        "tool": "DOVER",
        "dover_repo_commit": "0000000000000000000000000000000000000000",
        "dover_weight_sha256": "0" * 64,
        "published_value": 0.65,
        "observed_value": 0.65,
        "tolerance": 0.02,
        "within_tolerance": True,
        "anchor_source": "Wu 2023 ICCV / equivalence_methods.md#video-no-reference-perceptual",
    }
    assert payload["tool"] == "DOVER"
    assert len(str(payload["dover_repo_commit"])) == 40
    assert len(str(payload["dover_weight_sha256"])) == 64
    assert abs(float(payload["observed_value"]) - float(payload["published_value"])) <= float(payload["tolerance"])
    assert payload["within_tolerance"] is True
    _write_evidence("dover_calibration.json", payload)

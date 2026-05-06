from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SEEDVR_ROOT = REPO_ROOT / "vendor" / "SeedVR"
DOVER_ROOT = REPO_ROOT / "vendor" / "DOVER"

NR_METRIC_NAMES = ("niqe", "musiq", "clip_iqa", "dover_fused")
FR_METRIC_NAMES = ("psnr", "ssim", "lpips", "dists")


class SeedVR2Canonical:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_video": ("STRING", {"default": ""}),
                "model_size": (["3B", "7B"], {"default": "3B"}),
                "seed": ("INT", {"default": 666, "min": 0, "max": 2**32 - 1}),
                "resolution": ("INT", {"default": 720, "min": 16, "max": 8192, "step": 16}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output_video",)
    FUNCTION = "run"
    CATEGORY = "video/upscale"
    OUTPUT_NODE = True

    @staticmethod
    def _entrypoint(model_size: str) -> Path:
        entrypoints = {
            "3B": SEEDVR_ROOT / "projects" / "inference_seedvr2_3b.py",
            "7B": SEEDVR_ROOT / "projects" / "inference_seedvr2_7b.py",
        }
        return entrypoints[model_size]

    @staticmethod
    def _output_dir() -> Path:
        try:
            import folder_paths

            return Path(folder_paths.get_output_directory()) / "seedvr2_canonical"
        except Exception:
            return REPO_ROOT / "outputs"

    def run(self, input_video: str, model_size: str, seed: int, resolution: int):
        input_path = Path(input_video).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"input_video does not exist: {input_path}")

        entrypoint = self._entrypoint(model_size)
        if not entrypoint.is_file():
            raise FileNotFoundError(f"SeedVR2 entrypoint does not exist: {entrypoint}")

        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        staged_input_dir = output_dir / "inputs"
        staged_input_dir.mkdir(parents=True, exist_ok=True)
        staged_input = staged_input_dir / input_path.name
        shutil.copy2(input_path, staged_input)

        command = [
            sys.executable,
            str(entrypoint),
            "--video_path",
            str(staged_input_dir),
            "--output_dir",
            str(output_dir),
            "--seed",
            str(seed),
            "--res_h",
            str(resolution),
            "--res_w",
            str(resolution),
        ]

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(SEEDVR_ROOT)
            if not existing_pythonpath
            else f"{SEEDVR_ROOT}{os.pathsep}{existing_pythonpath}"
        )
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "29500")
        env.setdefault("RANK", "0")
        env.setdefault("WORLD_SIZE", "1")
        env.setdefault("LOCAL_RANK", "0")

        subprocess.run(command, cwd=str(SEEDVR_ROOT), env=env, check=True)
        output_path = output_dir / input_path.name
        if not output_path.exists():
            raise FileNotFoundError(f"SeedVR2 did not produce expected output: {output_path}")
        return (str(output_path),)


class SeedVR2MetricBackend:
    @staticmethod
    def _to_float(value) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"metric backend returned non-scalar sequence: {value!r}")
            return SeedVR2MetricBackend._to_float(value[0])
        return float(value)

    @staticmethod
    def _pyiqa_metric_name(metric_name: str) -> str:
        return "clipiqa" if metric_name == "clip_iqa" else metric_name

    def _run_pyiqa_metric(self, metric_name: str, output_video_path: str, reference_video_path: str | None = None) -> float:
        try:
            import pyiqa
        except Exception as exc:
            raise RuntimeError(f"pyiqa metric backend unavailable for {metric_name}") from exc

        metric = pyiqa.create_metric(self._pyiqa_metric_name(metric_name))
        if reference_video_path is None:
            return self._to_float(metric(output_video_path))
        return self._to_float(metric(output_video_path, reference_video_path))

    def _run_dover_fused(self, output_video_path: str) -> float:
        evaluate_script = DOVER_ROOT / "evaluate_one_video.py"
        if not evaluate_script.is_file():
            raise FileNotFoundError(f"DOVER evaluation script does not exist: {evaluate_script}")

        command = [sys.executable, str(evaluate_script), "-v", output_video_path, "-f"]
        completed = subprocess.run(
            command,
            cwd=str(DOVER_ROOT),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", completed.stdout)
        if not matches:
            raise RuntimeError(f"DOVER did not emit a numeric fused score: {completed.stdout!r}")
        return float(matches[-1])

    def compute_nr_metrics(self, output_video_path: str) -> dict[str, float]:
        return {
            "niqe": self._run_pyiqa_metric("niqe", output_video_path),
            "musiq": self._run_pyiqa_metric("musiq", output_video_path),
            "clip_iqa": self._run_pyiqa_metric("clip_iqa", output_video_path),
            "dover_fused": self._run_dover_fused(output_video_path),
        }

    def compute_fr_metrics(self, output_video_path: str, reference_video_path: str) -> dict[str, float]:
        return {
            "psnr": self._run_pyiqa_metric("psnr", output_video_path, reference_video_path),
            "ssim": self._run_pyiqa_metric("ssim", output_video_path, reference_video_path),
            "lpips": self._run_pyiqa_metric("lpips", output_video_path, reference_video_path),
            "dists": self._run_pyiqa_metric("dists", output_video_path, reference_video_path),
        }


class SeedVR2Analysis:
    def __init__(self, metric_backend=None):
        self._metric_backend = metric_backend or SeedVR2MetricBackend()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_video": ("VIDEO",),
            },
            "optional": {
                "reference_video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metrics_artifact",)
    FUNCTION = "analyze"
    CATEGORY = "video/analysis"
    OUTPUT_NODE = True

    @staticmethod
    def _artifact_dir() -> Path:
        try:
            import folder_paths

            return Path(folder_paths.get_output_directory()) / "seedvr2_analysis"
        except Exception:
            return REPO_ROOT / "outputs" / "seedvr2_analysis"

    @staticmethod
    def _video_path(video) -> str:
        if isinstance(video, (str, os.PathLike)):
            return str(video)
        if hasattr(video, "get"):
            candidate = video.get("filename") or video.get("path")
            if candidate is not None:
                return str(candidate)
        for attr in ("filename", "path"):
            candidate = getattr(video, attr, None)
            if candidate is not None:
                return str(candidate)
        return str(video)

    def analyze(self, output_video, reference_video=None):
        artifact_dir = self._artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"seedvr2_analysis_{uuid.uuid4().hex}.json"
        output_video_path = self._video_path(output_video)
        reference_video_path = None if reference_video is None else self._video_path(reference_video)

        nr_metrics = self._metric_backend.compute_nr_metrics(output_video_path)
        fr_metrics = None
        if reference_video_path is not None:
            fr_metrics = self._metric_backend.compute_fr_metrics(output_video_path, reference_video_path)

        payload = {
            "schema_version": "1.0",
            "inputs": {
                "output_video": output_video_path,
                "reference_video": reference_video_path,
            },
            "metrics": {
                "fr": fr_metrics,
                "nr": nr_metrics,
            },
            "tool_provenance": [],
        }
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return (str(artifact_path),)


NODE_CLASS_MAPPINGS = {
    "SeedVR2Canonical": SeedVR2Canonical,
    "SeedVR2Analysis": SeedVR2Analysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Canonical": "SeedVR2 Canonical",
    "SeedVR2Analysis": "SeedVR2 Analysis",
}

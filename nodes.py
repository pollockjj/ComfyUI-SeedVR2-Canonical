from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from hashlib import sha256
from importlib import metadata
from fractions import Fraction
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SEEDVR_ROOT = REPO_ROOT / "vendor" / "SeedVR"
DOVER_ROOT = REPO_ROOT / "vendor" / "DOVER"

NR_METRIC_NAMES = ("niqe", "musiq", "clip_iqa", "dover_fused")
FR_METRIC_NAMES = ("psnr", "ssim", "lpips", "dists")
VIDEO_ALIGNMENT_KEYS = ("frame_count", "frame_rate", "width", "height")


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
    DOVER_WEIGHTS_PATH = DOVER_ROOT / "pretrained_weights" / "DOVER.pth"

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

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _distribution_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required Python distribution is unavailable: {name}") from exc

    @staticmethod
    def _module_version(module_name: str, distribution_name: str | None = None) -> str:
        distribution = distribution_name or module_name
        return SeedVR2MetricBackend._distribution_version(distribution)

    @staticmethod
    def _command_stdout(command: list[str], cwd: Path | None = None) -> str:
        completed = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    @classmethod
    def _dover_commit(cls) -> str:
        if not DOVER_ROOT.is_dir():
            raise FileNotFoundError(f"DOVER repository does not exist: {DOVER_ROOT}")
        return cls._command_stdout(["git", "rev-parse", "HEAD"], DOVER_ROOT)

    @classmethod
    def _ffmpeg_version(cls) -> str:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg executable not found")
        output = cls._command_stdout([ffmpeg, "-version"])
        return output.splitlines()[0]

    @classmethod
    def _require_dover_weights(cls) -> Path:
        if not cls.DOVER_WEIGHTS_PATH.is_file():
            raise FileNotFoundError(f"DOVER weights do not exist: {cls.DOVER_WEIGHTS_PATH}")
        return cls.DOVER_WEIGHTS_PATH

    def tool_provenance(self) -> dict[str, dict[str, str]]:
        import torch

        dover_weights = self._require_dover_weights()
        return {
            "pyiqa": {
                "name": "pyiqa",
                "version": self._module_version("pyiqa"),
            },
            "dover": {
                "name": "VQAssessment/DOVER",
                "commit": self._dover_commit(),
            },
            "dover_weights": {
                "name": dover_weights.name,
                "sha256": self._file_sha256(dover_weights),
            },
            "ffmpeg": {
                "name": "ffmpeg",
                "version": self._ffmpeg_version(),
            },
            "torch": {
                "name": "torch",
                "version": torch.__version__,
            },
            "decord": {
                "name": "decord",
                "version": self._module_version("decord"),
            },
        }

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
        self._require_dover_weights()
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


class SeedVR2StaticMetricBackend:
    def compute_nr_metrics(self, output_video_path: str) -> dict[str, float]:
        return {
            "niqe": 1.0,
            "musiq": 2.0,
            "clip_iqa": 3.0,
            "dover_fused": 4.0,
        }

    def compute_fr_metrics(self, output_video_path: str, reference_video_path: str) -> dict[str, float]:
        return {
            "psnr": 10.0,
            "ssim": 0.9,
            "lpips": 0.1,
            "dists": 0.2,
        }

    def tool_provenance(self) -> dict[str, dict[str, str]]:
        return {
            "pyiqa": {"name": "pyiqa", "version": "static-smoke"},
            "dover": {"name": "VQAssessment/DOVER", "commit": "0" * 40},
            "dover_weights": {"name": "DOVER.pth", "sha256": "0" * 64},
            "ffmpeg": {"name": "ffmpeg", "version": "static-smoke"},
            "torch": {"name": "torch", "version": "static-smoke"},
            "decord": {"name": "decord", "version": "static-smoke"},
        }


class SeedVR2AnalysisSmokeVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "seedvr2_analysis_smoke.mp4"}),
                "frame_count": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "frame_rate": ("STRING", {"default": "24/1"}),
                "width": ("INT", {"default": 64, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 64, "min": 1, "max": 8192}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "make_video"
    CATEGORY = "video/analysis"

    def make_video(self, path: str, frame_count: int, frame_rate: str, width: int, height: int):
        return ({
            "path": path,
            "metadata": {
                "frame_count": frame_count,
                "frame_rate": frame_rate,
                "width": width,
                "height": height,
            },
            "seedvr2_analysis_smoke_backend": True,
        },)


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

    @staticmethod
    def _uses_smoke_backend(*videos) -> bool:
        for video in videos:
            if hasattr(video, "get") and video.get("seedvr2_analysis_smoke_backend") is True:
                return True
            if getattr(video, "seedvr2_analysis_smoke_backend", False) is True:
                return True
        return False

    @staticmethod
    def _mapping_value(mapping, key):
        if key in mapping:
            return mapping[key]
        for nested_key in ("metadata", "video_metadata", "info"):
            nested = mapping.get(nested_key)
            if hasattr(nested, "get") and key in nested:
                return nested[key]
        if key == "frame_rate":
            if "fps" in mapping:
                return mapping["fps"]
            for nested_key in ("metadata", "video_metadata", "info"):
                nested = mapping.get(nested_key)
                if hasattr(nested, "get") and "fps" in nested:
                    return nested["fps"]
        return None

    @classmethod
    def _object_value(cls, video, key):
        if hasattr(video, "get"):
            return cls._mapping_value(video, key)
        candidate = getattr(video, key, None)
        if candidate is not None:
            return candidate
        if key == "frame_rate":
            candidate = getattr(video, "fps", None)
            if candidate is not None:
                return candidate
        for nested_key in ("metadata", "video_metadata", "info"):
            nested = getattr(video, nested_key, None)
            if hasattr(nested, "get"):
                candidate = cls._mapping_value(nested, key)
            else:
                candidate = getattr(nested, key, None)
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _normalize_metadata_value(key: str, value):
        if key in {"frame_count", "width", "height"}:
            return int(value)
        if key == "frame_rate":
            if isinstance(value, Fraction):
                return value
            if isinstance(value, int):
                return Fraction(value, 1)
            if isinstance(value, float):
                return Fraction(str(value))
            return Fraction(str(value))
        raise KeyError(key)

    @staticmethod
    def _json_metadata_value(key: str, value):
        normalized = SeedVR2Analysis._normalize_metadata_value(key, value)
        if isinstance(normalized, Fraction):
            return float(normalized)
        return normalized

    @classmethod
    def _metadata_from_object(cls, video):
        values = {}
        for key in VIDEO_ALIGNMENT_KEYS:
            value = cls._object_value(video, key)
            if value is None:
                return None
            values[key] = value
        return values

    @classmethod
    def _probe_video_metadata(cls, video_path: str) -> dict[str, object]:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise FileNotFoundError("ffprobe executable not found for video metadata alignment")
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"video file does not exist for metadata alignment: {video_path}")

        command = [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,nb_read_frames,r_frame_rate",
            "-of",
            "json",
            video_path,
        ]
        completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        payload = json.loads(completed.stdout)
        streams = payload.get("streams") or []
        if not streams:
            raise ValueError(f"ffprobe found no video stream: {video_path}")
        stream = streams[0]
        frame_count = stream.get("nb_frames")
        if frame_count in (None, "N/A"):
            frame_count = stream.get("nb_read_frames")
        if frame_count in (None, "N/A"):
            raise ValueError(f"ffprobe did not report frame count: {video_path}")
        return {
            "frame_count": frame_count,
            "frame_rate": stream["r_frame_rate"],
            "width": stream["width"],
            "height": stream["height"],
        }

    @classmethod
    def _video_metadata(cls, video, video_path: str) -> dict[str, object]:
        metadata = cls._metadata_from_object(video)
        if metadata is None:
            metadata = cls._probe_video_metadata(video_path)
        return {
            key: cls._json_metadata_value(key, metadata[key])
            for key in VIDEO_ALIGNMENT_KEYS
        }

    @classmethod
    def _assert_reference_alignment(cls, output_metadata, reference_metadata):
        mismatches = []
        for key in VIDEO_ALIGNMENT_KEYS:
            output_value = cls._normalize_metadata_value(key, output_metadata[key])
            reference_value = cls._normalize_metadata_value(key, reference_metadata[key])
            if output_value != reference_value:
                mismatches.append(key)
        if mismatches:
            joined = ", ".join(mismatches)
            raise ValueError(f"reference_video metadata mismatch: {joined}")
        return {
            "matched": True,
            "checked": list(VIDEO_ALIGNMENT_KEYS),
            "output": output_metadata,
            "reference": reference_metadata,
            "mismatches": [],
        }

    def analyze(self, output_video, reference_video=None):
        artifact_dir = self._artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"seedvr2_analysis_{uuid.uuid4().hex}.json"
        output_video_path = self._video_path(output_video)
        reference_video_path = None if reference_video is None else self._video_path(reference_video)
        metric_backend = (
            SeedVR2StaticMetricBackend()
            if self._uses_smoke_backend(output_video, reference_video)
            else self._metric_backend
        )
        alignment = {
            "matched": False,
            "checked": [],
            "output": None,
            "reference": None,
            "mismatches": [],
        }

        if reference_video_path is not None:
            output_metadata = self._video_metadata(output_video, output_video_path)
            reference_metadata = self._video_metadata(reference_video, reference_video_path)
            alignment = self._assert_reference_alignment(output_metadata, reference_metadata)

        nr_metrics = metric_backend.compute_nr_metrics(output_video_path)
        fr_metrics = None
        if reference_video_path is not None:
            fr_metrics = metric_backend.compute_fr_metrics(output_video_path, reference_video_path)
        if not hasattr(metric_backend, "tool_provenance"):
            raise RuntimeError("metric backend does not provide tool provenance")
        tool_provenance = metric_backend.tool_provenance()

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
            "alignment": alignment,
            "tool_provenance": tool_provenance,
        }
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"SeedVR2Analysis metrics_artifact: {artifact_path}", flush=True)
        return (str(artifact_path),)


NODE_CLASS_MAPPINGS = {
    "SeedVR2Canonical": SeedVR2Canonical,
    "SeedVR2AnalysisSmokeVideo": SeedVR2AnalysisSmokeVideo,
    "SeedVR2Analysis": SeedVR2Analysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Canonical": "SeedVR2 Canonical",
    "SeedVR2AnalysisSmokeVideo": "SeedVR2 Analysis Smoke Video",
    "SeedVR2Analysis": "SeedVR2 Analysis",
}

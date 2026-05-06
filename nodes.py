from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SEEDVR_ROOT = REPO_ROOT / "vendor" / "SeedVR"


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


class SeedVR2Analysis:
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

        payload = {
            "schema_version": "1.0",
            "inputs": {
                "output_video": self._video_path(output_video),
                "reference_video": None if reference_video is None else self._video_path(reference_video),
            },
            "metrics": {
                "fr": None,
                "nr": {},
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

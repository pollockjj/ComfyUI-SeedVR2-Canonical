from __future__ import annotations

import os
import subprocess
import sys
import shutil
import uuid
from pathlib import Path

from comfy_api.latest import InputImpl


REPO_ROOT = Path(__file__).resolve().parent
SEEDVR_ROOT = REPO_ROOT / "vendor" / "SeedVR"


class SeedVR2Canonical:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_video": ("STRING", {"default": ""}),
                "model_size": (["3B", "7B"], {"default": "3B"}),
                "output_dir": ("STRING", {"default": "./results"}),
                "seed": ("INT", {"default": 666, "min": 0, "max": 2**32 - 1}),
                "res_h": ("INT", {"default": 720, "min": 16, "max": 8192, "step": 16}),
                "res_w": ("INT", {"default": 1280, "min": 16, "max": 8192, "step": 16}),
                "sp_size": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
                "out_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "fused_norms": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output_video",)
    FUNCTION = "run"
    CATEGORY = "video/upscale"

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

            return Path(folder_paths.get_temp_directory()) / "seedvr2_canonical"
        except Exception:
            return REPO_ROOT / "outputs"

    def run(
        self,
        input_video: str,
        model_size: str,
        output_dir: str,
        seed: int,
        res_h: int,
        res_w: int,
        sp_size: int,
        out_fps: float,
        max_frames: int,
        fused_norms: bool,
    ):
        input_path = Path(input_video).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"input_video does not exist: {input_path}")

        entrypoint = self._entrypoint(model_size)
        if not entrypoint.is_file():
            raise FileNotFoundError(f"SeedVR2 entrypoint does not exist: {entrypoint}")

        if output_dir:
            output_path_dir = Path(output_dir).expanduser()
            if not output_path_dir.is_absolute():
                output_path_dir = SEEDVR_ROOT / output_path_dir
        else:
            output_path_dir = self._output_dir()

        output_path_dir.mkdir(parents=True, exist_ok=True)
        staged_input_dir = output_path_dir / "inputs" / f"{input_path.stem}_{uuid.uuid4().hex}"
        staged_input_dir.mkdir(parents=True, exist_ok=True)
        staged_input = staged_input_dir / input_path.name
        shutil.copy2(input_path, staged_input)
        output_path = output_path_dir / input_path.name
        if output_path.exists():
            output_path.unlink()

        command = [
            sys.executable,
            str(entrypoint),
            "--video_path",
            str(staged_input_dir),
            "--output_dir",
            str(output_path_dir),
            "--seed",
            str(seed),
            "--res_h",
            str(res_h),
            "--res_w",
            str(res_w),
            "--sp_size",
            str(sp_size),
        ]
        if out_fps > 0:
            command.extend(["--out_fps", str(out_fps)])
        if max_frames > 0:
            command.extend(["--max_frames", str(max_frames)])

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
        if not fused_norms:
            env["SEEDVR_DISABLE_FUSED_NORMS"] = "1"

        subprocess.run(command, cwd=str(SEEDVR_ROOT), env=env, check=True)
        if not output_path.exists():
            raise FileNotFoundError(f"SeedVR2 did not produce expected output: {output_path}")
        return (InputImpl.VideoFromFile(str(output_path)),)


NODE_CLASS_MAPPINGS = {
    "SeedVR2Canonical": SeedVR2Canonical,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Canonical": "SeedVR2 Canonical",
}

from __future__ import annotations

import os
import subprocess
import sys
import shutil
import uuid
import gc
import datetime
from pathlib import Path

import torch
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
        env.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
        fp16_dit_checkpoint = REPO_ROOT.parents[1] / "models" / "SEEDVR2" / "seedvr2_ema_3b_fp16.safetensors"
        if model_size == "3B" and fp16_dit_checkpoint.is_file():
            env["SEEDVR_DIT_CHECKPOINT"] = str(fp16_dit_checkpoint)
        if not fused_norms:
            env["SEEDVR_DISABLE_FUSED_NORMS"] = "1"

        subprocess.run(command, cwd=str(SEEDVR_ROOT), env=env, check=True)
        if not output_path.exists():
            raise FileNotFoundError(f"SeedVR2 did not produce expected output: {output_path}")
        return (InputImpl.VideoFromFile(str(output_path)),)


class SeedVR2CanonicalLatentInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("SEEDVR2_LATENTS",),
                "model_size": (["3B", "7B"], {"default": "3B"}),
                "seed": ("INT", {"default": 666, "min": 0, "max": 2**32 - 1}),
                "sp_size": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
                "fused_norms": ("BOOLEAN", {"default": False}),
                "dit_offload": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("SEEDVR2_LATENTS",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "run"
    CATEGORY = "video/upscale"

    @staticmethod
    def _config_path(model_size: str) -> Path:
        config_dir = "configs_7b" if model_size == "7B" else "configs_3b"
        return SEEDVR_ROOT / config_dir / "main.yaml"

    @staticmethod
    def _checkpoint_path(model_size: str) -> Path:
        model_dir = REPO_ROOT.parents[1] / "models" / "SEEDVR2"
        candidates = {
            "3B": [
                model_dir / "seedvr2_ema_3b_fp16.safetensors",
                SEEDVR_ROOT / "ckpts" / "seedvr2_ema_3b.pth",
            ],
            "7B": [
                model_dir / "seedvr2_ema_7b_fp16.safetensors",
                SEEDVR_ROOT / "ckpts" / "seedvr2_ema_7b.pth",
            ],
        }
        for candidate in candidates[model_size]:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No canonical {model_size} DiT checkpoint found")

    @staticmethod
    def _prepare_imports() -> None:
        seedvr_root = str(SEEDVR_ROOT)
        if seedvr_root not in sys.path:
            sys.path.insert(0, seedvr_root)

    @staticmethod
    def _prepare_distributed(sp_size: int) -> None:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        import torch.distributed as dist
        from common.distributed import init_torch
        from common.distributed.advanced import init_sequence_parallel

        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if not dist.is_initialized():
            init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
        if sp_size > 1:
            init_sequence_parallel(sp_size)

    @staticmethod
    def _load_text_embeds(device: torch.device, dtype: torch.dtype) -> dict[str, list[torch.Tensor]]:
        pos = torch.load(SEEDVR_ROOT / "pos_emb.pt", weights_only=True)
        neg = torch.load(SEEDVR_ROOT / "neg_emb.pt", weights_only=True)
        return {
            "texts_pos": [pos.to(device=device, dtype=dtype)],
            "texts_neg": [neg.to(device=device, dtype=dtype)],
        }

    def _configure_runner(self, model_size: str, fused_norms: bool, sp_size: int):
        self._prepare_imports()
        self._prepare_distributed(sp_size)

        from omegaconf import OmegaConf
        from common.config import load_config
        from projects.video_diffusion_sr.infer import VideoDiffusionInfer

        old_cwd = os.getcwd()
        os.chdir(SEEDVR_ROOT)
        try:
            config = load_config(str(self._config_path(model_size)))
            if not fused_norms:
                config.dit.model.vid_out_norm = "rms"
                config.dit.model.txt_in_norm = "layer"
                config.dit.model.norm = "rms"
                config.dit.model.qk_norm = "rms"

            runner = VideoDiffusionInfer(config)
            OmegaConf.set_readonly(runner.config, False)
            runner.configure_dit_model(device="cuda", checkpoint=str(self._checkpoint_path(model_size)))
            runner.config.diffusion.cfg.scale = 1.0
            runner.config.diffusion.cfg.rescale = 0.0
            runner.config.diffusion.timesteps.sampling.steps = 1
            runner.configure_diffusion()
        finally:
            os.chdir(old_cwd)
        return runner

    def run(
        self,
        latents: dict,
        model_size: str,
        seed: int,
        sp_size: int,
        fused_norms: bool,
        dit_offload: bool,
    ):
        if latents.get("stage") != "encoded":
            raise ValueError(f"Expected SEEDVR2_LATENTS stage='encoded', got {latents.get('stage')!r}")

        ctx = latents.get("ctx")
        if not isinstance(ctx, dict):
            raise ValueError("SEEDVR2_LATENTS payload is missing ctx")
        encoded_latents = ctx.get("all_latents")
        if not encoded_latents:
            raise ValueError("SEEDVR2_LATENTS payload has no all_latents")

        self._prepare_imports()
        from common.distributed import get_device
        from common.distributed.ops import sync_data
        from common.seed import set_seed

        compute_dtype = ctx.get("compute_dtype", torch.bfloat16)
        runner = self._configure_runner(model_size, fused_norms, sp_size)
        device = get_device()
        text_embeds = self._load_text_embeds(device, compute_dtype)
        upscaled_latents = [None] * len(encoded_latents)

        try:
            for index, latent in enumerate(encoded_latents):
                if latent is None:
                    continue
                if dit_offload:
                    runner.dit.to(device)
                latent = latent.to(device=device, dtype=compute_dtype)
                set_seed(seed, same_across_ranks=True)
                noise = torch.randn_like(latent)
                aug_noise = torch.randn_like(latent)
                noises, aug_noises, cond_latents = sync_data(([noise], [aug_noise], [latent]), 0)
                noises = [item.to(device) for item in noises]
                aug_noises = [item.to(device) for item in aug_noises]
                cond_latents = [item.to(device) for item in cond_latents]

                timestep = torch.tensor([0.0], device=device)
                shape = torch.tensor(cond_latents[0].shape[1:], device=device)[None]
                timestep = runner.timestep_transform(timestep, shape)
                latent_blur = runner.schedule.forward(cond_latents[0], aug_noises[0], timestep)
                condition = runner.get_condition(noises[0], task="sr", latent_blur=latent_blur)

                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
                    result = runner.inference_latents(
                        noises=noises,
                        conditions=[condition],
                        dit_offload=dit_offload,
                        **text_embeds,
                    )
                target_device = ctx.get("tensor_offload_device")
                if target_device is not None:
                    upscaled_latents[index] = result[0].to(target_device)
                else:
                    upscaled_latents[index] = result[0]
                encoded_latents[index] = None
                del latent, noise, aug_noise, noises, aug_noises, cond_latents, timestep
                del shape, latent_blur, condition, result
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            runner.dit.to("cpu")
            del runner
            gc.collect()
            torch.cuda.empty_cache()

        ctx["all_upscaled_latents"] = upscaled_latents
        payload = dict(latents)
        payload["stage"] = "upscaled"
        payload["all_upscaled_latents"] = upscaled_latents
        payload["ctx"] = ctx
        return (payload,)


NODE_CLASS_MAPPINGS = {
    "SeedVR2Canonical": SeedVR2Canonical,
    "SeedVR2CanonicalLatentInference": SeedVR2CanonicalLatentInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2Canonical": "SeedVR2 Canonical",
    "SeedVR2CanonicalLatentInference": "SeedVR2 Canonical Latent Inference",
}

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=720)
    args = parser.parse_args()

    input_dir = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(path for path in input_dir.iterdir() if path.is_file())
    if not videos:
        raise FileNotFoundError(f"no input videos found under {input_dir}")

    for video in videos:
        shutil.copy2(video, output_dir / video.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

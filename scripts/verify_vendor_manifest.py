from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "vendor" / "SeedVR"
MANIFEST_PATH = REPO_ROOT / "vendor" / "SeedVR.upstream.sha256.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual_paths = sorted(
        str(path.relative_to(VENDOR_ROOT)).replace("\\", "/")
        for path in VENDOR_ROOT.rglob("*")
        if path.is_file()
    )

    expected_paths = sorted(expected)
    missing = sorted(set(expected_paths) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(expected_paths))
    mismatched = [
        rel_path
        for rel_path in expected_paths
        if rel_path in actual_paths and sha256(VENDOR_ROOT / rel_path) != expected[rel_path]
    ]

    print(f"upstream_commit: {manifest['upstream_commit']}")
    print(f"expected_files: {len(expected_paths)}")
    print(f"actual_files: {len(actual_paths)}")
    print(f"missing: {missing}")
    print(f"extra: {extra}")
    print(f"mismatched: {mismatched}")

    match = not missing and not extra and not mismatched
    print(f"vendor_manifest_match: {str(match).lower()}")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())

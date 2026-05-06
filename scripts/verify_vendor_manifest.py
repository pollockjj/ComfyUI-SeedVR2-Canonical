from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ENTRYPOINT = REPO_ROOT / "vendor" / "SeedVR" / "projects" / "inference_seedvr2_3b.py"
BANNED_TOKENS = (
    "CANONICAL_PHASE",
    "PROCESSING_BEFORE_DIT",
    "VAE_ENCODE_BEFORE",
    "VAE_ENCODE_AFTER",
    "memory_allocated",
)


def main() -> int:
    missing = not VENDOR_ENTRYPOINT.is_file()
    text = "" if missing else VENDOR_ENTRYPOINT.read_text(encoding="utf-8")
    matches = [token for token in BANNED_TOKENS if token in text]
    match = not missing and not matches

    print(f"vendor_entrypoint: {VENDOR_ENTRYPOINT.relative_to(REPO_ROOT)}")
    print(f"missing: {str(missing).lower()}")
    print(f"instrumentation_matches: {matches}")
    print(f"vendor_manifest_match: {str(match).lower()}")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())

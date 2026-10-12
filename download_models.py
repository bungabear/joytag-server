"""Download and verify the pinned official JoyTag ONNX artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen


MODEL_ID = "fancyfeast/joytag"
REVISION = "6b7f16331a6ccf0fdce37d5a9564715f6e772b22"
ARTIFACTS = {
    "config.json": "b53ee29a8f0d6353331f455fd640c70ef45e5bd979b656f7b9983cffa3724562",
    "top_tags.txt": "32b1963a234af848643b2bbf47d8eff1f1c7889406810c57b980f41b2b9e01d0",
    "model.onnx": "f85b7130e6e549b5b0822537007b7482e8c4c8e754c8d9a5bee08e27050e1097",
}
CHUNK_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_SEC = 60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_artifact(filename: str, output_dir: Path, expected_sha256: str) -> dict:
    """Reuse verified files, otherwise replace only a verified complete download."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{filename}"
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        print(f"Verified cached {filename}", flush=True)
        return {"sha256": expected_sha256, "bytes": destination.stat().st_size, "url": url}

    temporary_path: Path | None = None
    try:
        request = Request(url, headers={"User-Agent": "joytag-server-model-downloader/1.0"})
        digest = hashlib.sha256()
        downloaded_bytes = 0
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{filename}.", suffix=".part", dir=output_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
                for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                    temporary.write(chunk)
                    digest.update(chunk)
                    downloaded_bytes += len(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {filename}: expected {expected_sha256}, "
                f"received {actual_sha256}"
            )
        os.replace(temporary_path, destination)
        print(f"Downloaded and verified {filename} ({downloaded_bytes} bytes)", flush=True)
        return {"sha256": actual_sha256, "bytes": downloaded_bytes, "url": url}
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def download_models(output_dir: Path = Path("models")) -> dict:
    """Create a reproducible manifest after every pinned artifact is verified."""
    output_dir = Path(output_dir)
    files = {
        filename: download_artifact(filename, output_dir, expected_sha256)
        for filename, expected_sha256 in ARTIFACTS.items()
    }
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "files": files}
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=".manifest.",
            suffix=".part", dir=output_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(manifest, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_dir / "manifest.json")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models"), help="Model directory (default: models)")
    args = parser.parse_args()
    try:
        download_models(args.output)
    except (OSError, URLError, ValueError) as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

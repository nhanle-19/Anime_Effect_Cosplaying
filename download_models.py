#!/usr/bin/env python3
"""Download the runtime models that are intentionally not stored in Git."""

from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from pathlib import Path


MODELS_DIR = Path(__file__).parent / "models"
MODELS = {
    "hand_landmarker.task": {
        "sha256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        ),
    },
    "face_landmarker.task": {
        "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        ),
    },
    "rtmpose_hand_256.onnx": {
        "sha256": "39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29",
        "url": (
            "https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/"
            "a6e9fb8a9190efd0383b059033f45645180cc7df/rtmpose-m-hand.onnx"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, spec: dict[str, str]) -> None:
    destination = MODELS_DIR / name
    expected = spec["sha256"]
    if destination.exists() and sha256(destination) == expected:
        print(f"Ready: {destination}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    for attempt in range(1, 4):
        print(f"Downloading {name} (attempt {attempt}/3)...")
        request = urllib.request.Request(
            spec["url"], headers={"User-Agent": "Anime-Effect-Cosplaying model setup"}
        )
        try:
            with urllib.request.urlopen(
                request, timeout=60
            ) as response, temporary.open("wb") as file:
                while block := response.read(1024 * 1024):
                    file.write(block)
            break
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 2)

    actual = sha256(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {name}: expected {expected}, received {actual}"
        )
    temporary.replace(destination)
    print(f"Ready: {destination}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for name, spec in MODELS.items():
            download(name, spec)
    except Exception as exc:
        raise SystemExit(f"Model download failed: {exc}") from exc
    print("All runtime models are ready.")


if __name__ == "__main__":
    main()

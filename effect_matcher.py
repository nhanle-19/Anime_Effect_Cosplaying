"""Optional CLIP matcher for finding which library effect appears in an image."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def discover_effect_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    directories = [
        path
        for path in root.iterdir()
        if path.is_dir() and any(path.glob("*.png"))
    ]
    return sorted(directories, key=lambda path: path.name.casefold())


def effect_metadata(directory: Path) -> dict:
    metadata_path = directory / "effect.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def effect_description(directory: Path) -> str:
    metadata = effect_metadata(directory)
    description = metadata.get("description")
    if description:
        return str(description)
    return directory.name.replace("_", " ").replace("-", " ")


def effect_attachment(directory: Path) -> str | None:
    attachment = effect_metadata(directory).get("attachment")
    return str(attachment).casefold() if attachment else None


class ClipEffectMatcher:
    """Match expected-pose images against cached visual effect embeddings."""

    def __init__(
        self,
        directories: list[Path],
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ) -> None:
        if not directories:
            raise ValueError("No effect directories were provided")
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "VLM effect matching requires open_clip_torch. "
                "Install requirements-vlm.txt first."
            ) from exc

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.directories = directories
        self.descriptions = [effect_description(path) for path in directories]
        self.attachments: dict[Path, str] = {}
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        with torch.inference_mode():
            attachment_prompts = tokenizer(
                [
                    "a visual effect attached to a person's hand",
                    "a visual effect attached to a person's eyes",
                    "a visual effect attached to a person's face",
                    "a visual effect covering the whole video frame",
                ]
            ).to(self.device)
            attachment_embeddings = self.model.encode_text(attachment_prompts)
            attachment_embeddings /= attachment_embeddings.norm(dim=-1, keepdim=True)
            description_embeddings = self.model.encode_text(
                tokenizer(self.descriptions).to(self.device)
            )
            description_embeddings /= description_embeddings.norm(dim=-1, keepdim=True)
            target_names = ("hand", "eyes", "face", "frame")
            target_scores = description_embeddings @ attachment_embeddings.T
            for index, directory in enumerate(self.directories):
                self.attachments[directory] = (
                    effect_attachment(directory)
                    or target_names[int(target_scores[index].argmax().item())]
                )
            effect_embeddings = []
            for directory in self.directories:
                images = self._representative_effect_images(directory)
                batch = torch.stack([self.preprocess(image) for image in images]).to(
                    self.device
                )
                embeddings = self.model.encode_image(batch)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                embedding = embeddings.mean(dim=0)
                effect_embeddings.append(embedding / embedding.norm())
            self.effect_embeddings = torch.stack(effect_embeddings)

    @staticmethod
    def _representative_effect_images(directory: Path, count: int = 5) -> list[Image.Image]:
        files = sorted(directory.glob("*.png"))
        indices = np.linspace(0, len(files) - 1, min(count, len(files)), dtype=int)
        images = []
        for index in indices:
            frame = cv2.imread(str(files[index]), cv2.IMREAD_UNCHANGED)
            if frame is None:
                continue
            if frame.shape[2] == 4:
                maxval = 65535.0 if frame.dtype == np.uint16 else 255.0
                alpha = frame[..., 3:4].astype(np.float32) / maxval
                frame = (
                    frame[..., :3].astype(np.float32) / maxval * alpha * 255.0
                ).astype(np.uint8)
            rgb = cv2.cvtColor(frame[..., :3], cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
        if not images:
            raise ValueError(f"No readable effect frames found in {directory}")
        return images

    def match(self, frame: np.ndarray) -> tuple[Path, float, str]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            embedding = self.model.encode_image(image)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            scores = (embedding @ self.effect_embeddings.T).squeeze(0)
        index = int(scores.argmax().item())
        directory = self.directories[index]
        return directory, float(scores[index].item()), self.attachments[directory]

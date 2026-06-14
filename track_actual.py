#!/usr/bin/env python3
"""Live hand tracker with an animated effect that triggers when a hand matches
an ordered sequence of reference poses.

Hands are tracked with the YOLO + RTMPose hand pipeline. Each detected hand is
compared against reference poses (by default, every image in the input folder
in filename order) using a
translation/scale/rotation-invariant Procrustes disparity, so the match
tolerates deviation. With VLM matching, any pose mapped to an effect can trigger
an animated PNG sequence; without VLM matching, the final pose triggers the
configured effect.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from dwpose_runtime.detector import HAND_CONNECTIONS
from effect_matcher import ClipEffectMatcher, discover_effect_directories, effect_attachment
from hand_runtime import HandPoseDetector, draw_hands
from track_image import (
    FACE_MODEL,
    HAND_DET_MODEL,
    HAND_POSE_MODEL,
    INPUT_DIR,
    EFFECTS_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
)


HAND_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"
EFFECT_DIR = EFFECTS_DIR / "FootageCrate-Magic_Circle_Fire"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

PALM_INDICES = (0, 5, 9, 13, 17)
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
FACE_OUTLINE = (10, 152, 234, 454)


class MediaPipeFaceTracker:
    """Face landmarks used only to position effects; no eye glow is drawn."""

    def __init__(self, model_path: Path) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.start_time = time.monotonic()

    def close(self) -> None:
        self.landmarker.close()

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        height, width = frame.shape[:2]
        return np.asarray(
            [(point.x * width, point.y * height) for point in result.face_landmarks[0]],
            dtype=np.float32,
        )


class MediaPipeHandTracker:
    """Real-time 21-keypoint hand tracking via MediaPipe Hand Landmarker.

    Returns the same dict shape as the YOLO+RTMPose HandPoseDetector
    ({"hands", "hand_scores"}) with the same 21-keypoint ordering, so pose
    matching and drawing are unchanged.
    """

    def __init__(
        self,
        model_path: Path,
        num_hands: int = 2,
        detection_confidence: float = 0.5,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe hand model not found at {model_path}. See README.md setup."
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=detection_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.start_time = time.monotonic()

    def close(self) -> None:
        self.landmarker.close()

    def detect(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return {"hands": [], "hand_scores": []}
        hands = []
        for landmarks in result.hand_landmarks:
            hands.append([(lm.x, lm.y) for lm in landmarks])
        points = np.array(hands, dtype=np.float32)
        scores = np.ones(points.shape[:2], dtype=np.float32)
        return {"hands": points, "hand_scores": scores}


class EffectSequence:
    """Animated PNG sequence loaded as uint8 BGR + uint8 alpha, downscaled."""

    def __init__(self, directory: Path, max_dim: int = 512) -> None:
        files = sorted(glob.glob(os.path.join(str(directory), "*.png")))
        if not files:
            raise FileNotFoundError(f"No effect frames found in {directory}")
        self.bgr: list[np.ndarray] = []
        self.alpha: list[np.ndarray] = []
        for path in files:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                continue
            height, width = img.shape[:2]
            scale = max_dim / max(height, width)
            if scale < 1.0:
                img = cv2.resize(
                    img,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            maxval = 65535.0 if img.dtype == np.uint16 else 255.0
            normalized = img.astype(np.float32) / maxval
            self.bgr.append((normalized[..., :3] * 255).astype(np.uint8))
            self.alpha.append((normalized[..., 3] * 255).astype(np.uint8))
        if not self.bgr:
            raise ValueError(f"No RGBA frames decoded from {directory}")
        self.count = len(self.bgr)
        self.index = 0
        sample = self.bgr[0]
        self.aspect = sample.shape[1] / sample.shape[0]

    def advance_once(self) -> bool:
        """Advance without looping; return True after displaying the last frame."""
        if self.index >= self.count - 1:
            return True
        self.index += 1
        return False

    def reset(self) -> None:
        self.index = 0

    def current(self) -> tuple[np.ndarray, np.ndarray]:
        return self.bgr[self.index], self.alpha[self.index]


def overlay_effect(
    frame: np.ndarray,
    fg_bgr: np.ndarray,
    fg_alpha: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
) -> None:
    target_w, target_h = size
    if target_w < 2 or target_h < 2:
        return
    fg = cv2.resize(fg_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(fg_alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    alpha = (alpha.astype(np.float32) / 255.0)[..., None]

    cx, cy = center
    x0, y0 = int(cx - target_w / 2), int(cy - target_h / 2)
    x1, y1 = x0 + target_w, y0 + target_h

    fh, fw = frame.shape[:2]
    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(fw, x1), min(fh, y1)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return

    crop_w, crop_h = dst_x1 - dst_x0, dst_y1 - dst_y0
    fg_crop = fg[src_y0:src_y0 + crop_h, src_x0:src_x0 + crop_w]
    a_crop = alpha[src_y0:src_y0 + crop_h, src_x0:src_x0 + crop_w]
    roi = frame[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
    blended = roi * (1.0 - a_crop) + fg_crop.astype(np.float32) * a_crop
    frame[dst_y0:dst_y1, dst_x0:dst_x1] = blended.astype(np.uint8)


def attachment_geometry(
    attachment: str,
    hand_points: np.ndarray | None,
    face_points: np.ndarray | None,
    frame_size: tuple[int, int],
    effect_aspect: float,
    effect_scale: float,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    width, height = frame_size
    if attachment == "hand" and hand_points is not None:
        center = hand_points[list(PALM_INDICES)].mean(axis=0)
        base = float(max(np.ptp(hand_points, axis=0)))
    elif attachment == "eyes" and face_points is not None:
        points = face_points[list(LEFT_EYE_CORNERS + RIGHT_EYE_CORNERS)]
        center = points.mean(axis=0)
        base = float(np.ptp(points, axis=0)[0])
    elif attachment == "face" and face_points is not None:
        points = face_points[list(FACE_OUTLINE)]
        center = points.mean(axis=0)
        base = float(max(np.ptp(points, axis=0)))
    elif attachment == "frame":
        center = np.array([width / 2.0, height / 2.0], np.float32)
        base = float(min(width, height) / effect_scale)
    else:
        return None
    target_h = max(2, int(base * effect_scale))
    target_w = max(2, int(target_h * effect_aspect))
    return (int(center[0]), int(center[1])), (target_w, target_h)


def load_reference_pose(json_path: Path) -> tuple[np.ndarray, float | None]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    hands = data.get("hands", [])
    if not hands:
        raise ValueError(f"No hands found in reference {json_path}")
    best = max(hands, key=lambda h: sum(h.get("scores", [])))
    points = np.asarray(best["points_normalized"], dtype=np.float32)
    if len(points) != 21:
        raise ValueError(
            f"Reference {json_path} has {len(points)} hand points; expected 21. "
            "Regenerate it with track_image.py."
        )
    image = data.get("image", {})
    width, height = image.get("width"), image.get("height")
    aspect = float(width) / float(height) if width and height else None
    return points, aspect


def load_reference_metrics(
    json_paths: list[Path], image_paths: list[Path]
) -> list[np.ndarray]:
    if len(image_paths) not in (0, 1, len(json_paths)):
        raise ValueError(
            "Provide no --reference-image, one image for all poses, or one image "
            "per --reference."
        )

    references = []
    for index, json_path in enumerate(json_paths):
        pose, aspect = load_reference_pose(json_path)
        if aspect is None and image_paths:
            image_path = image_paths[0] if len(image_paths) == 1 else image_paths[index]
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read reference image: {image_path}")
            aspect = image.shape[1] / image.shape[0]
        references.append(to_metric(pose, aspect or 16.0 / 9.0))
    return references


def natural_sort_key(path: Path) -> list[tuple[int, int | str]]:
    parts = re.split(r"(\d+)", path.name.casefold())
    return [
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
    ]


def discover_input_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Input pose directory not found: {directory}")
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    images.sort(key=natural_sort_key)
    if not images:
        raise ValueError(f"No pose images found in {directory}")
    return images


def load_image_reference_metrics(
    image_paths: list[Path],
    detector: HandPoseDetector,
) -> list[np.ndarray]:
    references = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read pose image: {image_path}")
        result = detector.detect(image)
        if not len(result["hands"]):
            raise ValueError(f"No hand found in pose image: {image_path}")
        best_index = max(
            range(len(result["hands"])),
            key=lambda index: float(np.sum(result["hand_scores"][index])),
        )
        pose = np.asarray(result["hands"][best_index], dtype=np.float32)
        if len(pose) != 21:
            raise ValueError(
                f"Pose image {image_path} produced {len(pose)} hand points; expected 21."
            )
        references.append(to_metric(pose, image.shape[1] / image.shape[0]))
    return references


def load_preview_images(image_paths: list[Path], count: int) -> list[np.ndarray | None]:
    if not image_paths:
        return [None] * count
    if len(image_paths) not in (0, 1, count):
        return [None] * count
    paths = image_paths * count if len(image_paths) == 1 else image_paths
    return [cv2.imread(str(path)) for path in paths]


def draw_pose_guide(canvas: np.ndarray, pose: np.ndarray) -> None:
    height, width = canvas.shape[:2]
    points = pose.astype(np.float32).copy()
    mins, maxs = points.min(axis=0), points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    scale = min(width * 0.72 / span[0], height * 0.52 / span[1])
    points = (points - (mins + maxs) / 2.0) * scale
    points += np.array([width / 2.0, height * 0.57], np.float32)
    pixels = np.rint(points).astype(int)
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            canvas, tuple(pixels[start]), tuple(pixels[end]),
            (0, 230, 255), 4, cv2.LINE_AA,
        )
    for point in pixels:
        cv2.circle(canvas, tuple(point), 6, (255, 100, 210), -1, cv2.LINE_AA)


def overlay_pose_on_reference(image: np.ndarray, pose: np.ndarray) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    points = pose.astype(np.float32).copy()
    points[:, 0] /= width / height
    pixels = np.rint(points * np.array([width, height], np.float32)).astype(int)
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            annotated, tuple(pixels[start]), tuple(pixels[end]),
            (0, 230, 255), 4, cv2.LINE_AA,
        )
    for point in pixels:
        cv2.circle(annotated, tuple(point), 6, (255, 100, 210), -1, cv2.LINE_AA)
        cv2.circle(annotated, tuple(point), 7, (20, 20, 20), 1, cv2.LINE_AA)
    return annotated


def tracked_pose_image(
    pose: np.ndarray | None,
    disparity: float,
    matched: bool,
    height: int = 480,
    width: int = 480,
) -> np.ndarray:
    canvas = np.full((height, width, 3), (24, 24, 30), dtype=np.uint8)
    color = (0, 220, 80) if matched else (0, 180, 255)
    status = f"MATCH  d={disparity:.3f}" if matched else (
        f"TRACKED  d={disparity:.3f}" if pose is not None else "NO HAND"
    )
    cv2.putText(
        canvas, status, (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
    )
    if pose is not None:
        draw_pose_guide(canvas[48:], pose)
    return canvas


def expected_pose_panel(
    pose: np.ndarray,
    preview: np.ndarray | None,
    tracked_pose: np.ndarray | None,
    disparity: float,
    matched: bool,
    name: str,
    index: int,
    total: int,
    height: int,
    width: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), (24, 24, 30), dtype=np.uint8)
    cv2.putText(
        panel, "EXPECTED HAND POSE", (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA,
    )
    cv2.putText(
        panel, f"{index + 1}/{total}  {name[:32]}", (18, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA,
    )

    content_y = 82
    content_h = max(1, (height - content_y - 76) // 2)
    if preview is not None:
        annotated_preview = overlay_pose_on_reference(preview, pose)
        source_h, source_w = annotated_preview.shape[:2]
        scale = min((width - 28) / source_w, content_h / source_h)
        resized = cv2.resize(
            annotated_preview,
            (max(1, int(source_w * scale)), max(1, int(source_h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        x = (width - resized.shape[1]) // 2
        y = content_y + (content_h - resized.shape[0]) // 2
        panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized

    if preview is None:
        draw_pose_guide(panel[content_y:content_y + content_h], pose)

    tracked_y = content_y + content_h + 8
    cv2.line(panel, (14, tracked_y), (width - 14, tracked_y), (70, 70, 80), 1)
    tracked = tracked_pose_image(
        tracked_pose, disparity, matched, height=content_h, width=width - 28
    )
    panel[tracked_y + 6:tracked_y + 6 + content_h, 14:width - 14] = tracked

    available = width - 36
    spacing = available / max(total, 1)
    for step in range(total):
        center = (int(18 + spacing * (step + 0.5)), height - 25)
        color = (0, 220, 80) if step < index else (0, 220, 255) if step == index else (85, 85, 95)
        cv2.circle(panel, center, 7, color, -1, cv2.LINE_AA)
    return panel


def to_metric(points_normalized: np.ndarray, aspect: float) -> np.ndarray:
    """Undo per-axis normalization so shape is aspect-correct (square pixels)."""
    metric = points_normalized.copy()
    metric[:, 0] *= aspect
    return metric


def pose_disparity(live: np.ndarray, reference: np.ndarray) -> float:
    """Procrustes disparity in [0, 4]; 0 means identical shape up to
    translation, uniform scale, rotation and reflection."""
    if live.shape != reference.shape or len(live) < 3:
        return float("inf")
    a = live - live.mean(axis=0)
    b = reference - reference.mean(axis=0)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return float("inf")
    a /= norm_a
    b /= norm_b
    singular_values = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(max(0.0, 2.0 * (1.0 - singular_values.sum())))


def best_pose_match(
    hand_result: dict,
    reference: np.ndarray,
    frame_aspect: float,
    frame_size: tuple[int, int],
) -> tuple[float, np.ndarray | None]:
    width, height = frame_size
    best_disparity = float("inf")
    best_points_px = None
    for points in hand_result["hands"]:
        if len(points) != len(reference):
            continue
        live_metric = to_metric(points.astype(np.float32), frame_aspect)
        disparity = pose_disparity(live_metric, reference)
        if disparity < best_disparity:
            best_disparity = disparity
            best_points_px = points * np.array([width, height], np.float32)
    return best_disparity, best_points_px


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument("--video", type=Path, help="use a video file instead of the camera")
    parser.add_argument("-o", "--output", type=Path, help="save annotated video to this path")
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        dest="references",
        help="analysis JSON for a required pose; repeat in the order poses must be done",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="ordered pose-image folder used when --reference is not provided",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        action="append",
        dest="reference_images",
        help="fallback image for reference aspect ratio; repeat to match --reference",
    )
    parser.add_argument("--effect-dir", type=Path, default=EFFECT_DIR)
    parser.add_argument(
        "--vlm-effect-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use CLIP at startup to match input images with effects",
    )
    parser.add_argument("--effects-dir", type=Path, default=EFFECTS_DIR,
                        help="folder containing effect sequence directories")
    parser.add_argument("--effect-match-threshold", type=float, default=0.45,
                        help="minimum startup image/effect similarity")
    parser.add_argument("--effect-vlm-model", default="ViT-B-32")
    parser.add_argument("--effect-vlm-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--hand-model", type=Path, default=HAND_LANDMARKER_MODEL)
    parser.add_argument("--face-model", type=Path, default=FACE_MODEL)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--hand-confidence", type=float, default=0.5,
                        help="MediaPipe hand detection/tracking confidence")
    parser.add_argument("--hand-threshold", type=float, default=0.15,
                        help="score threshold used only for drawing the skeleton")
    parser.add_argument("--pose-tolerance", type=float, default=0.12,
                        help="max Procrustes disparity to count as a match (higher = more lenient)")
    parser.add_argument("--pose-hold-seconds", type=float, default=1.0,
                        help="seconds a pose without an effect must remain matched")
    parser.add_argument("--effect-pose-hold-seconds", type=float, default=0.25,
                        help="seconds an effect-mapped pose must remain matched")
    parser.add_argument("--pose-hold-frames", type=int,
                        help="override the time gate with an exact frame count")
    parser.add_argument("--effect-scale", type=float, default=2.4,
                        help="effect size relative to the hand's longer side")
    parser.add_argument("--effect-max-dim", type=int, default=512)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-hand", action="store_true", help="draw the hand skeleton")
    parser.add_argument(
        "--expected-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the current expected pose beside the live view",
    )
    parser.add_argument(
        "--tracked-pose-output",
        type=Path,
        default=OUTPUT_DIR / "tracked_hand_pose.png",
        help="latest tracked hand-pose visualization",
    )
    parser.add_argument("--max-frames", type=int, help="stop after this many frames")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.pose_hold_seconds <= 0 or args.effect_pose_hold_seconds <= 0:
        parser.error(
            "--pose-hold-seconds and --effect-pose-hold-seconds must be greater than 0"
        )
    if args.pose_hold_frames is not None and args.pose_hold_frames < 1:
        parser.error("--pose-hold-frames must be at least 1")
    if not 0.0 <= args.effect_match_threshold <= 1.0:
        parser.error("--effect-match-threshold must be between 0 and 1")
    if args.references:
        reference_metrics = load_reference_metrics(
            args.references, args.reference_images or []
        )
        pose_names = [path.name for path in args.references]
        pose_previews = load_preview_images(
            args.reference_images or [], len(reference_metrics)
        )
    else:
        pose_images = discover_input_images(args.input_dir)
        reference_detector = HandPoseDetector(HAND_DET_MODEL, HAND_POSE_MODEL)
        reference_metrics = load_image_reference_metrics(pose_images, reference_detector)
        pose_names = [path.name for path in pose_images]
        pose_previews = load_preview_images(pose_images, len(reference_metrics))
    print(f"Pose order ({len(pose_names)}): {' -> '.join(pose_names)}")
    args.tracked_pose_output.parent.mkdir(parents=True, exist_ok=True)

    effect_directories = (
        discover_effect_directories(args.effects_dir)
        if args.vlm_effect_match
        else [args.effect_dir]
    )
    if not effect_directories:
        raise SystemExit(
            f"No effect directories found in {args.effects_dir}. "
            "Provide a local effect library with --effects-dir."
        )
    active_effect_dir = (
        args.effect_dir if args.effect_dir in effect_directories else effect_directories[0]
    )
    try:
        effects = {
            active_effect_dir: EffectSequence(
                active_effect_dir, max_dim=args.effect_max_dim
            )
        }
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"{exc}. Provide a local RGBA PNG sequence with --effect-dir."
        ) from exc
    effect = effects[active_effect_dir]
    active_attachment = effect_attachment(active_effect_dir) or "hand"
    try:
        effect_matcher = (
            ClipEffectMatcher(
                effect_directories,
                model_name=args.effect_vlm_model,
                pretrained=args.effect_vlm_pretrained,
            )
            if args.vlm_effect_match
            else None
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    pose_effect_dirs: list[Path | None] = (
        [None] * len(reference_metrics)
        if effect_matcher is not None
        else [None] * (len(reference_metrics) - 1) + [active_effect_dir]
    )
    pose_effect_scores = [0.0] * len(reference_metrics)
    pose_attachments = [active_attachment] * len(reference_metrics)
    if effect_matcher is not None:
        for index, preview in enumerate(pose_previews):
            if preview is None:
                continue
            matched_dir, score, attachment = effect_matcher.match(preview)
            pose_effect_scores[index] = score
            if score >= args.effect_match_threshold:
                pose_effect_dirs[index] = matched_dir
                pose_attachments[index] = attachment
                result = f"{matched_dir.name} ({score:.3f}, on {attachment})"
            else:
                pose_effect_dirs[index] = None
                result = f"none ({score:.3f} below {args.effect_match_threshold:.3f})"
            print(f"Effect match: {pose_names[index]} -> {result}")
    hand_detector = MediaPipeHandTracker(
        args.hand_model, num_hands=args.num_hands,
        detection_confidence=args.hand_confidence,
    )
    face_tracker = MediaPipeFaceTracker(args.face_model)

    source = str(args.video) if args.video else args.camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps < 1.0:
        source_fps = 30.0
    required_hold_frames = args.pose_hold_frames or max(
        1, int(round(source_fps * args.pose_hold_seconds))
    )
    effect_hold_frames = args.pose_hold_frames or max(
        1, int(round(source_fps * args.effect_pose_hold_seconds))
    )
    print(
        f"Pose gates: {required_hold_frames / source_fps:.2f}s normally, "
        f"{effect_hold_frames / source_fps:.2f}s for effect-mapped poses "
        f"at {source_fps:.1f} FPS. "
        f"Tolerance: {args.pose_tolerance:.2f}"
    )

    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            hand_detector.close()
            face_tracker.close()
            raise SystemExit(f"Could not write output video: {args.output}")

    if args.show:
        cv2.namedWindow("Track Actual", cv2.WINDOW_NORMAL)

    frame_count = 0
    pose_index = 0
    pose_hold_frames = 0
    animation_active = False
    animation_next_pose_index = 0
    effect_center: tuple[int, int] | None = None
    effect_size: tuple[int, int] | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (
                args.max_frames is not None and frame_count >= args.max_frames
            ):
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            frame_aspect = width / height

            hand_result = hand_detector.detect(frame)
            face_landmarks = face_tracker.detect(frame)

            target_pose = reference_metrics[pose_index]
            best_disparity, best_points_px = best_pose_match(
                hand_result, target_pose, frame_aspect, (width, height)
            )
            target_matched = (
                best_points_px is not None and best_disparity <= args.pose_tolerance
            )
            tracked_preview = tracked_pose_image(
                best_points_px, best_disparity, target_matched
            )
            if not cv2.imwrite(str(args.tracked_pose_output), tracked_preview):
                raise RuntimeError(
                    f"Could not write tracked pose: {args.tracked_pose_output}"
                )

            if args.draw_hand:
                draw_hands(frame, hand_result, args.hand_threshold)

            if not animation_active and target_matched:
                pose_hold_frames += 1
                target_hold_frames = (
                    effect_hold_frames
                    if pose_effect_dirs[pose_index] is not None
                    else required_hold_frames
                )
                if pose_hold_frames >= target_hold_frames:
                    selected_dir = pose_effect_dirs[pose_index]
                    next_pose_index = (pose_index + 1) % len(reference_metrics)
                    if selected_dir is None:
                        pose_index = next_pose_index
                    else:
                        animation_active = True
                        animation_next_pose_index = next_pose_index
                        active_attachment = pose_attachments[pose_index]
                        if selected_dir != active_effect_dir:
                            effect.reset()
                            active_effect_dir = selected_dir
                            if selected_dir not in effects:
                                effects[selected_dir] = EffectSequence(
                                    selected_dir, max_dim=args.effect_max_dim
                                )
                            effect = effects[selected_dir]
                        geometry = attachment_geometry(
                            active_attachment,
                            best_points_px,
                            face_landmarks,
                            (width, height),
                            effect.aspect,
                            args.effect_scale,
                        )
                        if geometry is None:
                            active_attachment = "hand"
                            geometry = attachment_geometry(
                                active_attachment,
                                best_points_px,
                                face_landmarks,
                                (width, height),
                                effect.aspect,
                                args.effect_scale,
                            )
                        effect_center, effect_size = geometry
                        effect.reset()
                    pose_hold_frames = 0
            elif not animation_active:
                pose_hold_frames = 0

            effect_active = animation_active
            displayed_effect_frame = 0
            if effect_active:
                displayed_effect_frame = effect.index + 1
                live_geometry = attachment_geometry(
                    active_attachment,
                    best_points_px,
                    face_landmarks,
                    (width, height),
                    effect.aspect,
                    args.effect_scale,
                )
                if live_geometry is not None:
                    effect_center, effect_size = live_geometry
                fg_bgr, fg_alpha = effect.current()
                overlay_effect(
                    frame, fg_bgr, fg_alpha, effect_center, effect_size,
                )
                if effect.advance_once():
                    animation_active = False
                    pose_index = animation_next_pose_index
                    pose_hold_frames = 0
                    effect_center = None
                    effect_size = None
                    effect.reset()
            else:
                effect.reset()

            if effect_active:
                label = (
                    f"CAST  animation {displayed_effect_frame}/{effect.count}"
                )
            elif best_points_px is not None:
                target_hold_frames = (
                    effect_hold_frames
                    if pose_effect_dirs[pose_index] is not None
                    else required_hold_frames
                )
                label = (
                    f"{pose_index + 1}/{len(reference_metrics)} {pose_names[pose_index]}  "
                    f"hold={pose_hold_frames / source_fps:.1f}/"
                    f"{target_hold_frames / source_fps:.2f}s  d={best_disparity:.3f}"
                )
            else:
                label = (
                    f"{pose_index + 1}/{len(reference_metrics)} "
                    f"{pose_names[pose_index]}  no hand"
                )
            color = (0, 255, 0) if effect_active else (0, 180, 255)
            cv2.putText(
                frame,
                label,
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"tol={args.pose_tolerance:.2f}  Q/Esc: quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            if effect_matcher is not None:
                mapped_effect = pose_effect_dirs[pose_index]
                cv2.putText(
                    frame,
                    f"image effect={mapped_effect.name if mapped_effect else 'none'}  "
                    f"on={pose_attachments[pose_index]}  "
                    f"vlm={pose_effect_scores[pose_index]:.3f}",
                    (20, height - 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(frame)
            if args.show:
                display = frame
                if args.expected_pose:
                    panel_width = max(280, min(460, int(width * 0.42)))
                    panel = expected_pose_panel(
                        reference_metrics[pose_index],
                        pose_previews[pose_index],
                        best_points_px,
                        best_disparity,
                        target_matched,
                        pose_names[pose_index],
                        pose_index,
                        len(reference_metrics),
                        height,
                        panel_width,
                    )
                    display = np.hstack((frame, panel))
                cv2.imshow("Track Actual", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            frame_count += 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        hand_detector.close()
        face_tracker.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed {frame_count} frame(s).")


if __name__ == "__main__":
    main()

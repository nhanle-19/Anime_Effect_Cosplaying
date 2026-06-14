#!/usr/bin/env python3
"""Identify hands, eyes, and candidate visual-effect regions in one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dwpose_runtime.detector import hand_is_structurally_valid
from face_runtime import FaceEyeDetector, draw_eyes
from hand_runtime import HandPoseDetector, draw_hands


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
JSON_DIR = BASE_DIR / "json"
EFFECTS_DIR = BASE_DIR / "effects"

HAND_DET_MODEL = MODELS_DIR / "hand_yolov8n.onnx"
HAND_POSE_MODEL = MODELS_DIR / "rtmpose_hand_256.onnx"
FACE_MODEL = MODELS_DIR / "face_landmarker.task"


def find_effect_regions(
    frame: np.ndarray,
    min_area_ratio: float,
    max_regions: int,
    sat_threshold: int = 150,
    val_threshold: int = 190,
) -> list[dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    # Candidate VFX are bright and strongly colored. Higher thresholds keep only
    # genuinely vivid glows and reject soft lighting / muted clothing, which
    # matters when the mask is later reused to composite an effect onto a person.
    mask = ((saturation >= sat_threshold) & (value >= val_threshold)).astype(np.uint8) * 255
    # Light denoise, then a single tight close so region edges stay crisp.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = frame.shape[0] * frame.shape[1]
    min_area = image_area * min_area_ratio
    max_area = image_area * 0.35
    regions: list[dict] = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        regions.append(
            {
                "bbox": [int(x), int(y), int(width), int(height)],
                "area": int(area),
                "contour": contour,
            }
        )
        if len(regions) >= max_regions:
            break
    return regions


def draw_effect_regions(frame: np.ndarray, regions: list[dict]) -> None:
    overlay = frame.copy()
    for region in regions:
        contour = region["contour"]
        x, y, width, height = region["bbox"]
        cv2.drawContours(overlay, [contour], -1, (255, 80, 255), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 80, 255), 2)
        cv2.putText(
            frame,
            "effect?",
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 80, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)


def landmark_metadata(
    hand_result: dict, eye_result: dict, hand_threshold: float
) -> dict:
    hands = []
    for points, scores in zip(hand_result["hands"], hand_result["hand_scores"]):
        if not hand_is_structurally_valid(points, scores, hand_threshold):
            continue
        visible = scores >= hand_threshold
        hands.append(
            {
                "points_normalized": points.round(5).tolist(),
                "scores": scores.round(4).tolist(),
                "visible": visible.tolist(),
            }
        )

    eyes = []
    for eye in eye_result["eyes"]:
        points = np.asarray(eye["points_normalized"], dtype=np.float32)
        scores = np.asarray(eye["scores"], dtype=np.float32)
        eyes.append(
            {
                "points_normalized": points.round(5).tolist(),
                "scores": scores.round(4).tolist(),
            }
        )
    return {"hands": hands, "eyes": eyes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="path to the input picture")
    parser.add_argument("-o", "--output", type=Path, help="annotated output image")
    parser.add_argument("--json-output", type=Path, help="analysis JSON output")
    parser.add_argument("--hand-det-model", type=Path, default=HAND_DET_MODEL)
    parser.add_argument("--hand-pose-model", type=Path, default=HAND_POSE_MODEL)
    parser.add_argument("--face-model", type=Path, default=FACE_MODEL)
    parser.add_argument("--hand-threshold", type=float, default=0.15)
    parser.add_argument("--hand-score", type=float, default=0.3,
                        help="YOLO hand-box confidence threshold")
    parser.add_argument("--eye-threshold", type=float, default=0.3,
                        help="MediaPipe face detection confidence")
    parser.add_argument("--effects", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--effect-min-area", type=float, default=0.001)
    parser.add_argument("--effect-sat", type=int, default=150,
                        help="min HSV saturation for an effect (higher = sharper)")
    parser.add_argument("--effect-val", type=int, default=190,
                        help="min HSV value/brightness for an effect (higher = sharper)")
    parser.add_argument("--max-effects", type=int, default=8)
    parser.add_argument("--show", action="store_true", help="open an output preview")
    args = parser.parse_args()

    frame = cv2.imread(str(args.input))
    if frame is None:
        raise SystemExit(f"Could not read input image: {args.input}")

    original = frame.copy()
    hand_detector = HandPoseDetector(
        args.hand_det_model, args.hand_pose_model, score_threshold=args.hand_score
    )
    face_detector = FaceEyeDetector(
        args.face_model, detection_confidence=args.eye_threshold, video=False
    )
    try:
        hand_result = hand_detector.detect(original)
        eye_result = face_detector.detect(original)
    finally:
        face_detector.close()

    effects = (
        find_effect_regions(
            original,
            args.effect_min_area,
            args.max_effects,
            sat_threshold=args.effect_sat,
            val_threshold=args.effect_val,
        )
        if args.effects
        else []
    )
    hand_count = draw_hands(frame, hand_result, args.hand_threshold)
    eye_count = draw_eyes(frame, eye_result)
    draw_effect_regions(frame, effects)

    output = args.output or OUTPUT_DIR / f"{args.input.stem}_analyzed.jpg"
    json_output = args.json_output or JSON_DIR / f"{args.input.stem}_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise SystemExit(f"Could not write output image: {output}")

    metadata = landmark_metadata(hand_result, eye_result, args.hand_threshold)
    metadata["image"] = {"width": int(original.shape[1]), "height": int(original.shape[0])}
    metadata["effects"] = [
        {"bbox": region["bbox"], "area": region["area"]} for region in effects
    ]
    metadata["counts"] = {
        "hands": int(hand_count),
        "eyes": int(eye_count),
        "effect_regions": len(effects),
    }
    json_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"Saved {output}. Hands: {hand_count}. Eyes: {eye_count}. "
        f"Candidate effects: {len(effects)}."
    )
    print(f"Analysis: {json_output}")
    if args.show:
        cv2.imshow("Hand, Eye, and Effect Analysis", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

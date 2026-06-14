"""Hand-specific pose detection: YOLOv8 hand boxes + RTMPose hand keypoints.

The keypoint stage reuses DWPose's RTMPose top-down inference (the pose model is
a 21-keypoint hand RTMPose export), and drawing reuses DWPose's hand utilities.
DWPose itself is no longer used for whole-body detection here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from dwpose_runtime.detector import (
    HAND_CONNECTIONS,
    _draw_connections,
    _draw_points,
    hand_is_structurally_valid,
)
from dwpose_runtime.onnxpose import inference_pose

from .yolo_hand_det import detect_hands


class HandPoseDetector:
    def __init__(
        self,
        det_model: Path,
        pose_model: Path,
        device: str = "cpu",
        score_threshold: float = 0.3,
    ) -> None:
        for model in (det_model, pose_model):
            if not Path(model).exists():
                raise FileNotFoundError(f"Hand model not found: {model}")

        providers = (
            ["CPUExecutionProvider"]
            if device == "cpu"
            else ["CUDAExecutionProvider"]
        )
        provider_options = None if device == "cpu" else [{"device_id": 0}]
        self.session_det = ort.InferenceSession(
            str(det_model), providers=providers, provider_options=provider_options
        )
        self.session_pose = ort.InferenceSession(
            str(pose_model), providers=providers, provider_options=provider_options
        )
        self.score_threshold = score_threshold

    def detect(self, image: np.ndarray) -> dict:
        height, width = image.shape[:2]
        boxes = detect_hands(self.session_det, image, score_thr=self.score_threshold)
        if len(boxes) == 0:
            return {"hands": [], "hand_scores": []}

        keypoints, scores = inference_pose(
            self.session_pose, boxes.tolist(), image
        )
        points = keypoints.astype(np.float32)
        points[..., 0] /= width
        points[..., 1] /= height
        return {"hands": points, "hand_scores": scores}


def draw_hands(
    frame: np.ndarray, hand_result: dict, threshold: float = 0.15
) -> int:
    visible_hands = 0
    for points, scores in zip(hand_result["hands"], hand_result["hand_scores"]):
        if not hand_is_structurally_valid(points, scores, threshold):
            continue
        visible_hands += 1
        _draw_connections(
            frame, points, scores, HAND_CONNECTIONS, threshold, (0, 220, 255), 2
        )
        _draw_points(frame, points, scores, threshold, (255, 80, 180), 3)
    return visible_hands

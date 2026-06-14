"""Eye landmark detection via MediaPipe Face Landmarker.

Replaces DWPose for eyes only. Supports a per-image (IMAGE) running mode for
stills and a stateful VIDEO mode for frame sequences.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


LEFT_EYE_CONTOUR = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
)
RIGHT_EYE_CONTOUR = (
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
)
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)
EYE_LANDMARKS = (
    LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR + LEFT_IRIS + RIGHT_IRIS
)
IRIS_LANDMARKS = LEFT_IRIS + RIGHT_IRIS


class FaceEyeDetector:
    def __init__(
        self,
        model_path: Path,
        detection_confidence: float = 0.3,
        num_faces: int = 5,
        video: bool = False,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe face model not found: {model_path}. See README.md setup."
            )

        running_mode = (
            vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        )
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_faces=num_faces,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=detection_confidence,
            min_tracking_confidence=detection_confidence,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.video = video
        self.start_time = time.monotonic()

    def close(self) -> None:
        self.landmarker.close()

    def detect(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
            result = self.landmarker.detect_for_video(image, timestamp_ms)
        else:
            result = self.landmarker.detect(image)

        eyes = []
        for landmarks in result.face_landmarks:
            points = np.array(
                [(landmark.x, landmark.y) for landmark in landmarks],
                dtype=np.float32,
            )
            if len(points) <= max(EYE_LANDMARKS):
                continue
            eye_points = points[list(EYE_LANDMARKS)]
            eyes.append(
                {
                    "points_normalized": eye_points,
                    "scores": np.ones(len(eye_points), dtype=np.float32),
                }
            )
        return {"eyes": eyes}


def draw_eyes(
    frame: np.ndarray,
    eye_result: dict,
    color: tuple[int, int, int] = (80, 255, 255),
    radius: int = 2,
) -> int:
    height, width = frame.shape[:2]
    visible_eyes = 0
    for eye in eye_result["eyes"]:
        visible_eyes += 1
        for x, y in eye["points_normalized"]:
            cv2.circle(
                frame,
                (int(x * width), int(y * height)),
                radius,
                color,
                -1,
                cv2.LINE_AA,
            )
    return visible_eyes

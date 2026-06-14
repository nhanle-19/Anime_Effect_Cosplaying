"""DWPose inference and overlay drawing without a PyTorch dependency."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .wholebody import Wholebody


BODY_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16), (0, 15), (15, 17),
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
FACE_EYE_INDICES = tuple(range(36, 48))


class DWPoseDetector:
    def __init__(self, detector_model: Path, pose_model: Path) -> None:
        for model in (detector_model, pose_model):
            if not model.exists():
                raise FileNotFoundError(f"DWPose model not found: {model}")
        self.wholebody = Wholebody(str(detector_model), str(pose_model), device="cpu")

    def detect(self, image: np.ndarray) -> dict:
        height, width = image.shape[:2]
        candidate, score = self.wholebody(image)
        if candidate.size == 0:
            return {
                "bodies": [],
                "body_scores": [],
                "faces": [],
                "face_scores": [],
                "hands": [],
                "hand_scores": [],
            }

        points = candidate.astype(np.float32)
        points[..., 0] /= width
        points[..., 1] /= height
        return {
            "bodies": points[:, :18],
            "body_scores": score[:, :18],
            "faces": points[:, 24:92],
            "face_scores": score[:, 24:92],
            "hands": np.vstack((points[:, 92:113], points[:, 113:134])),
            "hand_scores": np.vstack((score[:, 92:113], score[:, 113:134])),
        }


def _pixel(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    return int(point[0] * width), int(point[1] * height)


def _draw_connections(
    frame: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    connections: tuple[tuple[int, int], ...],
    threshold: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    for start, end in connections:
        if scores[start] < threshold or scores[end] < threshold:
            continue
        cv2.line(
            frame,
            _pixel(points[start], width, height),
            _pixel(points[end], width, height),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    height, width = frame.shape[:2]
    for point, score in zip(points, scores):
        if score >= threshold:
            cv2.circle(
                frame, _pixel(point, width, height), radius, color, -1, cv2.LINE_AA
            )


def hand_is_structurally_valid(
    points: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    min_width: float = 0.025,
    min_height: float = 0.025,
) -> bool:
    visible = points[scores >= threshold]
    if len(visible) < 4:
        return False
    spread = np.ptp(visible, axis=0)
    unique_points = len(np.unique(np.round(visible, 4), axis=0))
    return bool(spread[0] >= min_width and spread[1] >= min_height and unique_points >= 4)


def draw_dwpose(frame: np.ndarray, pose: dict, threshold: float = 0.3) -> dict[str, int]:
    for points, scores in zip(pose["bodies"], pose["body_scores"]):
        _draw_connections(frame, points, scores, BODY_CONNECTIONS, threshold, (80, 255, 80), 3)
        _draw_points(frame, points, scores, threshold, (255, 100, 180), 4)

    visible_hands = 0
    for points, scores in zip(pose["hands"], pose["hand_scores"]):
        if not hand_is_structurally_valid(points, scores, threshold):
            continue
        visible_hands += 1
        _draw_connections(frame, points, scores, HAND_CONNECTIONS, threshold, (0, 220, 255), 2)
        _draw_points(frame, points, scores, threshold, (255, 80, 180), 3)

    visible_faces = 0
    for points, scores in zip(pose["faces"], pose["face_scores"]):
        if np.count_nonzero(scores >= threshold) < 4:
            continue
        visible_faces += 1
        _draw_points(frame, points, scores, threshold, (255, 220, 80), 2)

    return {
        "people": len(pose["bodies"]),
        "hands": visible_hands,
        "faces": visible_faces,
    }


def draw_face_eyes(frame: np.ndarray, pose: dict, threshold: float = 0.3) -> dict[str, int]:
    visible_faces = 0
    visible_eyes = 0
    for points, scores in zip(pose["faces"], pose["face_scores"]):
        face_visible = np.count_nonzero(scores >= threshold)
        eye_scores = scores[list(FACE_EYE_INDICES)]
        eye_visible = np.count_nonzero(eye_scores >= threshold)
        if face_visible < 4:
            continue

        visible_faces += 1
        visible_eyes += eye_visible >= 4
        _draw_points(frame, points, scores, threshold, (255, 220, 80), 2)
        _draw_points(
            frame,
            points[list(FACE_EYE_INDICES)],
            eye_scores,
            threshold,
            (80, 255, 255),
            3,
        )

    return {
        "people": len(pose["bodies"]),
        "faces": visible_faces,
        "eyes": visible_eyes,
    }


def draw_hands_eyes(
    frame: np.ndarray,
    pose: dict,
    hand_threshold: float = 0.15,
    eye_threshold: float = 0.3,
) -> dict[str, int]:
    visible_hands = 0
    for points, scores in zip(pose["hands"], pose["hand_scores"]):
        if not hand_is_structurally_valid(points, scores, hand_threshold):
            continue
        visible_hands += 1
        _draw_connections(
            frame,
            points,
            scores,
            HAND_CONNECTIONS,
            hand_threshold,
            (0, 220, 255),
            2,
        )
        _draw_points(frame, points, scores, hand_threshold, (255, 80, 180), 3)

    visible_faces = 0
    visible_eyes = 0
    for points, scores in zip(pose["faces"], pose["face_scores"]):
        eye_scores = scores[list(FACE_EYE_INDICES)]
        if np.count_nonzero(scores >= eye_threshold) >= 4:
            visible_faces += 1
        if np.count_nonzero(eye_scores >= eye_threshold) < 4:
            continue
        visible_eyes += 1
        _draw_points(
            frame,
            points[list(FACE_EYE_INDICES)],
            eye_scores,
            eye_threshold,
            (80, 255, 255),
            3,
        )

    return {
        "people": len(pose["bodies"]),
        "hands": visible_hands,
        "faces": visible_faces,
        "eyes": visible_eyes,
    }

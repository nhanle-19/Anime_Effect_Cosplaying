"""YOLOv8 hand bounding-box detection on top of ONNX Runtime."""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from dwpose_runtime.onnxdet import nms


def _letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = img.shape[:2]
    ratio = min(new_shape[0] / height, new_shape[1] / width)
    new_unpad = (int(round(width * ratio)), int(round(height * ratio)))
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    delta_w = (new_shape[1] - new_unpad[0]) / 2
    delta_h = (new_shape[0] - new_unpad[1]) / 2
    top, bottom = int(round(delta_h - 0.1)), int(round(delta_h + 0.1))
    left, right = int(round(delta_w - 0.1)), int(round(delta_w + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, ratio, (left, top)


def detect_hands(
    session: ort.InferenceSession,
    image: np.ndarray,
    score_thr: float = 0.3,
    nms_thr: float = 0.45,
) -> np.ndarray:
    """Return hand bounding boxes as an (N, 4) array of xyxy pixel coordinates.

    Assumes a single-class YOLOv8 detect head exported without baked-in NMS,
    i.e. an output shaped (1, 4 + num_classes, num_anchors) in input-pixel space.
    The orientation is auto-detected so (1, num_anchors, C) also works.
    """
    model_input = session.get_inputs()[0]
    shape = model_input.shape
    if isinstance(shape[2], int) and isinstance(shape[3], int):
        input_size = (shape[2], shape[3])
    else:
        input_size = (640, 640)

    padded, ratio, (pad_x, pad_y) = _letterbox(image, input_size)
    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]

    outputs = session.run(None, {model_input.name: blob})
    preds = outputs[0]
    if preds.ndim == 3:
        preds = preds[0]
    # Normalize to (num_anchors, C); the channel axis is the smaller one.
    if preds.shape[0] < preds.shape[1]:
        preds = preds.transpose(1, 0)

    boxes = preds[:, :4]
    class_scores = preds[:, 4:]
    scores = class_scores.max(axis=1)

    keep_mask = scores > score_thr
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

    xyxy[:, [0, 2]] -= pad_x
    xyxy[:, [1, 3]] -= pad_y
    xyxy /= ratio

    height, width = image.shape[:2]
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)

    keep = nms(xyxy, scores, nms_thr)
    return xyxy[keep].astype(np.float32)

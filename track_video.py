#!/usr/bin/env python3
"""Track hand keypoints (YOLO + RTMPose) and eyes (MediaPipe) in a video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from face_runtime import FaceEyeDetector, draw_eyes
from hand_runtime import HandPoseDetector, draw_hands
from track_image import FACE_MODEL, HAND_DET_MODEL, HAND_POSE_MODEL, OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="path to the input video")
    parser.add_argument("-o", "--output", type=Path, help="output video path")
    parser.add_argument("--hand-det-model", type=Path, default=HAND_DET_MODEL)
    parser.add_argument("--hand-pose-model", type=Path, default=HAND_POSE_MODEL)
    parser.add_argument("--face-model", type=Path, default=FACE_MODEL)
    parser.add_argument("--hand-threshold", type=float, default=0.15)
    parser.add_argument("--hand-score", type=float, default=0.3,
                        help="YOLO hand-box confidence threshold")
    parser.add_argument("--eye-threshold", type=float, default=0.3,
                        help="MediaPipe face detection confidence")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames")
    parser.add_argument("--show", action="store_true", help="preview while processing")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"Could not open input video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = args.output or OUTPUT_DIR / f"{args.input.stem}_hands.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not write output video: {output}")

    hand_detector = HandPoseDetector(
        args.hand_det_model, args.hand_pose_model, score_threshold=args.hand_score
    )
    face_detector = FaceEyeDetector(
        args.face_model, detection_confidence=args.eye_threshold, video=True
    )
    frame_count = hand_frames = eye_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (
                args.max_frames is not None and frame_count >= args.max_frames
            ):
                break

            hand_result = hand_detector.detect(frame)
            eye_result = face_detector.detect(frame)
            hand_count = draw_hands(frame, hand_result, args.hand_threshold)
            eye_count = draw_eyes(frame, eye_result)
            hand_frames += hand_count > 0
            eye_frames += eye_count > 0
            writer.write(frame)
            frame_count += 1

            if args.show:
                cv2.imshow("Hand & Eye Video Tracking", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        writer.release()
        face_detector.close()
        if args.show:
            cv2.destroyAllWindows()

    print(
        f"Saved {output} from {frame_count} frame(s). "
        f"Hand frames: {hand_frames}. Eye frames: {eye_frames}."
    )


if __name__ == "__main__":
    main()

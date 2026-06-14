# Live Hand Tracker with Pose-Triggered Effects

This project analyzes hands and eyes in images or video and includes a live
webcam demo. When a hand completes an ordered reference-pose sequence,
`track_actual.py` composites an animated PNG effect onto the tracked hand.

Example inputs, analysis JSON, and a tracked-pose output are included. Effect
frame libraries are intentionally not included because they are large and may
have separate licenses.

## Quick start

Python 3.10 or newer is recommended.

```bash
git clone <repository-url>
cd alimi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_models.py

# Analyze an included example image.
python track_image.py input/1.png --no-effects
```

`download_models.py` downloads about 65 MB of required MediaPipe and RTMPose
runtime models into `models/` and verifies their SHA-256 checksums. Downloaded
models are ignored by Git to keep clones small. Re-run the command whenever a
fresh checkout is missing models.

The example command writes:

- `output/1_analyzed.jpg`, an annotated image.
- `json/1_analysis.json`, normalized landmark data and detection counts.

### Recommended live workflow: VLM effect matching

For normal use, let the VLM choose the effect automatically. Install the
optional VLM dependencies, add one or more local effect folders, and run:

```bash
python -m pip install -r requirements-vlm.txt
python track_actual.py --vlm-effect-match --effects-dir /path/to/effect-library
```

At startup, the VLM compares every ordered image in `input/` with every effect
folder in the library. It automatically maps each reference image to the
closest effect or to `none`; users do not manually specify which reference
image needs which effect. When a mapped pose is completed, its effect plays and
the runner continues to the next pose.

Allow camera access when prompted. On Linux, close other applications using the
webcam if the camera cannot be opened. Press `Q` or `Esc` to quit.

## Add a pose sequence

Create a folder containing one image for each required hand pose. Every
supported image in the folder becomes part of the sequence and is loaded in
natural filename order:

```text
my_sequence/
  01_open_hand.jpg
  02_raise_fingers.png
  03_cast_with_fire_effect.png
```

Then run:

```bash
python track_actual.py \
  --input-dir my_sequence \
  --vlm-effect-match \
  --effects-dir /path/to/effect-library
```

Pose-image guidelines:

- Use `.bmp`, `.jpeg`, `.jpg`, `.png`, or `.webp`.
- Give files numeric prefixes to control the required order.
- Show one clear hand in each image; the highest-confidence detected hand is
  used as that pose's reference.
- Keep setup images visually clean when they should map to `none`.
- Make any effect-triggering pose image visually show the desired effect so the
  VLM can match it to the correct effect folder.
- Use a separate folder per sequence. Otherwise, every image in the folder is
  included.

No analysis JSON needs to be created for this workflow. The live tracker
detects the reference hand pose directly from each image at startup.

## Add effects

Create an effect library with one subdirectory per effect. Each effect
subdirectory must contain ordered RGBA PNG animation frames:

```text
my_effect_library/
  fire_circle/
    effect.json
    fire-00001.png
    fire-00002.png
    fire-00003.png
  galaxy_ball/
    effect.json
    galaxy-00001.png
    galaxy-00002.png
    galaxy-00003.png
```

Frames are loaded in filename order, so use zero-padded frame numbers. RGBA
transparency controls how each frame is composited over the camera image.

An optional `effect.json` describes the effect and controls attachment:

```json
{
  "description": "a fiery orange magic summoning circle around a hand",
  "attachment": "hand"
}
```

Supported attachment targets are `hand`, `eyes`, `face`, and `frame`. If
`attachment` is omitted, the VLM infers it from the description. Effect
selection itself is visual: CLIP compares the pose image with representative
PNG frames from each effect folder.

The `effects/` directory is ignored by Git, so local effect libraries are not
included in a push. They can live there or anywhere else supplied through
`--effects-dir`.

### Manual single-effect fallback

To skip VLM matching and always use one local effect for the final pose:

```bash
python track_actual.py --effect-dir /path/to/my-effect
```

### How it works

- Hands: YOLO hand detector + RTMPose 21-keypoint model.
- Pose match: each detected hand is compared to one or more ordered reference
  poses. By default, every image in `input/` is loaded in natural filename
  order using a
  translation/scale/rotation/reflection-invariant Procrustes disparity, so the
  match tolerates deviation. Lower disparity = closer match.
- Effect: the selected PNG sequence is alpha-composited onto the matching hand
  and always plays once from beginning to end before another sequence can start.
  Its attachment point is tracked throughout playback, so hand effects follow
  the moving hand.
- GUI: the live window shows the current expected pose image and sequence
  progress beside the camera feed, plus the currently tracked hand skeleton for
  direct comparison. The detected reference skeleton is overlaid on the
  expected image so incorrect reference tracking is visible. JSON-only
  references use a generated hand-skeleton preview when no image is provided.
- Tracked pose: the latest tracked hand skeleton is continually saved to
  `output/tracked_hand_pose.png`.

### Common options

```bash
# Test on a video file instead of the webcam, and save the result
python track_actual.py --vlm-effect-match --effects-dir /path/to/effect-library \
  --video clip.mp4 --no-mirror -o output/live-demo.mp4

# Loosen/tighten how close the pose must be (default 0.12, 0 = exact)
python track_actual.py --vlm-effect-match --effects-dir /path/to/effect-library \
  --pose-tolerance 0.25

# Bigger/smaller effect relative to the hand, draw the hand skeleton too
python track_actual.py --vlm-effect-match --effects-dir /path/to/effect-library \
  --effect-scale 3.0 --draw-hand

# Use a different camera
python track_actual.py --vlm-effect-match --effects-dir /path/to/effect-library \
  --camera 1

# Use a different ordered pose-image folder
python track_actual.py --vlm-effect-match --effects-dir /path/to/effect-library \
  --input-dir my_poses --pose-hold-seconds 1.5

# Hide the expected-pose side panel
python track_actual.py --vlm-effect-match --effects-dir /path/to/effect-library \
  --no-expected-pose

# Save the latest tracked-pose image somewhere else
python track_actual.py --vlm-effect-match --effects-dir /path/to/effect-library \
  --tracked-pose-output output/my_pose.png
```

### What VLM effect matching does

The CLIP matcher analyzes the expected images in `input/` at startup.
It compares each image against representative animation frames from every
effect directory and stores the closest effect for that pose. The VLM does not
run on live camera frames, so live performance still uses the hand and face
trackers rather than repeatedly running CLIP.

The first run downloads the configured CLIP weights. Startup prints mappings
for every expected image. Matches below the default `0.45` similarity threshold
are stored as `none`, so clean setup poses do not receive an effect. For
example, only images that visibly contain an effect may clear the threshold.
Every mapped pose triggers its selected animation; poses mapped to `none`
advance without one.

Typical startup output looks like:

```text
Effect match: 01_open_hand.jpg -> none (0.312 below 0.450)
Effect match: 02_raise_fingers.png -> none (0.338 below 0.450)
Effect match: 03_cast_with_fire_effect.png -> fire_circle (0.681, on hand)
```

Increase `--effect-match-threshold` if clean poses are receiving effects.
Decrease it if an intended effect pose is mapping to `none`. Without
`--vlm-effect-match`, the configured `--effect-dir` is used only for the final
pose without automatic selection.

The on-screen `d=` value is the current best disparity — watch it to pick a
`--pose-tolerance`. By default, poses without effects must remain matched for
1.0 seconds, while VLM-mapped effect poses use a shorter 0.25-second gate.
Change these with `--pose-hold-seconds` and
`--effect-pose-hold-seconds`, or use `--pose-hold-frames` for an exact
frame-count override. The effect sequence starts nearly transparent and builds
up, so a held effect pose looks like the effect is being cast. Poses advance
directly after their hold gates, so you can continuously move from one to the
next. Name files with numeric prefixes such as
`01_open.png`, `02_point.png`, and `03_cast.png` to control their order.
Explicit repeated `--reference` JSON arguments override the input-folder list.
When using JSON references with VLM matching, also repeat `--reference-image`
for each JSON file so the VLM has images to compare:

```bash
python track_actual.py \
  --reference json/pose_01_analysis.json \
  --reference-image input/pose_01.png \
  --reference json/pose_02_analysis.json \
  --reference-image input/pose_02.png \
  --vlm-effect-match \
  --effects-dir /path/to/effect-library
```

## Project layout

```
input/      ordered pose images used by track_actual.py
output/     example and generated annotated images/videos
json/       example and generated analysis JSON from track_image.py
effects/    local effect assets; ignored by Git
models/     small bundled detector plus ignored downloaded runtime models
```

The scripts default to these folders: `track_image.py` / `track_video.py` read
the input path you pass and write results into `output/` (and `json/`), while
`track_actual.py` defaults its reference pose and reference image to `json/`
and `input/`. Pass `--effect-dir` for a single local effect or `--effects-dir`
with `--vlm-effect-match` for a local effect library.

## Included examples

- `input/1.png`, `input/2.png`, and `input/3.jpg` form the default ordered
  pose-image sequence for the live demo. Add, remove, or rename images to
  change the sequence; numeric filenames control the order.
- `json/` contains example analysis files generated by `track_image.py`.
- `output/tracked_hand_pose.png` shows the live tracker's saved skeleton-output
  format.

Generated files in `output/` and `json/` are not ignored, so useful examples
can be committed deliberately. Review newly generated files before staging.

## Hand and Eye Analysis

Hands use a dedicated two-stage, hand-specific pipeline (no full body): a YOLO
hand-box detector followed by an RTMPose 21-keypoint hand model. Eyes use
MediaPipe Face Landmarker. The DWPose runtime is kept in `dwpose_runtime/` and
is reused only for its RTMPose decoding and hand-drawing helpers.

The hand pipeline uses two ONNX models:

- `models/hand_yolov8n.onnx` is the small bundled single-class YOLOv8 hand
  detector. It was exported from
  `Runware/adetailer`'s `hand_yolov8n.pt` via
  `YOLO(...).export(format="onnx", imgsz=640, opset=12, nms=False)`
  (output shape `(1, 5, 8400)`).
- `models/rtmpose_hand_256.onnx` is downloaded by `download_models.py`. It is
  an RTMPose-m hand keypoint model from `bukuroo/RTMPose-ONNX` with 21 points,
  256x256 input, and SimCC output.

The same setup command downloads MediaPipe's `hand_landmarker.task` for live
tracking and `face_landmarker.task` for face/eye tracking.

### Image

`track_image.py` analyzes one image frame. It draws hand landmarks, eye
landmarks, and candidate visual-effect regions, and writes a JSON file with
normalized hand/eye points, source image dimensions, and effect bounding boxes.
Pose references always retain all 21 hand landmark indices; regenerate older
analysis JSON files that report fewer points before using them in a sequence.

```bash
python track_image.py input/1.png
```

By default the result is saved to `output/1_analyzed.jpg` and
`json/1_analysis.json`. Choose another output path or open a preview with:

```bash
python track_image.py input/1.png -o output/1_preview.png --show
```

Effect regions are color/brightness candidates, not semantic VFX detections.
The HSV saturation/brightness cutoffs control how strict the match is — raise
them for sharper, more selective effect masks (useful when the mask is later
reused to composite an effect onto a person):

```bash
python track_image.py input/1.png --effect-sat 175 --effect-val 205
```

Defaults are `--effect-sat 150 --effect-val 190`. Disable effects entirely when
only hand and eye landmarks are needed:

```bash
python track_image.py input/1.png --no-effects
```

### Video

`track_video.py` runs the same hand and eye detection per frame and writes an
annotated video to `output/<name>_hands.mp4`.

```bash
python track_video.py input/clip.mp4
```

Use `python track_actual.py --help`, `python track_image.py --help`, or
`python track_video.py --help` for every available option.

## Preparing a GitHub push

This repository keeps the included input/output/JSON examples while excluding
effect libraries and downloaded runtime models. Before staging, review the
ignored files and working tree:

```bash
git status --ignored --short
git status --short
```

No project-level license has been selected yet; add one before publishing if
the repository is intended for reuse by others.

# Anime Effect Cosplaying

This is a fun project for trying anime hand-sign sequences in front of a
webcam and getting an effect when the pose matches.

It follows the hand poses in order, and CLIP tries to figure out which effect
goes with each pose. The effect can follow your hand, face, eyes, or cover the
whole frame.

<p align="center">
  <img src="input/3.jpg" alt="Anime effect pose" width="820">
</p>

## Example

<table>
  <tr>
    <th>Original anime frame</th>
    <th>What the tracker sees</th>
  </tr>
  <tr>
    <td><img src="input/3.jpg" alt="Anime visual effect input" width="520"></td>
    <td><img src="output/3_analyzed.jpg" alt="Analyzed anime visual effect" width="520"></td>
  </tr>
</table>

## How To Run

Python 3.10+ should work.

```bash
git clone https://github.com/nhanle-19/Anime_Effect_Cosplaying.git
cd Anime_Effect_Cosplaying

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_models.py
```

To test it on one image:

```bash
python track_image.py input/3.jpg
```

For the live camera version with automatic effect matching:

```bash
python -m pip install -r requirements-vlm.txt
python track_actual.py --vlm-effect-match --effects-dir effects
```

Press `Q` or `Esc` when you want to stop.

## Add Your Own Pose Sequence

Put one clear hand pose in each image and number them in the order you want:

```text
my_sequence/
  01_ready.jpg
  02_focus.png
  03_fireball.png
```

If a pose should trigger an effect, use an image that shows that effect. For
normal poses, use a clean image. The VLM handles the effect matching, so you do
not need to manually say which pose uses which effect.

```bash
python track_actual.py \
  --input-dir my_sequence \
  --vlm-effect-match \
  --effects-dir effects
```

## Add Effects

Make one folder for each animated effect. The PNG frames play in filename
order:

```text
effects/
  fire_circle/
    effect.json
    fire-00001.png
    fire-00002.png
    fire-00003.png
  galaxy_ball/
    effect.json
    galaxy-00001.png
    galaxy-00002.png
```

RGBA PNGs with transparent backgrounds work best. You can add `effect.json` to
say where the effect should appear:

```json
{
  "description": "a fiery magic circle around a hand",
  "attachment": "hand"
}
```

Attachments: `hand`, `eyes`, `face`, or `frame`.

The effect folders are ignored by Git because they can get very large and may
have their own licenses.

## More Commands

```bash
# Test on a video
python track_actual.py --video clip.mp4 --no-mirror \
  --vlm-effect-match --effects-dir effects -o output/demo.mp4

# Make pose matching more forgiving
python track_actual.py --vlm-effect-match --effects-dir effects \
  --pose-tolerance 0.25

# Always use one effect instead of VLM matching
python track_actual.py --effect-dir effects/fire_circle

# Analyze every frame of a video
python track_video.py input/clip.mp4
```

Run `python track_actual.py --help` to see the other options.

## How It Works

- MediaPipe tracks the live hand and face.
- YOLO + RTMPose find hand poses in images and videos.
- CLIP matches the pose images with effects.
- Pose matching still works if the hand moves, rotates, or changes size.

The larger models are not kept in the repository. `download_models.py`
downloads them after cloning.

# Hand Animations

Two webcam toys driven by hand tracking ([MediaPipe](https://github.com/google-ai-edge/mediapipe)):

- **`portal.py`**: a "filter portal" stretched between your two hands. Inside it, the live
  camera image is shown through an effect (thermal, sepia, pixel, …). Everything else stays normal.
- **`air_draw.py`**: pinch your thumb and index finger to draw in the air: freehand, lines,
  rectangles and circles.

## Requirements

- Linux with a webcam (tested on camera index `0`)
- Python 3 (tested on 3.14)
- Internet on first run, to download the hand model (~7.5 MB)

## Install

```bash
cd ~/Desktop/Saturday-Projects/Hand-Animations
python3 -m venv .venv
source .venv/bin/activate
pip install mediapipe==1.0.1
```

`mediapipe` brings OpenCV and numpy with it, so there is nothing else to install.
The hand model (`hand_landmarker.task`) downloads automatically the first time you run either app.

> **Every new terminal:** run `source .venv/bin/activate` first. If you skip it you'll get
> `ModuleNotFoundError: No module named 'cv2'`.

## Filter portal (`portal.py`)

```bash
python3 portal.py
```

Hold up **both hands** facing the camera. The portal opens between your fingertips.

### Gestures (either hand)

| Gesture | What it does |
|---|---|
| **Thumb touches pinky** | Next effect |
| **Thumb touches ring finger** | Next finger set (which fingertips form the portal) |

**Effects**, in order: thermal → purple → gray → invert → sepia → edges → pixel → blur → night →
**off** (portal hidden). One more thumb+pinky touch after "off" brings the portal back.

**Finger sets**, in order: thumb + index (start) → thumb + index + middle → index + middle → back to the start.

### Keys

Click the camera window first. Keys don't reach the app while the terminal has focus.

| Key | What it does |
|---|---|
| `f` | Next effect (same as thumb+pinky) |
| `m` | Next finger set (same as thumb+ring) |
| `q` / `Esc` | Quit |

### On-screen info

```
effect: thermal   (thumb+pinky or f)
fingers: thumb+index   (thumb+ring or m)
ring 1.20  pinky 1.45  (touch < 0.35)   q=quit
```

The last line shows, live, how close your thumb is to your ring and pinky fingers.
A touch counts when the number drops below `0.35`.

### Tuning the touch gesture

If touches don't register, or fire by accident, edit these two lines at the top of `portal.py`:

```python
TOUCH_ON  = 0.35   # raise if touches don't register, lower if they fire by accident
TOUCH_OFF = 0.55   # keep about 0.2 above TOUCH_ON
TOUCH_HOLD_FRAMES = 4   # frames a touch must be held; raise if things still switch by accident,
                        # lower if touches feel slow
```

A touch starts when a finger gets closer than `TOUCH_ON`, and must then stay closer than `TOUCH_OFF`
for `TOUCH_HOLD_FRAMES` frames (about 0.13 s). Whichever finger was closer on average in that time wins.
A thumb passing near a finger, or a one-frame tracking glitch, won't switch anything.

Watch the `ring` / `pinky` numbers on screen while touching to pick values that fit your hand.

### Camera speed

In a dim room, webcam auto-exposure slows the camera down to 15 fps. `portal.py` fixes the exposure
instead, which gives 30 fps (this camera's maximum), so touches and gestures are picked up twice as often.
The fps is shown on screen. Settings at the top of `portal.py`:

```python
FAST_CAMERA = True   # False = camera's own auto settings (15 fps in dim light)
EXPOSURE = 300       # 30 ms; must stay under ~330 for 30 fps. Lower = sharper motion but darker
TARGET_BRIGHTNESS = 120   # dim frames are brightened up to about this (0-255); bright ones are untouched
MAX_BRIGHTEN = 3.0        # limit for very dark rooms
```

Brightness adjusts itself: at night the image is lifted, in daylight it's left as the camera sees it.
If the picture is generally too dark or too bright for your taste, change `TARGET_BRIGHTNESS`.

Camera settings stay on the device after a program closes, so `portal.py` puts auto-exposure back
when you quit. If it ever gets killed instead of quitting with `q`, other apps may see a darker
image until you run `portal.py` and quit it normally once.

### Adding your own effect

Add one line to the `FILTERS` list in `portal.py`. Each effect is a name and a function that takes
the camera frame and returns a same-size image:

```python
("warm", lambda f: cv2.applyColorMap(f, cv2.COLORMAP_AUTUMN)),
```

Put it before the `("off", None)` line so "off" stays last.

## Air drawing (`air_draw.py`)

```bash
python3 air_draw.py
```

**Pinch** your thumb and index finger together to draw, and open them to stop. Both hands can draw at the same time.

| Key | What it does |
|---|---|
| `1` | Freehand |
| `2` | Line |
| `3` | Rectangle |
| `4` | Circle |
| `s` | Save a screenshot (`air_draw_<date>_<time>.png`) |
| `c` | Clear the drawing |
| `[` / `]` | Make pinch easier / harder to trigger |
| `q` / `Esc` | Quit |

With line, rectangle and circle, a **yellow preview** follows your hand while you pinch. When you
let go, the shape is drawn in red.

### Known issues

- With two hands in view, a hand can occasionally get the other hand's stroke, which leaves a stray line.
- If a hand leaves the frame while pinching, it can draw a line back to where it left when it returns.

## Run the tests

No camera needed:

```bash
python3 test_portal.py      # prints: ok
python3 test_air_draw.py    # prints: ok
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'cv2'` | Run `source .venv/bin/activate` first |
| `cannot open camera 0` | Another program is using the camera. Close it (including the other app here) and wait a second |
| Keys do nothing | Click the camera window so it has focus |
| Portal doesn't appear | Both hands must be in view, and the effect must not be "off" |
| Touches don't register | See [Tuning the touch gesture](#tuning-the-touch-gesture) |
| Portal feels slow / fps shows ~15 | Check `FAST_CAMERA = True` in `portal.py`; see [Camera speed](#camera-speed) |
| Camera dark in other apps | Run `portal.py` and quit with `q` once; that restores auto-exposure |
| Model download fails | Download [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) manually into this folder |

## Files

| File | What it is |
|---|---|
| `portal.py` | Filter portal app |
| `air_draw.py` | Air drawing app; also holds the shared helpers (`to_px`, `palm_width`, …) that `portal.py` uses |
| `test_portal.py`, `test_air_draw.py` | Tests, no camera needed |

import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from air_draw import to_px, ensure_model, MODEL_PATH, dist, palm_width

THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP, WRIST = 4, 8, 12, 16, 20, 0

# Fingertips that form each hand's side of the portal, listed from the inside
# of the hand outward. Thumb+ring touch (or m) moves to the next one.
CORNER_SETS = [
    ("thumb+index",        (THUMB_TIP, INDEX_TIP)),
    ("thumb+index+middle", (THUMB_TIP, INDEX_TIP, MIDDLE_TIP)),
    ("index+middle",       (INDEX_TIP, MIDDLE_TIP)),
]

# Thumb-to-fingertip touch, as a fraction of palm width. Two thresholds
# (hysteresis) so one touch = one change. Calibration knobs: watch the HUD.
TOUCH_ON  = 0.35
TOUCH_OFF = 0.55
# A touch counts only after it has been held this many frames (~0.13 s at
# 30 fps). Stops quick pass-bys and tracking glitches from switching anything.
# Raise if you still get accidental switches; lower if touches feel slow.
TOUCH_HOLD_FRAMES = 4
TOUCH_IDLE = (None, 0, 0.0, 0.0)   # (locked finger, frames held, ring sum, pinky sum)

# Camera speed. With auto-exposure in a dim room the camera slows to 15 fps
# (measured: it picks a 66 ms exposure). A fixed exposure keeps it at 30 fps --
# this camera's max -- so touches are seen twice as often. The image comes out
# darker in dim light, so auto_brighten lifts it back up. Linux/V4L2 values.
FAST_CAMERA = True
EXPOSURE = 300      # 30 ms (units of 100 us). Must stay under ~330 for 30 fps.
TARGET_BRIGHTNESS = 120   # 0-255: dim frames are brightened up to about this; brighter ones are left alone
MAX_BRIGHTEN = 3.0        # cap, so a very dark room doesn't turn into pure noise

SEPIA = np.array([[0.131, 0.534, 0.272],   # rows/cols are B, G, R
                  [0.168, 0.686, 0.349],
                  [0.189, 0.769, 0.393]])


def pixelate(f, block=16):
    h, w = f.shape[:2]
    small = cv2.resize(f, (max(1, w // block), max(1, h // block)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


FILTERS = [
    ("thermal",  lambda f: cv2.applyColorMap(f, cv2.COLORMAP_JET)),
    ("purple",   lambda f: cv2.addWeighted(f, 0.5, np.full_like(f, (160, 32, 120)), 0.5, 0)),
    ("gray",     lambda f: cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)),
    ("invert",   cv2.bitwise_not),
    ("sepia",    lambda f: cv2.transform(f, SEPIA)),
    ("edges",    lambda f: cv2.cvtColor(cv2.Canny(f, 80, 160), cv2.COLOR_GRAY2BGR)),
    ("pixel",    pixelate),
    ("blur",     lambda f: cv2.GaussianBlur(f, (31, 31), 0)),
    ("night",    lambda f: cv2.applyColorMap(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_DEEPGREEN)),
    ("off",      None),   # portal hidden; next thumb+pinky brings it back
]


def touch_ratio(px, tip):
    """Thumb-tip to `tip` distance over palm width. Small = touching."""
    w = palm_width(px)
    return dist(px[THUMB_TIP], px[tip]) / w if w >= 1.0 else 999


def auto_brighten(frame, gain):
    """Lift a dark frame toward TARGET_BRIGHTNESS; never darken a bright one.

    `gain` is carried between frames and eased 10% per frame, so lighting
    changes fade in over about a second instead of flickering. Returns
    (frame, gain); start with gain 1.0.
    """
    mean = frame[::8, ::8].mean()                 # a 1/64 sample is plenty for the average
    wanted = min(max(TARGET_BRIGHTNESS / max(mean, 1.0), 1.0), MAX_BRIGHTEN)
    gain = 0.9 * gain + 0.1 * wanted
    if gain > 1.01:
        frame = cv2.convertScaleAbs(frame, alpha=gain)
    return frame, gain


def touch_step(hands_px, state):
    """One frame of the thumb-touch gesture. One deliberate touch = one action.

    `state` is (locked, frames, ring_sum, pinky_sum); start from TOUCH_IDLE.
    - A touch starts when the thumb gets within TOUCH_ON of ring or pinky
      (either hand), then only has to stay within TOUCH_OFF. The pinky is short
      and often sits just above TOUCH_ON even when touching.
    - After TOUCH_HOLD_FRAMES frames it fires once, for whichever finger was
      nearer ON AVERAGE over those frames. Ring and pinky curl together and
      swap places frame to frame, so no single frame decides.
    - It then stays locked until the thumb is back beyond TOUCH_OFF.

    Returns (state, fired): `fired` is RING_TIP / PINKY_TIP on the frame a touch
    is accepted, else None.
    """
    locked, frames, ring_sum, pinky_sum = state
    # (ring, pinky) ratios of the hand whose thumb is nearest either finger
    ring, pinky = min(((touch_ratio(px, RING_TIP), touch_ratio(px, PINKY_TIP)) for px in hands_px),
                      key=min, default=(999, 999))
    closest = min(ring, pinky)

    if locked is not None:
        return (TOUCH_IDLE, None) if closest > TOUCH_OFF else (state, None)
    starting, holding = frames == 0, frames > 0
    if starting and closest >= TOUCH_ON or holding and closest > TOUCH_OFF:
        return TOUCH_IDLE, None
    frames, ring_sum, pinky_sum = frames + 1, ring_sum + ring, pinky_sum + pinky
    if frames >= TOUCH_HOLD_FRAMES:
        finger = RING_TIP if ring_sum < pinky_sum else PINKY_TIP
        return (finger, 0, 0.0, 0.0), finger
    return (None, frames, ring_sum, pinky_sum), None


def portal_points(hands_px, tips=CORNER_SETS[0][1]):
    """Portal corners from two hands, walked in order around the shape.

    The hand further left on screen is 'left' -- no handedness labels needed.
    Path: up the left hand's tips (thumb outward), then back down the right
    hand's tips in reverse, e.g. L thumb -> L index -> R index -> R thumb.
    """
    left, right = sorted(hands_px, key=lambda px: px[WRIST][0])
    return np.array([left[t] for t in tips] + [right[t] for t in reversed(tips)], np.int32)


def apply_portal(frame, pts, filter_fn):
    """Show filter_fn(frame) inside the polygon pts, in place."""
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    inside = mask > 0
    frame[inside] = filter_fn(frame)[inside]
    cv2.polylines(frame, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("cannot open camera 0")
        exit(1)
    if FAST_CAMERA:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)       # 1 = manual
        cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
    ))

    try:
        run_loop(cap, detector)
    finally:
        # Camera settings outlive this program: give other apps auto-exposure back.
        if FAST_CAMERA:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # 3 = auto
        cap.release()
        cv2.destroyAllWindows()


def run_loop(cap, detector):
    filter_idx = 0
    corner_idx = 0
    touch = TOUCH_IDLE
    fps, last = 0.0, time.perf_counter()
    gain = 1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if FAST_CAMERA:
            frame, gain = auto_brighten(frame, gain)
        frame = cv2.flip(frame, 1)
        now = time.perf_counter()
        fps, last = 0.9 * fps + 0.1 / max(now - last, 1e-6), now
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(time.perf_counter() * 1000))

        hands = [to_px(lm, w, h) for lm in result.hand_landmarks]

        # Either hand: thumb+ring -> next finger set, thumb+pinky -> next effect
        touch, fired = touch_step(hands, touch)
        if fired == RING_TIP:
            corner_idx = (corner_idx + 1) % len(CORNER_SETS)
        elif fired == PINKY_TIP:
            filter_idx = (filter_idx + 1) % len(FILTERS)
        ring = min((touch_ratio(px, RING_TIP) for px in hands), default=999)
        pinky = min((touch_ratio(px, PINKY_TIP) for px in hands), default=999)

        filter_name, filter_fn = FILTERS[filter_idx]
        corner_name, tips = CORNER_SETS[corner_idx]
        if len(hands) == 2 and filter_fn is not None:
            apply_portal(frame, portal_points(hands, tips), filter_fn)

        hud = [
            f"effect: {filter_name}   (thumb+pinky or f)",
            f"fingers: {corner_name}   (thumb+ring or m)",
            f"ring {ring:.2f}  pinky {pinky:.2f}  (touch < {TOUCH_ON})   {fps:4.1f} fps   q=quit",
        ]
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Portal de filtros", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('f'):
            filter_idx = (filter_idx + 1) % len(FILTERS)
        if key == ord('m'):
            corner_idx = (corner_idx + 1) % len(CORNER_SETS)


if __name__ == "__main__":
    main()

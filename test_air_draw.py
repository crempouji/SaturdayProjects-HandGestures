from air_draw import (
    dist, palm_width, pinch_ratio, fingers_up, commit_shape, pen_state_transition,
    PINCH_ON, PINCH_OFF, PIPS
)
import math
import numpy as np
import cv2

def make_hand(pinched, n_fingers, scale=1.0, offset=(0, 0)):
    """
    Fabricate a synthetic hand as list[tuple[int, int]], 21 landmarks.

    pinched: bool — thumb and index close together (pinched) or apart
    n_fingers: int — number of extended fingers (1-4)
    scale: float — overall hand size multiplier
    offset: tuple — (x, y) translation
    """
    ox, oy = offset

    # Base hand geometry (normalized, roughly palm-centred)
    base = [
        (0, 0),      # 0: wrist
        (-0.2, -0.3), # 1: thumb carp
        (-0.25, -0.35), # 2: thumb mcp
        (-0.25, -0.4), # 3: thumb pip
        (-0.1, -0.74) if pinched else (-0.2, -0.45),  # 4: thumb tip (close to index when pinched)
        (-0.1, -0.5), # 5: index mcp
        (-0.1, -0.6), # 6: index pip
        (-0.1, -0.65), # 7: index dip
        (-0.1, -0.75) if (n_fingers >= 1) else (-0.1, -0.3),  # 8: index tip
        (0, -0.65),   # 9: middle carp
        (0, -0.7),    # 10: middle pip
        (0, -0.75),   # 11: middle dip
        (0, -0.85) if (n_fingers >= 2) else (0, -0.3),  # 12: middle tip
        (0.1, -0.55), # 13: ring carp
        (0.1, -0.6),  # 14: ring pip
        (0.1, -0.65), # 15: ring dip
        (0.1, -0.75) if (n_fingers >= 3) else (0.1, -0.3),  # 16: ring tip
        (0.2, -0.45), # 17: pinky mcp
        (0.2, -0.5),  # 18: pinky pip
        (0.2, -0.55), # 19: pinky dip
        (0.2, -0.65) if (n_fingers >= 4) else (0.2, -0.3),  # 20: pinky tip
    ]

    # Scale and translate
    px = []
    for x, y in base:
        sx = int(x * scale * 100 + 320 + ox)  # 320 is approx midpoint in 640w
        sy = int(y * scale * 100 + 240 + oy)  # 240 is approx midpoint in 480h
        px.append((sx, sy))

    return px


def test_scale_invariance():
    """A pinched hand reads as pinched at scale 1.0 and 2.5."""
    pinched_1 = make_hand(pinched=True, n_fingers=1, scale=1.0)
    pinched_2 = make_hand(pinched=True, n_fingers=1, scale=2.5)

    ratio_1 = pinch_ratio(pinched_1)
    ratio_2 = pinch_ratio(pinched_2)

    # Both should be pinched (< PINCH_ON)
    assert ratio_1 < PINCH_ON, f"scale 1.0 pinched hand: ratio {ratio_1} >= {PINCH_ON}"
    assert ratio_2 < PINCH_ON, f"scale 2.5 pinched hand: ratio {ratio_2} >= {PINCH_ON}"

    # Open hand should not be pinched
    open_1 = make_hand(pinched=False, n_fingers=4, scale=1.0)
    open_2 = make_hand(pinched=False, n_fingers=4, scale=2.5)

    assert pinch_ratio(open_1) > PINCH_OFF, f"scale 1.0 open hand: ratio {pinch_ratio(open_1)} <= {PINCH_OFF}"
    assert pinch_ratio(open_2) > PINCH_OFF, f"scale 2.5 open hand: ratio {pinch_ratio(open_2)} <= {PINCH_OFF}"


def test_translation_invariance():
    """Verdicts unchanged with translation."""
    hand_a = make_hand(pinched=True, n_fingers=1, scale=1.0, offset=(0, 0))
    hand_b = make_hand(pinched=True, n_fingers=1, scale=1.0, offset=(300, 120))

    ratio_a = pinch_ratio(hand_a)
    ratio_b = pinch_ratio(hand_b)

    # Both should be pinched; ratios should be the same
    assert ratio_a < PINCH_ON
    assert ratio_b < PINCH_ON
    assert abs(ratio_a - ratio_b) < 0.01, f"Ratios differ: {ratio_a} vs {ratio_b}"


def test_finger_counting():
    """sum(fingers_up) == n_fingers for n in 1..4 at two scales."""
    for scale in [1.0, 2.5]:
        for n_fingers in range(1, 5):
            hand = make_hand(pinched=False, n_fingers=n_fingers, scale=scale)
            count = sum(fingers_up(hand))
            assert count == n_fingers, f"scale {scale}, n_fingers {n_fingers}: got {count}"


def test_hysteresis():
    """A ratio between PINCH_ON and PINCH_OFF preserves prior state; edges flip it."""
    mid = (PINCH_ON + PINCH_OFF) / 2
    assert pen_state_transition(mid, False, PINCH_ON, PINCH_OFF) is False, "Midpoint should keep pen up"
    assert pen_state_transition(mid, True, PINCH_ON, PINCH_OFF) is True, "Midpoint should keep pen down"
    assert pen_state_transition(PINCH_ON - 0.01, False, PINCH_ON, PINCH_OFF) is True, "Below ON should press"
    assert pen_state_transition(PINCH_OFF + 0.01, True, PINCH_ON, PINCH_OFF) is False, "Above OFF should release"


def test_shape_geometry():
    """Lines, rects, circles have correct geometry."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Line: anchor and cur should both be marked
    commit_shape(canvas, 2, (100, 100), (200, 200), (0, 0, 255), 1)
    nz = np.nonzero(canvas)
    assert (100, 100) in zip(nz[1], nz[0]), "Line start not marked"
    assert (200, 200) in zip(nz[1], nz[0]), "Line end not marked"

    # Rect: bounding box check
    canvas[:] = 0
    commit_shape(canvas, 3, (100, 100), (300, 200), (0, 0, 255), 1)
    nz = np.nonzero(canvas)
    assert len(nz[0]) > 0, "Rect: nothing drawn"
    y_min, y_max = nz[0].min(), nz[0].max()
    x_min, x_max = nz[1].min(), nz[1].max()
    assert abs(x_min - 100) <= 1, f"Rect x_min: {x_min}, expected 100"
    assert abs(x_max - 300) <= 1, f"Rect x_max: {x_max}, expected 300"
    assert abs(y_min - 100) <= 1, f"Rect y_min: {y_min}, expected 100"
    assert abs(y_max - 200) <= 1, f"Rect y_max: {y_max}, expected 200"

    # Circle: radius check
    canvas[:] = 0
    commit_shape(canvas, 4, (320, 240), (320, 140), (0, 0, 255), 1)
    nz = np.nonzero(canvas)
    assert len(nz[0]) > 0, "Circle: nothing drawn"
    y_min, y_max = nz[0].min(), nz[0].max()
    x_min, x_max = nz[1].min(), nz[1].max()
    # radius = 100, so bbox should be (220, 140) to (420, 340), within 2px
    assert abs(x_min - 220) <= 2, f"Circle x_min: {x_min}, expected 220"
    assert abs(x_max - 420) <= 2, f"Circle x_max: {x_max}, expected 420"
    assert abs(y_min - 140) <= 2, f"Circle y_min: {y_min}, expected 140"
    assert abs(y_max - 340) <= 2, f"Circle y_max: {y_max}, expected 340"

if __name__ == "__main__":
    try:
        test_scale_invariance()
        test_translation_invariance()
        test_finger_counting()
        test_hysteresis()
        test_shape_geometry()
        print("ok")
        exit(0)
    except AssertionError as e:
        print(f"FAIL: {e}")
        exit(1)

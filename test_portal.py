import numpy as np
from portal import (CORNER_SETS, FILTERS, TOUCH_ON, TOUCH_OFF,
                    portal_points, apply_portal, touch_ratio, touch_step, RING_TIP, PINKY_TIP,
                    TOUCH_IDLE, TOUCH_HOLD_FRAMES, auto_brighten, TARGET_BRIGHTNESS)

SETS = dict(CORNER_SETS)


def hand(wrist_x, thumb, index, middle=(0, 0)):
    px = [(0, 0)] * 21
    px[0], px[4], px[8], px[12] = (wrist_x, 400), thumb, index, middle
    return px


def touch_hand(thumb, ring, pinky, scale=1):
    """Palm width 100*scale (landmarks 5 and 17) with given thumb/ring/pinky tips."""
    px = [(0, 0)] * 21
    px[5], px[17] = (0, 0), (100 * scale, 0)
    px[4], px[16], px[20] = thumb, ring, pinky
    return px


if __name__ == "__main__":
    left = hand(100, thumb=(150, 300), index=(160, 200), middle=(120, 150))
    right = hand(500, thumb=(450, 300), index=(440, 200), middle=(480, 150))
    as_list = lambda pts: [list(p) for p in pts]

    # Starts with thumb+index; the three sets cycle in the order asked for.
    assert [name for name, _ in CORNER_SETS] == ["thumb+index", "thumb+index+middle", "index+middle"]

    # Corner order: up the left hand, back down the right -- whichever hand
    # MediaPipe detected first.
    expected = {
        "thumb+index":        [(150, 300), (160, 200), (440, 200), (450, 300)],
        "thumb+index+middle": [(150, 300), (160, 200), (120, 150), (480, 150), (440, 200), (450, 300)],
        "index+middle":       [(160, 200), (120, 150), (480, 150), (440, 200)],
    }
    for name, pts in expected.items():
        for pair in ([left, right], [right, left]):
            assert portal_points(pair, SETS[name]).tolist() == as_list(pts), name

    # Filter applies inside the polygon only.
    frame = np.full((480, 640, 3), 128, np.uint8)
    apply_portal(frame, portal_points([left, right], SETS["thumb+index"]), FILTERS[0][1])
    assert (frame[250, 300] != 128).any(), "inside should be filtered"
    assert (frame[450, 300] == 128).all(), "outside should be untouched"

    # Every real effect returns a same-shape uint8 image; "off" is the one None,
    # and it is last, so the app always starts with the portal visible.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    for name, fn in FILTERS[:-1]:
        out = fn(img)
        assert out.shape == img.shape and out.dtype == np.uint8, name
    assert FILTERS[-1] == ("off", None)

    open_hand = touch_hand(thumb=(0, 100), ring=(60, -150), pinky=(90, -120))

    # Touch ratio is scale-invariant.
    for scale in (1, 3):
        s = scale
        px = touch_hand(thumb=(90 * s, 60 * s), ring=(0, 0), pinky=(80 * s, 50 * s), scale=s)
        assert touch_ratio(px, 20) < TOUCH_ON, scale
        px[4] = (0, 100 * s)
        assert touch_ratio(px, 20) > TOUCH_OFF, scale

    # Touch gesture. A touch starts when the thumb gets within TOUCH_ON, must stay
    # within TOUCH_OFF for TOUCH_HOLD_FRAMES frames, then fires once for whichever
    # finger was nearer on average, and is locked until the thumb moves away.
    def two_tips(ring_r, pinky_r):
        return [touch_hand(thumb=(300, 300), ring=(300 + round(ring_r * 100), 300),
                           pinky=(300, 300 + round(pinky_r * 100)))]

    def fires(frames):
        state, log = TOUCH_IDLE, []
        for ring_r, pinky_r in frames:
            state, fired = touch_step(two_tips(ring_r, pinky_r), state)
            if fired is not None:
                log.append(fired)
        return log

    apart = (0.9, 0.9)
    hold = TOUCH_HOLD_FRAMES
    ring_near = (0.20, 0.24)
    pinky_near = (0.24, 0.20)

    # A held ring touch fires exactly once, on the hold-th frame -- even if the
    # fingers swap mid-touch or hover between the thresholds.
    assert fires([apart] + [ring_near] * hold) == [RING_TIP]
    assert fires([apart] + [ring_near] * (hold - 1)) == [], "not held long enough"
    assert fires([apart] + [ring_near] * hold + [pinky_near] * 10 + [(0.45, 0.44)] + [apart]) == [RING_TIP]

    # Passing the thumb near the ring finger for a moment does not switch fingers.
    assert fires([apart, (0.30, 1.2), (0.30, 1.2), apart]) == []

    # Ring and pinky taking turns being nearest still makes ONE touch, for the
    # finger that was nearer on average (real tracking jitter does this).
    mostly_pinky = [ring_near, pinky_near, pinky_near, (0.21, 0.20)]
    assert fires([apart] + mostly_pinky * 3 + [apart]) == [PINKY_TIP]

    # A pinky that dips under TOUCH_ON once, then sits just above it (short
    # finger, can't press flat against the thumb) still counts -- this is the
    # case that failed before: it only has to stay under TOUCH_OFF.
    assert fires([apart, (1.2, 0.34)] + [(1.2, 0.40), (1.2, 0.37), (1.2, 0.42)] * 2 + [apart]) == [PINKY_TIP]

    # Never getting under TOUCH_ON is not a touch, however long it lasts.
    assert fires([apart] + [(1.2, 0.40)] * 20 + [apart]) == []

    # Release, then a new touch fires again.
    assert fires([apart] + [ring_near] * hold + [apart] + [pinky_near] * hold + [apart]) == [RING_TIP, PINKY_TIP]

    # Either hand can make the touch; no hands means no touch.
    other = touch_hand(thumb=(82, 52), ring=(60, 50), pinky=(80, 50))
    state = TOUCH_IDLE
    for _ in range(hold):
        state, fired = touch_step([open_hand, other], state)
    assert fired == PINKY_TIP
    assert touch_step([], TOUCH_IDLE) == (TOUCH_IDLE, None)

    # Auto-brighten: a dark frame is lifted to about TARGET_BRIGHTNESS, a bright
    # one is left alone (never darkened, never blown out further).
    def settle(level):
        frame, gain = np.full((480, 640, 3), level, np.uint8), 1.0
        for _ in range(60):                       # ~2 s at 30 fps
            out, gain = auto_brighten(frame, gain)
        return out.mean()
    assert abs(settle(60) - TARGET_BRIGHTNESS) < 6, settle(60)
    assert settle(160) == 160, "bright frame must be untouched"
    assert settle(0) == 0, "black frame: no division by zero"

    print("ok")

import math
import time
import urllib.request
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

PINCH_ON   = 0.32
PINCH_OFF  = 0.42
SMOOTH     = 0.4
TOOL_HOLD_FRAMES = 5
PIPS       = [(8, 6), (12, 10), (16, 14), (20, 18)]  # index, middle, ring, pinky
TOOLS      = {1: "freehand", 2: "line", 3: "rect", 4: "circle"}
PEN_LABEL  = "Right"
SWAP_HANDS = False


def dist(a, b):
    """Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def to_px(hand_landmarks, w, h):
    """Convert normalized MediaPipe landmarks to pixel-space tuples."""
    px = []
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        px.append((x, y))
    return px


def palm_width(px):
    """Distance from index-MCP (5) to pinky-MCP (17)."""
    if len(px) < 18:
        return 0
    return dist(px[5], px[17])


def pinch_ratio(px):
    """Ratio of thumb-to-index distance over palm width."""
    w = palm_width(px)
    if w < 1.0:
        return 999  # Return a large value (unpinched) if palm is degenerate
    return dist(px[4], px[8]) / w


def fingers_up(px):
    """Return [index_up, middle_up, ring_up, pinky_up] based on distance from wrist."""
    wrist = px[0]
    fingers = []
    for tip_idx, pip_idx in PIPS:
        tip_dist = dist(px[tip_idx], wrist)
        pip_dist = dist(px[pip_idx], wrist)
        fingers.append(tip_dist > pip_dist)
    return fingers


def tool_from_count(count, current_tool):
    """Map finger count to tool ID. Return current_tool if count not in TOOLS."""
    return TOOLS.get(count, current_tool)


def commit_shape(img, tool, anchor, cur, color, thickness):
    """Draw a shape (tool 2/3/4) into img in-place. tool 1 (freehand) is not a shape."""
    if tool == 1:
        raise ValueError("commit_shape does not handle freehand (tool 1)")

    if tool == 2:  # line
        cv2.line(img, anchor, cur, color, thickness, cv2.LINE_AA)

    if tool == 3:  # rect
        x1, y1 = anchor
        x2, y2 = cur
        top_left = (min(x1, x2), min(y1, y2))
        bottom_right = (max(x1, x2), max(y1, y2))
        cv2.rectangle(img, top_left, bottom_right, color, thickness, cv2.LINE_AA)

    if tool == 4:  # circle
        radius = int(dist(anchor, cur))
        cv2.circle(img, anchor, radius, color, thickness, cv2.LINE_AA)


def ensure_model() -> None:
    """Download model if not present. Exit 1 on failure."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand_landmarker model...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"Download complete: {MODEL_PATH}")
        except Exception as e:
            print(f"Download failed. Manual download: {MODEL_URL}")
            print(f"Save to: {MODEL_PATH}")
            exit(1)


def main():
    """Main frame loop."""
    ensure_model()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("cannot open camera 0")
        exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # MediaPipe setup
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # State
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    pen_down = False
    prev_point = None
    anchor = None
    tool = 1  # Start with freehand
    tool_count = 0
    tool_hold = 0
    pinch_on_threshold = PINCH_ON
    pinch_off_threshold = PINCH_OFF
    ts = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror and convert to RGB
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms = int(time.perf_counter() * 1000)
        result = detector.detect_for_video(mp_image, ts_ms)

        # Extract hands with handedness
        hands = []
        if result.hand_landmarks and result.handedness:
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                px = to_px(landmarks, w, h)
                label = handedness[0].category_name
                hands.append({"px": px, "label": label})

        # Assign hands: one as pen, one as palette
        pen_hand = None
        palette_hand = None

        if len(hands) == 0:
            pen_down = False
        elif len(hands) == 1:
            pen_hand = hands[0]
        else:  # 2 hands
            target_label = PEN_LABEL
            if SWAP_HANDS:
                target_label = "Left" if PEN_LABEL == "Right" else "Right"

            for h in hands:
                if h["label"] == target_label:
                    pen_hand = h
                else:
                    palette_hand = h

        # Display
        cv2.imshow("Air Draw", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

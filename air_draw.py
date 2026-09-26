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


def pen_state_transition(ratio, pen_down, pinch_on, pinch_off):
    """Return new pen state (True/False) based on hysteresis."""
    if ratio < pinch_on and not pen_down:
        return True
    elif ratio > pinch_off and pen_down:
        return False
    return pen_down


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
    return count if count in TOOLS else current_tool


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
    canvas = None  # sized from the first frame
    tool = 1  # Start with freehand
    pinch_on_threshold = PINCH_ON
    pinch_off_threshold = PINCH_OFF

    # Per-hand state (track up to 2 hands)
    hand_states = {}  # {hand_id: {"pen_down": bool, "prev_point": tuple, "anchor": tuple}}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror and convert to RGB
        frame = cv2.flip(frame, 1)
        if canvas is None:
            canvas = np.zeros_like(frame)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms = int(time.perf_counter() * 1000)
        result = detector.detect_for_video(mp_image, ts_ms)

        # Extract hands with handedness
        hands = []
        if result.hand_landmarks and result.handedness:
            for idx, (landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                px = to_px(landmarks, w, h)
                label = handedness[0].category_name
                hands.append({"px": px, "label": label, "id": idx})

        # Process each hand independently
        for hand in hands:
            hand_id = hand["id"]
            px = hand["px"]

            if palm_width(px) < 1.0:
                continue  # Hand too small, skip

            # Initialize hand state if not exists
            if hand_id not in hand_states:
                hand_states[hand_id] = {"pen_down": False, "prev_point": None, "anchor": None}

            state = hand_states[hand_id]
            ratio = pinch_ratio(px)

            # Hysteresis state machine
            was_down = state["pen_down"]
            state["pen_down"] = pen_state_transition(ratio, state["pen_down"], pinch_on_threshold, pinch_off_threshold)

            if state["pen_down"] and not was_down:
                # Down-edge: record anchor, don't draw (no stray line from prior stroke)
                state["anchor"] = state["prev_point"] = px[8]
            elif was_down and not state["pen_down"]:
                # Release: commit at the last held point
                if tool != 1 and state["anchor"] is not None:
                    commit_shape(canvas, tool, state["anchor"], state["prev_point"], (0, 0, 255), 4)
            elif state["pen_down"]:
                # Held: draw freehand or prepare shape
                current = (int(SMOOTH * px[8][0] + (1 - SMOOTH) * state["prev_point"][0]),
                           int(SMOOTH * px[8][1] + (1 - SMOOTH) * state["prev_point"][1]))
                if tool == 1:  # freehand
                    cv2.line(canvas, state["prev_point"], current, (0, 0, 255), 4, cv2.LINE_AA)
                state["prev_point"] = current

        # Composite: mask-based to preserve colours
        mask = canvas.any(axis=2)
        frame[mask] = canvas[mask]

        # Draw preview shapes for all hands
        for hand in hands:
            hand_id = hand["id"]
            if hand_id in hand_states:
                state = hand_states[hand_id]
                if state["pen_down"] and tool != 1 and state["anchor"] is not None:
                    commit_shape(frame, tool, state["anchor"], state["prev_point"], (0, 255, 255), 4)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is ESC
            break
        elif key == ord('1'):
            tool = 1
            print("Tool: Freehand")
        elif key == ord('2'):
            tool = 2
            print("Tool: Line")
        elif key == ord('3'):
            tool = 3
            print("Tool: Rectangle")
        elif key == ord('4'):
            tool = 4
            print("Tool: Circle")
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"air_draw_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        elif key == ord('c'):
            canvas[:] = 0
            print("Canvas cleared")
        elif key == ord('['):
            pinch_on_threshold = max(0.1, pinch_on_threshold - 0.02)
            pinch_off_threshold = max(0.1, pinch_off_threshold - 0.02)
            print(f"Pinch sensitivity: {pinch_on_threshold:.2f}")
        elif key == ord(']'):
            pinch_on_threshold = min(0.9, pinch_on_threshold + 0.02)
            pinch_off_threshold = min(0.9, pinch_off_threshold + 0.02)
            print(f"Pinch sensitivity: {pinch_on_threshold:.2f}")

        # HUD: top-left, small text
        hud_lines = [
            f"Tool: {TOOLS.get(tool, '?')} (Press 1/2/3/4 to change)",
            f"Pinch threshold: {pinch_on_threshold:.2f} / {pinch_off_threshold:.2f}",
            f"Hands detected: {len(hands)}",
            f"Keys: 1-4=tool, s=save, c=clear, [/]=calibrate, q=quit",
        ]

        y = 20
        for line in hud_lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y += 15

        # Display
        cv2.imshow("Air Draw", frame)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

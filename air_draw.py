import math
import time
import urllib.request
import cv2
import numpy as np
import os

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


# Quick test (comment out when submitting)
if __name__ == "__main__":
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    test_px = [(100, 100)] + [(0, 0)] * 20  # Dummy hand
    test_px[5] = (100, 50)   # index-mcp
    test_px[17] = (200, 50)  # pinky-mcp
    test_px[4] = (110, 90)   # thumb-tip
    test_px[8] = (100, 80)   # index-tip

    print(f"dist((0,0), (3,4)): {dist((0, 0), (3, 4))} (expect ~5.0)")
    print(f"palm_width: {palm_width(test_px)} (expect ~100.0)")
    print(f"pinch_ratio: {pinch_ratio(test_px)} (expect ~0.1-0.2)")
    print(f"fingers_up: {fingers_up(test_px)}")
    print(f"tool_from_count(2, 1): {tool_from_count(2, 1)} (expect 2)")

    landmarks = [MockLandmark(0.5, 0.5)] * 21
    test_px_converted = to_px(landmarks, 640, 480)
    assert test_px_converted[0] == (320, 240), f"Expected (320, 240), got {test_px_converted[0]}"
    print(f"to_px test: {test_px_converted[0]} == (320, 240) ✓")

    # Test commit_shape
    print("\n--- Testing commit_shape ---")
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    commit_shape(canvas, 2, (100, 100), (200, 200), (0, 0, 255), 2)
    print(f"Line drawn: canvas has pixels at (100,100) = {canvas[100, 100]} (should be non-zero)")
    assert canvas.any(), "Line should have drawn pixels"

    canvas[:] = 0
    commit_shape(canvas, 3, (100, 100), (300, 200), (0, 0, 255), 2)
    print(f"Rect drawn: canvas has pixels at (150, 150) = {canvas[150, 150]} (should be non-zero)")
    assert canvas.any(), "Rectangle should have drawn pixels"

    canvas[:] = 0
    commit_shape(canvas, 4, (320, 240), (320, 140), (0, 0, 255), 2)
    print(f"Circle drawn: canvas has pixels = {canvas.any()} (should be True)")
    assert canvas.any(), "Circle should have drawn pixels"

    try:
        commit_shape(canvas, 1, (0, 0), (10, 10), (0, 0, 255), 2)
        print("ERROR: should have raised ValueError")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")

    print("\n--- Testing ensure_model ---")
    ensure_model()
    print("✓ ensure_model() completed")

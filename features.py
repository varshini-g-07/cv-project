"""Shared helpers: hand-keypoint -> geometric features (paper-style).

Features per hand (39 numbers):
  - 20 segment lengths   (wrist->base and each finger bone), divided by palm width
  - 15 joint angles      (3 per finger, radians)
  - 4 adjacent-fingertip distances, divided by palm width
"""
import numpy as np
import cv2

# MediaPipe landmark ids for each finger (wrist is id 0)
FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
TIPS = [4, 8, 12, 16, 20]

# Hand skeleton as (start, end) index pairs for cv2 drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# The frame is mirrored (selfie view) before MediaPipe sees it.
GESTURE_HAND = "Right"   # shows the number gesture 1-7
TRIGGER_HAND = "Left"    # pinch to play the chord


def landmarks_to_array(hand_landmarks, w, h):
    """NormalizedLandmark list -> (21, 3) numpy array."""
    return np.array(
        [[lm.x * w, lm.y * h, lm.z * w] for lm in hand_landmarks],
        dtype=np.float64,
    )


def pick_hand(result, wanted_label):
    """Return NormalizedLandmark list for 'Left' or 'Right' hand, or None."""
    if not result or not result.hand_landmarks:
        return None
    for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
        if handedness_list[0].category_name == wanted_label:
            return lm_list
    return None


def draw_hand(frame, lm_list, w, h, color=(0, 255, 0)):
    """Draw hand skeleton using plain cv2 lines (no mp.solutions needed)."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1)


def palm_width(pts):
    return float(np.linalg.norm(pts[5] - pts[17])) + 1e-9


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    return float(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))


def extract_features(pts):
    pw = palm_width(pts)
    feats = []
    for ids in FINGERS.values():
        chain = [0] + ids
        for a, b in zip(chain[:-1], chain[1:]):
            feats.append(np.linalg.norm(pts[a] - pts[b]) / pw)
        for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:]):
            feats.append(_angle(pts[a], pts[b], pts[c]))
    for a, b in zip(TIPS[:-1], TIPS[1:]):
        feats.append(np.linalg.norm(pts[a] - pts[b]) / pw)
    return np.array(feats, dtype=np.float32)


def pinch_ratio(pts):
    """Thumb-tip to index-tip distance / palm width (small = pinching)."""
    return float(np.linalg.norm(pts[4] - pts[8]) / palm_width(pts))

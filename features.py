"""Shared helpers: hand-keypoint -> geometric features (paper-style).

Features per hand (39 numbers):
  - 20 segment lengths   (wrist->base and each finger bone), divided by palm width
  - 15 joint angles      (3 per finger, radians)
  - 4 adjacent-fingertip distances, divided by palm width
"""
import numpy as np

# MediaPipe landmark ids for each finger (wrist is id 0)
FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
TIPS = [4, 8, 12, 16, 20]

# The frame is mirrored (selfie view) before MediaPipe sees it, so labels match
# the user's real hands. If left/right come out swapped on your setup, swap these.
GESTURE_HAND = "Left"    # shows the number gesture 1-7
TRIGGER_HAND = "Right"   # pinch to play the chord


def landmarks_to_array(hand_landmarks, w, h):
    """MediaPipe landmarks -> (21, 3) array. x/y are scaled by image size so
    distances are not distorted by the frame's aspect ratio."""
    return np.array(
        [[lm.x * w, lm.y * h, lm.z * w] for lm in hand_landmarks.landmark],
        dtype=np.float64,
    )


def pick_hand(results, wanted_label):
    """Return the landmarks of the hand with the given label, or None."""
    if not results.multi_hand_landmarks:
        return None
    for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
        if hd.classification[0].label == wanted_label:
            return lm
    return None


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
        for a, b in zip(chain[:-1], chain[1:]):            # 4 lengths
            feats.append(np.linalg.norm(pts[a] - pts[b]) / pw)
        for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:]):  # 3 angles
            feats.append(_angle(pts[a], pts[b], pts[c]))
    for a, b in zip(TIPS[:-1], TIPS[1:]):                  # 4 tip distances
        feats.append(np.linalg.norm(pts[a] - pts[b]) / pw)
    return np.array(feats, dtype=np.float32)


def pinch_ratio(pts):
    """Thumb-tip to index-tip distance relative to palm width (small = pinch)."""
    return float(np.linalg.norm(pts[4] - pts[8]) / palm_width(pts))

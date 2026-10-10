"""Webcam -> hand keypoints -> SVM gesture (left hand) -> pinch (right hand) plays chord.
Run after train_svm.py. Press q to quit."""
import collections, time
import cv2
import joblib
import mediapipe as mp
from audio import CHORDS, ChordPlayer
from features import (GESTURE_HAND, TRIGGER_HAND, extract_features,
                      landmarks_to_array, pick_hand, pinch_ratio)

PINCH_ON, PINCH_OFF = 0.30, 0.45   # hysteresis; tune for your camera


def main():
    model = joblib.load("models/svm.joblib")
    player = ChordPlayer()
    hands = mp.solutions.hands.Hands(
        max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)
    drawer = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    votes = collections.deque(maxlen=5)   # light temporal smoothing
    current, pinched, fps, t_prev = None, False, 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                drawer.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)

        left = pick_hand(res, GESTURE_HAND)
        if left is not None:
            feats = extract_features(landmarks_to_array(left, w, h))
            votes.append(int(model.predict(feats[None, :])[0]))
            current = collections.Counter(votes).most_common(1)[0][0]
        else:
            votes.clear(); current = None

        right = pick_hand(res, TRIGGER_HAND)
        if right is not None:
            ratio = pinch_ratio(landmarks_to_array(right, w, h))
            if not pinched and ratio < PINCH_ON:
                pinched = True
                if current is not None:
                    player.play(current)
            elif pinched and ratio > PINCH_OFF:
                pinched = False
        else:
            pinched = False

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        text = f"gesture: {current}  chord: {CHORDS[current][0] if current else '-'}  fps: {fps:.0f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("AIR gesture instrument", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

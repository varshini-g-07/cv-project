"""Record gesture samples from the webcam.

Usage:  python collect_data.py --person alice
Keys:   1-7 choose the gesture, SPACE start/stop recording, q quit.
Make your gesture with the LEFT hand (see GESTURE_HAND in features.py).
Vary distance, tilt, lighting and background between recordings.
Each person writes their own data/<person>.csv (used for leave-one-person-out).
"""
import argparse, csv, os
import cv2
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
from features import GESTURE_HAND, extract_features, landmarks_to_array, pick_hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    hands = mp_hands.Hands(
        max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)
    drawer = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    label, recording = 1, False
    counts = {k: 0 for k in range(1, 8)}

    with open(f"data/{args.person}.csv", "a", newline="") as fh:
        writer = csv.writer(fh)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hand = pick_hand(res, GESTURE_HAND)
            if hand is not None:
                drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                if recording:
                    feats = extract_features(landmarks_to_array(hand, w, h))
                    writer.writerow([label, *feats])
                    counts[label] += 1
            status = "REC" if recording else "paused"
            cv2.putText(frame, f"gesture {label} [{status}]  samples: {counts[label]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("collect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording
            if ord("1") <= key <= ord("7"):
                label = key - ord("0")
    cap.release()
    cv2.destroyAllWindows()
    print("Samples this session:", counts)


if __name__ == "__main__":
    main()
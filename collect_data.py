"""Record gesture samples from the webcam.

Usage:  python collect_data.py --person alice
Keys:   1-7 choose gesture, SPACE start/stop recording, q quit.
Either hand can be used for training (features are rotation & scale invariant).
Vary distance, tilt, lighting, and background while recording.
Data is saved to data/<person>.csv.
"""
import argparse, csv, os
import cv2
import mediapipe as mp
from features import GESTURE_HAND, extract_features, landmarks_to_array, pick_hand, draw_hand

BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
RunningMode        = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    csv_file = f"data/{args.person}.csv"

    # Preload existing counts if file exists
    counts = {k: 0 for k in range(1, 8)}
    if os.path.exists(csv_file):
        try:
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].isdigit():
                        lbl = int(row[0])
                        if lbl in counts:
                            counts[lbl] += 1
        except Exception:
            pass

    options = HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    cap = cv2.VideoCapture(0)
    label = 1
    recording = False

    with HandLandmarker.create_from_options(options) as detector:
        with open(csv_file, "a", newline="") as fh:
            writer = csv.writer(fh)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = detector.detect(mp_image)

                hand = None
                hand_type = None

                if result and result.hand_landmarks:
                    # Draw skeleton on all visible hands
                    for lm_list in result.hand_landmarks:
                        draw_hand(frame, lm_list, w, h)

                    # Prioritize GESTURE_HAND if present; otherwise use whichever hand is detected
                    hand = pick_hand(result, GESTURE_HAND)
                    if hand is not None:
                        hand_type = GESTURE_HAND
                    elif len(result.hand_landmarks) > 0:
                        hand = result.hand_landmarks[0]
                        try:
                            hand_type = result.handedness[0][0].category_name
                        except Exception:
                            hand_type = "Hand"

                # Record samples when active and a hand is detected
                if hand is not None and recording:
                    feats = extract_features(landmarks_to_array(hand, w, h))
                    writer.writerow([label, *feats])
                    fh.flush()
                    counts[label] += 1

                # On-screen UI overlay
                if recording:
                    status_text = f"● RECORDING GESTURE {label}"
                    status_color = (0, 0, 255)  # Red
                    hint = "SPACE = PAUSE"
                else:
                    status_text = f"PAUSED - Gesture {label}"
                    status_color = (0, 215, 255)  # Yellow
                    hint = "SPACE = START RECORDING"

                # Draw semi-transparent header bar for readability
                cv2.rectangle(frame, (10, 10), (460, 115), (20, 20, 20), -1)
                cv2.putText(frame, status_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)
                cv2.putText(frame, f"Samples for gesture {label}: {counts[label]}", (20, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if hand is not None:
                    hand_info = f"Hand: DETECTED ({hand_type})"
                    hand_color = (0, 255, 0)
                else:
                    hand_info = "Hand: NOT DETECTED (Place hand in camera)"
                    hand_color = (0, 140, 255)
                cv2.putText(frame, hand_info, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 1)

                # Footer bar
                cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)
                cv2.putText(frame, f"{hint} | Keys 1-7: Gesture | Q: Quit",
                            (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

                cv2.imshow("Gesture Data Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    recording = not recording
                if ord("1") <= key <= ord("7"):
                    label = key - ord("0")
                    recording = False

    cap.release()
    cv2.destroyAllWindows()
    print("\nRecorded samples in total for this person:")
    for k in range(1, 8):
        print(f"  Gesture {k}: {counts[k]} samples")


if __name__ == "__main__":
    main()
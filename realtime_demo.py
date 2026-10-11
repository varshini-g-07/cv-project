"""Webcam -> right hand (chords) -> left hand pinch (plays chord).
Run after train_svm.py. Press q to quit, s to swap hands."""
import collections, os, time
import cv2
import joblib
import mediapipe as mp
from audio import CHORDS, ChordPlayer
from features import (extract_features, landmarks_to_array, pinch_ratio, draw_hand)

BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
RunningMode        = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
PINCH_ON, PINCH_OFF = 0.32, 0.40
MODEL_FILE = "models/gesture_model.joblib"
STRUM_INTERVAL = 0.50  # seconds between auto-strums while holding pinch


def main():
    if not os.path.exists(MODEL_FILE):
        raise SystemExit(f"{MODEL_FILE} not found. Run python train_svm.py first.")
    model  = joblib.load(MODEL_FILE)
    player = ChordPlayer()

    options = HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    cap = cv2.VideoCapture(0)
    votes = collections.deque(maxlen=3)
    current, pinched = None, False
    last_played_chord = None
    last_strum_time = 0.0
    swapped = False

    try:
        with HandLandmarker.create_from_options(options) as detector:
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

                for lm_list in result.hand_landmarks:
                    draw_hand(frame, lm_list, w, h)

                # Physical hands on mirrored screen:
                # Screen-Left (x smaller) = Left hand (PINCH TRIGGER)
                # Screen-Right (x larger) = Right hand (CHORD GESTURE)
                g_hand = None
                t_hand = None

                if result and result.hand_landmarks:
                    if len(result.hand_landmarks) >= 2:
                        h0, h1 = result.hand_landmarks[0], result.hand_landmarks[1]
                        left_side = h0 if h0[0].x <= h1[0].x else h1
                        right_side = h1 if h0[0].x <= h1[0].x else h0

                        if not swapped:
                            t_hand, g_hand = left_side, right_side   # Left = pinch, Right = chord
                        else:
                            g_hand, t_hand = left_side, right_side
                    else:
                        single = result.hand_landmarks[0]
                        # For single-hand testing, detect gesture from the hand
                        g_hand = single
                        if single[0].x < 0.5:
                            t_hand = single

                # Gesture classification from right hand
                if g_hand is not None:
                    feats = extract_features(landmarks_to_array(g_hand, w, h))
                    votes.append(int(model.predict(feats[None, :])[0]))
                    current = collections.Counter(votes).most_common(1)[0][0]
                else:
                    votes.clear()
                    current = None

                # Pinch trigger from left hand
                now = time.time()
                if t_hand is not None:
                    ratio = pinch_ratio(landmarks_to_array(t_hand, w, h))
                    if ratio < PINCH_ON:
                        if not pinched:
                            pinched = True
                            if current is not None:
                                player.play(current)
                                last_played_chord = current
                                last_strum_time = now
                        else:
                            # Immediate chord switch while holding pinch
                            if current is not None and current != last_played_chord:
                                player.play(current)
                                last_played_chord = current
                                last_strum_time = now
                            # Continuous strumming while holding pinch
                            elif current is not None and (now - last_strum_time) >= STRUM_INTERVAL:
                                player.play(current)
                                last_strum_time = now
                    elif ratio > PINCH_OFF:
                        pinched = False
                        last_played_chord = None
                else:
                    pinched = False
                    last_played_chord = None

                # Only show chord name on the top left
                chord_name = CHORDS[current][0] if current else "-"
                cv2.putText(frame, f"chord: {chord_name}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

                cv2.imshow("AIR gesture instrument", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):  # Press 's' to swap anytime if desired
                    swapped = not swapped
    finally:
        player.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

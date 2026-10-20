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


def draw_badge(img, text, pt, bg_color, text_color=(255, 255, 255), scale=0.55, thickness=2):
    """Draw a readable badge with a background box on img."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 5
    x, y = pt
    x1 = max(0, x - pad)
    y1 = max(0, y - th - pad)
    x2 = min(img.shape[1] - 1, x + tw + pad)
    y2 = min(img.shape[0] - 1, y + baseline + pad)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)


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

                # Physical hands on mirrored screen:
                # Screen-Left (x smaller) = Left hand (PINCH TRIGGER / STRUM)
                # Screen-Right (x larger) = Right hand (CHORD GESTURE)
                g_hand = None
                t_hand = None

                if result and result.hand_landmarks:
                    if len(result.hand_landmarks) >= 2:
                        h0, h1 = result.hand_landmarks[0], result.hand_landmarks[1]
                        x0 = sum(lm.x for lm in h0) / len(h0)
                        x1 = sum(lm.x for lm in h1) / len(h1)
                        left_side = h0 if x0 <= x1 else h1
                        right_side = h1 if x0 <= x1 else h0

                        if not swapped:
                            t_hand, g_hand = left_side, right_side   # Left = strum, Right = chord
                        else:
                            g_hand, t_hand = left_side, right_side   # Swapped roles
                    else:
                        single = result.hand_landmarks[0]
                        hand_x = sum(lm.x for lm in single) / len(single)
                        is_left = hand_x < 0.5

                        if not swapped:
                            if is_left:
                                t_hand = single   # Left hand alone = Strum only (no chord classification)
                                g_hand = None
                            else:
                                g_hand = single   # Right hand alone = Chord only
                                t_hand = None
                        else:
                            if is_left:
                                g_hand = single   # Swapped: Left hand alone = Chord only
                                t_hand = None
                            else:
                                t_hand = single   # Swapped: Right hand alone = Strum only
                                g_hand = None

                # Gesture classification: ONLY run on the chord hand (g_hand)
                if g_hand is not None:
                    feats = extract_features(landmarks_to_array(g_hand, w, h))
                    votes.append(int(model.predict(feats[None, :])[0]))
                    current = collections.Counter(votes).most_common(1)[0][0]
                else:
                    votes.clear()
                    current = None

                # Pinch trigger: ONLY run on the strum hand (t_hand)
                now = time.time()
                strum_pts = None
                if t_hand is not None:
                    strum_pts = landmarks_to_array(t_hand, w, h)
                    ratio = pinch_ratio(strum_pts)
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
                                last_played_chord = current
                                last_strum_time = now
                    elif ratio > PINCH_OFF:
                        pinched = False
                        last_played_chord = None
                else:
                    pinched = False
                    last_played_chord = None

                # Draw hands with distinct role-based colors and labels
                if g_hand is not None:
                    chord_name = CHORDS[current][0] if current else "-"
                    draw_hand(frame, g_hand, w, h, color=(0, 255, 100))
                    wx = int(g_hand[0].x * w)
                    wy = int(g_hand[0].y * h)
                    badge_pos = (max(10, min(w - 180, wx - 40)), min(h - 15, wy + 25))
                    draw_badge(frame, f"CHORD: {chord_name}", badge_pos, (0, 130, 40))

                if t_hand is not None:
                    wx = int(t_hand[0].x * w)
                    wy = int(t_hand[0].y * h)
                    badge_pos = (max(10, min(w - 180, wx - 40)), min(h - 15, wy + 25))

                    if pinched:
                        draw_hand(frame, t_hand, w, h, color=(0, 255, 255))
                        # Highlight pinch contact point between thumb (4) and index (8)
                        if strum_pts is not None:
                            p4 = (int(strum_pts[4][0]), int(strum_pts[4][1]))
                            p8 = (int(strum_pts[8][0]), int(strum_pts[8][1]))
                            cv2.circle(frame, p4, 8, (0, 255, 255), -1)
                            cv2.circle(frame, p8, 8, (0, 255, 255), -1)
                            cv2.line(frame, p4, p8, (0, 255, 255), 3)
                        draw_badge(frame, "STRUM: PINCH!", badge_pos, (0, 180, 220))
                    else:
                        draw_hand(frame, t_hand, w, h, color=(0, 165, 255))
                        draw_badge(frame, "STRUM: READY", badge_pos, (0, 100, 180))

                # Top HUD Bar
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 55), (25, 25, 25), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                # Active chord indicator
                chord_name = CHORDS[current][0] if current else "-"
                chord_txt = f"CHORD: {chord_name}" if current else "CHORD: -"
                chord_col = (0, 255, 100) if current else (160, 160, 160)
                cv2.putText(frame, chord_txt, (w - 230, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, chord_col, 2, cv2.LINE_AA)

                # Strum status indicator
                if t_hand is not None:
                    if pinched:
                        strum_txt = "STRUM: PINCH (PLAYING)"
                        strum_col = (0, 255, 255)
                    else:
                        strum_txt = "STRUM: READY (OPEN)"
                        strum_col = (0, 200, 255)
                else:
                    strum_txt = "STRUM: WAITING..."
                    strum_col = (160, 160, 160)
                cv2.putText(frame, strum_txt, (20, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, strum_col, 2, cv2.LINE_AA)

                # Bottom control hint
                mode_str = "Hands: Left=Strum, Right=Chord" if not swapped else "Hands: Left=Chord, Right=Strum [SWAPPED]"
                hint_str = f"{mode_str}  |  [S] Swap  [Q] Quit"
                cv2.putText(frame, hint_str, (20, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

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

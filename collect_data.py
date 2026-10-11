"""Automated & interactive gesture data collector.

Usage:  python collect_data.py --person Thara
Controls:
  SPACE / Click button = Start / Pause recording
  Click [G1]..[G7] or Keys 1-7 = Switch gesture
  N / Click [NEXT] = Next gesture
  Q / ESC = Quit and save
"""
import argparse, csv, os, time
import cv2
import mediapipe as mp
from features import (extract_features, landmarks_to_array, draw_hand, pick_gesture_hand)

BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
RunningMode        = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
TARGET_PER_GESTURE = 150   # Samples per gesture goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    csv_file = f"data/{args.person}.csv"

    # Load existing sample counts
    counts = {k: 0 for k in range(1, 8)}
    session_added = {k: 0 for k in range(1, 8)}
    if os.path.exists(csv_file):
        try:
            with open(csv_file, "r") as f:
                for row in csv.reader(f):
                    if row and row[0].isdigit():
                        lbl = int(row[0])
                        if lbl in counts:
                            counts[lbl] += 1
        except Exception:
            pass

    # Start on the first incomplete gesture
    label = 1
    for k in range(1, 8):
        if counts[k] < TARGET_PER_GESTURE:
            label = k
            break

    options = HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.40,
        min_hand_presence_confidence=0.40,
        min_tracking_confidence=0.40,
    )

    cap = cv2.VideoCapture(0)
    recording = False
    countdown_start = 0.0
    countdown_seconds = 0
    window_name = f"Gesture Collector - {args.person}"
    cv2.namedWindow(window_name)

    mouse_click = [None]
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_click[0] = (x, y)
    cv2.setMouseCallback(window_name, on_mouse)

    with HandLandmarker.create_from_options(options) as detector:
        with open(csv_file, "a", newline="") as fh:
            writer = csv.writer(fh)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = time.time()

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = detector.detect(mp_image)

                hand = None
                if result and result.hand_landmarks:
                    for lm_list in result.hand_landmarks:
                        draw_hand(frame, lm_list, w, h)
                    # Same rule as the live demo: one hand -> use it; two hands -> use the
                    # one on the right of the mirrored screen (the gesture hand)
                    hand = pick_gesture_hand(result)

                # Check countdown state
                if countdown_seconds > 0:
                    elapsed = now - countdown_start
                    remaining = countdown_seconds - int(elapsed)
                    if remaining <= 0:
                        countdown_seconds = 0
                        recording = True
                    else:
                        recording = False

                # Record samples
                if hand is not None and recording:
                    feats = extract_features(landmarks_to_array(hand, w, h))
                    writer.writerow([label, *feats])
                    fh.flush()
                    counts[label] += 1
                    session_added[label] += 1

                    # Auto-advance when target reached for this gesture
                    if counts[label] >= TARGET_PER_GESTURE and session_added[label] >= 60:
                        recording = False
                        print(f"\aCompleted Gesture {label} ({counts[label]} samples)!")
                        # Advance to next incomplete gesture
                        next_gesture = None
                        for k in range(1, 8):
                            if counts[k] < TARGET_PER_GESTURE:
                                next_gesture = k
                                break
                        if next_gesture is not None:
                            label = next_gesture
                            # Start 3-second countdown to prepare hand
                            countdown_start = now
                            countdown_seconds = 3
                        else:
                            # All 7 gestures completed!
                            countdown_seconds = 0

                # --- Draw On-Screen UI ---
                # 1. Top Gesture Buttons [G1] .. [G7]
                btn_w = 60
                btn_h = 36
                for i in range(1, 8):
                    bx1 = 15 + (i - 1) * (btn_w + 8)
                    by1 = 10
                    bx2 = bx1 + btn_w
                    by2 = by1 + btn_h

                    is_active = (i == label)
                    is_done = (counts[i] >= TARGET_PER_GESTURE)
                    if is_active:
                        bg_col = (0, 180, 255) if not recording else (0, 0, 255)
                    elif is_done:
                        bg_col = (0, 140, 0)
                    else:
                        bg_col = (60, 60, 60)

                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_col, -1)
                    if is_active:
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
                    cv2.putText(frame, f"G{i}", (bx1 + 12, by1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                # Next button
                nx1 = 15 + 7 * (btn_w + 8)
                nx2 = nx1 + 75
                cv2.rectangle(frame, (nx1, 10), (nx2, 10 + btn_h), (80, 80, 160), -1)
                cv2.putText(frame, "NEXT >", (nx1 + 6, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # 2. Main Status Card
                card_y1, card_y2 = 55, 175
                cv2.rectangle(frame, (15, card_y1), (460, card_y2), (20, 20, 20), -1)

                if countdown_seconds > 0:
                    status_text = f"GET READY FOR G{label} IN: {int(countdown_seconds - (now - countdown_start)) + 1}"
                    status_color = (0, 200, 255)
                elif recording:
                    status_text = f"● RECORDING GESTURE {label}"
                    status_color = (0, 0, 255)
                else:
                    if all(counts[k] >= TARGET_PER_GESTURE for k in range(1, 8)):
                        status_text = "ALL 7 GESTURES COMPLETE!"
                        status_color = (0, 255, 0)
                    else:
                        status_text = f"PAUSED - Gesture {label}"
                        status_color = (0, 220, 255)

                cv2.putText(frame, status_text, (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                # Progress bar
                prog = min(counts[label] / TARGET_PER_GESTURE, 1.0)
                bar_x1, bar_y1 = 25, 102
                bar_w, bar_h = 320, 18
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + int(bar_w * prog), bar_y1 + bar_h), (0, 200, 0), -1)
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (180, 180, 180), 1)
                cv2.putText(frame, f"{counts[label]}/{TARGET_PER_GESTURE}", (bar_x1 + bar_w + 10, bar_y1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                # Hand detection feedback
                if hand is not None:
                    h_text = "Hand: DETECTED"
                    h_col = (0, 255, 0)
                else:
                    h_text = "Hand: NOT DETECTED (Show hand in camera)"
                    h_col = (0, 140, 255)
                cv2.putText(frame, h_text, (25, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.6, h_col, 1)

                # 3. Big Clickable Record Button
                rx1, ry1, rx2, ry2 = w - 210, 55, w - 15, 120
                r_col = (0, 0, 200) if recording else (0, 160, 0)
                r_txt = "PAUSE" if recording else "START REC"
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), r_col, -1)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
                cv2.putText(frame, r_txt, (rx1 + 25, ry1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                # 4. Bottom hint bar
                cv2.rectangle(frame, (0, h - 35), (w, h), (25, 25, 25), -1)
                cv2.putText(frame, "SPACE/Click: Record | Keys 1-7 or N: Gesture | Q: Quit & Save",
                            (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                # Handle mouse clicks
                if mouse_click[0] is not None:
                    mx, my = mouse_click[0]
                    mouse_click[0] = None
                    # Gesture buttons
                    for i in range(1, 8):
                        bx1 = 15 + (i - 1) * (btn_w + 8)
                        if bx1 <= mx <= bx1 + btn_w and 10 <= my <= 10 + btn_h:
                            label = i
                            recording = False
                            countdown_seconds = 0
                            break
                    # Next button
                    if nx1 <= mx <= nx2 and 10 <= my <= 10 + btn_h:
                        label = (label % 7) + 1
                        recording = False
                        countdown_seconds = 0
                    # Record button
                    if rx1 <= mx <= rx2 and ry1 <= my <= ry2:
                        countdown_seconds = 0
                        recording = not recording

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord(" "), ord("r"), ord("R"), 13):
                    countdown_seconds = 0
                    recording = not recording
                elif key in (ord("n"), ord("N")):
                    label = (label % 7) + 1
                    recording = False
                    countdown_seconds = 0
                elif key in (ord("p"), ord("P")):
                    label = 7 if label == 1 else label - 1
                    recording = False
                    countdown_seconds = 0
                elif ord("1") <= key <= ord("7"):
                    label = key - ord("0")
                    recording = False
                    countdown_seconds = 0
                elif 177 <= key <= 183:  # NumPad 1..7
                    label = key - 176
                    recording = False
                    countdown_seconds = 0
                elif 225 <= key <= 231:  # NumPad 1..7 (Windows)
                    label = key - 224
                    recording = False
                    countdown_seconds = 0

    cap.release()
    cv2.destroyAllWindows()
    print("\nRecorded samples in total for", args.person + ":")
    for k in range(1, 8):
        status_mark = "✓" if counts[k] >= TARGET_PER_GESTURE else " "
        print(f"  [{status_mark}] Gesture {k}: {counts[k]} samples")


if __name__ == "__main__":
    main()
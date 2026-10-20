"""Webcam -> right hand (chords) -> left hand pinch (plays chord).
Run after train_svm.py. Press q to quit, s to swap hands.

Play-along song test (for the report card):
  p = start the next song (Let It Be, Stand By Me, ...). The screen shows which
      chord to play next and how many fingers to show. Release the pinch between
      chords: every NEW pinch counts as one bar.
  n = skip the current bar (counts as a miss).
At the end of the song a report card is saved in reports/ and shown on screen."""
import collections, os, re, time
import cv2
import joblib
import mediapipe as mp
from audio import CHORDS, FS, ChordPlayer
from chord_judge import judge
from features import (extract_features, landmarks_to_array, pinch_ratio, draw_hand)
from songs import NAME_TO_NUMBER, SONGS

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


def draw_song_strip(img, song, chords, done, step):
    """Bottom strip: one box per bar. Green = right, red = wrong, yellow = play this now."""
    h, w = img.shape[:2]
    n = len(chords)
    box_w = min(90, (w - 40) // n)
    y0 = h - 110
    for i, c in enumerate(chords):
        x0 = 20 + i * box_w
        if i < len(done):
            col = (40, 160, 40) if done[i] else (40, 40, 200)
        elif i == step:
            col = (0, 200, 230)
        else:
            col = (90, 90, 90)
        cv2.rectangle(img, (x0 + 2, y0), (x0 + box_w - 2, y0 + 40), col, -1)
        cv2.putText(img, c, (x0 + 10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if step < n:
        target = chords[step]
        msg = f"PLAY-ALONG {song}: bar {step + 1}/{n} -> play {target} (show {NAME_TO_NUMBER[target]} fingers)"
    else:
        msg = f"PLAY-ALONG {song}: finished"
    draw_badge(img, msg, (20, y0 - 15), (60, 60, 60), scale=0.6)


def finish_song(session, player):
    """Save the play-along session and show its report card."""
    from report_card import make_report
    slug = re.sub(r"[^a-z0-9]+", "_", session["song"].lower()).strip("_")
    out = os.path.join("reports", f"song_{slug}_camera_{time.strftime('%Y%m%d_%H%M%S')}.png")
    lines = make_report(session, out, buffers=player.buffers)
    print("\n".join(lines))
    print("Saved", out)
    card = cv2.imread(out)
    if card is not None:
        cv2.imshow("Report card", card)


def main():
    if not os.path.exists(MODEL_FILE):
        raise SystemExit(f"{MODEL_FILE} not found. Run python train_svm.py first.")
    model  = joblib.load(MODEL_FILE)
    player = ChordPlayer()

    # Sound check at startup: the judge listens to every chord the instrument can play.
    sound_check = {n: judge(buf, FS, expected=CHORDS[n][0]) for n, buf in player.buffers.items()}
    for n in sorted(sound_check):
        print(f"Sound check, gesture {n}: {sound_check[n]['sentence']}")
    last_sound_msg = None

    # Play-along state
    song_names = list(SONGS)
    song_i = -1
    session = None          # dict while a song is being played
    t0 = None

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
                                r = sound_check[current]
                                last_sound_msg = (f"SOUND CHECK: judge heard {r['heard']} "
                                                  f"({'RIGHT' if r['ok'] else 'WRONG'})", r["ok"])
                                # Play-along: every new pinch = one bar of the song
                                if session is not None and len(session["steps"]) < len(SONGS[session["song"]]):
                                    if t0 is None:
                                        t0 = now
                                    k = len(session["steps"])
                                    session["steps"].append({"expected": SONGS[session["song"]][k],
                                                             "shown_number": int(current),
                                                             "t": round(now - t0, 3)})
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
                hint_str = f"{mode_str}  |  [S] Swap  [P] Play-along song  [N] Skip bar  [Q] Quit"
                cv2.putText(frame, hint_str, (20, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

                # Sound check badge for the last strum
                if last_sound_msg is not None:
                    msg, ok = last_sound_msg
                    draw_badge(frame, msg, (20, 85), (0, 130, 40) if ok else (0, 0, 180))

                # Play-along: count frames, draw the song strip, finish when all bars are played
                if session is not None:
                    session["frames_total"] += 1
                    session["frames_hand_seen"] += g_hand is not None
                    chords = SONGS[session["song"]]
                    done = [s["shown_number"] == NAME_TO_NUMBER[s["expected"]] for s in session["steps"]]
                    draw_song_strip(frame, session["song"], chords, done, len(session["steps"]))
                    if len(session["steps"]) == len(chords):
                        cv2.imshow("AIR gesture instrument", frame)
                        cv2.waitKey(1)
                        finish_song(session, player)
                        session = None

                cv2.imshow("AIR gesture instrument", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):  # Press 's' to swap anytime if desired
                    swapped = not swapped
                elif key == ord("p"):  # start the next play-along song
                    song_i = (song_i + 1) % len(song_names)
                    session = {"song": song_names[song_i], "mode": "camera", "steps": [],
                               "frames_total": 0, "frames_hand_seen": 0}
                    t0 = None
                    print(f"Play-along: {session['song']} -> {' '.join(SONGS[session['song']])}")
                elif key == ord("n") and session is not None:  # skip a bar (counts as a miss)
                    k = len(session["steps"])
                    if k < len(SONGS[session["song"]]):
                        session["steps"].append({"expected": SONGS[session["song"]][k],
                                                 "shown_number": None,
                                                 "t": round(time.time() - t0, 3) if t0 else 0.0})
    finally:
        player.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

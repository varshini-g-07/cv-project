"""Chord Lab: click buttons to compare OUR chord with a REAL guitar chord, by ear AND by eye.

For each chord the window shows:
  - which notes the chord needs (its "recipe"),
  - what the judge (chord_judge.py) heard in our chord and in the real recording,
  - for the last clicked clip: a "note meter" (how loud each of the 12 notes is),
    with the recipe notes shaded green. Tall bars inside the green = right chord.

    python chord_lab.py                 # uses real_chords/*.wav
Press q to quit.
"""
import argparse
import os

import cv2
import numpy as np

from audio import CHORDS, FS, ChordPlayer
from chord_judge import NOTE_NAMES, describe, find_real_recordings, judge, load_wav, recipe

W, H = 1180, 640
ROW_H, TOP = 66, 80
BTN_OURS = (150, 270)       # x-range of the "Play ours" buttons
BTN_REAL = (285, 405)       # x-range of the "Play real" buttons
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, WHITE, GREY = (60, 180, 60), (60, 60, 220), (255, 255, 255), (150, 150, 150)


def text(img, s, x, y, scale=0.55, color=WHITE, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def verdict(img, r, x, y):
    ok = r["ok"]
    text(img, r["heard"], x, y, 0.55, GREEN if ok else RED, 2)
    text(img, "RIGHT" if ok else "WRONG", x + 52, y, 0.55, GREEN if ok else RED, 2)


def draw_meter(img, title, meter, chord, x0, y0, w, h):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
    text(img, title, x0, y0 - 12, 0.6, WHITE, 2)
    bw = w / 12
    rec = recipe(chord)
    for p in range(12):
        x = int(x0 + p * bw)
        if p in rec:
            cv2.rectangle(img, (x, y0 + 1), (int(x + bw), y0 + h - 1), (40, 90, 40), -1)
        bh = int(meter[p] * (h - 30))
        cv2.rectangle(img, (x + 6, y0 + h - 25 - bh), (int(x + bw) - 6, y0 + h - 25), (230, 160, 40), -1)
        text(img, NOTE_NAMES[p], x + 8, y0 + h - 7, 0.5, WHITE)
    text(img, "green columns = notes the chord should have", x0, y0 + h + 22, 0.5, GREY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="real_chords")
    a = ap.parse_args()

    player = ChordPlayer()
    names = {n: CHORDS[n][0] for n in sorted(CHORDS)}
    real_paths = find_real_recordings(a.real_dir, list(names.values()))
    real = {c: load_wav(p[0], FS) for c, p in real_paths.items() if p}
    ours_r = {n: judge(player.buffers[n], FS, expected=c) for n, c in names.items()}
    real_r = {c: judge(y, FS, expected=c) for c, y in real.items()}
    selected = {"label": None, "meter": None, "chord": None}

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        row = (y - TOP) // ROW_H
        n = row + 1
        if n not in names:
            return
        c = names[n]
        if BTN_OURS[0] <= x <= BTN_OURS[1]:
            player.play(n)
            selected.update(label=f"Our {c}", meter=ours_r[n]["meter"], chord=c)
        elif BTN_REAL[0] <= x <= BTN_REAL[1] and c in real:
            player.play_buffer(real[c])
            selected.update(label=f"Real guitar {c}", meter=real_r[c]["meter"], chord=c)

    cv2.namedWindow("Chord Lab")
    cv2.setMouseCallback("Chord Lab", on_click)
    try:
        while True:
            img = np.full((H, W, 3), 30, np.uint8)
            text(img, "Chord Lab - is each chord right? (click to listen)", 20, 35, 0.8, WHITE, 2)
            text(img, "gesture / chord", 20, TOP - 10, 0.5, GREY)
            text(img, "needs notes", 420, TOP - 10, 0.5, GREY)
            text(img, "judge: ours", 540, TOP - 10, 0.5, GREY)
            text(img, "judge: real", 670, TOP - 10, 0.5, GREY)
            for n, c in names.items():
                y = TOP + (n - 1) * ROW_H
                text(img, f"{n}  {c}", 20, y + 38, 0.75, WHITE, 2)
                cv2.rectangle(img, (BTN_OURS[0], y + 12), (BTN_OURS[1], y + 52), (150, 90, 20), -1)
                text(img, "Play ours", BTN_OURS[0] + 12, y + 38, 0.6, WHITE, 2)
                if c in real:
                    cv2.rectangle(img, (BTN_REAL[0], y + 12), (BTN_REAL[1], y + 52), (20, 110, 160), -1)
                    text(img, "Play real", BTN_REAL[0] + 12, y + 38, 0.6, WHITE, 2)
                text(img, " ".join(NOTE_NAMES[p] for p in recipe(c)), 420, y + 38, 0.6, WHITE)
                verdict(img, ours_r[n], 540, y + 38)
                if c in real_r:
                    verdict(img, real_r[c], 670, y + 38)
            n_ok = sum(r["ok"] for r in ours_r.values())
            n_real = sum(r["ok"] for r in real_r.values())
            text(img, f"Judge on real guitars: {n_real} of {len(real_r)} right.  "
                      f"Our instrument: {n_ok} of {len(ours_r)} chords judged right.", 20, H - 20, 0.65,
                 GREEN if n_ok == len(ours_r) else RED, 2)
            if selected["meter"] is not None:
                c = selected["chord"]
                draw_meter(img, f"{selected['label']}  ({describe(c)})", selected["meter"], c,
                           810, 120, 350, 260)
            else:
                text(img, "Click a button to see", 820, 220, 0.6, GREY)
                text(img, "which notes are in the sound.", 820, 250, 0.6, GREY)
            cv2.imshow("Chord Lab", img)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        player.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

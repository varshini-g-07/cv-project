"""Whole-system "report card" for one song, written for a non-musician.

The instrument is a chain of four steps. The report gives each step a traffic
light and explains every mistake in one sentence, saying WHICH step caused it:

  1. Camera sees the hand
  2. Camera counts the right number of fingers   (gesture model)
  3. Number becomes the right chord               (chord table in audio.py)
  4. The sound really is that chord               (checked by the judge, chord_judge.py)

A session is a dict (saved as JSON by realtime_demo.py / song_test.py):
  {"song": "Let It Be (verse)", "mode": "camera" or "auto",
   "frames_total": 900, "frames_hand_seen": 870,
   "steps": [{"expected": "C", "shown_number": 1, "t": 0.0}, ...]}
"""
import json
import os

import numpy as np

from audio import CHORDS, DAMP_SAMPLES, FS, render_chord
from chord_judge import judge, notes_str, recipe
from songs import NAME_TO_NUMBER

GREEN, YELLOW, RED, GREY = "#2ca02c", "#f2b705", "#d62728", "#9e9e9e"
STEP_LEN = 1.0          # seconds of audio the judge listens to after each strum


def light(frac, good=0.95, ok=0.8):
    if frac is None:
        return GREY
    return GREEN if frac >= good else YELLOW if frac >= ok else RED


def chord_table_check():
    """Step 3: do the notes listed for each chord in audio.py match its textbook recipe?"""
    rows = []
    for n in sorted(CHORDS):
        name, midi = CHORDS[n]
        listed = sorted({m % 12 for m in midi})
        rows.append((n, name, listed == sorted(recipe(name)), listed))
    return rows


def render_session(steps, buffers=None):
    """Rebuild the sound of a session exactly like ChordPlayer plays it: each strum
    starts at its time; the previous chord is damped (80 ms fade) at the new strum."""
    if buffers is None:
        buffers = {n: render_chord(m) for n, (_, m) in CHORDS.items()}
    played = [s for s in steps if s.get("shown_number")]
    if not played:
        return np.zeros(FS, dtype=np.float32), buffers
    end = max(s["t"] for s in played) + 2.5
    mix = np.zeros(int(end * FS) + max(len(b) for b in buffers.values()), dtype=np.float32)
    for k, s in enumerate(played):
        b = buffers[s["shown_number"]].copy()
        if k + 1 < len(played):                       # damp at the next strum
            cut = int((played[k + 1]["t"] - s["t"]) * FS)
            if cut < len(b):
                b[cut: cut + DAMP_SAMPLES] *= np.linspace(1, 0, len(b[cut: cut + DAMP_SAMPLES]))
                b[cut + DAMP_SAMPLES:] = 0
        i = int(s["t"] * FS)
        mix[i: i + len(b)] += b
    mix /= max(1.0, float(np.max(np.abs(mix))) / 0.9)
    return mix, buffers


def evaluate(session, buffers=None):
    """Fill in, for every bar of the song, what was played and what the judge heard."""
    steps = session["steps"]
    audio, buffers = render_session(steps, buffers)
    for i, s in enumerate(steps):
        n = s.get("shown_number")
        s["expected_number"] = NAME_TO_NUMBER[s["expected"]]
        s["played"] = CHORDS[n][0] if n else None
        if n:
            nxt = steps[i + 1]["t"] if i + 1 < len(steps) and steps[i + 1].get("shown_number") else s["t"] + 5
            length = max(0.3, min(STEP_LEN, nxt - s["t"] - 0.05))
            r = judge(audio, FS, start=s["t"], length=length)
            s["heard"] = r["heard"]
        else:
            s["heard"] = None
    return audio


def explain(session):
    """Plain-English summary lines + one line per mistake."""
    steps, mode = session["steps"], session.get("mode", "camera")
    n = len(steps)
    right = sum(s["heard"] == s["expected"] for s in steps)
    lines = [f"Song: {session['song']}. {right} out of {n} chords came out right "
             f"(checked by listening to the sound, not by trusting the program)."]
    mistakes = []
    for i, s in enumerate(steps, 1):
        if s["heard"] == s["expected"]:
            continue
        want = f"bar {i} should be {s['expected']} ({s['expected_number']} fingers)"
        if not s.get("shown_number"):
            mistakes.append(f"- {want[0].upper() + want[1:]}: no chord was played (hand not seen).")
        elif s["shown_number"] != s["expected_number"]:
            mistakes.append(f"- {want[0].upper() + want[1:]}: the camera read {s['shown_number']} fingers, "
                            f"so the instrument played {s['played']}.  -> camera mistake (step 2)")
        elif s["heard"] != s["played"]:
            mistakes.append(f"- {want[0].upper() + want[1:]}: the camera was right, but the sound was "
                            f"heard as {s['heard']}.  -> sound mistake (step 4)")
    if mistakes:
        lines.append("Where the mistakes came from:")
        lines += mistakes
    else:
        lines.append("No mistakes.")
    if mode == "auto":
        lines.append("(Auto-play: chords were fed straight into the sound engine, no camera. "
                     "This tests the sound part on its own.)")
    return lines


def make_report(session, out_png, buffers=None, save_wav=True):
    """Evaluate a session and draw the one-page report card. Returns the text lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    audio = evaluate(session, buffers)
    steps = session["steps"]
    n = len(steps)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    stem = os.path.splitext(out_png)[0]
    if save_wav:
        from scipy.io import wavfile
        wavfile.write(stem + ".wav", FS, (audio * 32767).astype(np.int16))

    # Traffic lights
    camera = session.get("mode", "camera") == "camera"
    seen = (session["frames_hand_seen"] / session["frames_total"]
            if camera and session.get("frames_total") else None)
    played = [s for s in steps if s.get("shown_number")]
    fingers = (sum(s["shown_number"] == s["expected_number"] for s in played) / n) if camera else None
    table = chord_table_check()
    table_ok = sum(r[2] for r in table)
    sound_ok = sum(s["heard"] == s["played"] for s in played)
    stages = [
        ("1. Camera sees\nthe hand", seen,
         f"hand visible in {seen:.0%} of video frames" if seen is not None else "not used (auto-play)"),
        ("2. Camera counts\nthe fingers right", fingers,
         f"{sum(s['shown_number'] == s['expected_number'] for s in played)} of {n} bars"
         if fingers is not None else "not used (auto-play)"),
        ("3. Number becomes\nthe right chord", table_ok / len(table),
         f"{table_ok} of {len(table)} chords have the textbook notes"),
        ("4. Sound really is\nthat chord", sound_ok / max(1, len(played)),
         f"judge agreed on {sound_ok} of {len(played)} strums"),
    ]
    lines = explain(session)

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.955, f"AIR Guitar report card - {session['song']}"
             f"{' (camera play-along)' if camera else ' (auto-play, no camera)'}",
             ha="center", fontsize=16, weight="bold")

    # Song strip
    ax = fig.add_axes([0.14, 0.68, 0.82, 0.22])
    ax.set_xlim(0, n)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    ax.text(-0.02 * n, 1.6, "Song sheet", ha="right", va="center", fontsize=10)
    ax.text(-0.02 * n, 0.6, "Judge heard", ha="right", va="center", fontsize=10)
    for i, s in enumerate(steps):
        ok = s["heard"] == s["expected"]
        ax.add_patch(Rectangle((i + 0.05, 1.15), 0.9, 0.9, color="#e8e8e8"))
        ax.text(i + 0.5, 1.6, s["expected"], ha="center", va="center", fontsize=14, weight="bold")
        ax.add_patch(Rectangle((i + 0.05, 0.15), 0.9, 0.9, color=GREEN if ok else RED))
        ax.text(i + 0.5, 0.6, s["heard"] or "-", ha="center", va="center", fontsize=14,
                color="white", weight="bold")
        ax.text(i + 0.5, 2.3, f"bar {i + 1}", ha="center", fontsize=8, color="#555")
    right = sum(s["heard"] == s["expected"] for s in steps)
    fig.text(0.5, 0.635, f"{right} out of {n} chords came out right", ha="center", fontsize=15,
             weight="bold", color=light(right / n))

    # Chain of four steps
    ax2 = fig.add_axes([0.04, 0.36, 0.92, 0.24])
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    for i, (title, frac, detail) in enumerate(stages):
        ax2.add_patch(Circle((i + 0.5, 0.62), 0.16, color=light(frac)))
        ax2.text(i + 0.5, 0.88, title, ha="center", va="bottom", fontsize=10, weight="bold")
        ax2.text(i + 0.5, 0.2, detail, ha="center", fontsize=9, wrap=True)
        if i < 3:
            ax2.annotate("", xy=(i + 1.28, 0.62), xytext=(i + 0.72, 0.62),
                         arrowprops=dict(arrowstyle="->", lw=2, color="#555"))

    fig.text(0.05, 0.31, "\n".join(lines[:12]), va="top", fontsize=10, family="monospace")
    fig.text(0.05, 0.03, "Green = works, yellow = mostly works, red = needs fixing, grey = not tested here. "
             "The judge only listens to the sound; it does not know which chord was intended.",
             fontsize=8, color="#555")
    fig.savefig(out_png, dpi=100)
    plt.close(fig)

    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(session, f, indent=1)
    with open(stem + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
        for title, frac, detail in stages:
            f.write(f"{title.replace(chr(10), ' ')}: {detail}\n")
        f.write("\nChord table check (step 3):\n")
        for num, name, ok, listed in table:
            f.write(f"  gesture {num} -> {name}: notes {notes_str(listed)} "
                    f"{'match' if ok else 'DO NOT match'} the recipe {notes_str(recipe(name))}\n")
    return lines


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Redraw a report card from a saved session JSON.")
    ap.add_argument("session_json")
    a = ap.parse_args()
    with open(a.session_json, encoding="utf-8") as f:
        sess = json.load(f)
    out = os.path.splitext(a.session_json)[0] + ".png"
    print("\n".join(make_report(sess, out)))
    print("Saved", out)

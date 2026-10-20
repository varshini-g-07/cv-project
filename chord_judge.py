"""The "judge": an independent chord checker that only LISTENS to the sound.

It never looks at which chord the program meant to play. Like a guitar tuner,
it measures which musical notes are loudest in the audio and compares them to
chord "recipes" (C = C + E + G, Am = A + C + E, ...). It knows all 36 common
chords (12 starting notes x major / minor / diminished), not just our 7, so it
can also say "this is NOT a C, it sounds like Am".

Run it directly to check the judge itself on real guitar recordings and then
check our synthesized chords:
    python chord_judge.py                      # uses real_chords/*.wav
    python chord_judge.py --real_dir path/to/kaggle/guitar_chords   # more recordings
"""
import argparse
import os

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_TYPES = {"": [0, 4, 7], "m": [0, 3, 7], "dim": [0, 3, 6]}
TYPE_WORDS = {"": "major", "m": "minor", "dim": "diminished"}

# A plucked string also sounds softly at higher "overtones" (2x, 3x, 4x the
# frequency). The judge expects them, so they are not mistaken for extra notes.
POWER, SKIP, FMAX = 1, 0.03, 700    # spectrum power, skip after strum (s), top frequency (Hz)
OVERTONES = [(1, 1.0), (2, 0.6), (3, 0.45), (4, 0.3), (5, 0.2), (6, 0.15)]


def split_name(chord_name):
    """'Am' -> ('A', 'm'), 'Bdim' -> ('B', 'dim'), 'C' -> ('C', '')."""
    for suffix in ("dim", "m", ""):
        root = chord_name[: -len(suffix)] if suffix else chord_name
        if chord_name.endswith(suffix) and root in NOTE_NAMES:
            return root, suffix
    raise ValueError(f"Unknown chord name: {chord_name}")


def recipe(chord_name):
    """Pitch classes (0 = C ... 11 = B) that make up a chord, e.g. 'Am' -> [9, 0, 4]."""
    root, suffix = split_name(chord_name)
    r = NOTE_NAMES.index(root)
    return [(r + i) % 12 for i in CHORD_TYPES[suffix]]


def notes_str(pitch_classes):
    return ", ".join(NOTE_NAMES[p] for p in pitch_classes)


def describe(chord_name):
    """'Am' -> 'A minor'."""
    root, suffix = split_name(chord_name)
    return f"{root} {TYPE_WORDS[suffix]}"


def _template(pcs):
    t = np.zeros(12)
    for p in pcs:
        for h, w in OVERTONES:
            t[(p + int(round(12 * np.log2(h)))) % 12] += w
    return t / np.linalg.norm(t)


ALL_CHORDS = [root + s for root in NOTE_NAMES for s in CHORD_TYPES]
TEMPLATES = {name: _template(recipe(name)) for name in ALL_CHORDS}


def load_wav(path, fs_out=44100):
    """Read a WAV file as mono float32 at fs_out."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    fs, y = wavfile.read(path)
    if y.dtype.kind == "i":
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    elif y.dtype.kind == "u":
        y = (y.astype(np.float32) - 128) / 128
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if fs != fs_out:
        g = np.gcd(fs, fs_out)
        y = resample_poly(y, fs_out // g, fs // g).astype(np.float32)
    return y


def note_meter(y, fs, start=None, length=1.0):
    """How loud each of the 12 notes is in the clip (max = 1).

    Looks at `length` seconds starting just after the strum (or at `start`).
    """
    y = np.asarray(y, dtype=np.float64)
    if start is None:
        loud = np.flatnonzero(np.abs(y) > 0.2 * np.max(np.abs(y)) + 1e-12)
        i0 = int(loud[0]) if len(loud) else 0
    else:
        i0 = int(start * fs)
    seg = y[i0 + int(SKIP * fs): i0 + int((SKIP + length) * fs)]
    if len(seg) < 1024:
        return np.zeros(12)
    n_fft = 1 << int(np.ceil(np.log2(len(seg) * 4)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    band = (freqs > 70) & (freqs < FMAX)
    midi = 69 + 12 * np.log2(freqs[band] / 440.0)
    near = np.abs(midi - np.round(midi)) < 0.35          # close to a real note
    pcs = np.round(midi[near]).astype(int) % 12
    meter = np.bincount(pcs, weights=spec[band][near] ** POWER, minlength=12)
    meter = meter ** (1.0 / POWER)
    return meter / (meter.max() + 1e-12)


PRESENT = 0.15   # a note counts as "clearly present" at >= 15% of the loudest note


def judge(y, fs=44100, expected=None, start=None, length=1.0):
    """Name the chord in an audio clip.

    Returns a dict with the judge's answer (best of 36 chords), its second guess,
    the note meter, and (if `expected` is given) whether it matches.
    """
    meter = note_meter(y, fs, start=start, length=length)
    v = meter / (np.linalg.norm(meter) + 1e-12)
    scores = {name: float(v @ t) for name, t in TEMPLATES.items()}
    ranked = sorted(scores, key=scores.get, reverse=True)
    best = ranked[0]
    out = {
        "heard": best,
        "second": ranked[1],
        "score": scores[best],
        "second_score": scores[ranked[1]],
        "meter": meter,
    }
    if expected is not None:
        rec = recipe(expected)
        out["expected"] = expected
        out["ok"] = best == expected
        out["present"] = [p for p in rec if meter[p] >= PRESENT]
        out["missing"] = [p for p in rec if meter[p] < PRESENT]
        # strong notes that are not in the recipe AND not explained as overtones
        overtone_pcs = {(p + int(round(12 * np.log2(h)))) % 12 for p in rec for h, _ in OVERTONES}
        out["extra"] = [p for p in range(12) if p not in overtone_pcs and meter[p] >= 0.5]
        out["sentence"] = sentence(out)
    return out


def sentence(r):
    exp = r["expected"]
    rec = recipe(exp)
    s = f"{exp} ({describe(exp)}) needs the notes {notes_str(rec)}. "
    if r["missing"]:
        s += f"Missing: {notes_str(r['missing'])}. "
    else:
        s += f"All {len(rec)} are clearly there. "
    if r["extra"]:
        s += f"Extra note(s) not in the recipe: {notes_str(r['extra'])}. "
    if r["ok"]:
        return s + f"Out of 36 possible chords, the sound matches {exp} best. CORRECT"
    return s + (f"Out of 36 possible chords, the sound matches {r['heard']} "
                f"({notes_str(recipe(r['heard']))}) best, not {exp}. WRONG")


def similarity(meter_a, meter_b):
    a = meter_a / (np.linalg.norm(meter_a) + 1e-12)
    b = meter_b / (np.linalg.norm(meter_b) + 1e-12)
    return float(a @ b)


def find_real_recordings(real_dir, chord_names):
    """Map chord name -> list of WAV paths. Label = file name (C.wav) or parent folder (C/x.wav)."""
    found = {c: [] for c in chord_names}
    for root, _, files in os.walk(real_dir):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            stem = os.path.splitext(f)[0]
            parent = os.path.basename(root)
            for label in (stem, parent, stem.split("_")[0], stem.split("-")[0]):
                if label in found:
                    found[label].append(os.path.join(root, f))
                    break
    return found


def run_check(real_dir="real_chords", out_dir="reports"):
    """Check the judge on real guitars, then check our chords. Writes a picture + text summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from audio import CHORDS, FS, render_chord

    os.makedirs(out_dir, exist_ok=True)
    names = [CHORDS[n][0] for n in sorted(CHORDS)]
    lines = []

    # Step 1: does the judge itself work? Test it on real guitar recordings.
    real = find_real_recordings(real_dir, names)
    real_meters, n_ok, n_all = {}, 0, 0
    for c in names:
        ms = []
        for p in real[c]:
            r = judge(load_wav(p, FS), FS, expected=c)
            ms.append(r["meter"])
            n_ok += r["ok"]
            n_all += 1
        if ms:
            real_meters[c] = np.mean(ms, axis=0)
    lines.append("STEP 1 - Can we trust the judge?")
    lines.append(f"  We played {n_all} real guitar recordings to the judge. "
                 f"It named the right chord {n_ok} out of {n_all} times.")

    # Step 2: check every chord our instrument makes.
    ours = {}
    lines.append("")
    lines.append("STEP 2 - Is each chord our instrument plays the right chord?")
    for n in sorted(CHORDS):
        c, midi = CHORDS[n]
        r = judge(render_chord(midi), FS, expected=c)
        ours[c] = r
        lines.append(f"  Gesture {n}: {r['sentence']}")
    n_ours = sum(r["ok"] for r in ours.values())
    lines.append(f"  Result: {n_ours} out of {len(ours)} chords are correct.")

    # Step 3: is our chord closest to the SAME chord on a real guitar?
    if real_meters:
        lines.append("")
        lines.append("STEP 3 - Does our chord sound most like the same chord on a real guitar?")
        n_match = n_top2 = 0
        for c, r in ours.items():
            sims = {rc: similarity(r["meter"], m) for rc, m in real_meters.items()}
            ranked = sorted(sims, key=sims.get, reverse=True)
            n_match += ranked[0] == c
            n_top2 += c in ranked[:2]
            if ranked[0] == c:
                lines.append(f"  Our {c}: most similar real-guitar chord is {c}  [YES]")
            else:
                lines.append(f"  Our {c}: most similar real-guitar chord is {ranked[0]}, "
                             f"then {ranked[1]}  [{'CLOSE' if c in ranked[:2] else 'NO'}]")
        lines.append(f"  Result: {n_match} of {len(ours)} are most similar to the matching real chord "
                     f"({n_top2} of {len(ours)} in the top 2). A real guitarist spreads the notes "
                     f"over different strings, so this is a rougher check than step 2.")

    text = "\n".join(lines)
    with open(os.path.join(out_dir, "chord_check.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    # Picture: one row per chord. Grey bars = notes in the recipe.
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.0 * len(names)), sharex=True)
    x = np.arange(12)
    for ax, c in zip(axes, names):
        rec = recipe(c)
        for p in rec:
            ax.axvspan(p - 0.5, p + 0.5, color="#d9f2d9")
        ax.bar(x - 0.2, ours[c]["meter"], 0.4, color="#1f77b4", label="our instrument")
        if c in real_meters:
            ax.bar(x + 0.2, real_meters[c], 0.4, color="#ff7f0e", label="real guitar")
        ok = ours[c]["ok"]
        ax.set_title(f"{c} ({describe(c)}) - needs {notes_str(rec)} - judge heard our chord as "
                     f"{ours[c]['heard']}  {'CORRECT' if ok else 'WRONG'}",
                     fontsize=10, color="green" if ok else "red", loc="left")
        ax.set_yticks([])
        ax.set_ylim(0, 1.1)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(NOTE_NAMES)
    fig.suptitle("Which notes are in each chord? Green columns = notes the chord should have.\n"
                 "Tall bars inside the green columns = the chord is right.", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chord_check.png"), dpi=110)
    plt.close(fig)
    return text, ours, real_meters


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="real_chords",
                    help="folder of real guitar WAVs named C.wav, Am.wav ... or in C/, Am/ subfolders")
    ap.add_argument("--out_dir", default="reports")
    a = ap.parse_args()
    print(run_check(a.real_dir, a.out_dir)[0])
    print(f"\nSaved {a.out_dir}/chord_check.png and {a.out_dir}/chord_check.txt")

"""Song test without a camera: feed a song's chords straight into the sound engine,
then let the judge listen and check every bar. Produces a report card per song
(reports/song_*.png / .txt / .wav) that a non-musician can read.

    python song_test.py                       # all songs
    python song_test.py --song "Let It Be (verse)"
    python song_test.py --mistake 5           # deliberately play bar 5 wrong, to show
                                              # that the test really catches mistakes
"""
import argparse
import re

from audio import CHORDS, render_chord
from report_card import make_report
from songs import NAME_TO_NUMBER, SONGS

BAR = 1.5   # seconds per chord


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def auto_session(song, mistake_bar=None):
    steps = []
    for i, chord in enumerate(SONGS[song]):
        n = NAME_TO_NUMBER[chord]
        if mistake_bar == i + 1:
            n = n - 1 if n > 1 else 2          # pretend the camera read one finger off
        steps.append({"expected": chord, "shown_number": n, "t": i * BAR})
    return {"song": song, "mode": "auto", "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", choices=list(SONGS), help="default: all songs")
    ap.add_argument("--mistake", type=int, help="bar number to deliberately play wrong")
    a = ap.parse_args()
    buffers = {n: render_chord(m) for n, (_, m) in CHORDS.items()}
    for song in [a.song] if a.song else SONGS:
        sess = auto_session(song, a.mistake)
        out = f"reports/song_{slug(song)}_auto{'_mistake' if a.mistake else ''}.png"
        print("\n".join(make_report(sess, out, buffers)))
        print(f"Saved {out} (+ .wav to listen, .txt summary)\n")


if __name__ == "__main__":
    main()

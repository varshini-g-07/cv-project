"""Sound output: Karplus-Strong plucked-string chords, pre-rendered at startup."""
import numpy as np

FS = 44100

# Nashville number -> (chord name, MIDI notes) in the key of C major
CHORDS = {
    1: ("C",    [48, 52, 55, 60, 64]),
    2: ("Dm",   [50, 57, 62, 65]),
    3: ("Em",   [52, 59, 64, 67]),
    4: ("F",    [53, 57, 60, 65]),
    5: ("G",    [55, 59, 62, 67]),
    6: ("Am",   [57, 60, 64, 69]),
    7: ("Bdim", [59, 62, 65, 71]),
}


def midi_to_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def pluck(freq, dur=2.0, fs=FS, decay=0.996):
    n_delay = int(fs / freq)
    buf = np.random.uniform(-1, 1, n_delay)
    out = np.empty(int(dur * fs))
    for i in range(len(out)):
        k = i % n_delay
        out[i] = buf[k]
        buf[k] = decay * 0.5 * (buf[k] + buf[(k + 1) % n_delay])
    return out


def render_chord(notes, dur=2.0, strum_ms=25, fs=FS):
    delay = int(strum_ms / 1000 * fs)
    total = int(dur * fs) + delay * len(notes)
    mix = np.zeros(total)
    for i, m in enumerate(notes):
        tone = pluck(midi_to_freq(m), dur, fs)
        mix[i * delay : i * delay + len(tone)] += tone
    return (0.8 * mix / (np.max(np.abs(mix)) + 1e-9)).astype(np.float32)


class ChordPlayer:
    def __init__(self):
        print("Rendering chords...")
        self.buffers = {n: render_chord(notes) for n, (_, notes) in CHORDS.items()}

    def play(self, number):
        import sounddevice as sd
        sd.stop()                       # cut the previous chord
        sd.play(self.buffers[number], FS)   # non-blocking


if __name__ == "__main__":
    import time
    p = ChordPlayer()
    for n in range(1, 8):
        print(CHORDS[n][0]); p.play(n); time.sleep(1.2)

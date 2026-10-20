"""Sound output: Karplus-Strong plucked-string chords, pre-rendered at startup."""
import numpy as np

FS = 44100
DAMP_SAMPLES = int(0.08 * FS)   # old chord fades out over 80 ms when a new strum starts

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


def pluck(freq, dur=2.0, fs=FS, decay=0.996, seed=None):
    n_delay = int(fs / freq)
    # Seeded noise burst -> the instrument sounds identical every run, so the sound
    # checked by chord_judge.py is exactly the sound the live demo plays.
    buf = np.random.default_rng(seed).uniform(-1, 1, n_delay)
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
        tone = pluck(midi_to_freq(m), dur, fs, seed=m)
        mix[i * delay : i * delay + len(tone)] += tone
    return (0.8 * mix / (np.max(np.abs(mix)) + 1e-9)).astype(np.float32)


class ChordPlayer:
    def __init__(self):
        import sounddevice as sd
        import threading

        print("Rendering chords...")
        self.buffers = {n: render_chord(notes) for n, (_, notes) in CHORDS.items()}
        self._voices = []
        self._lock = threading.Lock()

        # Persistent low-latency audio stream to avoid stream startup delay
        self.stream = sd.OutputStream(
            samplerate=FS,
            channels=1,
            latency="low",
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        with self._lock:
            active = []
            for buf, pos, fade in self._voices:
                rem = len(buf) - pos
                chunk = min(frames, rem)
                if fade is not None:                       # damped voice: fade out
                    chunk = min(chunk, fade)
                    gain = np.linspace(fade, fade - chunk, chunk, endpoint=False) / DAMP_SAMPLES
                    outdata[:chunk, 0] += buf[pos : pos + chunk] * gain
                    fade -= chunk
                else:
                    outdata[:chunk, 0] += buf[pos : pos + chunk]
                pos += chunk
                if pos < len(buf) and (fade is None or fade > 0):
                    active.append([buf, pos, fade])
            self._voices = active
        np.clip(outdata, -1.0, 1.0, out=outdata)

    def play(self, number):
        """Immediately trigger chord with zero delay (non-blocking)."""
        buf = self.buffers.get(number)
        if buf is not None:
            with self._lock:
                # Like a guitarist re-strumming: chords still ringing are damped
                # (quick fade) so the old chord does not blur into the new one.
                for v in self._voices:
                    if v[2] is None:
                        v[2] = DAMP_SAMPLES
                self._voices.append([buf, 0, None])

    def play_buffer(self, buf):
        """Play any clip (e.g. a real guitar recording), stopping other sounds first
        so two clips can be compared cleanly by ear."""
        with self._lock:
            self._voices = [[np.asarray(buf, dtype=np.float32), 0, None]]

    def close(self):
        """Close audio stream cleanly."""
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


if __name__ == "__main__":
    import time
    p = ChordPlayer()
    for n in range(1, 8):
        print(CHORDS[n][0]); p.play(n); time.sleep(1.2)

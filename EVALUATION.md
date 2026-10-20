# How we check that the instrument is right (for non-musicians)

The professor asked two questions:

1. "When it plays C, how do I know it's really a C?"
2. "If you play a song with it, how do I know the song is right?"

Nobody has to judge by ear. The system checks itself.

## The idea in one sentence

We built a **judge**: a small program that only *listens* to the sound
and names the chord, without knowing which chord we meant to play. It
works like a guitar tuner app that tells you "in tune" when you can't hear
it yourself.

## How the judge works (no music knowledge needed)

- Every chord is a **recipe of notes**. C = the notes C + E + G. Am = A + C + E.
- Every note is a fixed **pitch** (frequency). For example, A = 440 vibrations per second.
- The judge measures how loud each of the 12 notes is in the sound. This is the "note meter" bar chart.
- It compares that against the recipes of **36 possible chords** (not just our 7) and picks the best match.
  So it can also say "this is not C, it sounds like Am".

## Can we trust the judge? (Step 1)

Before using the judge on our instrument, we tested it on **real guitar
recordings** (`real_chords/`). It named the right chord in **7 out of 7** recordings.

## Checks built into the system

| What to run | What it shows | Answers |
|---|---|---|
| `python chord_judge.py` | `reports/chord_check.png`: one bar chart per chord. Green columns = notes the chord should have. Tall bars inside the green = right chord. Also tests the judge on real guitars first. | Q1 |
| `python chord_lab.py` | Window with **Play ours / Play real** buttons for every chord, plus the judge's verdict for both. Click a button to see that clip's note meter. | Q1, by ear and by eye |
| `python song_test.py` | Plays 4 songs straight into the sound engine (no camera). The judge checks every bar. Saves a report card picture, a `.wav` to listen to, and a text summary for each song. | Q2 (sound part) |
| `python song_test.py --song "Let It Be (verse)" --mistake 5` | Plays bar 5 wrong on purpose. The report card turns that bar red and explains it. This proves the test can catch mistakes. | Trust in Q2 |
| `python realtime_demo.py`, then press **P** | **Play-along**: the screen shows which chord to play next and how many fingers to show. At the end, a **report card** pops up and is saved in `reports/`. | Q2 (whole system, with camera) |

Every strum in the live demo also shows a **SOUND CHECK** badge ("judge heard C, RIGHT").

## The report card (whole-system evaluation)

The instrument is a chain of four steps. Each step gets a traffic light:

```
1. Camera sees the hand -> 2. Camera counts the fingers -> 3. Number becomes the chord -> 4. Sound is that chord
```

- Steps 3 and 4 never make mistakes: the chord table is checked against the textbook recipes, and the judge checks the sound.
- So any red bar in a song tells you **which step failed**, in plain words:
  "Bar 2 should be G (5 fingers): the camera read 4 fingers, so the instrument played F -> camera mistake".
- That is the right message for a computer-vision project: the remaining errors are vision errors. For example, one teammate's 5-finger gesture is sometimes read as 4.

## Small changes to the instrument

- `audio.py`: the plucked-string noise is now seeded per note. The instrument sounds the same every run, so the sound the judge checks is exactly the sound the demo plays.
- `audio.py`: when a new chord is strummed, the old one fades out over 80 ms, like a guitarist's hand stopping the strings. Before, chords blurred into each other, and the judge heard G-after-C as C.
- `audio.py`: `ChordPlayer.play_buffer()` plays any clip, e.g. a real guitar recording.

## Limits (be honest in the presentation)

- The judge is tuned for plain major, minor and diminished chords, the ones the instrument uses.
- On 140 randomly varied versions of our 7 chords, the judge was right 136 times (97%). The fixed version the instrument actually uses is judged right 7/7.
- Only one real recording per chord. To test the judge on more real guitars, download the Kaggle "Guitar Chords V2" dataset and run
  `python chord_judge.py --real_dir <folder>`. Files are matched to chords by file name (`C.wav`) or folder name (`C/xxx.wav`).

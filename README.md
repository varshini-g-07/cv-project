# Gesture-controlled chord instrument (based on AIR Guitar, Ke & Chen)

Right hand shows a number gesture 1-7 -> classifier predicts it -> chord (C major, Nashville numbers).
Left hand pinches (thumb + index) -> the current chord is played. Press `s` in the demo to swap hands.

## Setup
    pip install -r requirements.txt
Also download MediaPipe's `hand_landmarker.task` model file (from the Hand Landmarker page of the
MediaPipe docs) and put it in this folder, next to the scripts.

## Workflow
1. `python collect_data.py --person alice`   (each member records all 7 gestures; use the same name
   every time. With two hands in view, the hand on the right of the screen is recorded.)
2. `python train_svm.py`                     (evaluates Extra Trees AND an SVM, then saves one model;
   `--model svm` saves the SVM, default saves Extra Trees)
3. `python realtime_demo.py`                 (live demo; loads models/gesture_model.joblib;
   press `p` for a play-along song test with a report card)

## Checking correctness (see EVALUATION.md)
- `python chord_judge.py`  - judge tested on real guitars, then on every chord we play -> reports/chord_check.png
- `python chord_lab.py`    - Play ours / Play real buttons + note meter for each chord
- `python song_test.py`    - songs played into the sound engine, checked bar by bar -> reports/song_*.png

## Reading the training output
- Random-split accuracy is optimistic (neighbouring frames leak across the split).
- Leave-one-person-out accuracy is the honest number: each person is held out in turn.
- Per-gesture recall and the confusion matrices show which gestures get mixed up.

## Files
- features.py       keypoints -> 39 geometric features; hand selection; drawing
- audio.py          Karplus-Strong chord synthesis + low-latency playback (`python audio.py` to test)
- collect_data.py   record labelled samples (writes data/<person>.csv)
- train_svm.py      evaluate + save the gesture model
- realtime_demo.py  full pipeline

## Still to build
- cnn_baseline.py: small CNN on cropped hand images; compare accuracy, FPS, latency, training time
- Robustness tests: lighting, distance, hand size, background
- Use a public dataset (e.g. Kaggle Fingers for gestures 1-5), as the professor requires

## Notes
- Features are rotation-invariant in the image plane, so rotating keypoints adds nothing;
  rotation augmentation only helps if you augment images before keypoint detection.
- Features are stored in the CSVs (not raw landmarks), so changing the features means re-recording.

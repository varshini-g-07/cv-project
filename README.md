# Gesture-controlled chord instrument (based on AIR Guitar, Ke & Chen)

Left hand shows a number gesture 1-7 -> SVM classifies it -> chord (C major, Nashville numbers).
Right hand pinches (thumb + index) -> the current chord is played.

## Setup
    pip install -r requirements.txt     # check MediaPipe's supported Python versions first

## Workflow
1. `python collect_data.py --person alice`   (each group member records their own data;
   record several sessions with different distance, tilt, lighting)
2. `python train_svm.py`                     (prints random-split vs leave-one-person-out accuracy)
3. `python realtime_demo.py`                 (live demo; FPS shown on screen)

## Files
- features.py       keypoints -> 39 geometric features; hand selection; pinch ratio
- audio.py          Karplus-Strong chord synthesis + playback (`python audio.py` to test sound)
- collect_data.py   record labelled samples
- train_svm.py      train + evaluate + save model
- realtime_demo.py  full pipeline

## Still to build (suggested split)
- cnn_baseline.py: small 3-layer CNN on cropped hand images; compare accuracy, FPS, latency, training time
- Robustness tests: lighting, distance, hand size, background
- Optional: hand detector (YOLO) in front of the pipeline; chord-shape mode

## Notes
- Written against MediaPipe's `mp.solutions.hands` API. If your installed version has removed
  it, switch to the newer MediaPipe Tasks HandLandmarker (needs a downloaded model file).
- Features are already rotation-invariant in the image plane, so rotating keypoints adds nothing;
  rotation augmentation only helps if you augment images before keypoint detection.
- Not yet run end to end on a webcam: the first thing to do is tune GESTURE_HAND/TRIGGER_HAND
  and the pinch thresholds.

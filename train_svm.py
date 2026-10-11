"""Train and evaluate the gesture classifier on data/*.csv.

Two accuracy numbers are printed for each model:
  random split          -> OPTIMISTIC: neighbouring video frames land on both sides of the split
  leave-one-person-out  -> HONEST: the model is tested on a person it never saw in training

Both Extra Trees and an SVM are evaluated every time, so you can compare them.
--model chooses which one is saved for the live demo (default: trees, as before).

Usage:  python train_svm.py               (saves the Extra Trees model)
        python train_svm.py --model svm   (saves the SVM instead)
"""
import argparse, glob, os
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_FILE = "models/gesture_model.joblib"
GESTURES = list(range(1, 8))


def load(folder="data"):
    X, y, g = [], [], []
    for path in sorted(glob.glob(f"{folder}/*.csv")):
        if os.path.getsize(path) == 0:
            print(f"Warning: {path} is empty, skipping it.")
            continue
        person = os.path.splitext(os.path.basename(path))[0]
        try:
            arr = np.loadtxt(path, delimiter=",", ndmin=2)
        except Exception as e:
            print(f"Warning: could not read {path} ({e}), skipping it.")
            continue
        if len(arr) > 0:
            X.append(arr[:, 1:])
            y.append(arr[:, 0].astype(int))
            g += [person] * len(arr)
    if not X:
        raise ValueError("No valid sample data found in data/*.csv")
    return np.vstack(X), np.concatenate(y), np.array(g)


def make_trees():
    return ExtraTreesClassifier(n_estimators=150, max_features="sqrt", random_state=42)


def make_svm():
    # The SVM needs its inputs scaled to similar ranges; tree models do not.
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))


MODELS = {"trees": ("Extra Trees", make_trees), "svm": ("SVM", make_svm)}


def evaluate(label, make_model, X, y, g, complete_people, show_matrices):
    print(f"\n===== {label} =====")

    # 1) random split (optimistic)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    split_acc = accuracy_score(yte, make_model().fit(Xtr, ytr).predict(Xte))
    print(f"Random-split accuracy:         {split_acc:.2%}  (optimistic)")

    # 2) leave-one-person-out (honest): hold each person out in turn
    if len(complete_people) < 2:
        print("Need at least 2 people with all 7 gestures for leave-one-person-out.")
        return
    mask = np.isin(g, complete_people)
    Xc, yc, gc = X[mask], y[mask], g[mask]
    pred = np.empty_like(yc)
    for person in complete_people:
        test = gc == person
        model = make_model().fit(Xc[~test], yc[~test])
        pred[test] = model.predict(Xc[test])

    print(f"Leave-one-person-out accuracy: {accuracy_score(yc, pred):.2%}  (honest)")
    cm = confusion_matrix(yc, pred, labels=GESTURES)
    recall = cm.diagonal() / cm.sum(axis=1)
    print("Per-gesture recall:  " + "   ".join(f"{k}: {r:.0%}" for k, r in zip(GESTURES, recall)))
    print("Per-person accuracy: " + "   ".join(
        f"{p}: {accuracy_score(yc[gc == p], pred[gc == p]):.0%}" for p in complete_people))

    if show_matrices:   # rows = true gesture, columns = predicted gesture
        print("\nConfusion matrix, all held-out people together:")
        print(cm)
        for p in complete_people:
            print(f"\nConfusion matrix, only {p} (held out):")
            print(confusion_matrix(yc[gc == p], pred[gc == p], labels=GESTURES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="trees",
                    help="which model to save for realtime_demo.py")
    args = ap.parse_args()

    X, y, g = load()
    people = sorted(set(g))
    print(f"Loaded {len(y)} samples across {len(people)} people ({', '.join(people)}).")
    print(f"Classes: {sorted(int(c) for c in set(y))}")

    # Only people who recorded all 7 gestures are used for evaluation and the saved model
    complete_people = []
    for p in people:
        got = sorted(int(c) for c in set(y[g == p]))
        if len(got) == 7:
            complete_people.append(p)
        else:
            print(f"Note: '{p}' has only {len(got)}/7 gestures {got}, so they are left out "
                  f"of the leave-one-person-out test and the saved model.")

    for key, (label, make_model) in MODELS.items():
        evaluate(label, make_model, X, y, g, complete_people, show_matrices=(key == args.model))

    # Fit the chosen model on all complete people and save it for the demo
    mask = np.isin(g, complete_people) if complete_people else np.ones(len(y), dtype=bool)
    name, make_model = MODELS[args.model]
    final_model = make_model().fit(X[mask], y[mask])
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, MODEL_FILE)
    print(f"\nSaved {MODEL_FILE} ({name}). realtime_demo.py will use this file.")


if __name__ == "__main__":
    main()

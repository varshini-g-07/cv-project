"""Train the gesture classifier on data/*.csv and report accuracy.

  random split          -> evaluates model accuracy on held-out test frames (~100%)
  leave-one-person-out  -> honest cross-validation across distinct people
"""
import glob, os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, train_test_split
from sklearn.ensemble import ExtraTreesClassifier


def load(folder="data"):
    X, y, g = [], [], []
    for path in sorted(glob.glob(f"{folder}/*.csv")):
        if os.path.getsize(path) == 0:
            continue
        person = os.path.splitext(os.path.basename(path))[0]
        try:
            arr = np.loadtxt(path, delimiter=",", ndmin=2)
            if len(arr) > 0:
                X.append(arr[:, 1:])
                y.append(arr[:, 0].astype(int))
                g += [person] * len(arr)
        except Exception:
            continue
    if not X:
        raise ValueError("No valid sample data found in data/*.csv")
    return np.vstack(X), np.concatenate(y), np.array(g)


def make_model():
    return ExtraTreesClassifier(n_estimators=150, max_features="sqrt", random_state=42)


def main():
    X, y, g = load()
    classes = sorted(set(y))
    people = sorted(set(g))
    print(f"Loaded {len(y)} samples across {len(people)} people ({', '.join(people)}).")
    print(f"Classes: {classes}\n")

    # Filter complete contributors (all 7 classes) for honest leave-one-person-out CV
    complete_people = []
    for p in people:
        p_classes = set(y[g == p])
        if len(p_classes) == 7:
            complete_people.append(p)
        else:
            print(f"Note: '{p}' has {len(p_classes)}/7 gestures recorded ({sorted(p_classes)}).")

    # Train-test random split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = make_model().fit(Xtr, ytr)
    split_acc = accuracy_score(yte, model.predict(Xte))
    print(f"\nRandom-split accuracy:         {split_acc * 100:.2f}% (Target: ~100%)")

    # Leave-one-person-out cross validation
    if len(complete_people) >= 2:
        mask = np.isin(g, complete_people)
        pred = cross_val_predict(make_model(), X[mask], y[mask], groups=g[mask], cv=LeaveOneGroupOut())
        loo_acc = accuracy_score(y[mask], pred)
        print(f"Leave-one-person-out accuracy: {loo_acc * 100:.2f}% (across {', '.join(complete_people)})")
        print("\nConfusion matrix (rows = true 1..7):")
        print(confusion_matrix(y[mask], pred))

    # Fit final production model on balanced data and save
    os.makedirs("models", exist_ok=True)
    if len(complete_people) >= 1:
        mask = np.isin(g, complete_people)
        final_model = make_model().fit(X[mask], y[mask])
    else:
        final_model = make_model().fit(X, y)
    joblib.dump(final_model, "models/svm.joblib")
    print("\nSaved models/svm.joblib successfully!\n")


if __name__ == "__main__":
    main()

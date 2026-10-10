"""Train the SVM on data/*.csv and report two kinds of accuracy.

  random split          -> optimistic (neighbouring video frames leak across the split)
  leave-one-person-out  -> honest estimate of how it works for a new person
"""
import glob, os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load(folder="data"):
    X, y, g = [], [], []
    for path in sorted(glob.glob(f"{folder}/*.csv")):
        person = os.path.splitext(os.path.basename(path))[0]
        arr = np.loadtxt(path, delimiter=",", ndmin=2)
        X.append(arr[:, 1:]); y.append(arr[:, 0].astype(int)); g += [person] * len(arr)
    return np.vstack(X), np.concatenate(y), np.array(g)


def make_model():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))


def main():
    X, y, g = load()
    print(f"{len(y)} samples, {len(set(g))} people, classes {sorted(set(y))}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    acc = accuracy_score(yte, make_model().fit(Xtr, ytr).predict(Xte))
    print(f"Random-split accuracy (optimistic): {acc:.3f}")

    if len(set(g)) >= 2:
        pred = cross_val_predict(make_model(), X, y, groups=g, cv=LeaveOneGroupOut())
        print(f"Leave-one-person-out accuracy:      {accuracy_score(y, pred):.3f}")
        print("Confusion matrix (rows = true 1..7):")
        print(confusion_matrix(y, pred))
    else:
        print("Record at least 2 people to get leave-one-person-out accuracy.")

    os.makedirs("models", exist_ok=True)
    joblib.dump(make_model().fit(X, y), "models/svm.joblib")
    print("Saved models/svm.joblib")


if __name__ == "__main__":
    main()

"""
train_model.py
--------------
Optional training script — trains a RandomForest classifier on collected
hand-landmark CSVs and saves the model to model/gesture_model.pkl.

Data format expected in data/ directory:
  Each CSV file is named after its label (e.g. "HELLO.csv", "YES.csv").
  Each row contains 63 floats: 21 landmarks × (x, y, z).

Usage:
  python train_model.py

The trained model is automatically picked up by main.py on next start.
"""

import os
import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(data_dir: str):
    """Load CSVs from data_dir.  Returns (X, y) numpy arrays."""
    X, y = [], []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv"):
            continue
        label = os.path.splitext(filename)[0].upper()
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 63:
                    continue
                X.append([float(v) for v in row[:63]])
                y.append(label)
    return np.array(X), np.array(y)


def train():
    if not os.path.isdir(DATA_DIR):
        print(f"[train] Data directory '{DATA_DIR}' not found.")
        print("  Create a 'data/' folder next to this script and put")
        print("  labelled CSVs there (one per sign, 63 floats per row).")
        return

    print(f"[train] Loading data from {DATA_DIR} ...")
    X, y = load_data(DATA_DIR)
    if len(X) == 0:
        print("[train] No data found. Exiting.")
        return

    print(f"[train] {len(X)} samples across {len(set(y))} classes: {sorted(set(y))}")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    labels = list(le.classes_)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Train
    print("[train] Training RandomForest ...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n[train] Classification report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "labels": labels}, f)
    print(f"\n[train] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()

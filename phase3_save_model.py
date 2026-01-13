import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("🔥 Phase 3.3 started")

# ---------------- CONFIG ----------------
FEATURES_DIR = "features"
MODELS_DIR = "models"
MERGED_FILE = os.path.join(FEATURES_DIR, "phase3_merged_dataset.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(MERGED_FILE)

# ---------------- LOAD MFCC FEATURES ----------------
X = []
y = []

for _, row in df.iterrows():
    mfcc_path = row["mfcc_file"]   # column from Phase 1
    label = row["true_label"]

    mfcc = np.load(mfcc_path)
    mfcc_mean = np.mean(mfcc, axis=1)  # same as Phase 3.2

    X.append(mfcc_mean)
    y.append(label)

X = np.array(X)
y = np.array(y)

# ---------------- ENCODE LABELS ----------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ---------------- TRAIN MODEL ----------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y_encoded)

# ---------------- SAVE ----------------
joblib.dump(model, os.path.join(MODELS_DIR, "emotion_vad_rf_model.pkl"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

print("💾 Model saved: models/emotion_vad_rf_model.pkl")
print("💾 Encoder saved: models/label_encoder.pkl")
print("✅ Phase 3.3 complete!")

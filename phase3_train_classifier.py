print("🔥 Phase 3.2 script started")
import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIG ----------------
DATA_PATH = "features/phase3_merged_dataset.csv"

print("📥 Loading merged dataset...")
df = pd.read_csv(DATA_PATH)

# ---------------- FEATURE EXTRACTION ----------------
X = []
y = []

print("🔧 Processing MFCC features...")

for _, row in df.iterrows():
    mfcc_path = row["mfcc_file"]

    # Load MFCC array
    mfcc = np.load(mfcc_path)

    # Statistical pooling (VERY important)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std])
    X.append(features)

    y.append(row["predicted_emotion"])

X = np.array(X)
y = np.array(y)

print("✅ Feature matrix shape:", X.shape)

# ---------------- LABEL ENCODING ----------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("📊 Training samples:", X_train.shape[0])
print("📊 Testing samples:", X_test.shape[0])

# ---------------- MODEL TRAINING ----------------
print("🚀 Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n✅ Phase 3.2 complete!")

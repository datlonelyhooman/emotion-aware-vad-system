import os
import joblib
import librosa
import numpy as np

MODEL_PATH = "models/emotion_vad_rf_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
AUDIO_DIR = "test_audio"

print("🔄 Loading model and encoder...")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

print("\n🎧 Running batch inference...\n")

for file in os.listdir(AUDIO_DIR):
    if file.endswith(".wav"):
        path = os.path.join(AUDIO_DIR, file)
        features = extract_features(path)
        pred = model.predict(features)
        emotion = label_encoder.inverse_transform(pred)[0]
        print(f"🗣️ {file} → {emotion}")

print("\n✅ Batch inference complete!")

import os
import numpy as np
import librosa
import joblib

# ---------------- CONFIG ----------------
MODEL_PATH = "models/emotion_vad_rf_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
TEST_AUDIO = r"C:\Users\Ria\Downloads\archive\Actor_01\03-01-01-01-01-01-01.wav"
SR = 16000
N_MFCC = 13

# ---------------- LOAD MODEL ----------------
print("🔄 Loading trained model...")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# ---------------- FEATURE EXTRACTION ----------------
def extract_mfcc(audio_path):
    signal, sr = librosa.load(audio_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean


# ---------------- INFERENCE ----------------
print("🎧 Running inference...")
features = extract_mfcc(TEST_AUDIO).reshape(1, -1)

pred = model.predict(features)
emotion = label_encoder.inverse_transform(pred)[0]

print("🎉 Prediction complete!")
print(f"🗣️ Detected Emotion: {emotion}")

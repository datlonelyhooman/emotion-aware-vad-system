import os
import librosa
import numpy as np
import joblib
import pandas as pd

# ---------------- CONFIG ----------------
AUDIO_DIR = "test_audio"   # put wav files here
MODEL_PATH = "models/emotion_vad_rf_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
OUTPUT_CSV = "phase4_batch_predictions.csv"
SAMPLE_RATE = 16000
N_MFCC = 13

print("🔄 Loading model and encoder...")
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

results = []

print("🎧 Running batch inference...")
for file in os.listdir(AUDIO_DIR):
    if not file.endswith(".wav"):
        continue

    path = os.path.join(AUDIO_DIR, file)

    # Load audio
    audio, sr = librosa.load(path, sr=SAMPLE_RATE)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)

    # Predict
    pred = model.predict(mfcc_mean)
    emotion = encoder.inverse_transform(pred)[0]

    results.append({
        "file": file,
        "predicted_emotion": emotion
    })

# Save results
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Batch inference complete!")
print(f"📄 Results saved to {OUTPUT_CSV}")

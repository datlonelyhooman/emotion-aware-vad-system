import os
import sys
import joblib
import numpy as np
import librosa

MODEL_PATH = "models/emotion_vad_rf_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 🔴 MUST BE 13
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

def main(folder_path):
    if not os.path.exists(folder_path):
        print("❌ Folder not found")
        return

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    print("\n🎧 Batch Emotion Analysis\n")

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            path = os.path.join(folder_path, file)
            features = extract_features(path)
            pred = model.predict(features)
            emotion = encoder.inverse_transform(pred)[0]

            print(f"{file} → 🗣️ {emotion}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase7_batch_inference.py <folder_path>")
    else:
        main(sys.argv[1])

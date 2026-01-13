import joblib
import librosa
import numpy as np
import sys
import os

MODEL_PATH = "models/emotion_vad_rf_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

def main():
    if len(sys.argv) != 2:
        print("❌ Usage: python phase6_cli_app.py <audio_file.wav>")
        return

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        print("❌ Audio file not found")
        return

    print("🔄 Loading model...")
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    print("🎧 Analyzing audio...")
    features = extract_features(audio_path)
    prediction = model.predict(features)
    emotion = encoder.inverse_transform(prediction)[0]

    print("\n🎉 RESULT")
    print(f"🗣️ Detected Emotion: {emotion}")

if __name__ == "__main__":
    main()

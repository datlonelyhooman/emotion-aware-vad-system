import streamlit as st
import joblib
import librosa
import numpy as np
import tempfile
import os

MODEL_PATH = "models/emotion_vad_rf_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

st.set_page_config(page_title="Emotion Detection", page_icon="🎧")

st.title("🎧 Speech Emotion Detection")
st.write("Upload a `.wav` file to detect emotion")

model, encoder = load_model()

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.audio(uploaded_file)

    if st.button("Analyze Emotion"):
        features = extract_features(temp_path)
        pred = model.predict(features)
        emotion = encoder.inverse_transform(pred)[0]

        st.success(f"🗣️ Detected Emotion: **{emotion.upper()}**")

        os.remove(temp_path)

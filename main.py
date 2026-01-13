import librosa
import librosa.display
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob

# ---------------------------
# Configuration
# ---------------------------
SAMPLE_RATE = 16000
N_MFCC = 13
FEATURES_DIR = "features"
PLOTS_DIR = "plots"  # optional folder for visualizations
TEST_SIZE = 0.2
RANDOM_STATE = 42
APPLY_NOISE_REDUCTION = False  # set True to reduce noise

# Path to your extracted RAVDESS dataset
DATASET_PATH = r"C:\Users\Ria\Downloads\archive"

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------
# Audio Processor Class
# ---------------------------
class AudioProcessor:
    def __init__(self, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def load_audio(self, file_path):
        signal, sr = librosa.load(file_path, sr=self.sr)
        return signal, sr

    def preprocess_audio(self, signal):
        """Optional noise reduction (can add library like noisereduce)."""
        # Currently disabled
        return signal

    def extract_mfcc(self, signal):
        return librosa.feature.mfcc(y=signal, sr=self.sr, n_mfcc=self.n_mfcc)

    def save_mfcc(self, mfcc, filename):
        path = os.path.join(FEATURES_DIR, filename)
        np.save(path, mfcc)
        return path

    def plot_waveform(self, signal, filename):
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(signal, sr=self.sr)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{filename}_waveform.png"))
        plt.close()

    def plot_mfcc(self, mfcc, filename):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, sr=self.sr, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title("MFCC")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{filename}_mfcc.png"))
        plt.close()


# ---------------------------
# Emotion Mapping
# ---------------------------
# RAVDESS filenames contain emotion codes in the format: 01-01-01-01-01-01-01.wav
# 3rd value indicates emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ---------------------------
# Process Audio & Extract Features
# ---------------------------
processor = AudioProcessor(sr=SAMPLE_RATE, n_mfcc=N_MFCC)
records = []

# Recursively find all .wav files in the dataset folder
audio_files = glob.glob(os.path.join(DATASET_PATH, "**", "*.wav"), recursive=True)
print(f"✅ Found {len(audio_files)} audio files.")

for i, file_path in enumerate(audio_files):
    try:
        # Extract emotion code from filename
        filename = os.path.basename(file_path)
        emotion_code = filename.split("-")[2]
        label = emotion_map.get(emotion_code, "unknown")
        base_filename = f"audio_{i}"

        # Load and preprocess
        signal, sr = processor.load_audio(file_path)
        signal = processor.preprocess_audio(signal)

        # Extract MFCC
        mfcc = processor.extract_mfcc(signal)
        save_name = f"mfcc_{i}.npy"
        save_path = processor.save_mfcc(mfcc, save_name)

        # Optional: save plots
        processor.plot_waveform(signal, base_filename)
        processor.plot_mfcc(mfcc, base_filename)

        # Record metadata
        records.append({
            "file": file_path,
            "mfcc_file": save_path,
            "label": label
        })

        if i % 50 == 0:
            print(f"Processed {i}/{len(audio_files)} files.")

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")

# ---------------------------
# Save Metadata CSV
# ---------------------------
df = pd.DataFrame(records)
metadata_csv = os.path.join(FEATURES_DIR, "mfcc_metadata.csv")
df.to_csv(metadata_csv, index=False)
print(f"✅ Saved metadata CSV at {metadata_csv}")

# ---------------------------
# Split into Train/Test
# ---------------------------
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df['label'], random_state=RANDOM_STATE)
train_csv = os.path.join(FEATURES_DIR, "train_metadata.csv")
test_csv = os.path.join(FEATURES_DIR, "test_metadata.csv")
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)
print(f"✅ Train/test split done. Train: {len(train_df)}, Test: {len(test_df)}")
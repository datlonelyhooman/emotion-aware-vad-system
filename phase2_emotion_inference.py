import os
import torch
import librosa
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm

# ---------------- CONFIG ----------------
MFCC_META_PATH = r"C:\Users\Ria\PycharmProjects\PythonProject\features\mfcc_metadata.csv"
OUTPUT_CSV = r"C:\Users\Ria\PycharmProjects\PythonProject\features\emotion_predictions.csv"
SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 5        # ⬅️ BIG SPEED BOOST
CHUNK_SAVE = 50

os.makedirs("features", exist_ok=True)

# ---------------- DEVICE ----------------
device = torch.device("cpu")  # Intel GPU not useful for torch
torch.set_num_threads(4)      # limit CPU thrashing

# ---------------- LOAD MODEL ----------------
print("🔍 Loading emotion model...")
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
).to(device)
model.eval()

# ---------------- RESUME SUPPORT ----------------
if os.path.exists(OUTPUT_CSV):
    done_df = pd.read_csv(OUTPUT_CSV)
    done_files = set(done_df["file"])
    print(f"🔁 Resuming from {len(done_files)} files")
else:
    done_df = pd.DataFrame(columns=["file", "true_label", "predicted_emotion"])
    done_files = set()
    print("🆕 Starting fresh")

# ---------------- PREDICTION FUNCTION ----------------
def predict_emotion(file_path):
    try:
        audio, _ = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=MAX_AUDIO_SEC
        )

        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        pred = torch.argmax(logits, dim=-1).item()
        return model.config.id2label[pred]

    except Exception as e:
        return f"error"

# ---------------- MAIN LOOP ----------------
meta_df = pd.read_csv(MFCC_META_PATH)
results_buffer = []

print(f"📂 Total files: {len(meta_df)}")

for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
    file_path = row["file"]
    label = row["label"]

    if file_path in done_files:
        continue

    emotion = predict_emotion(file_path)

    results_buffer.append({
        "file": file_path,
        "true_label": label,
        "predicted_emotion": emotion
    })

    if len(results_buffer) >= CHUNK_SAVE:
        temp = pd.DataFrame(results_buffer)
        done_df = pd.concat([done_df, temp], ignore_index=True)
        done_df.to_csv(OUTPUT_CSV, index=False)
        results_buffer.clear()
        print("💾 Chunk saved")

# Final save
if results_buffer:
    temp = pd.DataFrame(results_buffer)
    done_df = pd.concat([done_df, temp], ignore_index=True)
    done_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Phase 2 complete")

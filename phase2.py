import os
import torch
import librosa
import pandas as pd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm

# -------------------- CONFIG --------------------
DATASET_DIR = r"C:\Users\Ria\Downloads\archive\Actor_01" 
MFCC_META_PATH = r"C:\Users\Ria\PycharmProjects\PythonProject\features\mfcc_metadata.csv"
OUTPUT_CSV = "features/emotion_predictions.csv"
CHUNK_SIZE = 50  # save every 50 files

# -------------------- SAFE FOLDER CREATION --------------------
os.makedirs("features", exist_ok=True)

# -------------------- LOAD MODELS --------------------
print("🔍 Loading Wav2Vec2 processor + emotion model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
model.eval()

# -------------------- RESUME SUPPORT --------------------
if os.path.exists(OUTPUT_CSV):
    processed_df = pd.read_csv(OUTPUT_CSV)
    processed_files = set(processed_df["file"].tolist())
    print(f"🔁 Resuming: {len(processed_files)} files already processed.")
else:
    processed_df = pd.DataFrame(columns=["file", "true_label", "predicted_emotion"])
    processed_files = set()
    print("🆕 Starting fresh...")

# -------------------- EMOTION PREDICTION --------------------
def predict_emotion(file_path):
    try:
        speech, rate = librosa.load(file_path, sr=16000)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        return model.config.id2label[pred_id]
    except Exception as e:
        return f"Error: {e}"

# -------------------- MAIN LOOP --------------------
meta_df = pd.read_csv(MFCC_META_PATH)
print(f"📂 Total files in metadata: {len(meta_df)}")

new_rows = []

for i, row in tqdm(meta_df.iterrows(), total=len(meta_df)):

    file_path = row["file"]
    label = row["label"]

    # Skip already processed files
    if file_path in processed_files:
        continue

    # Predict emotion
    emotion = predict_emotion(file_path)

    # Store result
    new_rows.append({
        "file": file_path,
        "true_label": label,
        "predicted_emotion": emotion
    })

    # ---- SAVE PERIODICALLY (chunking) ----
    if len(new_rows) >= CHUNK_SIZE:
        temp_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([processed_df, temp_df], ignore_index=True)
        combined_df.to_csv(OUTPUT_CSV, index=False)
        processed_df = combined_df
        new_rows = []
        print("💾 Chunk saved...")

# -------------------- FINAL SAVE --------------------
if new_rows:
    temp_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([processed_df, temp_df], ignore_index=True)
    combined_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ FINAL SAVE done! File at: {OUTPUT_CSV}")
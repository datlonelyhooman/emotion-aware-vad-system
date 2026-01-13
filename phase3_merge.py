import pandas as pd
import os

# ---------------- CONFIG ----------------
FEATURES_DIR = "features"
MFCC_META = os.path.join(FEATURES_DIR, "mfcc_metadata.csv")
EMOTION_PRED = os.path.join(FEATURES_DIR, "emotion_predictions.csv")
OUTPUT_FILE = os.path.join(FEATURES_DIR, "phase3_merged_dataset.csv")

print("🔗 Loading Phase 1 (MFCC metadata)...")
mfcc_df = pd.read_csv(MFCC_META)

print("🔗 Loading Phase 2 (Emotion predictions)...")
emotion_df = pd.read_csv(EMOTION_PRED)

# ---------------- MERGE ----------------
print("🧩 Merging datasets on file path...")
merged_df = pd.merge(
    mfcc_df,
    emotion_df,
    on="file",
    how="inner"
)

# ---------------- SAVE ----------------
merged_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Phase 3.1 complete!")
print(f"📄 Merged dataset saved at: {OUTPUT_FILE}")
print(f"📊 Total samples: {len(merged_df)}")
print("📂 File exists:", os.path.exists(OUTPUT_FILE))

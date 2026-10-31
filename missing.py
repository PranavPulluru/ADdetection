import os
import csv

# =========================
# CONFIG
# =========================
LABELS_CSV = r"C:\Users\adipu\OneDrive\Desktop\ADmci_proj\labels.csv"
CLEANED_WAV_DIR = r"C:\Users\adipu\OneDrive\Desktop\ADmci_proj\data\processed\cleaned"

OUTPUT_CSV = os.path.join(CLEANED_WAV_DIR, "missing_wavs_root_only.csv")

FILENAME_COL = "filename"
SUFFIX = ".interviewee_only.wav"

# =========================
# Load labeled filenames (e.g. S001.wav)
# =========================
labeled = set()

with open(LABELS_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if FILENAME_COL not in reader.fieldnames:
        raise ValueError(
            f"CSV must contain column '{FILENAME_COL}'. "
            f"Found: {reader.fieldnames}"
        )

    for row in reader:
        if row[FILENAME_COL]:
            labeled.add(row[FILENAME_COL])  # EXACT string

# =========================
# Scan cleaned wavs
# =========================
missing_roots = set()

for name in os.listdir(CLEANED_WAV_DIR):
    if not name.lower().endswith(".wav"):
        continue
    if not name.endswith(SUFFIX):
        continue

    # Derive original root name (IN MEMORY ONLY)
    root_name = name.replace(SUFFIX, ".wav")

    if root_name not in labeled:
        missing_roots.add(root_name)  # store ROOT ONLY

missing = sorted(missing_roots)

# =========================
# Write output CSV
# =========================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([FILENAME_COL])
    for r in missing:
        writer.writerow([r])

# =========================
# Summary
# =========================
print(f"Labeled filenames in CSV: {len(labeled)}")
print(f"Missing ROOT wavs: {len(missing)}")
print(f"Wrote: {OUTPUT_CSV}")

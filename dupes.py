import os
from collections import defaultdict

ROOT_DIR = r"C:\Users\adipu\OneDrive\Desktop\ADmci_proj\data\raw\totalRaw"
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".aac")

# Maps
by_filename = defaultdict(list)
by_stem = defaultdict(list)

for dirpath, _, filenames in os.walk(ROOT_DIR):
    for name in filenames:
        if name.lower().endswith(AUDIO_EXTS) and not name.startswith("._"):
            full_path = os.path.join(dirpath, name)
            stem = os.path.splitext(name)[0]

            by_filename[name.lower()].append(full_path)
            by_stem[stem.lower()].append(full_path)

# === RESULTS ===
print("\n=== Exact duplicate filenames ===")
found = False
for name, paths in by_filename.items():
    if len(paths) > 1:
        found = True
        print(f"\n{name}")
        for p in paths:
            print("  ", p)

if not found:
    print("None")

print("\n=== Duplicate stems (same name, different extension or folder) ===")
found = False
for stem, paths in by_stem.items():
    if len(paths) > 1:
        found = True
        print(f"\n{stem}")
        for p in paths:
            print("  ", p)

if not found:
    print("None")

print("\nDone.")
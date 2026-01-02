import os
import numpy as np
import hashlib

# =====================
# PATHS
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGMEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "logmel")

# =====================
# HELPER FUNCTION
# =====================
def file_hash(path):
    """Compute SHA256 hash of a .npy file for comparison."""
    data = np.load(path)
    return hashlib.sha256(data.tobytes()).hexdigest()

# =====================
# MAIN DUPLICATE CHECK
# =====================
seen_hashes = {}
seen_names = {}
duplicates = []

for filename in os.listdir(LOGMEL_DIR):
    if not filename.lower().endswith(".npy"):
        continue
    
    file_path = os.path.join(LOGMEL_DIR, filename)
    
    # Check for duplicate names
    if filename in seen_names:
        duplicates.append((filename, "name"))
    else:
        seen_names[filename] = file_path
    
    # Check for duplicate content
    h = file_hash(file_path)
    if h in seen_hashes:
        duplicates.append((filename, "content"))
    else:
        seen_hashes[h] = file_path

# =====================
# REPORT
# =====================
if duplicates:
    print("Found duplicates:")
    for dup_file, dup_type in duplicates:
        print(f"{dup_file} → {dup_type}")
else:
    print("No duplicates found.")
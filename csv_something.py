import os
import sys
from pathlib import Path

# Import the standard library csv module explicitly to avoid shadowing
# (since this file is named csv.py, we need to import the stdlib module differently)
# Temporarily remove current directory from path to import the real csv module
_current_dir = os.path.dirname(os.path.abspath(__file__))
_removed_from_path = False
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
    _removed_from_path = True

# Import the real standard library csv module
import csv

# Restore the path if we removed it
if _removed_from_path:
    sys.path.insert(0, _current_dir)

# =========================
# CONFIGURATION
# =========================
# Get project root directory (one level up from scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output paths
DIARIZED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "diarized")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OG_LABELS_CSV = os.path.join(LOG_DIR, "ogLabels.csv")
OUTPUT_CSV = os.path.join(LOG_DIR, "file_labels.csv")

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# MAIN FUNCTION
# =========================
def create_labels_csv():
    """
    Create a CSV file with filenames from the diarized folder and their corresponding labels
    from ogLabels.csv.
    """
    print("=" * 60)
    print("Creating file labels CSV")
    print("=" * 60)
    
    # Step 1: Get all files from the diarized folder
    print(f"\n1. Reading files from: {DIARIZED_DIR}")
    if not os.path.exists(DIARIZED_DIR):
        print(f"   ❌ Error: Diarized directory does not exist: {DIARIZED_DIR}")
        return
    
    diarized_files = []
    if os.path.exists(DIARIZED_DIR):
        for file in os.listdir(DIARIZED_DIR):
            # Skip files that start with ._ (macOS resource fork files)
            if file.startswith('._'):
                continue
            file_path = os.path.join(DIARIZED_DIR, file)
            if os.path.isfile(file_path):
                diarized_files.append(file)
    
    if not diarized_files:
        print(f"   ⚠️  Warning: No files found in diarized folder")
        return
    
    print(f"   ✓ Found {len(diarized_files)} files")
    
    # Step 2: Load labels from ogLabels.csv
    print(f"\n2. Loading labels from: {OG_LABELS_CSV}")
    labels_dict = {}
    
    if os.path.exists(OG_LABELS_CSV):
        try:
            with open(OG_LABELS_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Get the column names
                fieldnames = reader.fieldnames
                print(f"   ✓ CSV columns found: {fieldnames}")
                
                # Try to identify filename and label columns
                # Common variations: filename, file_name, file, name, etc.
                filename_col = None
                label_col = None
                
                for col in fieldnames:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['file', 'name', 'filename']):
                        if filename_col is None:
                            filename_col = col
                    if any(keyword in col_lower for keyword in ['label', 'class', 'category']):
                        if label_col is None:
                            label_col = col
                
                # If not found, use first column as filename and second as label
                if filename_col is None:
                    filename_col = fieldnames[0] if fieldnames else None
                if label_col is None:
                    label_col = fieldnames[1] if len(fieldnames) > 1 else None
                
                print(f"   ✓ Using filename column: '{filename_col}'")
                print(f"   ✓ Using label column: '{label_col}'")
                
                # Read all rows and create a dictionary
                for row in reader:
                    if filename_col and label_col:
                        filename = row.get(filename_col, '').strip()
                        label = row.get(label_col, '').strip()
                        if filename:
                            labels_dict[filename] = label
                
                print(f"   ✓ Loaded {len(labels_dict)} label entries")
        except Exception as e:
            print(f"   ❌ Error reading ogLabels.csv: {e}")
    else:
        print(f"   ⚠️  Warning: ogLabels.csv not found at {OG_LABELS_CSV}")
        print(f"   Will create CSV with empty labels")
    
    # Step 3: Create output CSV with filenames and labels
    print(f"\n3. Creating output CSV: {OUTPUT_CSV}")
    
    matched_count = 0
    unmatched_count = 0
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        
        for filename in sorted(diarized_files):
            # Try to find matching label
            label = labels_dict.get(filename, '')
            
            # If exact match not found, try matching without extension
            if not label:
                base_name = os.path.splitext(filename)[0]
                # Try to find by base name (without extension)
                for key, value in labels_dict.items():
                    key_base = os.path.splitext(key)[0]
                    if key_base == base_name:
                        label = value
                        break
            
            # If still not found, try partial matching (filename contains key or vice versa)
            if not label:
                base_name = os.path.splitext(filename)[0]
                for key, value in labels_dict.items():
                    key_base = os.path.splitext(key)[0]
                    if base_name in key_base or key_base in base_name:
                        label = value
                        break
            
            if label:
                matched_count += 1
            else:
                unmatched_count += 1
                print(f"   ⚠️  No label found for: {filename}")
            
            writer.writerow([filename, label])
    
    print(f"\n   ✓ CSV created successfully!")
    print(f"   ✓ Matched labels: {matched_count}")
    print(f"   ⚠️  Unmatched files: {unmatched_count}")
    print(f"\n   Output saved to: {OUTPUT_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    create_labels_csv()


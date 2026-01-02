import os
import csv
import hashlib
from collections import defaultdict
from datetime import datetime

# =========================
# CONFIGURATION
# =========================
# Get project root directory (one level up from scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Diarized directory path
DIARIZED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "diarized")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DUPLICATES_CSV = os.path.join(LOG_DIR, "duplicates.csv")

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma', '.mp4', '.m4v']

# Create log directory
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# HELPER FUNCTIONS
# =========================
def file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of a file for content comparison.
    
    Args:
        file_path: Path to the file to hash
    
    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def save_duplicates_to_csv(
    duplicate_content: dict,
    duplicate_names: list,
    duplicate_names_case: list
):
    """
    Save duplicate file information to a CSV file.
    
    Args:
        duplicate_content: Dictionary mapping hash to list of (filename, file_path) tuples
        duplicate_names: List of (filename, original_path) tuples for exact name duplicates
        duplicate_names_case: List of (filename, original_name) tuples for case-insensitive duplicates
    """
    csv_rows = []
    
    # Add content duplicates
    for group_idx, (hash_value, files) in enumerate(duplicate_content.items(), 1):
        for filename, file_path in files:
            file_size = os.path.getsize(file_path)
            # Get other files in the same group (for reference)
            other_files = [f[0] for f in files if f[0] != filename]
            related_files = ", ".join(other_files) if other_files else ""
            
            csv_rows.append({
                "filename": filename,
                "file_path": file_path,
                "duplicate_type": "content",
                "group_id": f"content_group_{group_idx}",
                "related_files": related_files,
                "file_size_bytes": file_size,
                "file_size": format_size(file_size),
                "content_hash": hash_value[:16] + "..."  # First 16 chars for reference
            })
    
    # Add exact filename duplicates
    for dup_filename, original_path in duplicate_names:
        file_path = os.path.join(DIARIZED_DIR, dup_filename)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        csv_rows.append({
            "filename": dup_filename,
            "file_path": file_path,
            "duplicate_type": "exact_name",
            "group_id": "",
            "related_files": os.path.basename(original_path),
            "file_size_bytes": file_size,
            "file_size": format_size(file_size),
            "content_hash": ""
        })
    
    # Add case-insensitive filename duplicates
    for dup_filename, original_name in duplicate_names_case:
        if dup_filename != original_name:  # Only if actually different
            file_path = os.path.join(DIARIZED_DIR, dup_filename)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            csv_rows.append({
                "filename": dup_filename,
                "file_path": file_path,
                "duplicate_type": "case_insensitive_name",
                "group_id": "",
                "related_files": original_name,
                "file_size_bytes": file_size,
                "file_size": format_size(file_size),
                "content_hash": ""
            })
    
    # Write to CSV
    if csv_rows:
        with open(DUPLICATES_CSV, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["filename", "file_path", "duplicate_type", "group_id", 
                         "related_files", "file_size_bytes", "file_size", "content_hash"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"\n💾 Duplicate information saved to: {DUPLICATES_CSV}")
        print(f"   Total duplicate entries: {len(csv_rows)}\n")
    else:
        # Create empty CSV with headers if no duplicates
        with open(DUPLICATES_CSV, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["filename", "file_path", "duplicate_type", "group_id", 
                         "related_files", "file_size_bytes", "file_size", "content_hash"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print(f"\n💾 No duplicates found. Empty report saved to: {DUPLICATES_CSV}\n")

# =========================
# MAIN DUPLICATE CHECK
# =========================
def find_duplicates():
    """
    Find duplicate audio files in the diarized folder.
    Checks for both duplicate filenames and duplicate file content.
    """
    if not os.path.exists(DIARIZED_DIR):
        print(f"❌ Directory not found: {DIARIZED_DIR}")
        return
    
    # Dictionaries to track seen files
    seen_hashes = {}  # content_hash -> list of (filename, file_path)
    seen_names = {}  # filename -> first_file_path
    seen_names_lower = {}  # filename_lower -> (first_file_path, first_filename)
    
    # Lists to store duplicate information
    duplicate_content = {}  # content_hash -> list of (filename, file_path) for duplicates only
    duplicate_names = []  # list of (filename, original_filename)
    duplicate_names_case = []  # list of (filename, original_filename)
    
    # Get all audio files
    all_files = []
    for filename in os.listdir(DIARIZED_DIR):
        file_path = os.path.join(DIARIZED_DIR, filename)
        if os.path.isfile(file_path):
            # Check if it's a supported audio file
            if any(filename.lower().endswith(ext) for ext in SUPPORTED_AUDIO_EXTENSIONS):
                all_files.append((filename, file_path))
    
    if not all_files:
        print(f"No audio files found in {DIARIZED_DIR}")
        print(f"Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}")
        return
    
    print(f"Scanning {len(all_files)} audio file(s) in {DIARIZED_DIR}...\n")
    
    # Process each file
    for filename, file_path in all_files:
        # Check for exact duplicate filename
        if filename in seen_names:
            duplicate_names.append((filename, seen_names[filename]))
        else:
            seen_names[filename] = file_path
        
        # Check for case-insensitive duplicate filename
        filename_lower = filename.lower()
        if filename_lower in seen_names_lower:
            original_path, original_name = seen_names_lower[filename_lower]
            duplicate_names_case.append((filename, original_name))
        else:
            seen_names_lower[filename_lower] = (file_path, filename)
        
        # Check for duplicate content
        try:
            file_hash_value = file_hash(file_path)
            
            if file_hash_value in seen_hashes:
                # Add current file to the list
                seen_hashes[file_hash_value].append((filename, file_path))
                # Mark this hash as having duplicates
                duplicate_content[file_hash_value] = seen_hashes[file_hash_value]
            else:
                # First occurrence of this hash
                seen_hashes[file_hash_value] = [(filename, file_path)]
        except Exception as e:
            print(f"⚠️  Error processing {filename}: {e}")
            continue
    
    # =========================
    # REPORT RESULTS
    # =========================
    print("=" * 80)
    print("DUPLICATE AUDIO FILE REPORT")
    print("=" * 80)
    
    total_issues = len(duplicate_content) + len(duplicate_names) + len(duplicate_names_case)
    
    # Save duplicates to CSV
    save_duplicates_to_csv(duplicate_content, duplicate_names, duplicate_names_case)
    
    if total_issues == 0:
        print("\n✅ No duplicates found! All files are unique.\n")
        return
    
    # Report duplicate content
    if duplicate_content:
        print(f"\n📦 DUPLICATE CONTENT ({len(duplicate_content)} group(s)):")
        print("-" * 80)
        for idx, (hash_value, files) in enumerate(duplicate_content.items(), 1):
            print(f"\nGroup {idx} - Same content ({len(files)} file(s)):")
            for filename, file_path in files:
                file_size = os.path.getsize(file_path)
                print(f"  • {filename} ({format_size(file_size)})")
                print(f"    {file_path}")
    
    # Report exact duplicate filenames (shouldn't happen on same filesystem, but check anyway)
    if duplicate_names:
        print(f"\n📝 EXACT DUPLICATE FILENAMES ({len(duplicate_names)} pair(s)):")
        print("-" * 80)
        for dup_filename, original_path in duplicate_names:
            print(f"  • {dup_filename}")
            print(f"    Original: {original_path}")
    
    # Report case-insensitive duplicate filenames
    if duplicate_names_case:
        print(f"\n🔤 CASE-INSENSITIVE DUPLICATE FILENAMES ({len(duplicate_names_case)} pair(s)):")
        print("-" * 80)
        for dup_filename, original_name in duplicate_names_case:
            if dup_filename != original_name:  # Only show if actually different
                print(f"  • {dup_filename}")
                print(f"    Conflicts with: {original_name}")
    
    print("\n" + "=" * 80)
    print(f"Summary: {len(duplicate_content)} content duplicate group(s), "
          f"{len(duplicate_names)} exact filename duplicate(s), "
          f"{len(duplicate_names_case)} case-insensitive duplicate(s)")
    print("=" * 80 + "\n")

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    find_duplicates()


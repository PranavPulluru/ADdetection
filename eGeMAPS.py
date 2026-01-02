import os
import csv
import numpy as np
from typing import Optional
from datetime import datetime

# Try to import opensmile package
try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False
    print("⚠️  opensmile package not found. Please install it with: pip install opensmile")

# =========================
# CONFIGURATION
# =========================
# Get project root directory (one level up from scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output paths
DIARIZED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "diarized")
EGEMAPS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "eGeMAPS")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
FAILED_LOG = os.path.join(LOG_DIR, "failed_files.csv")

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma', '.mp4', '.m4v']

# Create output directories
os.makedirs(EGEMAPS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# HELPER FUNCTIONS
# =========================
def log_failure(filename: str, stage: str, error: str):
    """
    Append failed file information to failed_files.csv with timestamp and detailed error info.
    
    Args:
        filename: Name of the file that failed
        stage: Stage where failure occurred (e.g., "opensmile", "parse_output")
        error: Error message or description
    """
    file_exists = os.path.exists(FAILED_LOG)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get full file path for better debugging
    file_path = os.path.join(DIARIZED_DIR, filename)
    
    with open(FAILED_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "filename", "file_path", "stage", "error"])
        writer.writerow([timestamp, filename, file_path, stage, str(error)])


# =========================
# MAIN PROCESSING FUNCTION
# =========================
def process_audio_file(filename: str, smile) -> bool:
    """
    Process a single audio file to extract eGeMAPS features using openSMILE Python package.
    
    Args:
        filename: Name of the audio file to process
        smile: Initialized opensmile.Smile instance
    
    Returns:
        True if processing succeeded, False otherwise
    """
    input_path = os.path.join(DIARIZED_DIR, filename)
    base_name = os.path.splitext(filename)[0]
    
    # Output paths
    output_csv = os.path.join(EGEMAPS_DIR, f"{base_name}.csv")
    output_npy = os.path.join(EGEMAPS_DIR, f"{base_name}.npy")
    
    print(f"\nProcessing: {filename}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        log_failure(filename, "file_not_found", f"Input file does not exist: {input_path}")
        print(f"  ❌ Input file not found: {input_path}")
        return False
    
    # Extract features using openSMILE Python package
    try:
        # Process file - returns a pandas DataFrame
        features_df = smile.process_file(input_path)
        
        if features_df is None or features_df.empty:
            log_failure(filename, "extraction", "openSMILE returned empty features")
            print(f"  ❌ No features extracted")
            return False
        
        # Extract feature vector (flatten DataFrame to 1D array)
        # The DataFrame typically has one row with all features as columns
        feature_vector = features_df.values.flatten()
        
        if len(feature_vector) == 0:
            log_failure(filename, "extraction", "Feature vector is empty")
            print(f"  ❌ Feature vector is empty")
            return False
        
        # Save as CSV (preserve column names if available)
        features_df.to_csv(output_csv, index=False)
        
        # Save as numpy array for easier loading
        np.save(output_npy, feature_vector)
        
        print(f"  ✅ Extracted {len(feature_vector)} features | Saved to {base_name}.csv and {base_name}.npy")
        return True
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        log_failure(filename, "opensmile", error_msg)
        print(f"  ❌ Error extracting features: {error_msg}")
        return False

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("eGeMAPS Feature Extraction using openSMILE Python Package")
    print("=" * 60)
    print(f"Input directory: {DIARIZED_DIR}")
    print(f"Output directory: {EGEMAPS_DIR}")
    print(f"Log directory: {LOG_DIR}")
    print("=" * 60)
    
    # Check if opensmile package is available
    if not OPENSMILE_AVAILABLE:
        print("\n❌ opensmile package not found!")
        print("Please install it using:")
        print("  pip install opensmile")
        exit(1)
    
    # Initialize openSMILE with eGeMAPS feature set
    print("\nInitializing openSMILE with eGeMAPS feature set...")
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        print("✅ openSMILE initialized successfully")
    except Exception as e:
        print(f"\n❌ Failed to initialize openSMILE: {type(e).__name__}: {str(e)}")
        exit(1)
    
    # Check if input directory exists
    if not os.path.exists(DIARIZED_DIR):
        print(f"\n❌ Input directory does not exist: {DIARIZED_DIR}")
        print("Please run the diarization script first to generate audio files.")
        exit(1)
    
    # Get all supported audio files (skip files starting with ._)
    audio_files = [
        f for f in os.listdir(DIARIZED_DIR)
        if not f.startswith('._') and any(f.lower().endswith(ext) for ext in SUPPORTED_AUDIO_EXTENSIONS)
    ]
    
    if not audio_files:
        print(f"\n⚠️  No supported audio files found in {DIARIZED_DIR}")
        print(f"Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}")
    else:
        print(f"\nFound {len(audio_files)} audio file(s) to process\n")
        
        successful = 0
        failed = 0
        
        for filename in sorted(audio_files):
            if process_audio_file(filename, smile):
                successful += 1
            else:
                failed += 1
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {len(audio_files)}")
        print(f"Output directory: {EGEMAPS_DIR}")
        if failed > 0:
            print(f"Failed file log: {FAILED_LOG}")
        print("=" * 60)


import os
import csv
import torch
import torchaudio
from pyannote.audio import Pipeline
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime

# =========================
# CONFIGURATION
# =========================
# Get project root directory (one level up from scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output paths (absolute paths based on project root)
RAW_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "diarized")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
FAILED_LOG = os.path.join(LOG_DIR, "failed_files.csv")

# Hugging Face token for accessing pyannote models
HF_TOKEN = os.getenv("HF_TOKEN", "nope! it's a secret")

# Audio processing settings
SAMPLE_RATE = 16000

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma', '.mp4', '.m4v']

# Quality thresholds
MIN_TOTAL_SPEECH_SEC = 15.0  # Minimum total speech duration in seconds
MIN_DOMINANT_RATIO = 0.6  # Minimum ratio of dominant speaker to total speech (0.0-1.0)
MIN_FINAL_TOTAL_SEC = 10.0  # Minimum total duration after filtering

# Post-processing parameters
MIN_SEGMENT_DURATION = 0.2  # Minimum segment duration to keep (seconds)
MAX_GAP_TO_MERGE = 0.1  # Maximum gap between segments to merge them (seconds)
EDGE_BUFFER = 0.2  # Buffer to trim from segment edges (seconds) - ONLY applied to secondary speakers

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# DEVICE SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# LOAD DIARIZATION PIPELINE
# =========================
print("Loading speaker diarization pipeline...")
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )
    pipeline.to(device)
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    print("Please ensure HF_TOKEN is set correctly and you have access to the model.")
    raise

# =========================
# LOGGING FUNCTIONS
# =========================
def log_failure(filename: str, stage: str, error: str):
    """
    Append failed file information to failed_files.csv with timestamp and detailed error info.
    
    Args:
        filename: Name of the file that failed
        stage: Stage where failure occurred (e.g., "load_audio", "diarization")
        error: Error message or description
    """
    file_exists = os.path.exists(FAILED_LOG)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get full file path for better debugging
    file_path = os.path.join(RAW_AUDIO_DIR, filename)
    
    with open(FAILED_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "filename", "file_path", "stage", "error"])
        writer.writerow([timestamp, filename, file_path, stage, str(error)])

# =========================
# NEW ALGORITHM: TIME-BASED MASK APPROACH
# =========================

def create_speaker_mask(
    total_samples: int,
    segments: List[Tuple[float, float, str]],
    target_speaker: str
) -> torch.Tensor:
    """
    Create a binary mask for the target speaker using sample-level precision.
    This ensures clean separation - only samples where target speaker is active are marked.
    
    Args:
        total_samples: Total number of audio samples
        segments: List of (start_time, end_time, speaker_label) tuples
        target_speaker: The speaker label to create mask for
    
    Returns:
        Binary mask tensor (1 = target speaker, 0 = not target speaker)
    """
    mask = torch.zeros(total_samples, dtype=torch.bool)
    
    for start_time, end_time, speaker in segments:
        if speaker == target_speaker:
            start_sample = int(round(start_time * SAMPLE_RATE))
            end_sample = int(round(end_time * SAMPLE_RATE))
            
            # Ensure bounds are valid
            start_sample = max(0, min(start_sample, total_samples))
            end_sample = max(0, min(end_sample, total_samples))
            
            if end_sample > start_sample:
                mask[start_sample:end_sample] = True
    
    return mask

def identify_dominant_speaker(
    segments: List[Tuple[float, float, str]]
) -> Tuple[str, float, Dict[str, float]]:
    """
    Identify the dominant speaker using multiple validation methods.
    
    Args:
        segments: List of (start_time, end_time, speaker_label) tuples
    
    Returns:
        Tuple of (dominant_speaker_label, dominant_ratio, speaker_durations_dict)
    """
    speaker_durations = {}
    total_speech = 0.0
    
    # Calculate duration for each speaker
    for start, end, speaker in segments:
        duration = end - start
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
        total_speech += duration
    
    if total_speech == 0:
        return None, 0.0, {}
    
    # Find speaker with maximum duration
    dominant_speaker = max(speaker_durations, key=speaker_durations.get)
    dominant_duration = speaker_durations[dominant_speaker]
    dominant_ratio = dominant_duration / total_speech
    
    return dominant_speaker, dominant_ratio, speaker_durations

def filter_short_segments(
    segments: List[Tuple[float, float, str]],
    min_duration: float
) -> List[Tuple[float, float, str]]:
    """
    Remove segments shorter than minimum duration.
    
    Args:
        segments: List of (start_time, end_time, speaker_label) tuples
        min_duration: Minimum segment duration in seconds
    
    Returns:
        Filtered segments
    """
    return [
        (start, end, speaker)
        for start, end, speaker in segments
        if (end - start) >= min_duration
    ]

def merge_close_segments(
    segments: List[Tuple[float, float, str]],
    max_gap: float
) -> List[Tuple[float, float, str]]:
    """
    Merge segments of the same speaker that are very close together.
    Preserves natural pauses by only merging segments with gaps <= max_gap.
    
    Args:
        segments: List of (start_time, end_time, speaker_label) tuples
        max_gap: Maximum gap between segments to merge (seconds)
    
    Returns:
        Merged segments
    """
    if not segments:
        return []
    
    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    merged = [list(sorted_segments[0])]
    
    for current_start, current_end, current_speaker in sorted_segments[1:]:
        last_start, last_end, last_speaker = merged[-1]
        
        # Only merge if same speaker and gap is small
        if current_speaker == last_speaker:
            gap = current_start - last_end
            if gap <= max_gap:
                # Merge segments
                merged[-1][1] = current_end
            else:
                # Keep separate (natural pause preserved)
                merged.append([current_start, current_end, current_speaker])
        else:
            # Different speaker, keep separate
            merged.append([current_start, current_end, current_speaker])
    
    return [(s, e, sp) for s, e, sp in merged]

def apply_edge_buffer(
    segments: List[Tuple[float, float, str]],
    total_duration: float,
    buffer: float
) -> List[Tuple[float, float, str]]:
    """
    Apply a small buffer to segment edges to remove brief interjections.
    
    Args:
        segments: List of (start_time, end_time, speaker_label) tuples
        total_duration: Total audio duration in seconds
        buffer: Buffer size in seconds to trim from edges
    
    Returns:
        Segments with edges trimmed
    """
    buffered = []
    
    for start, end, speaker in segments:
        new_start = min(start + buffer, end - MIN_SEGMENT_DURATION)
        new_end = max(end - buffer, start + MIN_SEGMENT_DURATION)
        
        # Ensure valid bounds
        new_start = max(0, min(new_start, total_duration))
        new_end = max(0, min(new_end, total_duration))
        
        if new_end > new_start and (new_end - new_start) >= MIN_SEGMENT_DURATION:
            buffered.append((new_start, new_end, speaker))
    
    return buffered

def remove_overlapping_segments(
    dominant_segments: List[Tuple[float, float]],
    all_segments: List[Tuple[float, float, str]]
) -> List[Tuple[float, float]]:
    """
    Keep ALL dominant segments - do not remove any.
    The goal is to only remove secondary voice, not trim dominant voice.
    
    Args:
        dominant_segments: List of (start, end) tuples for dominant speaker
        all_segments: List of (start, end, speaker) tuples for all speakers
    
    Returns:
        All dominant segments (unchanged)
    """
    # Keep ALL dominant segments - don't remove any
    # The mask-based extraction will handle separation cleanly
    return dominant_segments.copy()

def extract_dominant_speaker_audio(
    waveform: torch.Tensor,
    segments: List[Tuple[float, float, str]],
    dominant_speaker: str
) -> torch.Tensor:
    """
    Extract audio for dominant speaker only, preserving natural pauses.
    Uses sample-level precision to ensure clean separation.
    
    Args:
        waveform: Input audio waveform (1D tensor)
        segments: List of (start_time, end_time, speaker_label) tuples
        dominant_speaker: Label of the dominant speaker
    
    Returns:
        Waveform with only dominant speaker audio, pauses preserved as silence
    """
    total_samples = len(waveform)
    
    # Create binary mask for dominant speaker
    dominant_mask = create_speaker_mask(total_samples, segments, dominant_speaker)
    
    # Create output waveform: silence everywhere, dominant speaker audio where mask is True
    output_waveform = torch.zeros_like(waveform)
    output_waveform[dominant_mask] = waveform[dominant_mask]
    
    return output_waveform

# =========================
# MAIN PROCESSING FUNCTION
# =========================
def process_audio_file(filename: str) -> bool:
    """
    Process a single audio file using the new algorithm.
    
    Args:
        filename: Name of the audio file to process
    
    Returns:
        True if processing succeeded, False otherwise
    """
    input_path = os.path.join(RAW_AUDIO_DIR, filename)
    
    # Output always as .wav regardless of input format
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}.wav")
    
    print(f"\nProcessing: {filename}")
    
    # -------------------------
    # Load and preprocess audio
    # -------------------------
    try:
        waveform, sr = torchaudio.load(input_path)
    except FileNotFoundError as e:
        log_failure(filename, "load_audio", f"File not found: {str(e)}")
        print(f"  ❌ Failed to load audio: File not found")
        return False
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        log_failure(filename, "load_audio", error_msg)
        print(f"  ❌ Failed to load audio: {error_msg}")
        return False
    
    # Resample if necessary
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    
    total_duration = len(waveform) / SAMPLE_RATE
    
    # -------------------------
    # Perform diarization
    # -------------------------
    try:
        audio_dict = {"waveform": waveform.unsqueeze(0), "sample_rate": SAMPLE_RATE}
        diarization_output = pipeline(audio_dict)
        annotation = diarization_output.speaker_diarization
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        log_failure(filename, "diarization", error_msg)
        print(f"  ❌ Diarization failed: {error_msg}")
        return False
    
    # -------------------------
    # Extract all segments with speaker labels
    # -------------------------
    all_segments = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        all_segments.append((segment.start, segment.end, speaker))
    
    if not all_segments:
        log_failure(filename, "no_segments", "No diarization segments detected")
        print(f"  ❌ No diarization segments detected")
        return False
    
    # -------------------------
    # Identify dominant speaker
    # -------------------------
    dominant_speaker, dominant_ratio, speaker_durations = identify_dominant_speaker(all_segments)
    
    if dominant_speaker is None:
        log_failure(filename, "no_dominant", "Could not identify dominant speaker")
        print(f"  ❌ Could not identify dominant speaker")
        return False
    
    # Calculate total speech duration
    total_speech = sum(speaker_durations.values())
    
    # Quality checks
    if total_speech < MIN_TOTAL_SPEECH_SEC:
        log_failure(filename, "too_short", f"Total speech: {total_speech:.1f}s (minimum: {MIN_TOTAL_SPEECH_SEC}s)")
        print(f"  ❌ Total speech too short: {total_speech:.1f}s")
        return False
    
    if dominant_ratio < MIN_DOMINANT_RATIO:
        log_failure(filename, "dominant_ratio", f"Ratio: {dominant_ratio:.2f} (minimum: {MIN_DOMINANT_RATIO})")
        print(f"  ❌ Dominant ratio too low: {dominant_ratio:.2f}")
        return False
    
    print(f"  Dominant speaker: {dominant_speaker} (ratio: {dominant_ratio:.2f})")
    
    # -------------------------
    # Filter and process segments
    # -------------------------
    # Step 1: Filter out very short segments
    filtered_segments = filter_short_segments(all_segments, MIN_SEGMENT_DURATION)
    
    if not filtered_segments:
        log_failure(filename, "filtered_out", "All segments filtered out")
        print(f"  ❌ All segments filtered out")
        return False
    
    # Step 2: Merge very close segments of the same speaker
    merged_segments = merge_close_segments(filtered_segments, MAX_GAP_TO_MERGE)
    
    # Step 3: Apply edge buffer ONLY to secondary speakers (not dominant)
    # This preserves all dominant speaker audio while removing brief interjections from secondary
    buffered_segments = []
    for start, end, speaker in merged_segments:
        if speaker == dominant_speaker:
            # Keep dominant speaker segments untouched - no edge buffering
            buffered_segments.append((start, end, speaker))
        else:
            # Apply edge buffer only to secondary speakers
            new_start = min(start + EDGE_BUFFER, end - MIN_SEGMENT_DURATION)
            new_end = max(end - EDGE_BUFFER, start + MIN_SEGMENT_DURATION)
            new_start = max(0, min(new_start, total_duration))
            new_end = max(0, min(new_end, total_duration))
            if new_end > new_start and (new_end - new_start) >= MIN_SEGMENT_DURATION:
                buffered_segments.append((new_start, new_end, speaker))
    
    if not buffered_segments:
        log_failure(filename, "edge_buffering", "All segments removed after edge buffering")
        print(f"  ❌ All segments removed after edge buffering")
        return False
    
    # Step 4: Extract dominant speaker segments only
    dominant_segments = [
        (start, end)
        for start, end, speaker in buffered_segments
        if speaker == dominant_speaker
    ]
    
    if not dominant_segments:
        log_failure(filename, "no_dominant_segments", "No dominant speaker segments found after filtering")
        print(f"  ❌ No dominant speaker segments found after filtering")
        return False
    
    # Step 5: Remove highly overlapping segments (conservative)
    # Label segments for overlap detection
    labeled_segments = [
        (start, end, "dominant" if speaker == dominant_speaker else "secondary")
        for start, end, speaker in buffered_segments
    ]
    
    cleaned_dominant = remove_overlapping_segments(dominant_segments, labeled_segments)
    
    if not cleaned_dominant:
        log_failure(filename, "overlap_removal", "All segments removed due to overlaps")
        print(f"  ❌ All segments removed due to overlaps")
        return False
    
    # Step 6: Final validation
    final_duration = sum(end - start for start, end in cleaned_dominant)
    if final_duration < MIN_FINAL_TOTAL_SEC:
        log_failure(
            filename,
            "final_duration",
            f"Final duration: {final_duration:.1f}s (minimum: {MIN_FINAL_TOTAL_SEC}s)"
        )
        print(f"  ❌ Final duration too short: {final_duration:.1f}s")
        return False
    
    # -------------------------
    # Extract dominant speaker audio with natural pauses
    # -------------------------
    # Convert cleaned segments back to labeled format for mask creation
    final_segments = [
        (start, end, dominant_speaker)
        for start, end in cleaned_dominant
    ]
    
    # Extract audio using sample-level mask (ensures clean separation)
    output_waveform = extract_dominant_speaker_audio(waveform, final_segments, dominant_speaker)
    
    # -------------------------
    # Save output
    # -------------------------
    try:
        torchaudio.save(output_path, output_waveform.unsqueeze(0), SAMPLE_RATE)
        
        # Statistics
        num_segments = len(cleaned_dominant)
        original_dominant_dur = sum(end - start for start, end in dominant_segments)
        
        print(
            f"  ✅ Saved | ratio: {dominant_ratio:.2f} "
            f"| original: {original_dominant_dur:.1f}s "
            f"| final: {final_duration:.1f}s "
            f"| segments: {num_segments}"
        )
        return True
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)} | Output path: {output_path}"
        log_failure(filename, "save_audio", error_msg)
        print(f"  ❌ Failed to save audio: {error_msg}")
        return False

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("Speaker Diarization Script (Redesigned Algorithm)")
    print("=" * 60)
    print(f"Input directory: {RAW_AUDIO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Quality thresholds:")
    print(f"  - Minimum total speech: {MIN_TOTAL_SPEECH_SEC}s")
    print(f"  - Minimum dominant ratio: {MIN_DOMINANT_RATIO}")
    print(f"  - Minimum final duration: {MIN_FINAL_TOTAL_SEC}s")
    print("=" * 60)
    
    # Get all supported audio files
    audio_files = [
        f for f in os.listdir(RAW_AUDIO_DIR)
        if any(f.lower().endswith(ext) for ext in SUPPORTED_AUDIO_EXTENSIONS)
    ]
    
    if not audio_files:
        print(f"No supported audio files found in {RAW_AUDIO_DIR}")
        print(f"Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}")
    else:
        print(f"\nFound {len(audio_files)} audio file(s) to process\n")
        
        successful = 0
        failed = 0
        
        for filename in sorted(audio_files):
            if process_audio_file(filename):
                successful += 1
            else:
                failed += 1
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {len(audio_files)}")
        print(f"Failed file log: {FAILED_LOG}")
        print("=" * 60)

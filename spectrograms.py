import os
import torch
import torchaudio
import numpy as np
from typing import List, Tuple
from datetime import datetime

# =========================
# CONFIGURATION
# =========================
# Get project root directory (one level up from scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output paths
DIARIZED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "diarized")
LOGMEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "logmel")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Audio processing settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 10.0  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# Mel spectrogram parameters
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 0.0
FMAX = SAMPLE_RATE / 2.0

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma', '.mp4', '.m4v']

# Create output directories
os.makedirs(LOGMEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# MEL SPECTROGRAM TRANSFORM
# =========================
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    power=2.0  # Power spectrogram (magnitude squared)
)

# =========================
# PROCESSING FUNCTIONS
# =========================
def segment_audio(waveform: torch.Tensor, sample_rate: int, chunk_duration: float) -> List[torch.Tensor]:
    """
    Segment audio waveform into fixed-duration chunks.
    The last chunk may be shorter than chunk_duration if the audio is not evenly divisible.
    
    Args:
        waveform: Input audio waveform (1D tensor)
        sample_rate: Sample rate of the audio
        chunk_duration: Duration of each chunk in seconds
    
    Returns:
        List of waveform chunks (each as 1D tensor, last one may be shorter)
    """
    chunk_samples = int(sample_rate * chunk_duration)
    total_samples = len(waveform)
    
    chunks = []
    start_idx = 0
    
    while start_idx < total_samples:
        end_idx = min(start_idx + chunk_samples, total_samples)
        chunk = waveform[start_idx:end_idx]
        
        chunks.append(chunk)
        start_idx = end_idx
    
    return chunks

def waveform_to_logmel(waveform: torch.Tensor) -> np.ndarray:
    """
    Convert audio waveform to log-mel spectrogram.
    
    Args:
        waveform: Input audio waveform (1D tensor)
    
    Returns:
        Log-mel spectrogram as numpy array (n_mels, time_frames)
    """
    # Ensure waveform is 2D (batch, samples) for the transform
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Compute mel spectrogram
    mel_spec = mel_spectrogram(waveform)
    
    # Convert to log scale (add small epsilon to avoid log(0))
    log_mel_spec = torch.log10(mel_spec + 1e-10)
    
    # Convert to numpy and squeeze batch dimension
    log_mel_spec_np = log_mel_spec.squeeze(0).numpy()
    
    return log_mel_spec_np

def process_audio_file(filename: str) -> Tuple[int, int]:
    """
    Process a single audio file: segment into chunks and convert to log-mel spectrograms.
    
    Args:
        filename: Name of the audio file to process
    
    Returns:
        Tuple of (successful_chunks, total_chunks)
    """
    input_path = os.path.join(DIARIZED_DIR, filename)
    base_name = os.path.splitext(filename)[0]
    
    print(f"\nProcessing: {filename}")
    
    # Load audio
    try:
        waveform, sr = torchaudio.load(input_path)
    except Exception as e:
        print(f"  ❌ Failed to load audio: {type(e).__name__}: {str(e)}")
        return (0, 0)
    
    # Resample if necessary
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    
    # Segment into 10-second chunks
    chunks = segment_audio(waveform, SAMPLE_RATE, CHUNK_DURATION)
    total_chunks = len(chunks)
    successful = 0
    
    print(f"  Found {total_chunks} chunk(s)")
    
    # Process each chunk
    for chunk_idx, chunk in enumerate(chunks):
        try:
            # Convert to log-mel spectrogram
            logmel = waveform_to_logmel(chunk)
            
            # Save as .npy file
            output_filename = f"{base_name}_chunk_{chunk_idx:04d}.npy"
            output_path = os.path.join(LOGMEL_DIR, output_filename)
            np.save(output_path, logmel)
            
            successful += 1
        except Exception as e:
            print(f"  ⚠️  Failed to process chunk {chunk_idx}: {type(e).__name__}: {str(e)}")
    
    print(f"  ✅ Successfully processed {successful}/{total_chunks} chunk(s)")
    
    return (successful, total_chunks)

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("Audio Segmentation and Log-Mel Spectrogram Generation")
    print("=" * 60)
    print(f"Input directory: {DIARIZED_DIR}")
    print(f"Output directory: {LOGMEL_DIR}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Chunk duration: {CHUNK_DURATION} seconds")
    print(f"Mel spectrogram parameters:")
    print(f"  - N_FFT: {N_FFT}")
    print(f"  - Hop length: {HOP_LENGTH}")
    print(f"  - N_MELS: {N_MELS}")
    print("=" * 60)
    
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
        
        total_successful = 0
        total_chunks = 0
        
        for filename in sorted(audio_files):
            successful, chunks = process_audio_file(filename)
            total_successful += successful
            total_chunks += chunks
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        print(f"Total chunks processed: {total_successful}/{total_chunks}")
        print(f"Output directory: {LOGMEL_DIR}")
        print("=" * 60)


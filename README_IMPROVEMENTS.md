# Improved Audio Preprocessing for Alzheimer's Disease Prediction

## Overview
This improved preprocessing script addresses inconsistencies in speaker diarization to ensure only the dominant speaker's (patient's) speech is preserved for eGeMAPS feature extraction.

## Key Improvements

### 1. **Post-Processing Filters**
   - **Minimum Segment Duration**: Removes dominant speaker segments shorter than 0.3 seconds (likely misclassified brief interjections)
   - **Gap Merging**: Merges dominant segments if the gap between them is less than 0.5 seconds (preserves natural pauses while removing artificial breaks)
   - **Edge Trimming**: Trims 0.5 seconds from the beginning and end of each segment to remove brief interviewer interjections

### 2. **Overlap Detection and Removal**
   - Identifies segments where dominant and secondary speakers overlap
   - Removes dominant segments that have >30% overlap with secondary speaker segments
   - Handles cases where diarization incorrectly labels overlapping speech

### 3. **Removed Secondary Speaker Saving**
   - No longer saves secondary speaker audio or timestamps (as per your requirements)
   - Focuses solely on preserving dominant speaker audio with natural pauses

### 4. **Enhanced Validation**
   - Validates total duration after all filtering steps
   - Ensures minimum final duration of 10 seconds
   - Comprehensive error logging at each stage

## Configuration Parameters

```python
MIN_DOMINANT_SEGMENT_SEC = 0.3  # Remove segments shorter than this
MAX_GAP_TO_MERGE_SEC = 0.5      # Merge segments if gap < this
EDGE_TRIM_SEC = 0.5              # Trim from segment edges
MIN_FINAL_TOTAL_SEC = 10.0       # Minimum total after filtering
```

## Usage

```bash
python scripts/preprocess_audio_improved.py
```

## How It Works

1. **Diarization**: Uses Pyannote speaker-diarization-3.1 to identify speakers
2. **Dominant Speaker Identification**: Finds speaker with most total speech time
3. **Segment Filtering**: Removes very short segments (< 0.3s)
4. **Gap Merging**: Merges close segments to preserve natural pauses
5. **Overlap Removal**: Removes segments with significant secondary speaker overlap
6. **Edge Trimming**: Trims edges to remove brief interjections
7. **Final Validation**: Ensures minimum duration requirements
8. **Audio Generation**: Creates waveform with only dominant speaker, preserving natural pauses

## Expected Results

- **Consistent Output**: All files should contain only dominant speaker audio
- **Natural Pauses**: Pauses between speech segments are preserved
- **No Interjections**: Brief interviewer speech at edges and interjections are removed
- **Better eGeMAPS Features**: More accurate speech-to-silence ratio calculations

## Troubleshooting

If you still see inconsistencies:

1. **Adjust MIN_DOMINANT_SEGMENT_SEC**: Increase to 0.5s if brief interjections persist
2. **Increase EDGE_TRIM_SEC**: Try 0.75s or 1.0s for more aggressive edge trimming
3. **Decrease MAX_GAP_TO_MERGE_SEC**: Use 0.3s to be more conservative about merging
4. **Check Failed Log**: Review `logs/failed_files.csv` for problematic files

## Advanced Options

For even more robust filtering, consider:
- Using VAD (Voice Activity Detection) as a secondary validation
- Implementing spectral analysis to detect speaker changes
- Using multiple diarization models and voting
- Manual review of edge cases


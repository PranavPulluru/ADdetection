"""
Build chunk-level labels from subject-level labels.

This script matches chunk files (log-mel spectrograms and eGeMAPS features)
to their parent subject labels, creating a comprehensive CSV file for training.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Handle csv module shadowing (similar to csv.py)
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


def parse_labels_csv(labels_csv_path: Path) -> Dict[str, int]:
    """
    Read labels.csv and create a mapping from subject_id to label.
    
    Args:
        labels_csv_path: Path to labels.csv file
        
    Returns:
        Dictionary mapping subject_id (str) to label (int)
        
    Raises:
        FileNotFoundError: If labels_csv_path doesn't exist
        ValueError: If CSV format is invalid
    """
    labels_map = {}
    
    if not labels_csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv_path}")
    
    with open(labels_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Validate required columns
        if 'filename' not in reader.fieldnames or 'label' not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain 'filename' and 'label' columns. "
                f"Found columns: {reader.fieldnames}"
            )
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
            filename = row['filename'].strip()
            label_str = row['label'].strip()
            
            # Skip empty rows
            if not filename or not label_str:
                continue
            
            # Extract subject_id from filename (e.g., "S001.wav" -> "S001")
            # Remove common audio extensions
            subject_id = filename.replace('.wav', '').replace('.mp3', '').replace('.flac', '')
            
            # Convert label to int
            try:
                label = int(label_str)
                if label not in [0, 1]:
                    print(f"Warning: Row {row_num}: Label '{label}' is not 0 or 1, skipping")
                    continue
            except ValueError:
                print(f"Warning: Row {row_num}: Invalid label '{label_str}', skipping")
                continue
            
            if subject_id in labels_map:
                print(f"Warning: Subject '{subject_id}' appears multiple times, using last occurrence")
            
            labels_map[subject_id] = label
    
    return labels_map


def extract_subject_id_from_chunk(chunk_filename: str) -> Optional[str]:
    """
    Extract subject_id from chunk filename.
    
    Args:
        chunk_filename: Filename like "S001_chunk_0000.npy" or "S001_chunk_0000"
        
    Returns:
        Subject ID (e.g., "S001") or None if format is invalid
    """
    # Remove extension if present
    name_without_ext = Path(chunk_filename).stem
    
    # Split at "_chunk_"
    if '_chunk_' in name_without_ext:
        subject_id = name_without_ext.split('_chunk_')[0]
        return subject_id
    
    return None


def find_egemaps_file(egemaps_dir: Path, subject_id: str) -> Tuple[Optional[Path], bool]:
    """
    Find eGeMAPS file for a subject (subject-level, not chunk-level).
    Only looks for .npy files, ignores .csv files.
    
    Args:
        egemaps_dir: Directory containing eGeMAPS files
        subject_id: Subject ID like "S001" (without extension or chunk suffix)
        
    Returns:
        Tuple of (file_path, is_missing):
        - file_path: Path to eGeMAPS file if found, None otherwise
        - is_missing: True if file is missing, False otherwise
    """
    # Look for subject-level .npy file only (ignore .csv)
    npy_path = egemaps_dir / f"{subject_id}.npy"
    if npy_path.exists():
        return npy_path, False
    
    # File not found
    return None, True


def scan_chunk_files(logmel_dir: Path) -> list[Tuple[str, Path]]:
    """
    Scan logmel directory for all .npy chunk files.
    
    Args:
        logmel_dir: Directory containing log-mel spectrogram files
        
    Returns:
        List of tuples (chunk_base_name, full_path)
        chunk_base_name is like "S001_chunk_0000" (without extension)
    """
    chunks = []
    
    if not logmel_dir.exists():
        print(f"Warning: Logmel directory does not exist: {logmel_dir}")
        return chunks
    
    # Find all .npy files
    for npy_file in logmel_dir.glob("*.npy"):
        chunk_base_name = npy_file.stem  # Remove .npy extension
        chunks.append((chunk_base_name, npy_file))
    
    return chunks


def build_chunk_labels(
    labels_map: Dict[str, int],
    chunks: list[Tuple[str, Path]],
    logmel_dir: Path,
    egemaps_dir: Path
) -> list[Dict[str, any]]:
    """
    Build chunk labels by matching chunks to subject labels.
    
    Args:
        labels_map: Dictionary mapping subject_id to label
        chunks: List of (chunk_base_name, logmel_path) tuples
        logmel_dir: Directory containing log-mel files
        egemaps_dir: Directory containing eGeMAPS files
        
    Returns:
        List of dictionaries with chunk information
    """
    chunk_data = []
    
    for chunk_base_name, logmel_path in chunks:
        # Extract subject_id from chunk name
        subject_id = extract_subject_id_from_chunk(chunk_base_name)
        
        if subject_id is None:
            print(f"Warning: Could not extract subject_id from '{chunk_base_name}', skipping")
            continue
        
        # Look up label
        if subject_id not in labels_map:
            print(f"Warning: Subject '{subject_id}' not found in labels, skipping chunk '{chunk_base_name}'")
            continue
        
        label = labels_map[subject_id]
        
        # Find corresponding eGeMAPS file (subject-level, not chunk-level)
        egemaps_path, egemaps_missing = find_egemaps_file(egemaps_dir, subject_id)
        
        # Convert paths to absolute paths for consistency
        logmel_abs_path = logmel_path.resolve()
        egemaps_abs_path = egemaps_path.resolve() if egemaps_path else None
        
        chunk_data.append({
            'file_id': chunk_base_name,
            'subject_id': subject_id,
            'label': label,
            'logmel_path': str(logmel_abs_path),
            'egemaps_path': str(egemaps_abs_path) if egemaps_abs_path else '',
            'egemaps_missing': egemaps_missing
        })
    
    return chunk_data


def write_output_csv(chunk_data: list[Dict[str, any]], output_path: Path):
    """
    Write chunk data to output CSV file.
    
    Args:
        chunk_data: List of dictionaries with chunk information
        output_path: Path to output CSV file
    """
    if not chunk_data:
        print("Warning: No chunk data to write!")
        return
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['file_id', 'subject_id', 'label', 'logmel_path', 'egemaps_path', 'egemaps_missing']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chunk_data)
    
    print(f"Successfully wrote {len(chunk_data)} rows to {output_path}")


def print_summary(
    total_chunks: int,
    chunks_matched: int,
    chunks_skipped: int,
    chunks_missing_egemaps: int
):
    """
    Print summary statistics.
    
    Args:
        total_chunks: Total number of logmel chunk files found
        chunks_matched: Number of chunks successfully matched to labels
        chunks_skipped: Number of chunks skipped due to missing subject label
        chunks_missing_egemaps: Number of matched chunks missing eGeMAPS files
    """
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total logmel chunks found:        {total_chunks}")
    print(f"Chunks matched to labels:         {chunks_matched}")
    print(f"Chunks skipped (missing label):   {chunks_skipped}")
    print(f"Chunks missing eGeMAPS:           {chunks_missing_egemaps}")
    if total_chunks > 0:
        match_rate = (chunks_matched / total_chunks) * 100
        print(f"Match rate:                       {match_rate:.1f}%")
    print("="*60)


def main():
    """Main function to orchestrate the chunk label building process."""
    # Get project root directory (one level up from scripts folder)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(
        description="Build chunk-level labels from subject-level labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_chunk_labels.py
  python build_chunk_labels.py --labels_csv data/labels.csv --out_csv chunk_labels.csv
  python build_chunk_labels.py --logmel_dir "C:\\path\\to\\logmel" --egemaps_dir "C:\\path\\to\\egemaps"
        """
    )
    
    # Set default paths relative to PROJECT_ROOT
    default_labels_csv = os.path.join(PROJECT_ROOT, "labels.csv")
    default_logmel_dir = os.path.join(PROJECT_ROOT, "data", "processed", "logmel")
    default_egemaps_dir = os.path.join(PROJECT_ROOT, "data", "processed", "eGeMAPS")
    default_out_csv = os.path.join(PROJECT_ROOT, "chunk_labels.csv")
    
    parser.add_argument(
        '--labels_csv',
        type=str,
        default=default_labels_csv,
        help=f'Path to labels.csv file (default: {default_labels_csv})'
    )
    
    parser.add_argument(
        '--logmel_dir',
        type=str,
        default=default_logmel_dir,
        help=f'Directory containing log-mel spectrogram .npy files (default: {default_logmel_dir})'
    )
    
    parser.add_argument(
        '--egemaps_dir',
        type=str,
        default=default_egemaps_dir,
        help=f'Directory containing eGeMAPS feature files (.csv or .npy) (default: {default_egemaps_dir})'
    )
    
    parser.add_argument(
        '--out_csv',
        type=str,
        default=default_out_csv,
        help=f'Output CSV file path (default: {default_out_csv})'
    )
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects for cross-platform compatibility
    labels_csv_path = Path(args.labels_csv).expanduser().resolve()
    logmel_dir = Path(args.logmel_dir).expanduser().resolve()
    egemaps_dir = Path(args.egemaps_dir).expanduser().resolve()
    output_path = Path(args.out_csv).expanduser().resolve()
    
    print("Building chunk labels...")
    print(f"Labels CSV:    {labels_csv_path}")
    print(f"Logmel dir:    {logmel_dir}")
    print(f"eGeMAPS dir:   {egemaps_dir}")
    print(f"Output CSV:    {output_path}")
    print()
    
    try:
        # Step 1: Parse labels CSV
        print("Reading labels CSV...")
        labels_map = parse_labels_csv(labels_csv_path)
        print(f"Loaded {len(labels_map)} subject labels")
        
        # Step 2: Scan for chunk files
        print(f"\nScanning logmel directory for chunk files...")
        chunks = scan_chunk_files(logmel_dir)
        total_chunks = len(chunks)
        print(f"Found {total_chunks} chunk files")
        
        # Step 3: Build chunk labels
        print(f"\nMatching chunks to labels...")
        chunk_data = build_chunk_labels(labels_map, chunks, logmel_dir, egemaps_dir)
        
        # Calculate statistics
        chunks_matched = len(chunk_data)
        chunks_skipped = total_chunks - chunks_matched
        chunks_missing_egemaps = sum(1 for chunk in chunk_data if chunk['egemaps_missing'])
        
        # Step 4: Write output CSV
        print(f"\nWriting output CSV...")
        write_output_csv(chunk_data, output_path)
        
        # Step 5: Print summary
        print_summary(
            total_chunks=total_chunks,
            chunks_matched=chunks_matched,
            chunks_skipped=chunks_skipped,
            chunks_missing_egemaps=chunks_missing_egemaps
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import json
import subprocess
from typing import List, Tuple, Dict, Optional

from faster_whisper import WhisperModel

# Progress bar
from tqdm import tqdm

# =========================
# CONFIG
# =========================
RAW_AUDIO_DIR = r"C:\Users\adipu\OneDrive\Desktop\ADmci_proj\data\test\raw"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")

OUT_DIR = r"C:\Users\adipu\OneDrive\Desktop\ADmci_proj\data\test\cleaned"

MODEL_SIZE = "small"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

MERGE_GAP_SEC = 0.20
MIN_CUT_SEC = 0.25
MIN_KEEP_SEC = 0.25
REENCODE_WAV = True
TARGET_SR = 16000

WRITE_FAILURE_LOG = True
FAILURE_LOG_PATH = os.path.join(OUT_DIR, "failed_files.csv")

# If you don't have ffmpeg on PATH, set this to full path, e.g. r"C:\ffmpeg\bin\ffmpeg.exe"
FFMPEG_EXE = "ffmpeg"

# =========================
# Interviewer text rules
# =========================
QUESTION_STARTS = (
    "what", "why", "how", "when", "where", "who", "which",
    "can you", "could you", "would you", "do you", "did you", "are you",
    "tell me", "describe", "explain", "please", "let's", "lets",
)
PROMPT_PHRASES = (
    "tell me about", "describe the", "what do you see", "what's happening",
    "take a look", "look at", "let’s talk about", "lets talk about",
    "next", "okay", "alright", "all right", "now",
)
ACK_TOKENS = (
    "mm-hmm", "mhm", "uh-huh", "uh huh", "yeah", "yep", "right", "okay", "ok", "sure"
)

ACK_MAX_DURATION_SEC = 0.80
ACK_MAX_WORDS = 3


# =========================
# Utilities
# =========================
def run_cmd(cmd: List[str]) -> None:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except FileNotFoundError as e:
        # Most common on Windows: ffmpeg not installed / not on PATH
        raise FileNotFoundError(
            f"{e}\n\nLooks like the executable was not found. "
            f"If this is ffmpeg, either install it or set FFMPEG_EXE to the full path "
            f"(e.g. r'C:\\ffmpeg\\bin\\ffmpeg.exe')."
        )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")


def normalize_text(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def count_words(t: str) -> int:
    t0 = normalize_text(t)
    if not t0:
        return 0
    t0 = t0.replace("-", " ")
    parts = [p for p in re.split(r"\s+", t0) if p]
    return len(parts)


# =========================
# Whisper model (load once)
# =========================
_WHISPER_MODEL: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
    return _WHISPER_MODEL


# =========================
# 1) Transcribe
# =========================
def transcribe(audio_path: str) -> Dict:
    model = get_whisper_model()
    segs, info = model.transcribe(audio_path, word_timestamps=False, vad_filter=True)

    out = []
    for s in segs:
        out.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip()
        })
    return {"duration": float(info.duration), "segments": out}


# =========================
# 2) Rule-based interviewer detection
# =========================
def looks_like_question(t: str) -> bool:
    t0 = normalize_text(t)
    if not t0:
        return False
    if t0.endswith("?"):
        return True
    for s in QUESTION_STARTS:
        if t0.startswith(s + " ") or t0 == s:
            return True
    for p in PROMPT_PHRASES:
        if p in t0:
            return True
    return False


def looks_like_ack(t: str, dur: float) -> bool:
    if dur > ACK_MAX_DURATION_SEC:
        return False
    t0 = normalize_text(t)
    if not t0:
        return False
    if count_words(t0) > ACK_MAX_WORDS:
        return False
    t0_clean = re.sub(r"[^\w\s-]", "", t0)
    t0_clean = re.sub(r"\s+", " ", t0_clean)
    ack_norm = {
        re.sub(r"\s+", " ", a.replace("-", " ").lower()).strip()
        for a in ACK_TOKENS
    }
    return t0_clean.replace("-", " ") in ack_norm


def merge_intervals(intervals: List[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for st, en in intervals[1:]:
        if st <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return [(a, b) for a, b in merged]


def label_interviewer_segments(segments: List[Dict]) -> List[Tuple[float, float]]:
    cut = []
    for seg in segments:
        st, en, text = seg["start"], seg["end"], seg["text"]
        dur = en - st
        if dur <= 0:
            continue
        if looks_like_question(text) or looks_like_ack(text, dur):
            cut.append((st, en))
    cut = [(a, b) for a, b in cut if (b - a) >= MIN_CUT_SEC]
    return merge_intervals(cut, MERGE_GAP_SEC)


def invert_intervals(cut: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    if not cut:
        return [(0.0, duration)]
    keep = []
    cur = 0.0
    for st, en in cut:
        if st > cur:
            keep.append((cur, st))
        cur = max(cur, en)
    if cur < duration:
        keep.append((cur, duration))
    return [(a, b) for a, b in keep if (b - a) >= MIN_KEEP_SEC]


# =========================
# 3) Hard-cut + concat
# =========================
def export_kept(audio_path: str, keep: List[Tuple[float, float]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_dir = os.path.join(OUT_DIR, "tmp_segments")
    os.makedirs(tmp_dir, exist_ok=True)

    seg_files = []
    for i, (st, en) in enumerate(keep):
        seg_out = os.path.join(tmp_dir, f"{os.path.basename(out_path)}_{i:04d}.wav")
        run_cmd([
            FFMPEG_EXE, "-y",
            "-ss", f"{st:.3f}", "-to", f"{en:.3f}",
            "-i", audio_path,
            "-ac", "1", "-ar", str(TARGET_SR),
            seg_out
        ])
        seg_files.append(seg_out)

    concat_list = os.path.join(tmp_dir, f"{os.path.basename(out_path)}.concat.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in seg_files:
            f.write(f"file '{os.path.abspath(p)}'\n")

    run_cmd([
        FFMPEG_EXE, "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-ac", "1", "-ar", str(TARGET_SR),
        out_path
    ])


# =========================
# 4) Batch processing
# =========================
def list_audio_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(AUDIO_EXTENSIONS) and not name.startswith("._"):
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def log_failure(path: str, err: str) -> None:
    if not WRITE_FAILURE_LOG:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    header_needed = not os.path.exists(FAILURE_LOG_PATH)
    with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("filepath,error\n")
        f.write(f"\"{path}\",\"{err.replace(chr(10), ' ').replace(chr(13), ' ')}\"\n")


def process_file(audio_path: str) -> None:
    base = os.path.splitext(os.path.basename(audio_path))[0]
    tr = transcribe(audio_path)

    # Save transcript (optional; comment out if you want faster runs / less disk usage)
    transcript_path = os.path.join(OUT_DIR, f"{base}.transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(tr, f, indent=2)

    keep = invert_intervals(label_interviewer_segments(tr["segments"]), tr["duration"])

    out_audio = os.path.join(OUT_DIR, f"{base}.interviewee_only.wav")
    export_kept(audio_path, keep, out_audio)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    audio_files = list_audio_files(RAW_AUDIO_DIR)
    print(f"Found {len(audio_files)} audio files under: {RAW_AUDIO_DIR}")

    # Load whisper once (warm-up) with a progress bar spinner message
    print("Loading Whisper model...")
    _ = get_whisper_model()
    print("Whisper model loaded.")

    failures = 0

    # Main progress bar for files
    for path in tqdm(audio_files, desc="Processing audio files", unit="file"):
        try:
            process_file(path)
        except Exception as e:
            failures += 1
            log_failure(path, str(e))
            # Update progress bar postfix so you can see failures accumulating
            tqdm.write(f"ERROR on {os.path.basename(path)}: {e}")

    print(f"\nDone. Failures: {failures}")
    if WRITE_FAILURE_LOG and failures > 0:
        print(f"See failure log: {FAILURE_LOG_PATH}")


if __name__ == "__main__":
    main()

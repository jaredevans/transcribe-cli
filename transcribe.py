#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil
import tempfile
import re
import math
from pathlib import Path

# ---------- DEFAULT CONFIG ----------
DEFAULT_WHISPER_CLI = os.environ.get("WHISPER_CLI", "/opt/homebrew/bin/whisper-cli")

# Default model directory and priority
MODELS_DIR = Path("~/.models/whisper").expanduser()
V3_MODEL = MODELS_DIR / "ggml-large-v3.bin"
V2_MODEL = MODELS_DIR / "ggml-large-v2.bin"

if V2_MODEL.exists():
    DEFAULT_MODEL_PATH = str(V2_MODEL)
else:
    DEFAULT_MODEL_PATH = str(V3_MODEL)

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

# Dynamic threads: use half of available cores, minimum 4, maximum 12
CPU_COUNT = os.cpu_count() or 8
DEFAULT_THREADS = os.environ.get("THREADS", str(max(4, min(12, CPU_COUNT // 2))))
DEFAULT_BEAM_SIZE = os.environ.get("BEAM_SIZE", "8")
DEFAULT_DETECT_DURATION = os.environ.get("DETECT_DURATION", "30")

DEFAULT_MIN_DUR_MS = int(os.environ.get("MIN_DUR_MS", "500"))
DEFAULT_DEDUP_WINDOW_MS = int(os.environ.get("DEDUP_WINDOW_MS", "1500"))
DEFAULT_ALLOW_OVERLAP_MS = int(os.environ.get("ALLOW_OVERLAP_MS", "50"))

BASE_DECODE_ARGS = [
    "-mc", "256",
    "-ml", "60",
    "-tp", "0.50",
    "-tpi", "0.10",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Auto-detect spoken language, then transcribe (if English) or translate (if other) to SRT.")
    parser.add_argument("input", help="Input audio or video file")
    parser.add_argument("--threads", default=DEFAULT_THREADS, help=f"Number of threads (default: {DEFAULT_THREADS})")
    parser.add_argument("--beam-size", default=DEFAULT_BEAM_SIZE, help=f"Beam size (default: {DEFAULT_BEAM_SIZE})")
    parser.add_argument("--detect-duration", default=DEFAULT_DETECT_DURATION, help=f"Seconds for midpoint language sample (default: {DEFAULT_DETECT_DURATION})")
    parser.add_argument("--model-version", choices=["v2", "v3"], default="v2", help="Whisper model version to use (default: v2)")
    parser.add_argument("--no-post", action="store_true", help="Skip SRT post-processing")
    parser.add_argument("--transcribe", action="store_true", help="Force English transcription regardless of detected language")
    return parser.parse_args()

def check_requirements(whisper_cli, model_path, input_file):
    if not os.path.isfile(input_file):
        sys.exit(f"Error: file not found: {input_file}")
    if not os.access(whisper_cli, os.X_OK):
        sys.exit(f"Error: whisper-cli not executable: {whisper_cli}")
    if not os.path.isfile(model_path):
        sys.exit(f"Error: model file not found: {model_path}")
    if not shutil.which("ffmpeg"):
        sys.exit("Error: ffmpeg not in PATH")
    if not shutil.which("ffprobe"):
        sys.exit("Error: ffprobe not in PATH")

def get_duration(file_path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", file_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return float(output)
    except (subprocess.CalledProcessError, ValueError):
        return 0.0

def ms_to_timestamp(ms):
    if ms < 0: ms = 0
    ms = int(ms)
    total_seconds = ms // 1000
    remainder_ms = ms % 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{remainder_ms:03d}"

def parse_timestamp(ts_str):
    # HH:MM:SS,mmm
    try:
        parts = re.split(r'[:,]', ts_str)
        if len(parts) != 4:
            return 0
        h, m, s, ms = map(int, parts)
        return (h * 3600 + m * 60 + s) * 1000 + ms
    except ValueError:
        return 0

def normalize_text(t):
    if not t:
        return ""
    # Remove punctuation and lowercase
    t = re.sub(r'[^\w\s]', '', t).lower()
    # Collapse multiple spaces and strip
    return " ".join(t.split())

def rebalance_sentences(blocks, max_chars=88, min_dur_ms=500):
    """
    Advanced two-pass rebalancing:
    1. Split blocks that contain multiple sentences.
    2. Merge adjacent blocks that are fragments of the same sentence or small sentences
       that fit together within max_chars.
    """
    if not blocks:
        return []

    # Regex for sentence end: . ? ! followed by space or end of string.
    # Includes lookbehind for common abbreviations.
    abbrevs = r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bMs)(?<!\bJr)(?<!\bSr)(?<!\bSt)(?<!\bJan)(?<!\bFeb)(?<!\bMar)(?<!\bApr)(?<!\bJun)(?<!\bJul)(?<!\bAug)(?<!\bSep)(?<!\bOct)(?<!\bNov)(?<!\bDec)"
    sentence_end_re = re.compile(abbrevs + r'([.?!]+)(\s+|$)', re.IGNORECASE)

    # --- Pass 1: Split ---
    split_blocks = []
    for blk in blocks:
        text = blk['text'].strip()
        matches = list(sentence_end_re.finditer(text))
        has_internal_split = any(m.end() < len(text) for m in matches)
        
        if has_internal_split:
            start_time = blk['start']
            end_time = blk['end']
            duration = end_time - start_time
            full_text_len = len(text)
            
            last_idx = 0
            for match in matches:
                split_idx = match.end()
                if split_idx >= len(text): 
                    continue
                    
                sub_text = text[last_idx:split_idx].strip()
                if not sub_text:
                    continue
                
                # Assign time proportionally to character length, with a floor
                sub_dur = max(200, int(duration * (len(sub_text) / full_text_len)))
                split_blocks.append({
                    'start': start_time,
                    'end': min(end_time, start_time + sub_dur),
                    'text': sub_text
                })
                start_time += sub_dur
                last_idx = split_idx
            
            # Catch trailing fragment
            rem_text = text[last_idx:].strip()
            if rem_text:
                split_blocks.append({
                    'start': max(start_time, blk['start']),
                    'end': end_time,
                    'text': rem_text
                })
        else:
            split_blocks.append(blk)

    # --- Pass 2: Merge ---
    merged_blocks = []
    if not split_blocks:
        return []

    curr = split_blocks[0]
    for i in range(1, len(split_blocks)):
        nxt = split_blocks[i]
        
        # A block is finished if it ends in punctuation
        is_curr_finished = any(curr['text'].strip().endswith(p) for p in ['.', '!', '?', '"', '”'])
        
        # Check if they are very close in time (gap < 1.5s)
        gap = nxt['start'] - curr['end']
        is_close = gap < 1500
        combined_text = (curr['text'] + " " + nxt['text']).strip()
        
        # Normalize for repeat detection
        n_curr = normalize_text(curr['text'])
        n_nxt = normalize_text(nxt['text'])
        
        should_merge = False
        if len(combined_text) <= max_chars:
            # If they are constructive repeats (identical or one contained in other),
            # DO NOT merge them. This allows Pass 3 to dedup them while keeping 
            # the original timing of the first one.
            is_constructive_repeat = False
            if n_curr and n_nxt:
                if n_curr == n_nxt:
                    is_constructive_repeat = True
                elif (n_nxt in n_curr or n_curr in n_nxt) and is_close:
                    is_constructive_repeat = True

            if not is_constructive_repeat:
                if not is_curr_finished:
                    # Always merge if current is a fragment (it needs to be completed)
                    should_merge = True
                # Optional: also merge if they are very close and both are short
                # elif is_close and (len(curr['text']) < 40 and len(nxt['text']) < 40):
                #     should_merge = True

        if should_merge:
            curr['text'] = combined_text
            curr['end'] = nxt['end']
        else:
            merged_blocks.append(curr)
            curr = nxt
    
    merged_blocks.append(curr)

    # Final polish: ensure minimum duration and non-zero intervals
    for blk in merged_blocks:
        if (blk['end'] - blk['start']) < min_dur_ms:
            blk['end'] = blk['start'] + min_dur_ms
            
    return merged_blocks

def post_process_srt(srt_path, min_dur_ms, dedup_window_ms, allow_overlap_ms):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Normalize line endings and split by double newlines
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    raw_blocks = content.split('\n\n')
    
    parsed_blocks = []
    for block in raw_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        ts_line_idx = -1
        for i, line in enumerate(lines):
            if '-->' in line:
                ts_line_idx = i
                break
        
        if ts_line_idx == -1:
            continue

        ts_line = lines[ts_line_idx]
        text = "\n".join(l.strip() for l in lines[ts_line_idx+1:] if l.strip())
        
        if not text:
            continue

        ts_parts = ts_line.split('-->')
        start_ms = parse_timestamp(ts_parts[0].strip())
        end_ms = parse_timestamp(ts_parts[1].strip())
        
        parsed_blocks.append({'start': start_ms, 'end': end_ms, 'text': text})

    # Pass 1: Temporal cleanup
    cleaned_blocks = []
    have_prev = False
    prev_end = 0

    for blk in parsed_blocks:
        start, end, text = blk['start'], blk['end'], blk['text']
        if have_prev and start < (prev_end - allow_overlap_ms):
            start = prev_end
        cleaned_blocks.append({'start': start, 'end': end, 'text': text})
        prev_end = end
        have_prev = True

    # Pass 2: Rebalance (Split/Merge sentences)
    rebalanced_blocks = rebalance_sentences(cleaned_blocks, min_dur_ms=min_dur_ms)

    # Pass 3: Final Dedup (consecutive repeated text)
    final_blocks = []
    if rebalanced_blocks:
        curr = rebalanced_blocks[0]
        final_blocks.append(curr)
        for i in range(1, len(rebalanced_blocks)):
            nxt = rebalanced_blocks[i]
            
            n_curr = normalize_text(curr['text'])
            n_nxt = normalize_text(nxt['text'])
            
            is_repeat = False
            if not n_nxt:
                is_repeat = True
            elif n_curr:
                if n_curr == n_nxt:
                    is_repeat = True
                elif (n_nxt in n_curr or n_curr in n_nxt) and (nxt['start'] - curr['end'] < dedup_window_ms):
                    # Constructive repetition (subset/superset within window)
                    is_repeat = True

            if is_repeat:
                # Skip duplicate. Keep the first one's timing.
                continue

            final_blocks.append(nxt)
            curr = nxt

    # Write back to SRT
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, block in enumerate(final_blocks, 1):
            f.write(f"{i}\n")
            f.write(f"{ms_to_timestamp(block['start'])} --> {ms_to_timestamp(block['end'])}\n")
            f.write(f"{block['text']}\n\n")

def main():
    args = parse_args()
    
    input_file = args.input
    whisper_cli = DEFAULT_WHISPER_CLI
    
    # Model directory
    models_dir = Path("~/.models/whisper").expanduser()
    v3_model = models_dir / "ggml-large-v3.bin"
    v2_model = models_dir / "ggml-large-v2.bin"
    
    if args.model_version == "v2" and v2_model.exists():
        model_path = str(v2_model)
    elif args.model_version == "v3" and v3_model.exists():
        model_path = str(v3_model)
    elif v2_model.exists():
        model_path = str(v2_model)
    else:
        # Fallback to whatever exists or default
        model_path = str(v3_model) if v3_model.exists() else DEFAULT_MODEL_PATH
    
    check_requirements(whisper_cli, model_path, input_file)
    
    input_path = Path(input_file).resolve()
    basename = input_path.stem
    # If input has multiple extensions (like .tar.gz), stem only removes the last one. 
    # The bash script uses ${INPUT%.*}, which removes the last extension.
    # Python's Path.stem behaves similarly for simple cases (file.mp4 -> file).
    
    output_prefix = input_path.parent / basename
    output_srt = output_prefix.with_suffix(".srt")
    
    # Temp files cleanup
    temp_files = []
    
    try:
        # ---------- AUDIO EXTRACTION ----------
        ext = input_path.suffix.lower().lstrip('.')
        audio_src = str(input_path)
        
        if ext not in ['wav', 'mp3', 'ogg', 'flac']:
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav.close()
            temp_files.append(tmp_wav.name)
            
            print("🎙️  Extracting full audio with ffmpeg → 16 kHz mono WAV ...")
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(input_path), "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", tmp_wav.name
            ], check=True)
            audio_src = tmp_wav.name

        # ---------- LANGUAGE DETECTION ----------
        # Force C locale
        env = os.environ.copy()
        env["LC_ALL"] = "C"

        detected_lang = "auto"
        if args.transcribe:
            detected_lang = "en"
            print("✅ Forcing language: en (due to --transcribe)")
        else:
            print(f"🌍 Detecting spoken language (using {args.detect_duration}s sample from midpoint)...")
            
            tmp_detect = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_detect.close()
            temp_files.append(tmp_detect.name)
            
            duration_sec = get_duration(audio_src)
            start_time = 0
            if duration_sec > 0:
                start_time = int(duration_sec / 2)
                
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(start_time), "-t", args.detect_duration,
                "-i", audio_src, "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", tmp_detect.name
            ], check=True)
            
            # Run whisper detection
            detect_cmd = [
                whisper_cli, "-m", model_path, "-f", tmp_detect.name, "-dl", "-t", args.threads
            ]
            
            try:
                result = subprocess.run(detect_cmd, capture_output=True, text=True, env=env)
                output_log = result.stdout + result.stderr
            except subprocess.CalledProcessError as e:
                output_log = e.stdout + e.stderr if e.stdout else ""
                if e.stderr: output_log += e.stderr

            # Grep equivalent: language: [a-z]{2}
            match = re.search(r'language: ([a-z]{2})', output_log)
            if match:
                detected_lang = match.group(1)
                print(f"✅ Detected language: {detected_lang}")
            else:
                print("⚠️  Could not detect language — defaulting to 'auto'.")

        # ---------- TRANSLATE/TRANSCRIBE FULL AUDIO ----------
        action = "Transcribing" if detected_lang == "en" else "Translating"
        print()
        print(f"🎧 {action} '{input_path.name}' ({detected_lang} → English subtitles)...")
        print(f"Model: {Path(model_path).name}")
        print(f"Output: {output_srt.name}")
        print()
        
        # whisper-cli often has a hard limit of 8 for beam size (decoders)
        requested_beam = int(args.beam_size)
        if requested_beam > 8:
            print(f"⚠️  Note: Capping beam size at 8 (requested {requested_beam}) to match whisper-cli limits.")
            effective_beam = "8"
        else:
            effective_beam = str(requested_beam)

        # Primary run
        cmd = [
            whisper_cli,
            "-m", model_path,
            "-f", audio_src,
            "-l", detected_lang,
            "-osrt",
            "-of", str(output_prefix), 
            "-t", args.threads,
            "-bs", effective_beam,
            "--best-of", effective_beam
        ]
        
        if detected_lang != "en":
            cmd.append("-tr")
            
        cmd += BASE_DECODE_ARGS
        cmd += ["--prompt", "Transcribe accurately." if detected_lang == "en" else "Translate accurately."]
        
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError:
            print("⚠️ whisper-cli failed. Retrying once with safer defaults...")
            fallback_cmd = [
                whisper_cli,
                "-m", model_path,
                "-f", audio_src,
                "-l", detected_lang,
                "-osrt",
                "-of", str(output_prefix),
                "-t", args.threads,
                "-bs", "5",
                "-pp"
            ]
            if detected_lang != "en":
                fallback_cmd.append("-tr")
            
            subprocess.run(fallback_cmd, check=True, env=env)

        # ---------- POST-PROCESS SRT ----------
        if not args.no_post and output_srt.exists():
            post_process_srt(
                output_srt, 
                DEFAULT_MIN_DUR_MS, 
                DEFAULT_DEDUP_WINDOW_MS, 
                DEFAULT_ALLOW_OVERLAP_MS
            )

        print()
        print(f"✅ Done: {output_srt.name}")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Cleanup
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    main()

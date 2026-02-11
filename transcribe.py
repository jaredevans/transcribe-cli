#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil
import tempfile
import re
import math
from collections import Counter
from pathlib import Path

# ---------- DEFAULT CONFIG ----------
DEFAULT_WHISPER_CLI = os.environ.get("WHISPER_CLI", "/opt/homebrew/bin/whisper-cli")

# Default model directory and priority
MODELS_DIR = Path("~/.models/whisper").expanduser()
V3_MODEL = MODELS_DIR / "ggml-large-v3.bin"
V2_MODEL = MODELS_DIR / "ggml-large-v2.bin"

# Dynamic threads: use half of available cores, minimum 4, maximum 12
CPU_COUNT = os.cpu_count() or 8
DEFAULT_THREADS = int(os.environ.get("THREADS", max(4, min(12, CPU_COUNT // 2))))
DEFAULT_BEAM_SIZE = int(os.environ.get("BEAM_SIZE", 8))
DEFAULT_BEST_OF = int(os.environ.get("BEST_OF", 5))
DEFAULT_DETECT_DURATION = int(os.environ.get("DETECT_DURATION", 30))

# ---------- PROMPT PRESETS ----------
PROMPT_PRESETS = {
    "film":   "Dialogue from a film. Translate with natural, idiomatic English subtitles. Keep translations concise for readability.",
    "anime":  "Japanese anime dialogue. Translate with natural English subtitles. Preserve character names, honorifics, and attack names. Ignore background music and sound effects.",
    "street": "Informal recording with background noise. Translate conversational speech accurately, ignoring background audio.",
    "talk":   "Presentation or speech. Translate accurately, preserving names, titles, and key terminology.",
}

BASE_DECODE_PARAMS = {
    "-mc":  "256",
    "-ml":  "60",
    "-tp":  "0.50",
    "-tpi": "0.10",
}

PRESET_DECODE_OVERRIDES = {
    "anime": {
        "--no-speech-thold": "0.6",
        "--entropy-thold":   "1.8",
    },
    "street": {
        "-tp":               "0.40",
        "-tpi":              "0.08",
        "--no-speech-thold": "0.3",
        "--entropy-thold":   "1.8",
    },
    "talk": {
        "-tp":             "0.55",
        "--entropy-thold": "2.0",
    },
}

DEFAULT_MIN_DUR_MS = int(os.environ.get("MIN_DUR_MS", "500"))
DEFAULT_MAX_DUR_MS = int(os.environ.get("MAX_DUR_MS", "15000"))
DEFAULT_DEDUP_WINDOW_MS = int(os.environ.get("DEDUP_WINDOW_MS", "1500"))
DEFAULT_ALLOW_OVERLAP_MS = int(os.environ.get("ALLOW_OVERLAP_MS", "50"))
DEFAULT_MAX_CHARS = int(os.environ.get("MAX_CHARS", "120"))

# Whisper supported languages (ISO 639-1 codes)
SUPPORTED_LANGUAGES = {
    "af": "Afrikaans",    "am": "Amharic",      "ar": "Arabic",       "as": "Assamese",
    "az": "Azerbaijani",  "ba": "Bashkir",      "be": "Belarusian",   "bg": "Bulgarian",
    "bn": "Bengali",      "bo": "Tibetan",      "br": "Breton",       "bs": "Bosnian",
    "ca": "Catalan",      "cs": "Czech",         "cy": "Welsh",        "da": "Danish",
    "de": "German",       "el": "Greek",         "en": "English",      "es": "Spanish",
    "et": "Estonian",     "eu": "Basque",        "fa": "Persian",      "fi": "Finnish",
    "fo": "Faroese",     "fr": "French",        "gl": "Galician",     "gu": "Gujarati",
    "ha": "Hausa",       "haw": "Hawaiian",     "he": "Hebrew",       "hi": "Hindi",
    "hr": "Croatian",    "ht": "Haitian Creole","hu": "Hungarian",    "hy": "Armenian",
    "id": "Indonesian",  "is": "Icelandic",     "it": "Italian",      "ja": "Japanese",
    "jw": "Javanese",    "ka": "Georgian",      "kk": "Kazakh",       "km": "Khmer",
    "kn": "Kannada",     "ko": "Korean",        "la": "Latin",        "lb": "Luxembourgish",
    "ln": "Lingala",     "lo": "Lao",           "lt": "Lithuanian",   "lv": "Latvian",
    "mg": "Malagasy",    "mi": "Maori",         "mk": "Macedonian",   "ml": "Malayalam",
    "mn": "Mongolian",   "mr": "Marathi",       "ms": "Malay",        "mt": "Maltese",
    "my": "Myanmar",     "ne": "Nepali",        "nl": "Dutch",        "nn": "Nynorsk",
    "no": "Norwegian",   "oc": "Occitan",       "pa": "Punjabi",      "pl": "Polish",
    "ps": "Pashto",      "pt": "Portuguese",    "ro": "Romanian",     "ru": "Russian",
    "sa": "Sanskrit",    "sd": "Sindhi",        "si": "Sinhala",      "sk": "Slovak",
    "sl": "Slovenian",   "sn": "Shona",         "so": "Somali",       "sq": "Albanian",
    "sr": "Serbian",     "su": "Sundanese",     "sv": "Swedish",      "sw": "Swahili",
    "ta": "Tamil",       "te": "Telugu",        "tg": "Tajik",        "th": "Thai",
    "tk": "Turkmen",     "tl": "Tagalog",       "tr": "Turkish",      "tt": "Tatar",
    "uk": "Ukrainian",   "ur": "Urdu",          "uz": "Uzbek",        "vi": "Vietnamese",
    "yi": "Yiddish",     "yo": "Yoruba",        "zh": "Chinese",
}

def _format_lang_list():
    """Format supported languages for help text."""
    items = [f"{code} ({name})" for code, name in sorted(SUPPORTED_LANGUAGES.items())]
    return ", ".join(items)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-detect spoken language, then transcribe (if English) or translate (if other) to SRT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "prompt presets:\n"
            "  film     Dialogue from a film (default for translation)\n"
            "  anime    Japanese anime dialogue, preserves honorifics & names\n"
            "  street   Informal/noisy recording, tighter hallucination thresholds\n"
            "  talk     Presentation or speech, preserves terminology\n"
            "  (any other string is used as a custom prompt)\n"
            "\n"
            "examples:\n"
            "  transcribe.py movie.mkv                          Auto-detect language, translate to English\n"
            "  transcribe.py movie.mkv --prompt film             Explicit film preset (same as default)\n"
            "  transcribe.py episode.mkv --prompt anime          Anime preset with honorific preservation\n"
            "  transcribe.py dashcam.mp4 --prompt street         Noisy recording with tighter thresholds\n"
            "  transcribe.py lecture.mp4 --prompt talk            Lecture/speech preset\n"
            "  transcribe.py interview.mp4 --prompt \"Medical terminology. Preserve drug names.\"  Custom prompt\n"
            "  transcribe.py movie.mkv --beam-size 5 --best-of 3 Independent beam/best-of\n"
            "  transcribe.py movie.mkv --override-lang ja        Force Japanese detection\n"
            "  transcribe.py podcast.mp3 --transcribe            Force English transcription\n"
            "  transcribe.py movie.mkv --direct-transcribe       Keep original language in subtitles\n"
            "\n"
            f"supported languages for --override-lang:\n  {_format_lang_list()}"
        )
    )
    parser.add_argument("input", help="Input audio or video file")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=f"Number of threads (default: {DEFAULT_THREADS})")
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE, help=f"Beam size (default: {DEFAULT_BEAM_SIZE})")
    parser.add_argument("--best-of", type=int, default=DEFAULT_BEST_OF, help=f"Best-of candidates for decoding (default: {DEFAULT_BEST_OF})")
    parser.add_argument("--detect-duration", type=int, default=DEFAULT_DETECT_DURATION, help=f"Seconds for language detection sample (default: {DEFAULT_DETECT_DURATION})")
    parser.add_argument("--prompt", default=None,
                        help=f"Prompt preset ({', '.join(PROMPT_PRESETS.keys())}) or custom prompt string (default: film for translate, generic for transcribe)")
    parser.add_argument("--model-version", choices=["v2", "v3"], default="v2", help="Whisper model version to use (default: v2)")
    parser.add_argument("--no-post", action="store_true", help="Skip SRT post-processing")
    parser.add_argument("--transcribe", action="store_true", help="Force English transcription regardless of detected language")
    parser.add_argument("--override-lang", metavar="LANG", choices=SUPPORTED_LANGUAGES.keys(),
                        help="Override detected language with the specified language code (skips auto-detection)")
    parser.add_argument("--direct-transcribe", action="store_true",
                        help="Transcribe in the detected/overridden language without translating to English")
    return parser.parse_args()

def resolve_model_path(model_version):
    """Select model path based on version preference with fallbacks."""
    if model_version == "v2" and V2_MODEL.exists():
        return str(V2_MODEL)
    if model_version == "v3" and V3_MODEL.exists():
        return str(V3_MODEL)
    if V2_MODEL.exists():
        return str(V2_MODEL)
    if V3_MODEL.exists():
        return str(V3_MODEL)
    return os.environ.get("MODEL_PATH", str(V3_MODEL))

def resolve_prompt(prompt_arg, is_translate):
    """Resolve --prompt value to (preset_name_or_None, prompt_text)."""
    if prompt_arg is None:
        if is_translate:
            return "film", PROMPT_PRESETS["film"]
        return None, "Transcribe accurately."
    if prompt_arg in PROMPT_PRESETS:
        return prompt_arg, PROMPT_PRESETS[prompt_arg]
    return None, prompt_arg

def build_decode_args(preset_name):
    """Merge BASE_DECODE_PARAMS with any preset overrides into a flat CLI arg list."""
    params = {**BASE_DECODE_PARAMS, **PRESET_DECODE_OVERRIDES.get(preset_name or "", {})}
    return [item for k, v in params.items() for item in (k, v)]

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
        print("⚠️  Could not determine file duration via ffprobe — language detection will sample from file start.")
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
                elif (n_nxt in n_curr or n_curr in n_nxt) and is_close and min(len(n_curr), len(n_nxt)) > 40:
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

def post_process_srt(srt_path, min_dur_ms, max_dur_ms, dedup_window_ms, allow_overlap_ms, max_chars):
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

    # Pass 1: Temporal cleanup — fix overlaps and cap abnormally long durations
    cleaned_blocks = []
    have_prev = False
    prev_end = 0

    for blk in parsed_blocks:
        start, end, text = blk['start'], blk['end'], blk['text']
        if have_prev and start < (prev_end - allow_overlap_ms):
            start = prev_end
        # Cap duration — a single subtitle >15s is likely hallucination over silence/music
        if (end - start) > max_dur_ms:
            end = start + max_dur_ms
        cleaned_blocks.append({'start': start, 'end': end, 'text': text})
        prev_end = end
        have_prev = True

    # Pass 2: Rebalance (Split/Merge sentences)
    rebalanced_blocks = rebalance_sentences(cleaned_blocks, max_chars=max_chars, min_dur_ms=min_dur_ms)

    # Pass 3: Dedup — consecutive repeats and A-B-A-B alternating patterns
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
                elif (n_nxt in n_curr or n_curr in n_nxt) and (nxt['start'] - curr['end'] <= dedup_window_ms) and min(len(n_curr), len(n_nxt)) > 40:
                    # Constructive repetition (subset/superset within window)
                    is_repeat = True

            # Alternating pattern: check up to 3 positions back in final_blocks
            # Catches A-B-A-B, A-B-C-A, and similar short-cycle repetitions
            if not is_repeat:
                for lookback in range(2, min(4, len(final_blocks) + 1)):
                    if lookback > len(final_blocks):
                        break
                    prev_blk = final_blocks[-lookback]
                    n_prev = normalize_text(prev_blk['text'])
                    if n_prev and n_nxt == n_prev and (nxt['start'] - prev_blk['end'] <= dedup_window_ms):
                        is_repeat = True
                        break

            if is_repeat:
                # Skip duplicate. Keep the first one's timing.
                continue

            final_blocks.append(nxt)
            curr = nxt

    # Pass 4: Tighten start times on overlong subtitles
    # Whisper often starts segments early (during music/silence before dialogue).
    # Estimate comfortable reading duration from word count and shift start forward
    # so the subtitle appears closer to when the speech actually occurs.
    MS_PER_WORD = 400
    MIN_READING_MS = 1500
    OVERLONG_RATIO = 3.0

    for blk in final_blocks:
        word_count = len(blk['text'].split())
        reading_ms = max(MIN_READING_MS, word_count * MS_PER_WORD)
        actual_dur = blk['end'] - blk['start']
        if actual_dur > reading_ms * OVERLONG_RATIO:
            # Shift start forward so subtitle appears just before the end
            blk['start'] = blk['end'] - reading_ms

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
    model_path = resolve_model_path(args.model_version)
    
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
        if args.override_lang:
            detected_lang = args.override_lang
            lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)
            print(f"✅ Forcing language: {detected_lang} ({lang_name}) (due to --override-lang)")
        elif args.transcribe:
            detected_lang = "en"
            print("✅ Forcing language: en (due to --transcribe)")
        else:
            duration_sec = get_duration(audio_src)
            detect_dur = str(args.detect_duration)

            # Multi-sample detection: 25%, 50%, 75% of duration (majority vote)
            if duration_sec > args.detect_duration * 2:
                sample_points = [0.25, 0.50, 0.75]
                print(f"🌍 Detecting spoken language (3-point sampling at 25%/50%/75%)...")
            else:
                sample_points = [0.50]
                print(f"🌍 Detecting spoken language (using {args.detect_duration}s sample from midpoint)...")

            detected_langs = []
            for frac in sample_points:
                tmp_detect = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_detect.close()
                temp_files.append(tmp_detect.name)

                start_time = int(duration_sec * frac) if duration_sec > 0 else 0

                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(start_time), "-t", detect_dur,
                    "-i", audio_src, "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", tmp_detect.name
                ], check=True)

                detect_cmd = [
                    whisper_cli, "-m", model_path, "-f", tmp_detect.name, "-dl", "-t", str(args.threads)
                ]

                try:
                    result = subprocess.run(detect_cmd, capture_output=True, text=True, env=env)
                    output_log = result.stdout + result.stderr
                except subprocess.CalledProcessError as e:
                    output_log = (e.stdout or "") + (e.stderr or "")

                match = re.search(r'language: ([a-z]{2})', output_log)
                if match:
                    detected_langs.append(match.group(1))

            if detected_langs:
                winner, count = Counter(detected_langs).most_common(1)[0]
                detected_lang = winner
                if len(sample_points) > 1:
                    print(f"✅ Detected language: {detected_lang} ({count}/{len(detected_langs)} samples)")
                else:
                    print(f"✅ Detected language: {detected_lang}")
            else:
                print("⚠️  Could not detect language — defaulting to 'auto'.")

        # ---------- TRANSLATE/TRANSCRIBE FULL AUDIO ----------
        should_translate = detected_lang != "en" and not args.direct_transcribe
        if args.direct_transcribe:
            action = "Transcribing"
            lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)
            target_desc = f"{lang_name} subtitles"
        elif detected_lang == "en":
            action = "Transcribing"
            target_desc = "English subtitles"
        else:
            action = "Translating"
            target_desc = "English subtitles"
        print()
        print(f"🎧 {action} '{input_path.name}' ({detected_lang} → {target_desc})...")
        print(f"Model: {Path(model_path).name}")
        print(f"Output: {output_srt.name}")
        print()
        
        # whisper-cli often has a hard limit of 8 for beam size (decoders)
        if args.beam_size > 8:
            print(f"⚠️  Note: Capping beam size at 8 (requested {args.beam_size}) to match whisper-cli limits.")
            effective_beam = 8
        else:
            effective_beam = args.beam_size

        # Resolve prompt and decode parameters
        preset_name, prompt_text = resolve_prompt(args.prompt, should_translate)
        decode_args = build_decode_args(preset_name)

        # Primary run
        cmd = [
            whisper_cli,
            "-m", model_path,
            "-f", audio_src,
            "-l", detected_lang,
            "-osrt",
            "-of", str(output_prefix),
            "-t", str(args.threads),
            "-bs", str(effective_beam),
            "--best-of", str(args.best_of)
        ]

        if should_translate:
            cmd.append("-tr")

        cmd += decode_args
        cmd += ["--prompt", prompt_text]
        
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
                "-t", str(args.threads),
                "-bs", "5",
                "-pp"
            ]
            if should_translate:
                fallback_cmd.append("-tr")
            
            subprocess.run(fallback_cmd, check=True, env=env)

        # ---------- POST-PROCESS SRT ----------
        if not args.no_post and output_srt.exists():
            post_process_srt(
                output_srt,
                DEFAULT_MIN_DUR_MS,
                DEFAULT_MAX_DUR_MS,
                DEFAULT_DEDUP_WINDOW_MS,
                DEFAULT_ALLOW_OVERLAP_MS,
                DEFAULT_MAX_CHARS
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

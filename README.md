# transcribe-cli

A high-performance, opinionated wrapper for `whisper-cli` (Whisper.cpp) optimized for translating videos into English subtitles (`.srt`).

It automates the entire pipeline: audio extraction, language detection, translation, and advanced subtitle post-processing.

## Features

*   **Auto-Language Detection:** Multi-sample detection at 25%/50%/75% of the file with majority vote for reliable results. Falls back to single midpoint sample for short files.
*   **Prompt Presets:** Built-in presets (`film`, `anime`, `street`, `talk`) with tuned decode parameters per content type, or supply a custom prompt string.
*   **Smart Subtitle Rebalancing:**
    *   **Splits** long blocks containing multiple sentences into separate entries.
    *   **Merges** sentence fragments forward to keep thought units together (up to 120 chars).
    *   **Respects** short, punchy dialogue (e.g., "No!", "What?") by keeping them standalone.
*   **Hallucination Protection:**
    *   Defaults to `large-v2` (more stable).
    *   Per-preset decode thresholds (entropy, no-speech, token probability) tuned for each content type.
    *   Defaults to **Beam Size 8** with **Best-of 5** (independently configurable).
*   **Performance:**
    *   Optimized for Apple Silicon (Metal) and multi-core CPUs.
    *   Dynamic thread allocation based on your hardware.
*   **SRT Post-Processing:**
    *   Clamps minimum durations to prevent flickering.
    *   Caps maximum durations (15s) to trim hallucinated segments over silence/music.
    *   **Smart Deduplication:** Removes exact duplicates, constructive repetitions, and A-B-A-B alternating patterns.
    *   **Overlong subtitle tightening:** Shifts start times forward when subtitle duration far exceeds reading time, fixing whisper's tendency to start segments early.

## Requirements

*   **Python 3.6+**
*   **ffmpeg** & **ffprobe**
*   **whisper-cli**

### Model Selection

The script looks for Whisper models in `~/.models/whisper/`. It prioritizes:
1.  `ggml-large-v2.bin` (default & preferred for stability)
2.  `ggml-large-v3.bin` (fallback)

You can specify the version via `--model-version` or provide a direct path using the `MODEL_PATH` environment variable.

## Usage

```bash
transcribe.py [options] input_video.mp4
```

### Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `input` | Path to audio/video file. | (Required) |
| `--prompt` | Prompt preset (`film`, `anime`, `street`, `talk`) or custom prompt string. | `film` (translate), generic (transcribe) |
| `--model-version` | Whisper model version (`v2` or `v3`). `v2` is more stable. | `v2` |
| `--beam-size` | Beam size for decoding. | `8` |
| `--best-of` | Best-of candidates for decoding. | `5` |
| `--transcribe` | Force English transcription regardless of detected language. | `False` |
| `--override-lang LANG` | Override detected language with a specific language code (skips auto-detection). | (auto-detect) |
| `--direct-transcribe` | Transcribe in the detected/overridden language without translating to English. | `False` |
| `--threads` | Number of CPU threads to use. | Auto (half of cores) |
| `--detect-duration`| Duration (sec) to sample for language detection. | `30` |
| `--no-post` | Skip the subtitle post-processing step. | `False` |

### Prompt Presets

Each preset provides a tailored prompt and tuned decode parameters:

| Preset | Description | Tuned Parameters |
| :--- | :--- | :--- |
| `film` | Dialogue from a film (default for translation) | Base defaults |
| `anime` | Japanese anime dialogue, preserves honorifics & names | Higher no-speech threshold, tighter entropy |
| `street` | Informal/noisy recording | Lower token probability, tighter entropy, lower no-speech threshold |
| `talk` | Presentation or speech, preserves terminology | Higher token probability, tighter entropy |

Any string that doesn't match a preset name is used as a custom prompt with base decode parameters.

### Environment Variables

You can override defaults using environment variables:

*   `WHISPER_CLI`: Path to the `whisper-cli` binary.
*   `MODEL_PATH`: Direct path to a `.bin` model file.
*   `THREADS`: Number of threads to use.
*   `BEAM_SIZE`: Default beam size.
*   `BEST_OF`: Default best-of candidates.
*   `DETECT_DURATION`: Seconds to sample for language detection.
*   `MIN_DUR_MS`: Minimum subtitle duration in milliseconds (default: `500`).
*   `MAX_DUR_MS`: Maximum subtitle duration in milliseconds (default: `15000`).
*   `DEDUP_WINDOW_MS`: Time window for merging constructive repeats (default: `1500`).
*   `MAX_CHARS`: Maximum characters for merged subtitle lines (default: `120`).

### Supported Languages

Run `transcribe.py -h` to see the full list of 99 supported language codes (ISO 639-1). Common examples:

`en` (English), `es` (Spanish), `fr` (French), `de` (German), `it` (Italian), `pt` (Portuguese), `ja` (Japanese), `ko` (Korean), `zh` (Chinese), `ar` (Arabic), `ru` (Russian), `hi` (Hindi), `nl` (Dutch), `pl` (Polish), `sv` (Swedish), `tr` (Turkish), `vi` (Vietnamese), `th` (Thai)

### Examples

**Standard run (auto-detect and translate to English):**
```
transcribe.py video.mp4
```

**Anime with v3 model:**
```
transcribe.py episode.mkv --prompt anime --model-version v3
```

**Noisy street recording with tighter hallucination thresholds:**
```
transcribe.py dashcam.mp4 --prompt street
```

**Lecture/presentation:**
```
transcribe.py lecture.mp4 --prompt talk
```

**Custom prompt:**
```
transcribe.py interview.mp4 --prompt "Medical terminology. Preserve drug names."
```

**Independent beam-size and best-of:**
```
transcribe.py video.mp4 --beam-size 5 --best-of 3
```

**Override language (skip auto-detection, translate to English):**
```
transcribe.py --override-lang es video.mp4
```

**Direct transcribe in original language (no translation):**
```
transcribe.py --override-lang es --direct-transcribe video.mp4
```

**Direct transcribe with auto-detection (keeps detected language):**
```
transcribe.py --direct-transcribe video.mp4
```

## How it Works

1.  **Extract:** ffmpeg converts input to 16kHz mono WAV.
2.  **Detect:** Multi-sample language detection (3 points at 25%/50%/75%) with majority vote. Falls back to single midpoint sample for short files.
3.  **Transcribe:** Runs `whisper-cli` with preset-tuned parameters (prompt, entropy threshold, token probability, no-speech threshold).
4.  **Post-Process:**
    *   Fixes temporal overlaps and caps abnormally long durations (>15s).
    *   **Rebalances:** Splits multi-sentence blocks and merges fragments to keep thought units together.
    *   **Smart Deduplication:** Removes consecutive repeats, constructive repetitions (partial overlaps), and alternating A-B-A-B patterns.
    *   **Overlong tightening:** Shifts start times forward on subtitles whose duration far exceeds their word count, fixing early-start artifacts.
    *   Writes the final `.srt` file to the same directory as the input.

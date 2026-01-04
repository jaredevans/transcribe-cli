# transcribe-cli

A high-performance, opinionated wrapper for `whisper-cli` (Whisper.cpp) optimized for translating movies, anime, and videos into English subtitles (`.srt`).

It automates the entire pipeline: audio extraction, language detection, translation, and advanced subtitle post-processing.

## Features

*   **Auto-Language Detection:** Samples audio from the midpoint to detect the spoken language before transcribing.
*   **Smart Subtitle Rebalancing:**
    *   **Splits** long blocks containing multiple sentences into separate entries.
    *   **Merges** sentence fragments forward to keep thought units together.
    *   **Respects** short, punchy dialogue (e.g., "No!", "What?") by keeping them standalone.
*   **Hallucination Protection:**
    *   Defaults to `large-v2` (more stable for anime/silence).
    *   Enables **VAD** (Voice Activity Detection) to ignore background noise/music.
    *   Defaults to **Beam Size 8** for high-quality translation, but can be lowered to 1 (greedy) if loops persist.
*   **Performance:**
    *   Optimized for Apple Silicon (Metal) and multi-core CPUs.
    *   Dynamic thread allocation based on your hardware.
*   **SRT Formatting:**
    *   Clamps minimum durations to prevent flickering.
    *   Removes duplicate lines.
    *   Ensures strict character limits (default 60 chars) for readability.

## Requirements

*   **Python 3.6+**
*   **ffmpeg** & **ffprobe** (must be in your PATH)
*   **whisper-cli** (from [whisper.cpp](https://github.com/ggerganov/whisper.cpp))
*   **Whisper Models:** `ggml-large-v2.bin` (default) or `ggml-large-v3.bin` in `~/.models/whisper/`.

## Usage

```bash
./transcribe.py [options] input_video.mp4
```

### Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `input` | Path to audio/video file. | (Required) |
| `--model-version` | Whisper model version (`v2` or `v3`). `v2` is more stable. | `v2` |
| `--threads` | Number of CPU threads to use. | Auto (half of cores) |
| `--beam-size` | Beam size for decoding. `1` is safer for loops, `8` is better quality. | `8` |
| `--detect-duration`| Duration (sec) to sample for language detection. | `30` |
| `--no-post` | Skip the subtitle rebalancing/cleanup step. | `False` |

### Examples

**Standard run:**
```bash
./transcribe.py video.mp4
```

**Force use of Large-V3 model:**
```bash
./transcribe.py --model-version v3 video.mp4
```

**High-Accuracy Mode (Riskier for silence):**
If you have clear audio (like a podcast) and want maximum translation quality:
```bash
./transcribe.py --beam-size 5 video.mp4
```

## How it Works

1.  **Extract:** ffmpeg converts input to 16kHz mono WAV.
2.  **Detect:** A 30s sample from the middle is analyzed to determine the source language.
3.  **Transcribe:** Runs `whisper-cli` with tuned parameters (context window, temperature, VAD).
4.  **Post-Process:**
    *   Parses the raw SRT.
    *   Dedups repetitive lines.
    *   **Rebalances:** Moves "orphan" text (sentence fragments at the end of a block) to the next block to ensure sentences are complete.
    *   Writes the final `.srt` file to the same directory as the input.
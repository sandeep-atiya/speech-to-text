"""Command-line interface.

    python transcribe.py                       # every recording in recordings/
    python transcribe.py call.wav other.mp3    # specific files or folders
    python transcribe.py --format txt --format srt --limit-seconds 60 call.wav
"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from transcriber.audio import find_recordings
from transcriber.config import FORMAT_CHOICES, LANGUAGE_CHOICES, MODEL_CHOICES, Settings
from transcriber.engine import Transcriber
from transcriber.writer import format_line, write_transcript

log = logging.getLogger("transcriber")


def build_parser(defaults: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe Hindi/Urdu call recordings into Hinglish with faster-whisper.",
    )
    parser.add_argument("inputs", nargs="*", type=Path,
                        help=f"recordings or folders (default: {defaults.recordings_dir})")
    parser.add_argument("-o", "--output-dir", type=Path, default=defaults.transcripts_dir,
                        help="where transcripts are written (default: %(default)s)")
    parser.add_argument("-f", "--format", action="append", choices=FORMAT_CHOICES, dest="formats",
                        help="output format, repeatable (default: txt)")
    parser.add_argument("-m", "--model", default=defaults.model_size,
                        help=f"Whisper model: {', '.join(MODEL_CHOICES)} or a local path (default: %(default)s)")
    parser.add_argument("-l", "--language", choices=LANGUAGE_CHOICES, default=defaults.language,
                        help="force the spoken language instead of detecting it (default: %(default)s)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=defaults.device)
    parser.add_argument("--compute-type", default=defaults.compute_type,
                        help="auto, int8, int8_float16, float16, float32 (default: %(default)s)")
    parser.add_argument("--cpu-threads", type=int, default=defaults.cpu_threads,
                        help="CPU threads for the model, 0 = library default (default: %(default)s)")
    parser.add_argument("--beam-size", type=int, default=defaults.beam_size)
    parser.add_argument("--chunk-seconds", type=int, default=defaults.chunk_seconds,
                        help="max seconds of speech per decoded chunk, 1-30 (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--limit-seconds", type=float, default=defaults.limit_seconds,
                        help="only transcribe the first N seconds of each file, for quick checks")
    parser.add_argument("-q", "--quiet", action="store_true", help="do not print transcript lines")
    return parser


def settings_from_args(args: argparse.Namespace, defaults: Settings) -> Settings:
    return replace(
        defaults,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        beam_size=args.beam_size,
        chunk_seconds=args.chunk_seconds,
        batch_size=args.batch_size,
        language=args.language,
        transcripts_dir=args.output_dir,
        formats=tuple(args.formats or defaults.formats),
        limit_seconds=args.limit_seconds,
    )


def configure_output() -> None:
    # Hinglish is ASCII, but detected text can contain other scripts; never crash on it,
    # even when the output is redirected to a file on Windows.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("faster_whisper", "huggingface_hub", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    configure_output()
    defaults = Settings()
    args = build_parser(defaults).parse_args(argv)
    try:
        settings = settings_from_args(args, defaults)
        recordings = find_recordings(args.inputs, settings.recordings_dir)
    except (ValueError, FileNotFoundError) as exc:
        log.error("error: %s", exc)
        return 2
    if not recordings:
        log.error("No recordings found. Put audio files in %s or pass them as arguments.", settings.recordings_dir)
        return 1

    transcriber = Transcriber(settings)
    print_line = (lambda segment: None) if args.quiet else (lambda segment: print(format_line(segment)))

    failures = 0
    for path in recordings:
        log.info("\n=== %s ===", path.name)
        try:
            transcript = transcriber.transcribe(path, on_segment=print_line)
        except Exception:  # keep going with the other recordings
            failures += 1
            log.exception("Failed to transcribe %s", path.name)
            continue
        written = write_transcript(transcript, settings.transcripts_dir, settings.formats)
        speed = transcript.audio_seconds / transcript.elapsed_seconds if transcript.elapsed_seconds else 0.0
        log.info("(%.0fs of audio in %.0fs, %.1fx realtime) saved: %s",
                 transcript.audio_seconds, transcript.elapsed_seconds, speed,
                 ", ".join(p.name for p in written))
    return 1 if failures else 0

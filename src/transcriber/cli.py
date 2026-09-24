"""Command-line interface.

hinglish-transcribe                         # every new recording in recordings/
hinglish-transcribe call.wav other.mp3      # specific files or folders
hinglish-transcribe -f txt -f srt --limit-seconds 60 call.wav
hinglish-transcribe --watch                 # keep running, transcribe new files as they arrive
"""

import argparse
import io
import logging
import sys
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from transcriber.audio import find_recordings
from transcriber.config import FORMAT_CHOICES, LANGUAGE_CHOICES, MODEL_CHOICES, TIMESTAMP_CHOICES, Settings
from transcriber.engine import Segment, Transcriber
from transcriber.watch import watch
from transcriber.writer import format_line, transcript_paths, write_transcript

log = logging.getLogger("transcriber")


def build_parser(defaults: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hinglish-transcribe",
        description="Transcribe Hindi/Urdu call recordings into Hinglish with faster-whisper.",
    )
    parser.add_argument(
        "inputs", nargs="*", type=Path, help=f"recordings or folders (default: {defaults.recordings_dir})"
    )
    out = parser.add_argument_group("output")
    out.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=defaults.transcripts_dir,
        help="where transcripts are written (default: %(default)s)",
    )
    out.add_argument(
        "-f",
        "--format",
        action="append",
        choices=FORMAT_CHOICES,
        dest="formats",
        help="output format, repeatable (default: txt)",
    )
    out.add_argument(
        "--timestamps",
        choices=TIMESTAMP_CHOICES,
        default=defaults.timestamps,
        help="line timestamps as 00:15 (clock) or 15.57s (seconds) (default: %(default)s)",
    )
    out.add_argument("--force", action="store_true", help="re-transcribe recordings that already have a transcript")
    out.add_argument("-q", "--quiet", action="store_true", help="do not print transcript lines")
    out.add_argument("--no-progress", action="store_true", help="hide the progress bar")

    model = parser.add_argument_group("model")
    model.add_argument(
        "-m",
        "--model",
        default=defaults.model_size,
        help=f"Whisper model: {', '.join(MODEL_CHOICES)} or a local path (default: %(default)s)",
    )
    model.add_argument(
        "-l",
        "--language",
        choices=LANGUAGE_CHOICES,
        default=defaults.language,
        help="force the spoken language instead of detecting it (default: %(default)s)",
    )
    model.add_argument("--device", choices=("auto", "cpu", "cuda"), default=defaults.device)
    model.add_argument(
        "--compute-type",
        default=defaults.compute_type,
        help="auto, int8, int8_float16, float16, float32 (default: %(default)s)",
    )
    model.add_argument(
        "--cpu-threads",
        type=int,
        default=defaults.cpu_threads,
        help="CPU threads for the model, 0 = library default (default: %(default)s)",
    )
    model.add_argument("--beam-size", type=int, default=defaults.beam_size)
    model.add_argument("--batch-size", type=int, default=defaults.batch_size)

    chunking = parser.add_argument_group("chunking")
    chunking.add_argument(
        "--chunk-seconds",
        type=int,
        default=defaults.chunk_seconds,
        help="max seconds of speech per decoded chunk, 1-30; cuts are placed in pauses (default: %(default)s)",
    )
    chunking.add_argument(
        "--speech-threshold",
        type=float,
        default=defaults.speech_threshold,
        help="voice detector sensitivity, 0-1: lower keeps quieter speech (default: %(default)s)",
    )
    chunking.add_argument(
        "--skip-silence-seconds",
        type=float,
        default=defaults.skip_silence_seconds,
        help="only silences at least this long are left out of the transcript (default: %(default)s)",
    )

    vocab = parser.add_argument_group("vocabulary")
    vocab.add_argument(
        "--glossary",
        type=Path,
        help="text file of names Whisper should recognise, one per line (default: ./glossary.txt if present)",
    )
    vocab.add_argument(
        "--custom-words",
        type=Path,
        help='JSON file of extra spellings {"देवनागरी": "hinglish"} (default: ./custom_words.json if present)',
    )

    run = parser.add_argument_group("run")
    run.add_argument(
        "--limit-seconds",
        type=float,
        default=defaults.limit_seconds,
        help="only transcribe the first N seconds of each file, for quick checks",
    )
    run.add_argument(
        "--watch",
        action="store_true",
        help="after the current files, keep running and transcribe new recordings as they arrive",
    )
    run.add_argument(
        "--poll-seconds",
        type=float,
        default=10.0,
        help="how often --watch checks the folder (default: %(default)s)",
    )
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
        speech_threshold=args.speech_threshold,
        skip_silence_seconds=args.skip_silence_seconds,
        batch_size=args.batch_size,
        language=args.language,
        glossary_path=args.glossary,
        custom_words_path=args.custom_words,
        transcripts_dir=args.output_dir,
        formats=tuple(args.formats or defaults.formats),
        timestamps=args.timestamps,
        force=args.force,
        limit_seconds=args.limit_seconds,
    )


def configure_output() -> None:
    # Hinglish is ASCII, but detected text can contain other scripts; never crash on it,
    # even when the output is redirected to a file on Windows.
    if isinstance(sys.stdout, io.TextIOWrapper):
        # line_buffering: lines show up immediately even when output is redirected to a file
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("faster_whisper", "huggingface_hub", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def has_transcript(path: Path, settings: Settings) -> bool:
    return all(p.is_file() for p in transcript_paths(path, settings.transcripts_dir, settings.formats))


class ConsoleProgress:
    """Console output for one recording: a progress bar with ETA (on stderr) and each line as it arrives."""

    def __init__(self, show_lines: bool, show_bar: bool, timestamps: str) -> None:
        self.show_lines = show_lines
        self.show_bar = show_bar
        self.timestamps = timestamps
        self.bar: tqdm | None = None

    def start(self, audio_seconds: float) -> None:
        self.bar = tqdm(
            total=round(audio_seconds),
            unit="s",
            disable=not self.show_bar,
            leave=False,
            dynamic_ncols=True,
            bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
        )

    def segment(self, segment: Segment) -> None:
        if self.bar is not None:
            self.bar.update(max(0.0, min(segment.end, self.bar.total) - self.bar.n))
        if self.show_lines:
            tqdm.write(format_line(segment, self.timestamps))

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def transcribe_recordings(
    transcriber: Transcriber, recordings: list[Path], settings: Settings, args: argparse.Namespace
) -> int:
    """Transcribe each recording, printing progress and saving the result. Returns the number of failures."""
    failures = 0
    for path in recordings:
        log.info("\n=== %s ===", path.name)
        progress = ConsoleProgress(
            show_lines=not args.quiet,
            show_bar=not args.no_progress and sys.stderr.isatty(),
            timestamps=settings.timestamps,
        )
        try:
            transcript = transcriber.transcribe(path, on_segment=progress.segment, on_start=progress.start)
        except Exception:  # keep going with the other recordings
            failures += 1
            log.exception("Failed to transcribe %s", path.name)
            continue
        finally:
            progress.close()
        written = write_transcript(transcript, settings.transcripts_dir, settings.formats, settings.timestamps)
        speed = transcript.audio_seconds / transcript.elapsed_seconds if transcript.elapsed_seconds else 0.0
        log.info(
            "(%.0fs of audio in %.0fs, %.1fx realtime) saved: %s",
            transcript.audio_seconds,
            transcript.elapsed_seconds,
            speed,
            ", ".join(p.name for p in written),
        )
    return failures


def watched_folder(inputs: list[Path], settings: Settings) -> Path:
    """--watch works on exactly one folder: the one given, or the recordings folder."""
    if len(inputs) > 1 or (inputs and not inputs[0].is_dir()):
        raise ValueError("--watch takes a single folder (or none, for the recordings folder)")
    return inputs[0].resolve() if inputs else settings.recordings_dir


def main(argv: list[str] | None = None) -> int:
    configure_output()
    defaults = Settings()
    args = build_parser(defaults).parse_args(argv)
    try:
        settings = settings_from_args(args, defaults)
        recordings = find_recordings(args.inputs, settings.recordings_dir)
        folder = watched_folder(args.inputs, settings) if args.watch else None
    except (ValueError, FileNotFoundError) as exc:
        log.error("error: %s", exc)
        return 2

    pending = recordings if settings.force else [p for p in recordings if not has_transcript(p, settings)]
    skipped = len(recordings) - len(pending)
    if skipped:
        log.info("Skipping %d recording(s) that already have a transcript (use --force to redo them).", skipped)
    if not pending and folder is None:
        if not recordings:
            log.error("No recordings found. Put audio files in %s or pass them as arguments.", settings.recordings_dir)
            return 1
        log.info("Nothing to do.")
        return 0

    try:
        transcriber = Transcriber(settings)
    except (ValueError, FileNotFoundError) as exc:  # bad glossary / custom words file
        log.error("error: %s", exc)
        return 2

    failures = transcribe_recordings(transcriber, pending, settings, args) if pending else 0

    if folder is not None:

        def handle_new(paths: list[Path]) -> None:
            transcribe_recordings(transcriber, paths, settings, args)

        try:
            watch(
                folder,
                is_done=lambda path: has_transcript(path, settings),
                handle=handle_new,
                poll_seconds=args.poll_seconds,
            )
        except KeyboardInterrupt:
            log.info("\nStopped watching.")
    return 1 if failures else 0

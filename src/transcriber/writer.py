"""Writing transcripts to disk in txt, srt and json formats."""

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from transcriber.engine import Segment, Transcript


def clock(seconds: float) -> str:
    """12.9 -> '00:12', 3725.0 -> '1:02:05'."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def format_line(segment: Segment, timestamps: str = "clock") -> str:
    """One console/txt line: [00:12 -> 00:15] text  (or [12.34s -> 15.67s] text)."""
    if timestamps == "seconds":
        return f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
    return f"[{clock(segment.start)} -> {clock(segment.end)}] {segment.text}"


def to_txt(transcript: Transcript, timestamps: str = "clock") -> str:
    return "".join(format_line(s, timestamps) + "\n" for s in transcript.segments)


def _srt_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def to_srt(transcript: Transcript) -> str:
    blocks = []
    for index, s in enumerate(transcript.segments, start=1):
        blocks.append(f"{index}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{s.text}\n")
    return "\n".join(blocks)


def to_json(transcript: Transcript) -> str:
    data = asdict(transcript)
    data["source"] = str(transcript.source)
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def transcript_paths(source: Path, out_dir: Path, formats: Iterable[str]) -> list[Path]:
    """Where the transcript of a recording goes: <out_dir>/<recording name>.hinglish.<format>."""
    return [out_dir / f"{source.stem}.hinglish.{fmt}" for fmt in formats]


def write_transcript(
    transcript: Transcript,
    out_dir: Path,
    formats: Iterable[str],
    timestamps: str = "clock",
) -> list[Path]:
    """Save the transcript in each format and return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in transcript_paths(transcript.source, out_dir, formats):
        fmt = path.suffix.lstrip(".")
        if fmt == "txt":
            content = to_txt(transcript, timestamps)
        elif fmt == "srt":
            content = to_srt(transcript)
        else:
            content = to_json(transcript)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written

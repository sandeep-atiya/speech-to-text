"""Writing transcripts to disk in txt, srt and json formats."""

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from transcriber.engine import Segment, Transcript


def format_line(segment: Segment) -> str:
    """One console/txt line: [12.34s -> 15.67s] text"""
    return f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"


def to_txt(transcript: Transcript) -> str:
    return "".join(format_line(s) + "\n" for s in transcript.segments)


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


FORMATTERS = {"txt": to_txt, "srt": to_srt, "json": to_json}


def write_transcript(transcript: Transcript, out_dir: Path, formats: Iterable[str]) -> list[Path]:
    """Save the transcript as <out_dir>/<recording name>.hinglish.<format> for each format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{transcript.source.stem}.hinglish.{fmt}"
        path.write_text(FORMATTERS[fmt](transcript), encoding="utf-8")
        written.append(path)
    return written

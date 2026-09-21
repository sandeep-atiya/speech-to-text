"""Tests for the txt / srt / json writers."""

import json
from pathlib import Path

from transcriber.engine import Segment, Transcript
from transcriber.writer import format_line, to_json, to_srt, to_txt, write_transcript


def make_transcript(tmp_path: Path) -> Transcript:
    return Transcript(
        source=tmp_path / "call.wav",
        detected_language="hi",
        detected_probability=0.9,
        language="hi",
        audio_seconds=70.0,
        elapsed_seconds=50.0,
        segments=[
            Segment(0.94, 15.57, "hello ji namaste"),
            Segment(3661.5, 3665.25, "theek hai"),
        ],
    )


def test_format_line() -> None:
    assert format_line(Segment(0.94, 15.57, "hello ji")) == "[0.94s -> 15.57s] hello ji"


def test_txt_has_one_line_per_segment(tmp_path: Path) -> None:
    text = to_txt(make_transcript(tmp_path))
    assert text == "[0.94s -> 15.57s] hello ji namaste\n[3661.50s -> 3665.25s] theek hai\n"


def test_srt_timestamps_and_numbering(tmp_path: Path) -> None:
    srt = to_srt(make_transcript(tmp_path))
    assert srt.startswith("1\n00:00:00,940 --> 00:00:15,570\nhello ji namaste\n")
    assert "2\n01:01:01,500 --> 01:01:05,250\ntheek hai\n" in srt


def test_json_is_valid_and_complete(tmp_path: Path) -> None:
    data = json.loads(to_json(make_transcript(tmp_path)))
    assert data["language"] == "hi"
    assert data["source"].endswith("call.wav")
    assert data["segments"][1] == {"start": 3661.5, "end": 3665.25, "text": "theek hai"}


def test_write_transcript_creates_one_file_per_format(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    written = write_transcript(make_transcript(tmp_path), out_dir, ["txt", "srt", "json"])
    assert [p.name for p in written] == ["call.hinglish.txt", "call.hinglish.srt", "call.hinglish.json"]
    assert all(p.is_file() for p in written)
    assert written[0].read_text(encoding="utf-8").startswith("[0.94s -> 15.57s]")

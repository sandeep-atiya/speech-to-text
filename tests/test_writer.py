"""Tests for the txt / srt / json writers."""

import json
from pathlib import Path

from transcriber.engine import Segment, Transcript
from transcriber.writer import clock, format_line, to_json, to_srt, to_txt, transcript_paths, write_transcript


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


def test_clock_format() -> None:
    assert clock(0.94) == "00:00"
    assert clock(75.9) == "01:15"
    assert clock(3725.0) == "1:02:05"


def test_format_line_styles() -> None:
    segment = Segment(0.94, 15.57, "hello ji")
    assert format_line(segment) == "[00:00 -> 00:15] hello ji"
    assert format_line(segment, "seconds") == "[0.94s -> 15.57s] hello ji"


def test_txt_has_one_line_per_segment(tmp_path: Path) -> None:
    transcript = make_transcript(tmp_path)
    assert to_txt(transcript) == "[00:00 -> 00:15] hello ji namaste\n[1:01:01 -> 1:01:05] theek hai\n"
    assert to_txt(transcript, "seconds").startswith("[0.94s -> 15.57s] hello ji namaste\n")


def test_srt_timestamps_and_numbering(tmp_path: Path) -> None:
    srt = to_srt(make_transcript(tmp_path))
    assert srt.startswith("1\n00:00:00,940 --> 00:00:15,570\nhello ji namaste\n")
    assert "2\n01:01:01,500 --> 01:01:05,250\ntheek hai\n" in srt


def test_json_is_valid_and_complete(tmp_path: Path) -> None:
    data = json.loads(to_json(make_transcript(tmp_path)))
    assert data["language"] == "hi"
    assert data["source"].endswith("call.wav")
    assert data["segments"][1] == {"start": 3661.5, "end": 3665.25, "text": "theek hai"}


def test_transcript_paths(tmp_path: Path) -> None:
    paths = transcript_paths(Path("x/call.wav"), tmp_path, ["txt", "json"])
    assert paths == [tmp_path / "call.hinglish.txt", tmp_path / "call.hinglish.json"]


def test_write_transcript_creates_one_file_per_format(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    written = write_transcript(make_transcript(tmp_path), out_dir, ["txt", "srt", "json"], "seconds")
    assert [p.name for p in written] == ["call.hinglish.txt", "call.hinglish.srt", "call.hinglish.json"]
    assert all(p.is_file() for p in written)
    assert written[0].read_text(encoding="utf-8").startswith("[0.94s -> 15.57s]")

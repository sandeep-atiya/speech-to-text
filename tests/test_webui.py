"""Tests for the web UI helpers (the page itself needs gradio and a browser)."""

import json
from pathlib import Path

import pytest

from transcriber.config import Settings
from transcriber.engine import Segment
from transcriber.webui import (
    PLAYER_ID,
    WebApp,
    file_url,
    load_saved_segments,
    render_player_html,
    render_transcript_html,
)


def test_player_html_points_at_the_served_file(tmp_path: Path) -> None:
    path = tmp_path / "call.wav"
    assert render_player_html(None) == ""
    html = render_player_html(path)
    assert f'id="{PLAYER_ID}"' in html
    assert file_url(path) in html


def test_transcript_html_escapes_text_and_seeks_on_click() -> None:
    html = render_transcript_html([Segment(75.5, 80.0, "a <b> & c")])
    assert "a &lt;b&gt; &amp; c" in html
    assert "[01:15]" in html
    assert "p.currentTime=75.50" in html
    assert "No speech" in render_transcript_html([])


def test_saved_segments_round_trip(tmp_path: Path) -> None:
    saved = tmp_path / "call.hinglish.json"
    saved.write_text(json.dumps({"segments": [{"start": 1.0, "end": 2.5, "text": "namaste"}]}), encoding="utf-8")
    assert load_saved_segments(saved) == [Segment(1.0, 2.5, "namaste")]


def test_webapp_lists_recordings_and_finds_saved_transcripts(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    transcripts = tmp_path / "transcripts"
    recordings.mkdir()
    (recordings / "b.wav").touch()
    (recordings / "a.mp3").touch()
    app = WebApp(Settings(recordings_dir=recordings, transcripts_dir=transcripts))

    assert app.recording_names() == ["a.mp3", "b.wav"]
    assert app.resolve_input("a.mp3", None) == recordings / "a.mp3"
    assert app.resolve_input("a.mp3", str(tmp_path / "up.wav")) == tmp_path / "up.wav"
    with pytest.raises(ValueError):
        app.resolve_input(None, None)

    player, html, files, status = app.show_existing("a.mp3")
    assert files is None and "No transcript yet" in status and PLAYER_ID in player

    transcripts.mkdir()
    for ext in ("txt", "srt"):
        (transcripts / f"a.hinglish.{ext}").touch()
    (transcripts / "a.hinglish.json").write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "hello ji"}]}), encoding="utf-8"
    )
    player, html, files, status = app.show_existing("a.mp3")
    assert "hello ji" in html and files is not None and len(files) == 3 and "Saved" in status

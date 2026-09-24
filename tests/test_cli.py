"""Tests for command-line parsing, skipping and error handling (no model is loaded)."""

from pathlib import Path

import pytest

from transcriber.cli import build_parser, has_transcript, main, settings_from_args
from transcriber.config import Settings


def parse(argv: list[str]) -> Settings:
    defaults = Settings()
    return settings_from_args(build_parser(defaults).parse_args(argv), defaults)


def test_defaults_match_settings() -> None:
    assert parse([]) == Settings()


def test_flags_override_settings(tmp_path: Path) -> None:
    settings = parse(
        [
            "call.wav",
            "-f", "srt", "-f", "json",
            "--model", "large-v3",
            "--language", "hi",
            "--chunk-seconds", "10",
            "--speech-threshold", "0.4",
            "--skip-silence-seconds", "2",
            "--limit-seconds", "60",
            "--timestamps", "seconds",
            "--force",
            "--glossary", "names.txt",
            "--custom-words", "spellings.json",
            "--output-dir", str(tmp_path),
        ]
    )  # fmt: skip
    assert settings.formats == ("srt", "json")
    assert settings.model_size == "large-v3"
    assert settings.language == "hi"
    assert settings.chunk_seconds == 10
    assert settings.speech_threshold == 0.4
    assert settings.skip_silence_seconds == 2.0
    assert settings.limit_seconds == 60
    assert settings.timestamps == "seconds"
    assert settings.force is True
    assert settings.glossary_path == Path("names.txt")
    assert settings.custom_words_path == Path("spellings.json")
    assert settings.transcripts_dir == tmp_path


def test_invalid_flag_value_is_a_usage_error(tmp_path: Path) -> None:
    assert main(["--chunk-seconds", "99", str(tmp_path)]) == 2


def test_missing_input_is_a_usage_error(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing.wav")]) == 2


def test_empty_recordings_folder_returns_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "recordings").mkdir()
    monkeypatch.chdir(tmp_path)
    assert main([]) == 1


def test_has_transcript_requires_every_format(tmp_path: Path) -> None:
    settings = Settings(transcripts_dir=tmp_path, formats=("txt", "srt"))
    (tmp_path / "call.hinglish.txt").touch()
    assert has_transcript(Path("call.wav"), settings) is False
    (tmp_path / "call.hinglish.srt").touch()
    assert has_transcript(Path("call.wav"), settings) is True


def test_already_transcribed_recordings_are_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "recordings").mkdir()
    (tmp_path / "recordings" / "call.wav").touch()
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "call.hinglish.txt").touch()
    monkeypatch.chdir(tmp_path)

    assert main([]) == 0  # nothing to do, so no model is loaded


def test_watch_needs_a_single_folder(tmp_path: Path) -> None:
    (tmp_path / "a.wav").touch()
    assert main(["--watch", str(tmp_path / "a.wav")]) == 2
    assert main(["--watch", str(tmp_path), str(tmp_path)]) == 2


def test_bad_custom_words_file_is_a_usage_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "recordings").mkdir()
    (tmp_path / "recordings" / "call.wav").touch()
    (tmp_path / "custom_words.json").write_text("[1, 2]", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main([]) == 2

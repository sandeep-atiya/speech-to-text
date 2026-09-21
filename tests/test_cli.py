"""Tests for command-line parsing and error handling (no model is loaded)."""

from pathlib import Path

import pytest

from transcriber.cli import build_parser, main, settings_from_args
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
            "--limit-seconds", "60",
            "--output-dir", str(tmp_path),
        ]
    )  # fmt: skip
    assert settings.formats == ("srt", "json")
    assert settings.model_size == "large-v3"
    assert settings.language == "hi"
    assert settings.chunk_seconds == 10
    assert settings.limit_seconds == 60
    assert settings.transcripts_dir == tmp_path


def test_invalid_flag_value_is_a_usage_error(tmp_path: Path) -> None:
    assert main(["--chunk-seconds", "99", str(tmp_path)]) == 2


def test_missing_input_is_a_usage_error(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing.wav")]) == 2


def test_empty_recordings_folder_returns_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "recordings").mkdir()
    monkeypatch.chdir(tmp_path)
    assert main([]) == 1

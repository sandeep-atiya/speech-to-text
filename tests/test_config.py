"""Tests for settings validation."""

from dataclasses import replace

import pytest

from transcriber.config import Settings


def test_defaults_are_valid() -> None:
    settings = Settings()
    assert settings.model_size == "turbo"
    assert settings.language == "auto"
    assert settings.formats == ("txt",)
    assert settings.recordings_dir.is_absolute()


@pytest.mark.parametrize(
    "override",
    [
        {"language": "fr"},
        {"formats": ("txt", "pdf")},
        {"chunk_seconds": 0},
        {"chunk_seconds": 31},
        {"english_threshold": 1.5},
    ],
)
def test_invalid_values_are_rejected(override: dict) -> None:
    with pytest.raises(ValueError):
        replace(Settings(), **override)

"""Tests for the glossary and custom spelling files."""

from pathlib import Path

import pytest

from transcriber.glossary import hotwords_from, load_custom_words, load_glossary


def test_glossary_ignores_blank_lines_and_comments(tmp_path: Path) -> None:
    glossary = tmp_path / "glossary.txt"
    glossary.write_text("# products\nगोंद सिया\n\n  काला गोंद  # the black one\nAshwagandha\n", encoding="utf-8")

    assert load_glossary(glossary) == ["गोंद सिया", "काला गोंद", "Ashwagandha"]
    assert hotwords_from(load_glossary(glossary)) == "गोंद सिया, काला गोंद, Ashwagandha"


def test_missing_files_mean_no_vocabulary() -> None:
    assert load_glossary(None) == []
    assert hotwords_from([]) is None
    assert load_custom_words(None) == {}


def test_custom_words_are_loaded_and_trimmed(tmp_path: Path) -> None:
    words = tmp_path / "custom_words.json"
    words.write_text('{"सुलेमान": " suleman ", "माजून": "majun"}', encoding="utf-8")

    assert load_custom_words(words) == {"सुलेमान": "suleman", "माजून": "majun"}


def test_custom_words_must_be_a_string_object(tmp_path: Path) -> None:
    words = tmp_path / "custom_words.json"
    words.write_text('["सुलेमान"]', encoding="utf-8")

    with pytest.raises(ValueError):
        load_custom_words(words)

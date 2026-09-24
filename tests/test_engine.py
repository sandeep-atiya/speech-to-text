"""Tests for the transcription engine with a fake model and pipeline (no weights needed)."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from transcriber.config import SAMPLE_RATE, Settings
from transcriber.engine import Transcriber, resolve_device
from transcriber.hinglish import to_hinglish


class FakeModel:
    """Pretends Whisper detected the given language."""

    def __init__(self, language: str, probability: float) -> None:
        self.info = SimpleNamespace(language=language, language_probability=probability)

    def transcribe(self, audio, **kwargs):
        return iter([]), self.info


class FakePipeline:
    """Returns Devanagari segments and records how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        segments = [
            SimpleNamespace(start=0.5, end=4.0, text=" हेलो जी नमस्ते ", tokens=[1, 2, 3]),
            SimpleNamespace(start=4.0, end=9.5, text=" कितने वक्त से दर्द है ", tokens=[4, 5, 6, 7]),
        ]
        return iter(segments), None


class TruncatingPipeline:
    """First call: one segment that used up the whole token budget. Later calls: one segment per clip."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        clips = kwargs["clip_timestamps"]
        if len(self.calls) == 1:
            segments = [SimpleNamespace(start=clips[0]["start"], end=clips[0]["end"], text=" हेलो ", tokens=[1, 2, 3])]
        else:
            texts = [" हेलो ", " नमस्ते "]
            segments = [
                SimpleNamespace(start=clip["start"], end=clip["end"], text=texts[i], tokens=[1])
                for i, clip in enumerate(clips)
            ]
        return iter(segments), None


@pytest.fixture
def silence(monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    audio = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)
    monkeypatch.setattr("transcriber.engine.load_audio", lambda path, limit_seconds=0.0: audio)
    return audio


def make_transcriber(
    settings: Settings, language: str = "hi", probability: float = 0.9
) -> tuple[Transcriber, FakePipeline]:
    pipeline = FakePipeline()
    return Transcriber(settings, model=FakeModel(language, probability), pipeline=pipeline), pipeline


def test_resolve_device_auto_picks_a_concrete_pair() -> None:
    device, compute_type = resolve_device("auto", "auto")
    assert (device, compute_type) in {("cpu", "int8"), ("cuda", "float16")}
    assert resolve_device("cpu", "float32") == ("cpu", "float32")


@pytest.mark.parametrize(
    ("setting", "detected", "probability", "expected"),
    [
        ("auto", "hi", 0.9, "hi"),
        ("auto", "en", 0.95, "en"),  # clearly English
        ("auto", "en", 0.66, "hi"),  # Hindi with English words still counts as Hindi
        ("auto", "ur", 0.8, "hi"),
        ("en", "hi", 0.9, "en"),  # forced
        ("hi", "en", 0.99, "hi"),  # forced
    ],
)
def test_choose_language(setting: str, detected: str, probability: float, expected: str) -> None:
    transcriber, _ = make_transcriber(replace(Settings(), language=setting))
    assert transcriber.choose_language(detected, probability) == expected


def test_transcribe_converts_segments_to_hinglish_and_reports_progress(silence: np.ndarray) -> None:
    settings = replace(Settings(), chunk_seconds=12, beam_size=3, glossary_path=None)
    transcriber, pipeline = make_transcriber(settings)
    seen = []
    lengths = []

    transcript = transcriber.transcribe(Path("call.wav"), on_segment=seen.append, on_start=lengths.append)

    assert [s.text for s in transcript.segments] == ["hello ji namaste", "kitne waqt se dard hai"]
    assert seen == transcript.segments
    assert lengths == [10.0]
    assert transcript.language == "hi"
    assert transcript.detected_probability == 0.9
    assert transcript.audio_seconds == 10.0
    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert (call["language"], call["beam_size"], call["batch_size"]) == ("hi", 3, 8)
    # the voice detector finds nothing in silence, so the whole recording is decoded as one chunk
    assert call["clip_timestamps"] == [{"start": 0.0, "end": 10.0}]
    assert transcriber.token_budget is None  # no real tokenizer, so the overflow check is off


def test_chunk_that_fills_the_token_budget_is_decoded_again_in_halves(silence: np.ndarray) -> None:
    pipeline = TruncatingPipeline()
    transcriber = Transcriber(Settings(), model=FakeModel("hi", 0.9), pipeline=pipeline)
    transcriber.token_budget = 3

    transcript = transcriber.transcribe(Path("call.wav"))

    assert len(pipeline.calls) == 2
    halves = pipeline.calls[1]["clip_timestamps"]
    assert len(halves) == 2
    assert halves[0]["start"] == 0.0 and halves[1]["end"] == 10.0
    assert halves[0]["end"] == halves[1]["start"]
    assert [s.text for s in transcript.segments] == ["hello", "namaste"]
    assert [(s.start, s.end) for s in transcript.segments] == [(0.0, halves[0]["end"]), (halves[1]["start"], 10.0)]


def test_glossary_becomes_hotwords_and_custom_words_apply(silence: np.ndarray, tmp_path: Path) -> None:
    glossary = tmp_path / "glossary.txt"
    glossary.write_text("गोंद सिया\nहकीम सुलेमान खान\n", encoding="utf-8")
    words = tmp_path / "custom_words.json"
    words.write_text('{"ज़ुबानी": "zubani"}', encoding="utf-8")
    settings = replace(Settings(), glossary_path=glossary, custom_words_path=words)

    transcriber, pipeline = make_transcriber(settings)
    transcriber.transcribe(Path("call.wav"))

    assert transcriber.hotwords == "गोंद सिया, हकीम सुलेमान खान"
    assert pipeline.calls[0]["hotwords"] == "गोंद सिया, हकीम सुलेमान खान"
    assert to_hinglish("ज़ुबानी") == "zubani"

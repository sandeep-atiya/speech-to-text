"""Model loading, language detection and transcription."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import ctranslate2
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel

from transcriber.audio import duration_seconds, load_audio
from transcriber.config import SAMPLE_RATE, Settings
from transcriber.hinglish import to_hinglish

log = logging.getLogger(__name__)

# Whisper detects the language from the first 30 s of audio.
DETECTION_SECONDS = 30


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    source: Path
    detected_language: str
    detected_probability: float
    language: str  # language actually used for decoding
    audio_seconds: float
    elapsed_seconds: float
    segments: list[Segment] = field(default_factory=list)


def resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    """Turn "auto" choices into concrete values.

    float16 is only supported on GPU; int8 is the fastest option on CPU.
    """
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


class Transcriber:
    """Loads a Whisper model once and transcribes recordings into Hinglish.

    Usage:
        transcriber = Transcriber(Settings())
        transcript = transcriber.transcribe(Path("recordings/call.wav"))
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device, self.compute_type = resolve_device(settings.device, settings.compute_type)
        log.info("Loading '%s' on %s (%s)...", settings.model_size, self.device, self.compute_type)
        started = time.perf_counter()
        self.model = WhisperModel(
            settings.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=settings.cpu_threads,
        )
        # The batched pipeline decodes each voice-activity chunk on its own, so a long
        # Devanagari passage can never overrun the decoder's token limit.
        self.pipeline = BatchedInferencePipeline(model=self.model)
        log.info("Model loaded in %.1fs", time.perf_counter() - started)

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Return Whisper's (language, probability) guess for the start of the audio.

        Nothing is decoded: transcribe() only runs the encoder until segments are read.
        """
        head = audio[: DETECTION_SECONDS * SAMPLE_RATE]
        _, info = self.model.transcribe(head, language=None)
        return info.language, info.language_probability

    def choose_language(self, detected: str, probability: float) -> str:
        if self.settings.language != "auto":
            return self.settings.language
        if detected == "en" and probability >= self.settings.english_threshold:
            return "en"
        return "hi"

    def transcribe(self, path: Path, on_segment: Callable[[Segment], None] | None = None) -> Transcript:
        """Transcribe one recording. on_segment is called for each line as soon as it is ready."""
        started = time.perf_counter()
        audio = load_audio(path, self.settings.limit_seconds)

        detected, probability = self.detect_language(audio)
        language = self.choose_language(detected, probability)
        log.info("Detected language '%s' (%.2f) -> transcribing as '%s'", detected, probability, language)

        raw_segments, _ = self.pipeline.transcribe(
            audio,
            language=language,
            beam_size=self.settings.beam_size,
            chunk_length=self.settings.chunk_seconds,
            batch_size=self.settings.batch_size,
        )
        segments: list[Segment] = []
        for raw in raw_segments:
            segment = Segment(raw.start, raw.end, to_hinglish(raw.text.strip()))
            segments.append(segment)
            if on_segment is not None:
                on_segment(segment)

        return Transcript(
            source=path,
            detected_language=detected,
            detected_probability=probability,
            language=language,
            audio_seconds=duration_seconds(audio),
            elapsed_seconds=time.perf_counter() - started,
            segments=segments,
        )

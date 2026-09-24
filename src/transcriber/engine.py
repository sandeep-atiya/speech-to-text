"""Model loading, language detection and transcription."""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import ctranslate2
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.tokenizer import Tokenizer

from transcriber.audio import duration_seconds, load_audio
from transcriber.chunking import Chunk, halve, loudness, plan_chunks
from transcriber.cleanup import collapse_repeats
from transcriber.config import SAMPLE_RATE, Settings
from transcriber.glossary import hotwords_from, load_custom_words, load_glossary
from transcriber.hinglish import add_words, to_hinglish

log = logging.getLogger(__name__)

# Whisper detects the language from the first 30 s of audio.
DETECTION_SECONDS = 30
# CTranslate2 stops generating after 224 tokens (half the decoder's 448 positions, like
# OpenAI's sample_len) whatever max_length or max_new_tokens say. That is about 13 s of
# fast Hindi in Devanagari, and the audio after the cut-off is silently dropped.
MAX_OUTPUT_TOKENS = 224
# A chunk shorter than this that still overflows the decoder is a repetition loop, not speech.
MIN_SPLIT_SECONDS = 2.0


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


class SpeechModel(Protocol):
    """The part of faster_whisper.WhisperModel this module uses (lets tests pass a fake)."""

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]: ...


class SpeechPipeline(Protocol):
    """The part of faster_whisper.BatchedInferencePipeline this module uses."""

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]: ...


def resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    """Turn "auto" choices into concrete values.

    float16 is only supported on GPU; int8 is the fastest option on CPU.
    """
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def load_model(settings: Settings) -> tuple[WhisperModel, str, str]:
    """Load the Whisper model described by settings; returns (model, device, compute_type)."""
    device, compute_type = resolve_device(settings.device, settings.compute_type)
    log.info("Loading '%s' on %s (%s)...", settings.model_size, device, compute_type)
    started = time.perf_counter()
    model = WhisperModel(
        settings.model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=settings.cpu_threads,
    )
    log.info("Model loaded in %.1fs", time.perf_counter() - started)
    return model, device, compute_type


def token_budget(model: Any, hotwords: str | None) -> int | None:
    """How many text tokens the decoder may emit per chunk; None for a stand-in model without a tokenizer.

    The decoder's 448 positions are shared with the prompt (hotwords and control tokens),
    and generation stops at MAX_OUTPUT_TOKENS anyway. Whisper normally stops by itself
    with an end token, so a chunk that fills the whole budget was cut off, and the audio
    after the cut was never written down.
    """
    hf_tokenizer = getattr(model, "hf_tokenizer", None)
    if hf_tokenizer is None:
        return None
    tokenizer = Tokenizer(hf_tokenizer, True, task="transcribe", language="hi")
    prompt = model.get_prompt(tokenizer, [], without_timestamps=True, hotwords=hotwords)
    return min(MAX_OUTPUT_TOKENS, int(model.max_length) - len(prompt))


class Transcriber:
    """Loads a Whisper model once and transcribes recordings into Hinglish.

    Usage:
        transcriber = Transcriber(Settings())
        transcript = transcriber.transcribe(Path("recordings/call.wav"))

    A model and pipeline can be injected (tests do this to avoid loading 1.6 GB of weights).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        model: SpeechModel | None = None,
        pipeline: SpeechPipeline | None = None,
    ) -> None:
        self.settings = settings

        custom_words = load_custom_words(settings.custom_words_file)
        if custom_words:
            add_words(custom_words)
            log.info("Loaded %d custom spellings from %s", len(custom_words), settings.custom_words_file)
        glossary = load_glossary(settings.glossary_file)
        self.hotwords = hotwords_from(glossary)
        if glossary:
            log.info("Loaded %d glossary names from %s", len(glossary), settings.glossary_file)

        if model is None:
            model, self.device, self.compute_type = load_model(settings)
        else:
            self.device, self.compute_type = resolve_device(settings.device, settings.compute_type)
        self.model = model
        self.token_budget = token_budget(model, self.hotwords)
        # The batched pipeline decodes each chunk on its own; chunking.py chooses the chunks.
        self.pipeline = pipeline if pipeline is not None else BatchedInferencePipeline(model=model)

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Return Whisper's (language, probability) guess for the start of the audio.

        Nothing is decoded: transcribe() only runs the encoder until segments are read.
        """
        head = audio[: DETECTION_SECONDS * SAMPLE_RATE]
        _, info = self.model.transcribe(head, language=None)
        return info.language, info.language_probability

    def choose_language(self, detected: str, probability: float, forced: str | None = None) -> str:
        """The language to decode with: a forced choice, else Hindi unless the audio is clearly English."""
        forced = forced or self.settings.language
        if forced != "auto":
            return forced
        if detected == "en" and probability >= self.settings.english_threshold:
            return "en"
        return "hi"

    def transcribe(
        self,
        path: Path,
        on_segment: Callable[[Segment], None] | None = None,
        on_start: Callable[[float], None] | None = None,
        language: str | None = None,
    ) -> Transcript:
        """Transcribe one recording.

        on_start receives the audio length in seconds once the file is decoded;
        on_segment is called for each line as soon as it is ready;
        language overrides the configured language for this call ("auto", "hi", "en").
        """
        started = time.perf_counter()
        audio = load_audio(path, self.settings.limit_seconds)
        audio_seconds = duration_seconds(audio)
        if on_start is not None:
            on_start(audio_seconds)

        detected, probability = self.detect_language(audio)
        language = self.choose_language(detected, probability, language)
        log.info("Detected language '%s' (%.2f) -> transcribing as '%s'", detected, probability, language)

        loud = loudness(audio)
        chunks = plan_chunks(
            audio,
            self.settings.chunk_seconds,
            self.settings.speech_threshold,
            self.settings.skip_silence_seconds,
            loud,
        )
        speech_seconds = sum(chunk.seconds for chunk in chunks)
        log.info(
            "Decoding %.0fs of speech in %d chunk(s), skipping %.0fs of silence",
            speech_seconds,
            len(chunks),
            audio_seconds - speech_seconds,
        )

        segments: list[Segment] = []
        for raw in self.decode(audio, loud, chunks, language):
            segment = Segment(raw.start, raw.end, collapse_repeats(to_hinglish(raw.text.strip())))
            segments.append(segment)
            if on_segment is not None:
                on_segment(segment)

        return Transcript(
            source=path,
            detected_language=detected,
            detected_probability=probability,
            language=language,
            audio_seconds=audio_seconds,
            elapsed_seconds=time.perf_counter() - started,
            segments=segments,
        )

    def decode(self, audio: np.ndarray, loud: np.ndarray, chunks: list[Chunk], language: str) -> Iterator[Any]:
        """Raw faster-whisper segments for the chunks, in order.

        A chunk that fills the whole token budget was cut off by the decoder, so it is
        decoded again as two halves (and those again, if needed) until every word is in.
        """
        if not chunks:
            return
        raw_segments, _ = self.pipeline.transcribe(
            audio,
            language=language,
            beam_size=self.settings.beam_size,
            batch_size=self.settings.batch_size,
            hotwords=self.hotwords,
            clip_timestamps=[{"start": chunk.start, "end": chunk.end} for chunk in chunks],
        )
        for raw in raw_segments:
            if self.ran_out_of_tokens(raw) and raw.end - raw.start >= MIN_SPLIT_SECONDS:
                log.warning(
                    "[%.1fs -> %.1fs] filled the decoder's token budget; decoding it in two parts", raw.start, raw.end
                )
                yield from self.decode(audio, loud, halve(Chunk(raw.start, raw.end), loud), language)
            else:
                yield raw

    def ran_out_of_tokens(self, raw: Any) -> bool:
        """True when a chunk's text stopped at the token limit (measured: 224 tokens, 223 with a shorter max_length)."""
        return self.token_budget is not None and len(raw.tokens) >= self.token_budget - 1

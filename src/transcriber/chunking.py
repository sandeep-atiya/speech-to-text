"""Cutting a recording into the chunks Whisper decodes one at a time.

Whisper decodes at most 30 s at once, and Devanagari uses so many tokens that even 30 s
can overflow the decoder. faster-whisper's own splitter therefore cuts speech at a length
limit wherever that falls: in the middle of a word when nobody pauses, leaving a short
remainder that is decoded on its own with no context. Both lose words.

This module plans the chunks instead:

* Silero VAD finds the speech. Only silences of at least skip_silence_seconds are left
  out; shorter pauses are decoded together with the speech around them, so a quiet
  caller the detector is unsure about is never dropped.
* Speech longer than chunk_seconds is cut into evenly sized pieces, each cut placed at
  the quietest moment near the even split, i.e. in a gap between words. Even sizes
  mean no 1 s leftovers.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

from transcriber.config import SAMPLE_RATE

log = logging.getLogger(__name__)

FRAME_SECONDS = 0.02  # loudness is measured per 20 ms frame...
SMOOTH_FRAMES = 3  # ...and averaged over 60 ms, so one quiet frame inside a word is not taken for a pause
SEARCH_FRACTION = 0.25  # a cut may move this fraction of chunk_seconds either side of the even split
SPEECH_PAD_SECONDS = 0.4  # kept around each stretch of speech so its first and last word are whole
MIN_SPEECH_SECONDS = 0.25  # shorter blips on their own are clicks, not words

Samples = tuple[int, int]  # a [start, end) range of samples


@dataclass(frozen=True)
class Chunk:
    """A [start, end) range of the recording, in seconds."""

    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


def speech_regions(audio: np.ndarray, threshold: float, skip_silence_seconds: float) -> list[Samples]:
    """Sample ranges of speech, separated only by silences of at least skip_silence_seconds."""
    options = VadOptions(
        threshold=threshold,
        min_speech_duration_ms=int(MIN_SPEECH_SECONDS * 1000),
        max_speech_duration_s=float("inf"),
        min_silence_duration_ms=int(skip_silence_seconds * 1000),
        speech_pad_ms=int(SPEECH_PAD_SECONDS * 1000),
    )
    regions = get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE)
    return [(int(region["start"]), int(region["end"])) for region in regions]


def loudness(audio: np.ndarray) -> np.ndarray:
    """Smoothed RMS level of each FRAME_SECONDS frame."""
    frame = int(FRAME_SECONDS * SAMPLE_RATE)
    count = len(audio) // frame
    if count == 0:
        return np.zeros(0)
    frames = audio[: count * frame].reshape(count, frame).astype(np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return np.convolve(rms, np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES, mode="same")


def quietest_sample(loud: np.ndarray, lo: int, hi: int) -> int:
    """The start of the quietest frame in samples [lo, hi); the middle when no whole frame fits."""
    frame = int(FRAME_SECONDS * SAMPLE_RATE)
    first, last = math.ceil(lo / frame), hi // frame
    if last <= first:
        return (lo + hi) // 2
    return int(first + np.argmin(loud[first:last])) * frame


def split_evenly(region: Samples, chunk_samples: int, loud: np.ndarray) -> list[Samples]:
    """Cut a region into evenly sized pieces of at most chunk_samples, each cut at the quietest nearby moment."""
    start, end = region
    pieces: list[Samples] = []
    while end - start > chunk_samples:
        remaining = end - start
        target = remaining / math.ceil(remaining / chunk_samples)  # even share for this piece
        slack = SEARCH_FRACTION * chunk_samples
        cut = quietest_sample(
            loud, int(start + target - slack), int(min(start + chunk_samples, start + target + slack))
        )
        pieces.append((start, cut))
        start = cut
    pieces.append((start, end))
    return pieces


def halve(chunk: Chunk, loud: np.ndarray) -> list[Chunk]:
    """Split a chunk in two at the quietest moment in its middle half."""
    start, end = int(chunk.start * SAMPLE_RATE), int(chunk.end * SAMPLE_RATE)
    middle, slack = start + (end - start) // 2, int(SEARCH_FRACTION * (end - start))
    cut = quietest_sample(loud, middle - slack, middle + slack)
    return [Chunk(start / SAMPLE_RATE, cut / SAMPLE_RATE), Chunk(cut / SAMPLE_RATE, end / SAMPLE_RATE)]


def plan_chunks(
    audio: np.ndarray,
    chunk_seconds: float,
    speech_threshold: float,
    skip_silence_seconds: float,
    loud: np.ndarray | None = None,
) -> list[Chunk]:
    """The chunks to decode, in order. Every moment of speech is in exactly one chunk."""
    if len(audio) == 0:
        return []
    regions = speech_regions(audio, speech_threshold, skip_silence_seconds)
    if not regions:
        log.warning("The voice detector found no speech; decoding the whole recording anyway.")
        regions = [(0, len(audio))]
    if loud is None:
        loud = loudness(audio)
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    return [
        Chunk(start / SAMPLE_RATE, end / SAMPLE_RATE)
        for region in regions
        for start, end in split_evenly(region, chunk_samples, loud)
    ]

"""Tests for chunk planning: cuts land in pauses, pieces are even, nothing is left out."""

import numpy as np
import pytest

from transcriber import chunking
from transcriber.chunking import Chunk, halve, loudness, plan_chunks, quietest_sample, split_evenly
from transcriber.config import SAMPLE_RATE

FRAME = int(chunking.FRAME_SECONDS * SAMPLE_RATE)


def noise_with_gaps(seconds: float, gaps: list[tuple[float, float]] | None = None, seed: int = 0) -> np.ndarray:
    """Loud noise (stands in for speech) with silent gaps at the given (start, end) seconds."""
    rng = np.random.default_rng(seed)
    audio = rng.uniform(-0.5, 0.5, int(seconds * SAMPLE_RATE)).astype(np.float32)
    for start, end in gaps or []:
        audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = 0.0
    return audio


def in_seconds(pieces: list[tuple[int, int]]) -> list[tuple[float, float]]:
    return [(start / SAMPLE_RATE, end / SAMPLE_RATE) for start, end in pieces]


def test_loudness_is_one_value_per_frame_and_low_in_gaps() -> None:
    loud = loudness(noise_with_gaps(2.0, [(1.0, 1.5)]))

    assert len(loud) == 100
    assert loud[60] < loud[20] / 100  # the frame at 1.2 s is inside the gap
    assert len(loudness(np.zeros(0, dtype=np.float32))) == 0


def test_quietest_sample_is_frame_aligned_and_inside_the_window() -> None:
    loud = np.ones(100)
    loud[42] = 0.0

    assert quietest_sample(loud, 30 * FRAME, 60 * FRAME) == 42 * FRAME
    assert quietest_sample(loud, 45 * FRAME, 60 * FRAME) == 45 * FRAME  # 42 is outside the window
    assert quietest_sample(loud, 100, 200) == 150  # no whole frame fits: the middle


def test_split_evenly_cuts_in_the_pauses() -> None:
    # 40 s of speech with short pauses at 13 s and 26.5 s: three pieces, both cuts in a pause
    audio = noise_with_gaps(40.0, [(13.0, 13.2), (26.5, 26.7)])

    pieces = in_seconds(split_evenly((0, len(audio)), 15 * SAMPLE_RATE, loudness(audio)))

    assert len(pieces) == 3
    assert pieces[0][0] == 0.0 and pieces[-1][1] == 40.0
    assert all(pieces[i][1] == pieces[i + 1][0] for i in range(2))  # contiguous: nothing lost
    assert 13.0 <= pieces[0][1] <= 13.2
    assert 26.5 <= pieces[1][1] <= 26.7


def test_split_evenly_makes_even_pieces_instead_of_a_tiny_leftover() -> None:
    audio = noise_with_gaps(17.3)  # no pauses at all

    pieces = in_seconds(split_evenly((0, len(audio)), 15 * SAMPLE_RATE, loudness(audio)))

    assert len(pieces) == 2
    assert all(4.9 <= end - start <= 12.5 for start, end in pieces)


def test_split_evenly_keeps_a_short_region_whole() -> None:
    audio = noise_with_gaps(5.0)

    assert split_evenly((0, len(audio)), 15 * SAMPLE_RATE, loudness(audio)) == [(0, len(audio))]


def test_halve_splits_in_a_pause_near_the_middle() -> None:
    audio = noise_with_gaps(10.0, [(6.2, 6.3)])

    halves = halve(Chunk(0.0, 10.0), loudness(audio))

    assert len(halves) == 2
    assert halves[0].start == 0.0 and halves[1].end == 10.0
    assert halves[0].end == halves[1].start
    assert 6.2 <= halves[0].end <= 6.3


def test_halve_always_gives_two_pieces_even_without_a_pause() -> None:
    halves = halve(Chunk(2.0, 12.0), loudness(np.zeros(SAMPLE_RATE * 12, dtype=np.float32)))

    assert len(halves) == 2
    assert halves[0].start == 2.0 and halves[1].end == 12.0
    assert 4.5 <= halves[0].end <= 9.5


def test_plan_chunks_keeps_detected_speech_and_splits_long_stretches(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = noise_with_gaps(40.0, [(5.0, 10.0), (23.0, 23.2)])
    regions = [(0, 5 * SAMPLE_RATE), (10 * SAMPLE_RATE, 40 * SAMPLE_RATE)]
    monkeypatch.setattr(chunking, "speech_regions", lambda audio, threshold, skip: regions)

    chunks = plan_chunks(audio, chunk_seconds=15, speech_threshold=0.3, skip_silence_seconds=3.0)

    assert chunks[0] == Chunk(0.0, 5.0)
    assert chunks[1].start == 10.0
    assert 23.0 <= chunks[1].end <= 23.2  # the 30 s stretch is first cut in its pause
    assert chunks[-1].end == 40.0
    assert all(chunk.seconds <= 15 for chunk in chunks)
    assert all(chunks[i].end == chunks[i + 1].start for i in range(1, len(chunks) - 1))


def test_plan_chunks_decodes_everything_when_no_speech_is_found() -> None:
    silence = np.zeros(SAMPLE_RATE * 4, dtype=np.float32)  # the real voice detector runs on this

    assert plan_chunks(silence, 15, 0.3, 3.0) == [Chunk(0.0, 4.0)]


def test_plan_chunks_of_empty_audio_is_empty() -> None:
    assert plan_chunks(np.zeros(0, dtype=np.float32), 15, 0.3, 3.0) == []

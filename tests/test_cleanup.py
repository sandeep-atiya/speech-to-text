"""Tests for transcript cleanup."""

import pytest

from transcriber.cleanup import collapse_repeats


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # a looping phrase is cut to one occurrence
        (
            "hello, hari prasaad ji, hari prasaad ji, hari prasaad ji, hari prasaad ji, to",
            "hello, hari prasaad ji, to",
        ),
        # a phrase said twice is natural speech and stays
        ("theek hai theek hai bhai", "theek hai theek hai bhai"),
        # single words: three in a row stay, more are cut to two
        ("ji ji ji", "ji ji ji"),
        ("ji ji ji ji ji ji ji", "ji ji"),
        # nothing to do
        ("aap kaise hain", "aap kaise hain"),
        ("", ""),
    ],
)
def test_collapse_repeats(text: str, expected: str) -> None:
    assert collapse_repeats(text) == expected

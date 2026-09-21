"""Devanagari (Hindi/Urdu) -> Hinglish conversion.

    >>> to_hinglish("मुझे capsule चाहिए")
    'mujhe capsule chahiye'

Spellings for common words and English loanwords live in words.py (edit that table to
change a spelling); everything else is produced by the rules in rules.py. Extra spellings
can be added at runtime with add_words(), e.g. from a custom_words.json file.
"""

from collections.abc import Mapping

from transcriber.hinglish.rules import to_hinglish, word_to_hinglish
from transcriber.hinglish.words import WORDS

__all__ = ["WORDS", "add_words", "to_hinglish", "word_to_hinglish"]


def add_words(spellings: Mapping[str, str]) -> None:
    """Add or override spellings in the table for this process."""
    WORDS.update(spellings)

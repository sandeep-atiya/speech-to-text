"""Devanagari (Hindi/Urdu) -> Hinglish conversion.

    >>> to_hinglish("मुझे capsule चाहिए")
    'mujhe capsule chahiye'

Spellings for common words and English loanwords live in words.py (edit that table to
change a spelling); everything else is produced by the rules in rules.py.
"""

from transcriber.hinglish.rules import to_hinglish, word_to_hinglish
from transcriber.hinglish.words import WORDS

__all__ = ["to_hinglish", "word_to_hinglish", "WORDS"]

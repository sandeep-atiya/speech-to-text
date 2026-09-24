"""Tests for the Devanagari -> Hinglish converter."""

import pytest

from transcriber.hinglish import to_hinglish, word_to_hinglish


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # spelling table
        ("में", "mein"),
        ("नहीं", "nahi"),
        ("कैप्सूल", "capsule"),
        ("टीवी", "TV"),
        ("सिर्फ", "sirf"),  # Urdu loan: "f", not the native "ph"
        ("तकलीफ़", "takleef"),
        ("युनानी", "unani"),  # short-u spelling Whisper also uses
        # schwa deletion
        ("समझाने", "samjhane"),
        ("समझ", "samajh"),
        ("कितने", "kitne"),
        ("लोग", "log"),
        # long "aa" only in closed syllables
        ("बात", "baat"),
        ("नाम", "naam"),
        ("करना", "karna"),
        ("चाहते", "chahte"),
        ("आदमी", "aadmi"),
        # nasal signs
        ("सुरंजान", "suranjaan"),
        ("अश्वगंधा", "ashvagandha"),
        ("करेंगे", "karenge"),
        ("बताएं", "batayein"),
        ("संभव", "sambhav"),
        # vowel sequences
        ("गए", "gaye"),
        ("भाई", "bhai"),
        ("जाओ", "jao"),
        # nukta letters
        ("ज़रूर", "zaroor"),
        ("मर्ज़", "marz"),
        ("पढ़ना", "parhna"),
    ],
)
def test_word(word: str, expected: str) -> None:
    assert word_to_hinglish(word) == expected


@pytest.mark.parametrize(
    ("devanagari", "expected"),
    [
        (
            "हम लोग एक मॉक कॉल करके आपको समझाने की कोशिश करेंगे",
            "hum log ek mock call karke aapko samjhane ki koshish karenge",
        ),
        (
            "कितने वक्त से आपको जोड़ों में दर्द है",
            "kitne waqt se aapko jodon mein dard hai",
        ),
        (
            "हकीम साहब बताते हैं सुरंजान डालो इसमें अश्वगंधा डालो हल्दी डालो",
            "hakim sahab batate hain suranjaan dalo ismein ashvagandha dalo haldi dalo",
        ),
        (
            "दूसरी कैटेगरी जो लोग हकीम साहब की दवा से फ़ायदा हुआ है",
            "doosri category jo log hakim sahab ki dawa se fayda hua hai",
        ),
        (
            "क्या मेरी बात उनसे नहीं हो सकती है। ठीक है 2 महीने",
            "kya meri baat unse nahi ho sakti hai. theek hai 2 mahine",
        ),
    ],
)
def test_sentence(devanagari: str, expected: str) -> None:
    assert to_hinglish(devanagari) == expected


def test_latin_text_and_punctuation_pass_through() -> None:
    assert to_hinglish("aap already कैप्सूल भी खा रहे हैं?") == "aap already capsule bhi kha rahe hain?"
    assert to_hinglish("Hello, world!") == "Hello, world!"
    assert to_hinglish("") == ""


def test_digits_and_danda() -> None:
    assert to_hinglish("२०२६ में।") == "2026 mein."


def test_invisible_characters_are_removed() -> None:
    zero_width_space, replacement_char = "​", "�"
    assert to_hinglish(f"ठीक{zero_width_space} है{replacement_char}") == "theek hai"

"""Unit tests for the Devanagari -> Hinglish converter.

Run from the project folder:  python -m unittest discover tests
"""

import unittest

from transcriber.hinglish import to_hinglish, word_to_hinglish


class SpellingTableTests(unittest.TestCase):
    def test_common_words_use_table_spelling(self):
        self.assertEqual(word_to_hinglish("में"), "mein")
        self.assertEqual(word_to_hinglish("नहीं"), "nahi")
        self.assertEqual(word_to_hinglish("कैप्सूल"), "capsule")
        self.assertEqual(word_to_hinglish("टीवी"), "TV")


class RuleTests(unittest.TestCase):
    def test_schwa_deletion(self):
        self.assertEqual(word_to_hinglish("समझाने"), "samjhane")
        self.assertEqual(word_to_hinglish("समझ"), "samajh")
        self.assertEqual(word_to_hinglish("कितने"), "kitne")
        self.assertEqual(word_to_hinglish("लोग"), "log")

    def test_long_a_only_in_closed_syllables(self):
        self.assertEqual(word_to_hinglish("बात"), "baat")
        self.assertEqual(word_to_hinglish("नाम"), "naam")
        self.assertEqual(word_to_hinglish("करना"), "karna")
        self.assertEqual(word_to_hinglish("चाहते"), "chahte")
        self.assertEqual(word_to_hinglish("आदमी"), "aadmi")

    def test_nasal_signs(self):
        self.assertEqual(word_to_hinglish("सुरंजान"), "suranjaan")
        self.assertEqual(word_to_hinglish("अश्वगंधा"), "ashvagandha")
        self.assertEqual(word_to_hinglish("करेंगे"), "karenge")
        self.assertEqual(word_to_hinglish("बताएं"), "batayein")
        self.assertEqual(word_to_hinglish("संभव"), "sambhav")

    def test_vowel_sequences(self):
        self.assertEqual(word_to_hinglish("गए"), "gaye")
        self.assertEqual(word_to_hinglish("भाई"), "bhai")
        self.assertEqual(word_to_hinglish("जाओ"), "jao")

    def test_nukta_letters(self):
        self.assertEqual(word_to_hinglish("ज़रूर"), "zaroor")
        self.assertEqual(word_to_hinglish("मर्ज़"), "marz")
        self.assertEqual(word_to_hinglish("पढ़ना"), "parhna")


class SentenceTests(unittest.TestCase):
    def test_full_sentences(self):
        cases = {
            "हम लोग एक मॉक कॉल करके आपको समझाने की कोशिश करेंगे":
                "hum log ek mock call karke aapko samjhane ki koshish karenge",
            "कितने वक्त से आपको जोड़ों में दर्द है":
                "kitne waqt se aapko jodon mein dard hai",
            "हकीम साहब बताते हैं सुरंजान डालो इसमें अश्वगंधा डालो हल्दी डालो":
                "hakim sahab batate hain suranjaan dalo ismein ashvagandha dalo haldi dalo",
            "दूसरी कैटेगरी जो लोग हकीम साहब की दवा से फ़ायदा हुआ है":
                "doosri category jo log hakim sahab ki dawa se fayda hua hai",
            "क्या मेरी बात उनसे नहीं हो सकती है। ठीक है 2 महीने":
                "kya meri baat unse nahi ho sakti hai. theek hai 2 mahine",
        }
        for devanagari, expected in cases.items():
            with self.subTest(devanagari=devanagari):
                self.assertEqual(to_hinglish(devanagari), expected)

    def test_latin_text_and_punctuation_pass_through(self):
        self.assertEqual(to_hinglish("aap already कैप्सूल भी खा रहे हैं?"), "aap already capsule bhi kha rahe hain?")
        self.assertEqual(to_hinglish("Hello, world!"), "Hello, world!")
        self.assertEqual(to_hinglish(""), "")

    def test_digits_and_danda(self):
        self.assertEqual(to_hinglish("२०२६ में।"), "2026 mein.")

    def test_invisible_characters_are_removed(self):
        self.assertEqual(to_hinglish("ठीक​ है�"), "theek hai")


if __name__ == "__main__":
    unittest.main()

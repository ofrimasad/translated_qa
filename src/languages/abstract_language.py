from typing import List

from utils.translation_utils import WordSpliter, SentenceSpliter


class Language:

    symbol = ""
    alphabet = ""
    valid_chars = ".,:;'\"()[]{}?!@#$%&- \t"

    @classmethod
    def pre_translation_callback(cls, text: str) -> str:
        return text

    @classmethod
    def post_translation_callback(cls, text: str) -> str:
        return text

    @classmethod
    def is_lang(cls, text: str) -> bool:
        count = 0
        language_symbols = cls.valid_chars + cls.alphabet
        max_count = min(len(text), 20)
        for c in text[:max_count]:
            count += c in language_symbols

        return count / max_count > 0.8

    @classmethod
    def split_to_words(cls, text: str) -> List[str]:
        return WordSpliter.word_split(text)

    @classmethod
    def split_to_sentences(cls, text: str) -> List[str]:
        return SentenceSpliter.sentence_split(text)



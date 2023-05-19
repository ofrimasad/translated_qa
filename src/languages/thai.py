from typing import List

from languages.abstract_language import Language
from utils.translation_utils import ThaiSentenceSpliter, ThaiWordSpliter


class Thai(Language):

    symbol = "th"

    @classmethod
    def split_to_words(cls, text: str) -> List[str]:
        return ThaiWordSpliter.word_split(text)

    @classmethod
    def split_to_sentences(cls, text: str) -> List[str]:
        return ThaiSentenceSpliter.sentence_split(text)
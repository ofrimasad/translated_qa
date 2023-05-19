from typing import List

from languages.abstract_language import Language
from utils.translation_utils import ChineseWordSpliter, ChineseSentenceSpliter


class Chinese(Language):

    symbol = "zh-CN"

    @classmethod
    def split_to_words(cls, text: str) -> List[str]:
        return ChineseWordSpliter.word_split(text)

    @classmethod
    def split_to_sentences(cls, text: str) -> List[str]:
        return ChineseSentenceSpliter.sentence_split(text)
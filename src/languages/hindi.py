import re
from typing import List

from languages.abstract_language import Language
from utils.translation_utils import IndicSentenceSpliter


class Hindi(Language):

    symbol = "hi"

    @classmethod
    def split_to_sentences(cls, text: str) -> List[str]:
        return IndicSentenceSpliter.sentence_split(text)
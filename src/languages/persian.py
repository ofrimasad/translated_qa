import re

from languages.abstract_language import Language


class Persian(Language):

    symbol = "fa"
    alphabet = ""  # TODO
    diacritics = "".join([chr(i) for i in range(1611, 1619)])  # tashkīl


    @classmethod
    def post_translation_callback(cls, text: str) -> str:
        return cls.remove_diacritics(text)

    @classmethod
    def remove_diacritics(cls, text: str):
        return re.sub(fr"[{cls.diacritics}]", '', text)


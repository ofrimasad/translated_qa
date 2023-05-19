import re

from languages.abstract_language import Language


class Hebrew(Language):

    symbol = "iw"
    alphabet = "אבגדהוזחטיכלמנסעפצקרשתףץםן"
    niqqud = "".join([chr(i) for i in range(1456, 1470)])


    @classmethod
    def post_translation_callback(cls, text: str) -> str:
        return cls.remove_niqqud(text)

    @classmethod
    def remove_niqqud(cls, text: str):
        return re.sub(fr"[{cls.niqqud}]", '', text)


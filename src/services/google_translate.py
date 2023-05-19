import time
from typing import Type

from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests

from languages.abstract_language import Language
from services.abstract_service import AbstractTranslationService


class GoogleTranslate(AbstractTranslationService):

    TEXT_SEPARATOR = "\n\n"
    DELAYS = [0, 1, 5, 30, 30]

    def __init__(self, source: Type[Language], target: Type[Language]):
        super().__init__(source, target)
        self.translator = GoogleTranslator(source=source.symbol, target=target.symbol)

    def translate(self, text: str) -> str:
        for delay in self.DELAYS:
            time.sleep(delay)
            try:
                translation =  self.translator.translate(text)
                return translation
            except (TooManyRequests, RuntimeError):
                continue
        raise TooManyRequests



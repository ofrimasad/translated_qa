from typing import Type

from languages.abstract_language import Language
from services.abstract_service import AbstractTranslationService


class CounterService(AbstractTranslationService):

    def __init__(self, source: Type[Language], target: Type[Language]):
        super().__init__(source, target)
        self.num_calls = 0
        self.num_chars = 0

    def translate(self, text: str) -> str:
        self.num_calls += 1
        self.num_chars += (len(text))
        return text

    def __repr__(self):
        return f'Total calls:\t\t{self.num_calls}\n Total chars:\t\t{self.num_chars}'

from typing import Type

from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

from languages.abstract_language import Language
from services.abstract_service import AbstractTranslationService


class ModelTranslate(AbstractTranslationService):

    TEXT_SEPARATOR = "\n\n"

    def __init__(self, source: Type[Language], target: Type[Language]):
        super().__init__(source, target)

        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B", src_lang=source.symbol, device=0)
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
        self.source = source
        self.target = target
        self.model.to("cuda:0")

    def translate(self, text: str) -> str:
        encoded_zh = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_zh, forced_bos_token_id=self.tokenizer.get_lang_id(self.target.symbol))
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    def translate_together(self, texts: list) -> list:
        encoded = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=2000).to("cuda:0")
        generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.get_lang_id(self.target.symbol))
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


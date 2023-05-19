from typing import Union, Type

from tensorboard.compat.tensorflow_stub.errors import UnimplementedError

from languages.abstract_language import Language


def _handle_exceptions(texts: list):
    if texts[0] == 'tanrı': texts[0] = 'tanrı.'
    return texts


class AbstractTranslationService:

    def __init__(self, source: Type[Language], target: Type[Language]):
        self.source = source
        self.target = target

    def translate(self, text: str) -> str:
        raise UnimplementedError

    def translate_together(self, texts: list) -> list:
        """
        concatenate the texts in the list (separated by @separator).
        translate the concatenated text
        split he result by @separator

        if the concatenated text len exceeds @max_len, it is assumed the the first text (index 0 in the list)
        is the context and should be concatenated to each sub section of the list. the list will be translated
        in several calls

        :param texts: list of texts. texts[1] is assumed to be the context
        :return: translated list with the same len
        """
        texts = _handle_exceptions(texts)
        full_text = self.separator.join(texts)
        if len(texts) == 1 or len(texts[1]) > self.max_len:
            return texts

        if len(full_text) < self.max_len:
            full_text = self.source.pre_translation_callback(full_text)
            full_text_translated = self.translate(full_text)
            full_text_translated = self.target.post_translation_callback(full_text_translated)
            translated_list = full_text_translated.split(self.separator)
        else:
            # if text is too long - recurse in two parts
            assert len(texts) > 2
            center = len(texts) // 2
            sublist_a = self.translate_together(texts[0:center])
            sublist_b = self.translate_together(texts[1:2] + texts[center:])
            translated_list = sublist_a + sublist_b[1:]

        if len(texts) != len(translated_list):
            raise RuntimeError
        return translated_list

    def __call__(self, text: Union[str, list]):
        if isinstance(text, list):
            return self.translate_together(text)
        else:
            self.translate(text)

    @property
    def separator(self) -> str:
        return "\n\n"

    @property
    def max_len(self) -> int:
        return 5000

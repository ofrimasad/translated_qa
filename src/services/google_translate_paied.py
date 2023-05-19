# Imports the Google Cloud Translation library
from typing import Type

from google.cloud import translate

from languages.abstract_language import Language
from services.abstract_service import AbstractTranslationService


class GoogleTranslateP(AbstractTranslationService):

    TEXT_SEPARATOR = "\n\n"

    def __init__(self, source: Type[Language], target: Type[Language]):
        super().__init__(source, target)
        self.translator = translate.TranslationServiceClient()
        self.source_symbol = source.symbol
        self.target_symbol = target.symbol
        self.parent = f"projects/norm-deconv/locations/global"

    def translate(self, text: str) -> str:

        response = self.translator.translate_text(
            request={
                "parent": self.parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": self.source_symbol,
                "target_language_code": self.target_symbol,
            }
        )

        # Display the translation for each input text provided
        return response.translations[0].translated_text

    @property
    def separator(self) -> str:
        return "<lbr>"

# Initialize Translation client
def translate_text(text="YOUR_TEXT_TO_TRANSLATE", project_id="norm-deconv"):
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "en-US",
            "target_language_code": "iw",
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        print("Translated text: {}".format(translation.translated_text))

# context = """
# The <aca>Normans</aca> (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who <aaf>in the <aae>10th and 11th centuries</aae> gave their name to <acb>Normandy</acb>, a region in <aaa>France</aaa>. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from <aai>Denmark, Iceland and Norway</aai> who, under their leader <abc>Rollo</abc>, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in <abh>the first half of the <abg>10th</abi> century</abg>, and it continued to evolve over the succeeding centuries.
# """
# translate_text(context)
from languages.arabic import Arabic
from languages.bengali import Bengali
from languages.chinese import Chinese
from languages.czech import Czech
from languages.dutch import Dutch
from languages.english import English
from languages.finnish import Finnish
from languages.french import French
from languages.german import German
from languages.greek import Greek
from languages.hebrew import Hebrew
from languages.hindi import Hindi
from languages.persian import Persian
from languages.russian import Russian
from languages.spanish import Spanish
from languages.swedish import Swedish
from languages.thai import Thai
from languages.turkish import Turkish
from languages.vietnamese import Vietnamese

__all__ = [English, Spanish, Hebrew, Persian, Arabic, Chinese, German, Greek, Hindi, Russian, Thai, Turkish, Vietnamese,
           French, Swedish, Finnish, Dutch, Czech, Bengali]

LANGUAGES = {}

for lang in __all__:
    LANGUAGES[lang.symbol] = lang

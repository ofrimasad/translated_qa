import re

from languages.abstract_language import Language


class Spanish(Language):

    symbol = "es"
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    connection_articles = ['un', 'una', 'unos', 'unas', 'el']


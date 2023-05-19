import re
from string import ascii_letters
from typing import List, Optional
import subprocess

import numpy as np
import string

from pythainlp import sent_tokenize
from transformers import BertTokenizerFast

from pythainlp.tokenize import word_tokenize
import pynlpir
from indicnlp.tokenize import sentence_tokenize

class DictionaryLink:
    def __init__(self, object: dict, label: str):
        self.object = object
        self.label = label

    def __set__(self, instance, value):
        instance.object[instance.label] = value

    def set(self, value):
        self.object[self.label] = value


class TextList:

    def __init__(self):
        self.links = []
        self.texts = []

    def append(self, dictionary: dict, name: str):
        self.links.append(DictionaryLink(dictionary, name))
        self.texts.append(dictionary[name].strip())

    def lists(self):
        return self.texts, self.links


class AlephBertTokenizerFast(BertTokenizerFast):

    def __call__(self, *args, **kwargs):
        res = super().__call__(*args, **kwargs)
        res['token_type_ids'] = [0] * len(res['token_type_ids'])
        return res

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        res = super().create_token_type_ids_from_sequences(token_ids_0, token_ids_1)
        return len(res) * [0]


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


class HtmlTagger:

    def __init__(self, text):
        self.next = 0
        self.map = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        self.cleaner = re.compile('<.*?>')
        self.starter = re.compile('<.*?>')
        self.ender = re.compile('</.*?>')
        self.shift_mapping = np.zeros([len(text) + 1], dtype=np.int)
        self.index_to_tag_map = {}

    def get_tags(self):
        tag = f'{self.map[self.next // 100]}{self.map[self.next % 100 // 10]}{self.map[self.next % 10]}'
        self.next += 1
        return f'<{tag}>', f'</{tag}>'

    def insert_tags(self, text: str, start: int, end: int, with_shift: bool = False):
        start_tag, end_tag = self.get_tags()

        orig_start = start
        orig_end = end

        start_tag_exist = False
        end_tag_exist = False

        if start in self.index_to_tag_map:
            start_tag = self.index_to_tag_map[start]
            start_tag_exist = True

        if end in self.index_to_tag_map:
            end_tag = self.index_to_tag_map[end]
            end_tag_exist = True

        if not end_tag_exist:
            if with_shift:
                end += self.get_text_shift(text, end)
            text = text[:end] + end_tag + text[end:]
            self.shift_mapping[orig_end + 1:] += 6
            self.index_to_tag_map[orig_end] = end_tag

        if not start_tag_exist:

            if with_shift:
                start += self.get_text_shift(text, start)

            # if 'in the' in text[start-7:start]:
            #     start -= 4

            text = text[:start] + start_tag + text[start:]
            self.shift_mapping[orig_start:] += 5
            self.index_to_tag_map[orig_start] = start_tag

        return text, start_tag, end_tag

    def clean(self, text):
        cleantext = re.sub(self.cleaner, '', text)
        return cleantext

    def get_text_shift(self, text, index):
        return self.shift_mapping[index]

    def get_text_unshift(self, text, index):
        sub = text[:index]
        end_tags = len(re.findall(self.ender, sub))
        start_tags = len(re.findall(self.starter, sub)) - end_tags

        return index-(end_tags * 6 + start_tags * 5)

    def fix_tags(self, text: str):
        tags = re.findall('</? ?[a-k]{3} ?>', text)
        for tag in tags:
            if ' ' in tag:
                fixed_tag = tag.replace(' ', '')
                text = text.replace(tag, fixed_tag)

        for tag in tags:
            if '/>' in tag:
                fixed_tag = tag.replace('/>', '>')
                text = text.replace(tag, fixed_tag)

        return text


class SentenceSpliter:


    abbreviations = ["St.", "No.", "Dr.", "Mr.", "Jr.", "vs.", "Inc.", "Mrs.", "Op.", "Corp.", "A.L.A.", "J.C.B.", "R.D.W.", "E.J.S.", "Bros.", "J.A.D.",
                     "Ecl.", "Col.", "T.R.M.", "U.S.C.", "Gen.", "Mt.", "Sch.", "ca.", "PT.", "Oct.", "Nov.", "Co.", "Lt.", "D.A.T.S.", "Lk.", "Capt.", "Adm.",
                     "Rev.", "Fr.", "BC–c."]

    for first in ascii_letters:
        abbreviations.append(f'{first}.')
        for second in ascii_letters:
            abbreviations.append(f'{first}.{second}.')

    for n in range(10):
        abbreviations.append(f'{n}.')

    with_brackets = []
    for abb in abbreviations:
        with_brackets.append(f'({abb}')
        with_brackets.append(f'"{abb}')

    abbreviations.extend(with_brackets)

    @classmethod
    def sentence_split(cls, s: str, full_stop: str = ".") -> List[str]:
        tokens = s.split(" ")
        out = []
        current = []
        for t in tokens:
            if t.endswith(full_stop) and t not in cls.abbreviations:
                current.append(t)
                out.append(" ".join(current))
                current = []
            else:
                current.append(t)
        if len(current) > 0:
            out.append(" ".join(current))
        return out

class WordSpliter:

    @classmethod
    def word_split(cls, s: str) -> List[str]:
        return s.split(" ")


class ThaiSentenceSpliter(SentenceSpliter):
    @classmethod
    def sentence_split(cls, s: str, full_stop: str = ".") -> List[str]:
        return sent_tokenize(s, keep_whitespace=False)

class ChineseSentenceSpliter(SentenceSpliter):
    @classmethod
    def sentence_split(cls, s: str, full_stop: str = "。") -> List[str]:
        return super().sentence_split(s, "。")


class IndicSentenceSpliter(SentenceSpliter):
    @classmethod
    def sentence_split(cls, s: str, full_stop: str = ".") -> List[str]:
        return sentence_tokenize.sentence_split(s, lang='hi')



class ThaiWordSpliter(WordSpliter):

    @classmethod
    def word_split(cls, s: str) -> List[str]:
        words = word_tokenize(s, engine="newmm", keep_whitespace=False)
        words = [w.strip() for w in words]
        words = [w for w in words if w not in string.punctuation]
        return [w for w in words if w != " "]


class ChineseWordSpliter(WordSpliter):
    pynlpir.open()
    @classmethod
    def word_split(cls, s: str) -> List[str]:
        try:
            words = pynlpir.segment(s, pos_tagging=False)
        except:
            words = [s]
        words = [w.strip() for w in words]
        words = [w for w in words if w not in string.punctuation]
        return words
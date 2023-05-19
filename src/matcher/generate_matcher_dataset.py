import argparse
import json
import logging
import random
import re
import sys
from os import path
from typing import List

import numpy as np
import os
from deep_translator.exceptions import NotValidPayload, NotValidLength, RequestError, TranslationNotFound
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from languages import LANGUAGES
from languages.abstract_language import Language
from languages.english import English
from services.google_translate import GoogleTranslate
from utils.convert_to_hf import squad_to_huggingface


class Stats:

    def __init__(self):
        self.num_sentences = 0
        self.num_possible_questions = 0
        self.num_possible_questions_enq = 0
        self.num_impossible_questions = 0

    def __str__(self):
        _str = ''
        _str += f'\nnumber of sentences: {self.num_sentences}'
        _str += f'\nnumber of possible questions: {self.num_possible_questions}'
        _str += f'\nnumber of impossible questions: {self.num_impossible_questions}'

        return _str


def random_positive_normal_int(low: int, high: int, scale: int = 3):
    x = np.random.normal(scale=scale)
    return int(np.clip(np.round(np.abs(x)), low, high))


def get_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(level)
    return logger


def add_impossibles(_paragraphs: list, _stats: Stats, next_id: int, count: bool = True):
    for paragraph_index, paragraph in enumerate(_paragraphs):
        num_impossible_to_add = max(1, len(paragraph['qas']) // 2)

        for i in range(num_impossible_to_add):
            other_paragraph = _paragraphs[(paragraph_index + 1024 + i) % len(_paragraphs)]
            other_question = other_paragraph['qas'][0]['question']

            # just make sure this question is not in the context (at least is original form)
            if other_question in paragraph['context']:
                continue

            legit_answer = paragraph['qas'][0]['answers'][0]
            paragraph['qas'].append({'id': f'ss{target.symbol}{next_id}',
                                     'is_impossible': True,
                                     'question': other_question,
                                     'plausible_answers': [{'text': legit_answer['text'],
                                                            'answer_start': legit_answer['answer_start']}]})
            next_id += 1
            if count:
                _stats.num_impossible_questions += 1
                writer.add_scalar(tag='generated_impossibles', scalar_value=_stats.num_impossible_questions, global_step=global_step)


def init_logger(_opt) -> logging.Logger:
    _logger = get_logger('data generator', level=logging.DEBUG if _opt.d else logging.INFO)

    _logger.info(f'Language: {_opt.language_sym}')
    _logger.info(f'English question: {_opt.enq}')
    _logger.info(f'Max len: {_opt.max_len_for_translation}')
    _logger.info(f'Min word count per sentence: {_opt.minimum_words_in_sentence}')
    _logger.info(f'Num Phrases per sentence: {_opt.num_phrases_in_sentence}')
    _logger.info(f'Max attempts per sentence: {_opt.max_attempts}')
    _logger.info(f'Max span: {_opt.max_span}')
    _logger.info(f'Scale: {_opt.scale}')
    _logger.info(f'Max sections: {_opt.max_sections}')
    _logger.info(f'Translated: {_opt.translated}')

    return _logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json', type=str)
    parser.add_argument('language_sym', type=str)
    parser.add_argument('--enq', action='store_true', help='question in english')
    parser.add_argument('-d', action='store_true', help='debug mode')
    parser.add_argument('--translated', action='store_true', help='is the context already translated')
    parser.add_argument('-r', '--readable', action='store_true', help='readable json output format')
    parser.add_argument('--hf', action='store_true', help='output in huggingface format')
    parser.add_argument('--max_len_for_translation', type=int, default=4500, help='maximum number of characters to send to translation')
    parser.add_argument('--minimum_words_in_sentence', type=int, default=15, help='minimum words allowed in a valid sentence')
    parser.add_argument('--num_phrases_in_sentence', type=int, default=4, help='number of phrases to extract from each sentence')
    parser.add_argument('--max_attempts', type=int, default=30, help='maximum number of attempts per sentence')
    parser.add_argument('--max_span', type=int, default=25, help='maximum number of words in a phrase')
    parser.add_argument('--scale', type=int, default=9, help='standard deviation of the distribution of number of words in a phrase')
    parser.add_argument('--max_sections', type=int, default=600, help='maximum number of original sections')
    parser.add_argument('--out_dir', type=str, help='output directory')
    parser.add_argument('--max_possible', type=int, default=500000, help='maximum number of possible answers')

    return parser.parse_args()


def init_writer(_opt: argparse.Namespace, target: Language) -> SummaryWriter:
    phase = 'dev' if 'dev' in opt.input_json else 'train'
    writer = SummaryWriter(log_dir=f'./data_gen_logs/{target.symbol}_{phase}{"_enq" if _opt.enq else ""}')
    writer.add_hparams(hparam_dict=_opt.__dict__, metric_dict={})
    return writer

def find_shortest(start_with:str, end_with:str, text:str) -> List[str]:
    start_loc = [i.regs[0][0] for i in re.finditer(re.escape(start_with), text)]
    end_loc = [i.regs[0][1] for i in re.finditer(re.escape(end_with), text)]

    min_len = 100000000
    min_text = ""
    for s in start_loc:
        for e in end_loc:
            if e > s and e - s < min_len:
                min_len = e - s
                min_text = text[s:e]
    return [min_text]


if __name__ == "__main__":

    opt = parse_args()

    logger = init_logger(opt)
    random.seed(42)
    stats = Stats()
    target = LANGUAGES[opt.language_sym]

    writer = init_writer(opt, target)

    output_json_enq = opt.input_json.replace('.json', f'_matcher_{target.symbol}_enq.json')
    output_json = opt.input_json.replace('.json', f'_matcher_{target.symbol}.json')

    if opt.hf:
        output_json_enq = output_json_enq.replace("v2.0", "v2.0hf")
        output_json = output_json.replace("v2.0", "v2.0hf")

    if opt.out_dir is not None:
        os.makedirs(opt.out_dir, exist_ok=True)
        output_json = path.join(opt.out_dir, path.split(output_json)[-1])
        output_json_enq = path.join(opt.out_dir, path.split(output_json_enq)[-1])

    translator = GoogleTranslate(source=English, target=target)
    inverse_translator = GoogleTranslate(source=target, target=English)

    with open(opt.input_json) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    new_paragraphs = []
    new_data = [{'paragraphs': new_paragraphs}]
    new_paragraphs_enq = []
    new_data_enq = [{'paragraphs': new_paragraphs_enq}]
    next_id = 10000000000000000000

    global_step = 0
    if len(data) > opt.max_sections:
        data = data[0: opt.max_sections]

    try:
        for ind, subject in enumerate(data):
            writer.add_scalar(tag='sections', scalar_value=ind, global_step=global_step)
            logger.info(f'Section {ind + 1} / {len(data)}\n')

            paragraphs = subject['paragraphs']

            if stats.num_possible_questions > opt.max_possible and stats.num_possible_questions_enq > opt.max_possible:
                break

            for paragraph in tqdm(paragraphs):
                global_step += 1
                original_context = paragraph['context']
                if len(original_context) > opt.max_len_for_translation:
                    original_context = original_context[:opt.max_len_for_translation]

                try:
                    if opt.translated:
                        translated_context = original_context
                    else:
                        translated_context = translator.translate(original_context)
                    translated_sentences = target.split_to_sentences(translated_context)
                except (NotValidPayload, NotValidLength, RequestError, RuntimeError, TranslationNotFound) as e:
                    logger.debug('cant translate')
                    continue

                phrases = []
                answer_starts = []
                contexts = []

                for sentence in translated_sentences:
                    translated_tokens = target.split_to_words(sentence)

                    # use only sentences longer than opt.minimum_words_in_sentence words
                    if len(translated_tokens) < opt.minimum_words_in_sentence:
                        logger.debug('too short')
                        continue

                    qas = []
                    used_spans = []

                    for attempt in range(opt.max_attempts):
                        # randomly select the start token and span
                        span = random_positive_normal_int(1, min(opt.max_span, len(translated_tokens) - 3), opt.scale)
                        start = random.randint(0, len(translated_tokens) - span - 1)
                        if (start, span) in used_spans:
                            logger.debug('used spans')
                            continue

                        used_spans.append((start, span))

                        if span == 1:
                            phrase_translated = translated_tokens[start]
                        else:
                            instances = find_shortest(translated_tokens[start], translated_tokens[start + span], sentence)
                                # re.findall(rf"{re.escape(translated_tokens[start])}.*?{re.escape(translated_tokens[start + span - 1])}",
                                #                    sentence)  # " ".join(translated_tokens[start: start + span])
                            # if len(instances) != 1:
                            #     logger.debug('multiple instance')
                            #     continue
                            if len(instances) < 1:
                                logger.debug('cant find')
                                continue
                            phrase_translated = instances[0]

                        # make sure phrase is not a digits
                        if phrase_translated.isdigit():
                            logger.debug('digits')
                            continue

                        # make sure only 1 instance of this phrase is in the text
                        instances = re.findall(re.escape(phrase_translated), sentence)
                        if len(instances) != 1:
                            logger.debug('multiple instance')
                            continue

                        phrases.append(phrase_translated)
                        contexts.append(sentence)
                        added_space = 1 if start > 0 else 0
                        answer_start = sentence.index(phrase_translated)  ##len(" ".join(translated_tokens[:start])) + added_space
                        answer_starts.append(answer_start)

                if len(phrases) == 0:
                    continue

                try:
                    inv_translated_phrases = inverse_translator.translate_together(phrases)

                    re_translated_phrases = translator.translate_together(inv_translated_phrases)
                except (NotValidPayload, NotValidLength, RequestError, RuntimeError, TranslationNotFound) as e:
                    logger.debug('cant translate')
                    continue

                if len(inv_translated_phrases) != len(phrases):
                    continue

                last_context = contexts[0]
                qas = []
                qas_enq = []
                for phrase_translated, answer_start, context, phrase_translated_again, inv_translated_phrase in zip(phrases, answer_starts, contexts,
                                                                                                                    re_translated_phrases,
                                                                                                                    inv_translated_phrases):

                    # if context == last_context and len(qas) >= opt.num_phrases_in_sentence:
                    #     continue

                    if context != last_context: # when context changed - save new paragraph
                        if len(qas) > 0:
                            stats.num_sentences += 1
                            writer.add_scalar(tag='generated_instances', scalar_value=stats.num_possible_questions, global_step=global_step)
                            new_paragraphs.append({'context': last_context, 'qas': qas})
                        if len(qas_enq) > 0:
                            new_paragraphs_enq.append({'context': last_context, 'qas': qas_enq})

                        qas = []
                        qas_enq = []

                    last_context = context

                    # make sure qa is valid
                    if context[answer_start:answer_start + len(phrase_translated)] != phrase_translated:
                        logger.debug('wrong index')
                        continue

                    # if inv_translated_phrase != phrase_translated:  # removed since it makes the matcher dismiss names and non-translatable words
                    if len(qas_enq) <= opt.num_phrases_in_sentence:
                        qas_enq.append({'id': f'mm{target.symbol}{next_id}',
                                        'is_impossible': False,
                                        'question': inv_translated_phrase,
                                        'answers': [{'text': phrase_translated,
                                                     'answer_start': answer_start}]})
                        stats.num_possible_questions_enq += 1

                    if len(qas) <= opt.num_phrases_in_sentence:
                        if phrase_translated_again != phrase_translated:
                            qas.append({'id': f'mm{target.symbol}{next_id}',
                                        'is_impossible': False,
                                        'question': phrase_translated_again,
                                        'answers': [{'text': phrase_translated,
                                                     'answer_start': answer_start}]})
                            stats.num_possible_questions += 1
                        else:
                            logger.debug('same translation')

                    next_id += 1


        # add impossible
        add_impossibles(new_paragraphs, stats, next_id=70000000000000000000)
        add_impossibles(new_paragraphs_enq, stats, next_id=80000000000000000000, count=False)

    finally:
        writer.close()
        logger.info('Saving to file')
        with open(output_json, 'w') as json_out:
            if opt.hf:
                new_data = squad_to_huggingface(new_data)
            full_doc['data'] = new_data
            json.dump(full_doc, json_out, ensure_ascii=False, indent=3 if opt.readable else None)
            logger.info(f'file saved: {output_json}')
            logger.info(stats)
        if opt.enq:
            with open(output_json_enq, 'w') as json_out:
                if opt.hf:
                    new_data_enq = squad_to_huggingface(new_data_enq)
                full_doc['data'] = new_data_enq
                json.dump(full_doc, json_out, ensure_ascii=False, indent=3 if opt.readable else None)
                logger.info(f'file saved: {output_json_enq}')
                logger.info(stats)

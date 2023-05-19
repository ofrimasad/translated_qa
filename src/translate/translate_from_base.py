import argparse
import json
import os
import re

from tqdm import tqdm

from matcher.smart_match import ModelMatcher, CorrelationMatcher

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('base_input_path', type=str, help='path of the base file')
    parser.add_argument('lang', type=str, help='language symbol')
    parser.add_argument('model_path', type=str, help='path of the trained matcher')
    parser.add_argument('--match_thresh', type=float, default=0.05, help='threshold for matcher')
    parser.add_argument('--output_dir', type=str, help='path for output dir')
    parser.add_argument('--from_en', action='store_true', help='match from english')

    opt = parser.parse_args()

    for k, v in opt.__dict__.items():
        print(f'{k}:\t{v}')

    matcher = ModelMatcher(model_name_or_path=opt.model_path)
    cor_matcher = CorrelationMatcher('bert-base-multilingual-cased')

    with open(opt.base_input_path) as json_file:
        full_doc = json.load(json_file)

    if opt.output_dir:
        os.makedirs(opt.output_dir, exist_ok=True)

    data = full_doc['data']
    new_data = []
    multiple = 0
    cant_find = 0
    wc_success = []
    wc_fail = []
    wc_success_cor = []
    wc_fail_cor = []
    wc = []
    require_translation, not_require_translation, success = 0, 0, 0

    from_en = opt.from_en
    for si, subject in enumerate(tqdm(data)):
        title = subject['title'] if 'title' in subject else ''
        paragraphs = subject['paragraphs']

        for paragraph in paragraphs:

            for qa in paragraph['qas']:

                # drop empty context
                if paragraph["context"] == "":
                    continue

                new_item = {
                    "context": paragraph["context"],
                    "title": title,
                    'question': qa['question'],
                    'id': qa['id']
                }

                if 'is_impossible' in qa and qa['is_impossible']:
                    new_item['answers'] = {'text': [], 'answer_start': []}
                else:
                    new_item['answers'] = {'text': [], 'answer_start': []}
                    # assert len(qa['answers']) == 1
                    for ans in qa['answers']:
                        if 'need_replace' in ans and ans['need_replace']:
                            require_translation += 1
                            # drop empty answer
                            if ans['text'] == "":
                                continue

                            translated_context = ans['translated_context']
                            if translated_context is None or translated_context == "":
                                continue
                            original_text = ans['original_text'] if from_en else ans['translated_text']
                            translated_text = ans['translated_text']
                            translated_offset = ans['translated_offset']
                            new_answer, score = matcher.match(translated_context, original_text)
                            cor_score = 0
                            length = len(original_text.split(" "))

                            # if long sentence and failed to match
                            if score < opt.match_thresh and length > 15:
                                new_answer, cor_score = cor_matcher.match(translated_context, translated_text)

                            if score > opt.match_thresh or cor_score > 0.5:

                                translated_start_indices = [_.start() for _ in re.finditer(re.escape(new_answer), translated_context)]
                                if len(translated_start_indices) == 1:
                                    success += 1

                                    new_item['answers']['text'].append(new_answer)
                                    new_item['answers']['answer_start'].append(translated_start_indices[0] + translated_offset)
                                else:
                                    multiple += 1
                            else:
                                cant_find += 1
                        else:
                            not_require_translation += 1
                            new_item['answers']['text'].append(ans['text'])
                            new_item['answers']['answer_start'].append(ans['answer_start'])

                if len(new_item['answers']['text']) > 0:
                    new_data.append(new_item)

    print(f'multiple: {multiple}')
    print(f'cant_find: {cant_find}')
    phase = 'dev' if 'dev' in opt.base_input_path else 'train'
    out_dir = opt.output_dir or os.path.dirname(opt.base_input_path)
    output_path = f'{out_dir}/{phase}_v1.0hf_{opt.lang}_{opt.match_thresh:.2f}{"_enq" if opt.from_en else ""}.json'
    print(f'require: {require_translation}')
    print(f'not require {not_require_translation}')
    print(f'success: {success}')

    with open(output_path, 'w') as json_out:
        full_doc['data'] = new_data
        full_doc['version'] = 'v1.0'
        json.dump(full_doc, json_out, ensure_ascii=False)
        print(f'file saved: {output_path}')

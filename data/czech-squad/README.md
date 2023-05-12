# Czech Translation of SQuAD 2.0 and 1.1

The Czech translation of SQuAD 2.0 and SQuAD 1.1 datasets
contains automatically translated texts, questions and
answers from the training set and the development set
of the respective datasets.

The test set is missing, because it is not publicly available.

## Licence

The data is released under the CC BY-NC-SA 4.0 license.

## Sizes

| Dataset         | English Questions | Czech Questions |
|:----------------|------------------:|----------------:|
| SQuAD 2.0 train |           130,319 |         107,088 |
| SQuAD 2.0 dev   |            11,873 |          10,845 |
| SQuAD 1.1 train |            87,599 |          64,164 |
| SQuAD 1.1 dev   |            10,570 |           8,739 |


## Format

The original `JSON` format is kept, with several modifications:
- The answer attribute `text` is exactly as present in the context of the question. The translated answer, not neccesarily present in the context, is available as `text_translated`.
- The plausible answers are not translated. Therefore, their attributes `text` and `answer_start` are missing, but the original English text is available as `text_en`.

## Citing

If you use the dataset, please cite the following paper (the exact format was not available during the submission of the dataset):
- Kateřina Macková and Straka Milan: Reading Comprehension in Czech via Machine Translation and Cross-lingual Transfer, presented at TSD 2020, Brno, Czech Republic, September 8-11 2020.

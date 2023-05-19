from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


class Matcher:

    def __init__(self, model_name_or_path: str):
        pass

    def match(self, context: str, phrase: str) -> (str, float):
        raise RuntimeError('unimplemented')


class CorrelationMatcher(Matcher):

    def __init__(self, model_name_or_path: str = 'onlplab/alephbert-base'):
        super().__init__(model_name_or_path)
        self.model = SentenceTransformer(model_name_or_path)

    @staticmethod
    def _sub_sentence(sentence: str, n: int):
        sentence_list = sentence.split(' ')
        results = []

        for start in range(n):
            next = start
            for _ in range((len(sentence_list) - start) // n):
                results.append(" ".join(sentence_list[next: next + n]))
                next += n

        return results

    def match(self, context: str, phrase: str):
        phrase_len = len(phrase.split(' '))
        candidates = self._sub_sentence(context, phrase_len)
        if phrase_len > 1:
            candidates.extend(self._sub_sentence(context, phrase_len - 1))
        candidates.extend(self._sub_sentence(context, phrase_len + 1))
        if len(candidates) == 0:
            return "", 0

        embeddings_candidates = self.model.encode(candidates, show_progress_bar=False, batch_size=32, convert_to_tensor=True)
        embeddings_phrase = self.model.encode([phrase], show_progress_bar=False, batch_size=32, convert_to_tensor=True)

        paraphrases = util.semantic_search(embeddings_phrase, embeddings_candidates)
        return candidates[paraphrases[0][0]['corpus_id']], paraphrases[0][0]['score']


class ModelMatcher(Matcher):
    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            device=0
        )

    def match(self, context: str, phrase: str) -> (str, float):
        res = self.qa_pipeline({
            'context': context,
            'question': phrase
        })
        return res['answer'], res['score']

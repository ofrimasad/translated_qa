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


if __name__ == "__main__":
    smart_match = CorrelationMatcher()

    context = 'מבחינה סטטיסטית, מעליות עם כבלים הן בטוחות ביותר. שיא הבטיחות שלהם אינו מתעלה על ידי כל מערכת רכב אחרת. בשנת 1998, ההערכה הייתה שכשמונה מיליוניית אחוז אחד (1 ל-12 מיליון) מהנסיעות במעלית גורמות לאנומליה, והרוב המכריע של אלו היו דברים מינוריים כמו הדלתות שלא נפתחו. מתוך 20 עד 30 מקרי מוות הקשורים למעלית בכל שנה, רובם קשורים לתחזוקה - למשל, טכנאים שנשענים יותר מדי לתוך הפיר או נקלעים בין חלקים נעים, ורוב השאר מיוחסים לסוגים אחרים של תאונות, כגון אנשים שפוסעים בעיוורון דרך דלתות הנפתחות לפירים ריקים או נחנקים על ידי צעיפים שנתפסו בדלתות. למעשה, לפני פיגועי הטרור של ה-11 בספטמבר, תקרית הנפילה החופשית היחידה הידועה במעלית מודרנית עם כבלים התרחשה בשנת 1945 כאשר מפציץ B-25 פגע בבניין האמפייר סטייט בערפל, וניתק את הכבלים של תא מעלית, שנפל מהקומה ה-75 עד לתחתית הבניין, ופצע קשה (אם כי לא הרג) את הדייר היחיד - מפעיל המעלית. עם זאת, אירעה תקרית בשנת 2007 בבית חולים לילדים בסיאטל, שם מעלית ללא חדר מכונות של ThyssenKrupp ISIS נפלה חופשית עד שבלמים הבטיחות הופעלו. זה נבע מפגם בתכנון שבו הכבלים היו מחוברים בנקודה משותפת אחת, ולחבלי הקוולר הייתה נטייה להתחמם יתר על המידה ולגרום להחלקה (או, במקרה זה, נפילה חופשית). אמנם זה אפשרי (אם כי לא סביר בצורה יוצאת דופן) שכבל של מעלית ייקרע, כל המעליות בעידן המודרני הותקנו במספר התקני בטיחות שמונעים מהמעלית פשוט ליפול חופשית ולהתרסק. תא מעלית נשא בדרך כלל על ידי 2 עד 6 (עד 12 או יותר במתקנים גבוהים) כבלים או רצועות הרמה, שכל אחת מהן מסוגלת בעצמה לתמוך בעומס המלא של המעלית בתוספת של עשרים וחמישה אחוזים יותר במשקל. בנוסף, קיים מכשיר אשר מזהה האם המעלית יורדת מהר יותר מהמהירות המיועדת המרבית שלה; אם זה קורה, ההתקן גורם לנעלי בלם נחושת (או סיליקון ניטריד במתקנים גבוהים) להיצמד לאורך המסילות האנכיות בפיר, ולעצור את המעלית במהירות, אך לא באופן פתאומי עד כדי פגיעה. מכשיר זה נקרא המושל, והוא הומצא על ידי אלישע גרייבס אוטיס. בנוסף, מותקן שמן/הידראולי או קפיץ או פוליאוריטן או שמן טלסקופי/חיץ הידראולי או שילוב (בהתאם לגובה הנסיעה ומהירות הנסיעה) בתחתית הפיר (או בתחתית הקבינה ולעיתים גם בתא הנהג). החלק העליון של הקבינה או הפיר) כדי לרכך מעט כל פגיעה. עם זאת, בתאילנד, בנובמבר 2012, אישה נהרגה במעלית נפילה חופשית, במה שדווח כ"מוות המוכר הראשון שנגרם על ידי נפילה במעלית".'

    print(smart_match.match(context, "אנשים שהולכים"))
    print(smart_match.match(context, "אירע מקרה"))
    print(smart_match.match(context, "אירעה, התקרית"))
    print(smart_match.match(context, "התקני בטיחות אשר מונעים מהמעלית פשוט ליפול חופשית ולהתרסק"))
    print(smart_match.match(context, "אוגר סיבירי"))

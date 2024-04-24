import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class DataPreparation:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def lower_case(self, data):
        for item in data:
            item['text_in_lower'] = item['text'].lower()
        return data


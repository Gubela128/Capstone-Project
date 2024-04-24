import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class DataPreparation:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def lower_case(self, data):
        for item in data:
            item['text_in_lower'] = item['text'].lower()
        return data

    def remove_stopwords(self, data):
        for item in data:
            item['text_without_stopwords'] = ""
            for word in item['text_in_lower'].split():
                if word not in self.stop_words:
                    if item['text_without_stopwords'] == "":
                        item['text_without_stopwords'] = word
                    else:
                        item['text_without_stopwords'] = item['text_without_stopwords'] + " " + word
        return data

    def stem_text(self, data):
        for item in data:
            item['stemmed_data'] = self.stem_text_string(item['text_without_stopwords'])
        return data

    def stem_text_string(self, text):
        tokens = word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed_tokens)

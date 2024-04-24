import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class DataPreparation:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def lower_case(self, data):
        return data.lower()

    def remove_stopwords(self, data):
        def remove_stopwords_from_text(text):
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
            return ' '.join(filtered_tokens)

        data['text'] = data['text'].apply(remove_stopwords_from_text)
        return data

    def lemmatize_text(self, data):
        def lemmatize_text_string(text):
            tokens = word_tokenize(text)
            lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            return ' '.join(lemmatized_tokens)

        data['text'] = data['text'].apply(lemmatize_text_string)
        return data

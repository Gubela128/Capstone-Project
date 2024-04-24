import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')


class DataPreparation:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
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
            item['stemmed_data'] = self.stem_text_string(item['text_without_special_characters'])
        return data

    def stem_text_string(self, text):
        tokens = word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed_tokens)

    def lemmatize_text(self, data):
        for item in data:
            word_tokens = word_tokenize(item['text_without_special_characters'])
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in word_tokens]
            item['lemmatized_text'] = " ".join(lemmatized_words)
        return data
    
    def remove_special_characters(self, data):
        for item in data:
            item['text_without_special_characters'] = ""
            for word in item['text_without_stopwords']:
                for character in word:
                    if (character >= chr(65) and character <= chr(90)) or (character >= chr(97) and character <= chr(122) or character == " "):
                        item['text_without_special_characters'] += character
            # Remove consecutive spaces
            item['text_without_special_characters'] = ' '.join(item['text_without_special_characters'].split())
        return data

    def pos_tagging(self, data):
        for item in data:
            item['text_pos_tagged'] = nltk.pos_tag(word_tokenize(item['lemmatized_text']))
        return data
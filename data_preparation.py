import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


class DataPreparation:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def lower_case(self, data):
        for item in data:
            item['text_in_lower'] = item['text'].lower()
        return data

    def remove_stopwords(self, data):
        for item in data:
            item['text_without_stopwords'] = ' '.join([word for word in item['text_in_lower'].split() if word not in self.stop_words or word == 'not'])
        return data

    def lemmatize_text(self, data):
        for item in data:
            word_tokens = word_tokenize(item['text_without_special_characters'])
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in word_tokens]
            item['lemmatized_text'] = ' '.join(lemmatized_words)
        return data

    def remove_special_characters(self, data):
        for item in data:
            item['text_without_special_characters'] = ''.join([char for char in item['text_without_stopwords'] if char not in string.punctuation])
        return data

    def handle_negations(self, data):
        negation_words = {"not", "never", "no", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
                          "havent", "hasnt", "hadnt", "cannot", "cant", "couldnt", "shouldnt", "wont", "wouldnt",
                          "aint", "neither", "nor", "none", "nobody", "nothing", "nowhere", "hardly", "scarcely",
                          "barely"}
        for item in data:
            words = item['lemmatized_text'].split()
            new_words = []
            skip_next = False
            for i in range(len(words)):
                if skip_next:
                    skip_next = False
                    continue
                if words[i] in negation_words and i + 1 < len(words):
                    new_word = words[i] + "_" + words[i + 1]
                    new_words.append(new_word)
                    skip_next = True
                else:
                    new_words.append(words[i])
            item['negation_handled_text'] = ' '.join(new_words)
        return data

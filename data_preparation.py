import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class DataPreparation:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = {'few', 'further', 'ours', 'above', 'was', 'on', 'there', 'd', 'up', 'very', 
                           'have', 'yourself', 'those', 'over', 'then', "you'll", 'should', 'more', 
                           'o', 'hadn', 'is', 'both', 'all',  'has', 'it', "should've", 'am', 'each', 'will', 
                           'the', 'who', 'than', "she's", 'doing', 'own', 'only', 'your', 'which', 'same', 're', 
                           'himself', 'these', 'off', 'does', 'they', 'too', 'she', 'here', 'until', 'll', 'won', 
                           'being', 'her', 'most', 'against', 'whom', 'yourselves', 'between', 'after', 'isn', 
                           't', 'if', 'did', 'through', 'shan', 'such', "you've", "shan't", 'him', 'before', 
                           'of', 'been', 'any', 'doesn', 'other', 'about', 'or', 'once', 'why', 'some', 've', 
                           'just', 'we', 'his', 'this', 'ma', 'hers', 'were', 'having', 'can', 'as', 
                           'my', "it's", 'y', 'he', "you're", 'down', 'i', 'm', 'into', 'had', 'a', 'during', 
                           'at', 'itself', 'now', 'myself', 'me', 'our', 'ain', 'again', 'you', 
                           'with', 'when', 'that', 'so', 'where', 'do', 'because', 'out', 'an', 'while', 
                           "you'd", 's', 'aren', 'their', 'and', 'its', 'are', 'how', 'below', 'from', 
                           'in', 'theirs', 'be', 'them', 'what', "that'll", 'but', 'to', 'haven', 'ourselves', 
                           'under', 'yours', 'for', 'themselves', 'herself', 'by'}
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
                    if (chr(65) <= character <= chr(90)) or (chr(97) <= character <= chr(122) or character == " "):
                        item['text_without_special_characters'] += character
            item['text_without_special_characters'] = ' '.join(item['text_without_special_characters'].split())
        return data

    def pos_tagging(self, data):
        for item in data:
            item['text_pos_tagged'] = nltk.pos_tag(word_tokenize(item['lemmatized_text']))
        return data

    # def handle_negations(self, data):
    #     negation_words = {"not", "never", "no", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    #                       "havent", "hasnt", "hadnt"}
    #     for item in data:
    #         words = item['lemmatized_text'].split()
    #         new_words = []
    #         skip_next = False
    #         for i in range(len(words)):
    #             if skip_next:
    #                 skip_next = False
    #                 continue
    #             if words[i] in negation_words and i + 1 < len(words):
    #                 new_word = words[i] + "_" + words[i + 1]
    #                 new_words.append(new_word)
    #                 skip_next = True
    #             else:
    #                 new_words.append(words[i])
    #         item['negation_handled_text'] = ' '.join(new_words)
    #     return data

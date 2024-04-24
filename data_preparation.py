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
# obj = DataPreparation()
# data = obj.remove_stopwords()
# def remove_stopwords(text):
#     word_tokens = word_tokenize(text) # Tokenize the text - split it into words
#     filtered_text = [word for word in word_tokens if not word in stop_words]
#     return " ".join(filtered_text)

# # Load your JSON data
# with open('training.json') as f:
#     data = json.load(f)

# # Remove stopwords from each text
# for item in data:
#     item['text_without_stopwords'] = remove_stopwords(item['text_in_lower'])

# # Save the updated data back to the JSON file
# with open('training.json', 'w') as f:
#     json.dump(data, f)

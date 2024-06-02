import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


class NaiveBayesEmotionClassifier:
    def __init__(self):
        self.emotion_classes = {
            0: 'sadness',
            1: 'joy',
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        }
        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(self.emotion_classes.values()))

    def train(self, data):
        texts = [sample['negation_handled_text'] for sample in data]
        labels = [self.emotion_classes[sample['label']] for sample in data]
        encoded_labels = self.label_encoder.transform(labels)
        self.model.fit(texts, encoded_labels)

    def predict(self, negation_handled_text):
        predicted_label = self.model.predict([negation_handled_text])[0]
        return self.label_encoder.inverse_transform([predicted_label])[0]




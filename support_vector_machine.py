from sklearn import svm
import numpy as np


class SupportVectorMachine:
    def __init__(self):
        self.emotion_classes = {
            0: 'sadness',
            1: 'joy',
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        }
        self.model = svm.LinearSVC(dual='auto')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        predictions = self.model.predict(X)
        return [self.emotion_classes[int(prediction)] for prediction in predictions]

    def predict_with_probabilities(self, X):
        decision_values = self.model.decision_function(X)
        probabilities = self.softmax(decision_values)

        predictions = np.argmax(probabilities, axis=1)
        predicted_emotions = [self.emotion_classes[int(prediction)] for prediction in predictions]

        # Convert prediction probabilities to a dictionary
        probabilities_dicts = [{self.emotion_classes[i]: prob for i, prob in enumerate(probs)} for probs in
                               probabilities]

        return predicted_emotions, probabilities_dicts

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

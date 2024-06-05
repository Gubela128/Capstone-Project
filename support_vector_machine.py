from sklearn import svm


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
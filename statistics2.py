from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from data_preparation import DataPreparation
from emotion_detection import EmotionDetection
from support_vector_machine import SupportVectorMachine
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    emotion_detection = EmotionDetection()
    data_preparation = DataPreparation()
    training_data = emotion_detection.read_data('data/prepared_training_data.json')

    X = [item['lemmatized_text'] for item in training_data]
    y = [item['label'] for item in training_data]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = SupportVectorMachine()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Map y_test to the corresponding emotion classes
    y_test = [classifier.emotion_classes[int(label)] for label in y_test]

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()

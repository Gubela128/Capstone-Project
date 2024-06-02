from sklearn.feature_extraction.text import TfidfVectorizer
from data_preparation import DataPreparation
from emotion_detection import EmotionDetection
from support_vector_machine import SupportVectorMachine


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def main():
    emotion_detection = EmotionDetection()
    training_data = emotion_detection.read_data('data/prepared_training_data.json')

    X_train = [item['lemmatized_text'] for item in training_data]
    y_train = [item['label'] for item in training_data]

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()

    classifier = SupportVectorMachine()
    classifier.fit(X_train, y_train)

    while True:
        text = input("Enter a text: ")
        if text.lower() == "exit":
            break

        text_vector = vectorizer.transform([text]).toarray()
        predicted_emotion = classifier.predict(text_vector)[0]
        predicted_emotion_label = get_key_from_value(classifier.emotion_classes, predicted_emotion)

        print(f"Predicted Emotion Label: {predicted_emotion_label}")
        print(f"Predicted Emotion: {predicted_emotion}")


if __name__ == "__main__":
    main()

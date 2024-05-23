from sklearn.feature_extraction.text import TfidfVectorizer
from data_preparation import DataPreparation
from emotion_detection import EmotionDetection
from support_vector_machine import SupportVectorMachine

def main():
    emotion_detection = EmotionDetection()
    data_preparation = DataPreparation()
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

        text = vectorizer.transform([text]).toarray()
        predicted_emotion = classifier.predict(text)
        print("Predicted Emotion:", predicted_emotion)

if __name__ == "__main__":
    main()
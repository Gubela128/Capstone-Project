from naive_bayes_classifier_with_library import NaiveBayesEmotionClassifier


def preprocess_text(data_preparation, text):
    preprocessed_data = [{
        'text': text,
        'label': -1,
        'text_in_lower': '',
        'text_without_stopwords': '',
        'text_without_special_characters': '',
        'lemmatized_text': '',
        'negation_handled_text': ''
    }]
    preprocessed_data = data_preparation.lower_case(preprocessed_data)
    preprocessed_data = data_preparation.remove_stopwords(preprocessed_data)
    preprocessed_data = data_preparation.remove_special_characters(preprocessed_data)
    preprocessed_data = data_preparation.lemmatize_text(preprocessed_data)
    preprocessed_data = data_preparation.handle_negations(preprocessed_data)
    return preprocessed_data[0]['negation_handled_text']


def main():
    from data_preparation import DataPreparation
    from emotion_detection import EmotionDetection

    emotion_detection = EmotionDetection()
    data_preparation = DataPreparation()
    classifier = NaiveBayesEmotionClassifier()

    training_data = emotion_detection.read_data('data/prepared_training_data.json')
    classifier.train(training_data)

    while True:
        text = input("Enter a text: ")
        if text.lower() == "exit":
            break

        negation_handled_text = preprocess_text(data_preparation, text)
        predicted_emotion = classifier.predict(negation_handled_text)
        print("Predicted Emotion:", predicted_emotion)


if __name__ == "__main__":
    main()
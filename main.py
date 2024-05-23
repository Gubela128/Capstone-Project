from data_preparation import DataPreparation
from emotion_detection import EmotionDetection
from naive_bayes_classifier import NaiveBayesClassifier


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def main():
    emotion_detection = EmotionDetection()
    data_preparation = DataPreparation()
    classifier = NaiveBayesClassifier()

    training_data = emotion_detection.read_data('data/prepared_training_data.json')
    classifier.calculate_prior_probability(training_data)
    classifier.calculate_likelihood_probabilities(training_data)

    while True:
        text = input("Enter a text: ")
        if text.lower() == "exit":
            break

        preprocessed_data = [{
            'text': text,
            'label': -1,
            'text_in_lower': '',
            'text_without_stopwords': '',
            'text_without_special_characters': '',
            'lemmatized_text': '',
            'negation_handled_text': '',
            'stemmed_data': '',
            'text_pos_tagged': []
        }]
        preprocessed_data = data_preparation.lower_case(preprocessed_data)
        preprocessed_data = data_preparation.remove_stopwords(preprocessed_data)
        preprocessed_data = data_preparation.remove_special_characters(preprocessed_data)
        preprocessed_data = data_preparation.lemmatize_text(preprocessed_data)
        preprocessed_data = data_preparation.handle_negations(preprocessed_data)
        print(preprocessed_data)
        negation_handled_text = preprocessed_data[0]['negation_handled_text']
        print(negation_handled_text)
        predicted_emotion = classifier.predict(negation_handled_text)
        print("Predicted Emotion:", predicted_emotion, get_key_from_value(classifier.emotion_classes, predicted_emotion))


if __name__ == "__main__":
    main()

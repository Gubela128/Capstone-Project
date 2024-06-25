from sklearn.metrics import classification_report
from data_preparation import DataPreparation
from training_data_preparation import TrainingDataPreparation
from naive_bayes_classifier import NaiveBayesClassifier


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def main():
    emotion_detection = TrainingDataPreparation()
    data_preparation = DataPreparation()
    classifier = NaiveBayesClassifier()

    training_data = emotion_detection.read_data('data/prepared_training_data.json')
    classifier.calculate_prior_probability(training_data)
    classifier.calculate_likelihood_probabilities(training_data)

    test_data = emotion_detection.read_data('data/training.json')

    y_true = []
    y_pred = []
    print('statistics of naive_bayes_classifier')
    for item in test_data:
        text = item['text']
        true_label = item['label']

        preprocessed_data = [{
            'text': text,
            'label': -1,
            'text_in_lower': '',
            'text_without_stopwords': '',
            'text_without_special_characters': '',
            'lemmatized_text': '',
            'negation_handled_text': '',
        }]
        preprocessed_data = data_preparation.lower_case(preprocessed_data)
        preprocessed_data = data_preparation.remove_stopwords(preprocessed_data)
        preprocessed_data = data_preparation.remove_special_characters(preprocessed_data)
        preprocessed_data = data_preparation.lemmatize_text(preprocessed_data)
        preprocessed_data = data_preparation.handle_negations(preprocessed_data)

        negation_handled_text = preprocessed_data[0]['negation_handled_text']
        predicted_emotion = classifier.predict(negation_handled_text)

        y_true.append(classifier.emotion_classes[true_label])
        y_pred.append(predicted_emotion)

    print(classification_report(y_true, y_pred, target_names=classifier.emotion_classes.values()))


if __name__ == "__main__":
    main()

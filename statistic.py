from data_preparation import DataPreparation
from emotion_detection import EmotionDetection
from naive_bayes_classifier import NaiveBayesClassifier
import json


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def main():
    emotion_detection = EmotionDetection()
    data_preparation = DataPreparation()
    classifier = NaiveBayesClassifier()

    # Prepare training data
    training_data = emotion_detection.read_data('data/prepared_training_data.json')
    classifier.calculate_prior_probability(training_data)
    classifier.calculate_likelihood_probabilities(training_data)

    # Test data processing and evaluation
    test_data = emotion_detection.read_data('data/test.json')
    correct_count = 0
    total_count = 0

    for item in test_data:
        text = item['text']
        true_label = item['label']

        # Skip test samples with labels not in emotion_classes
        if true_label not in classifier.emotion_classes:
            print(f"Skipping sample with unknown label: {true_label}")
            continue

        preprocessed_data = [{
            'text': text,
            'label': -1,
            'text_in_lower': '',
            'text_without_stopwords': '',
            'text_without_special_characters': '',
            'lemmatized_text': '',
            'stemmed_data': '',
            'text_pos_tagged': []
        }]
        preprocessed_data = data_preparation.lower_case(preprocessed_data)
        preprocessed_data = data_preparation.remove_stopwords(preprocessed_data)
        preprocessed_data = data_preparation.remove_special_characters(preprocessed_data)
        preprocessed_data = data_preparation.lemmatize_text(preprocessed_data)
        preprocessed_data = data_preparation.handle_negations(preprocessed_data)

        negation_handled_text = preprocessed_data[0]['negation_handled_text']
        predicted_emotion = classifier.predict(negation_handled_text)

        if classifier.emotion_classes[true_label] == predicted_emotion:
            correct_count += 1

        total_count += 1

    if total_count > 0:
        accuracy = correct_count / total_count
    else:
        accuracy = 0

    print("Accuracy: ", accuracy)
    print("Correct Count: ", correct_count)
    print("Total Test Samples: ", total_count)


if __name__ == "__main__":
    main()

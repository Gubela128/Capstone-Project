import json
import random
from sklearn.model_selection import train_test_split
from data_preparation import DataPreparation
from emotion_detection import EmotionDetection
from naive_bayes_classifier import NaiveBayesClassifier
from support_vector_machine import SupportVectorMachine
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


def split_data(data, train_size, test_size, eval_size):
    random.shuffle(data)
    train_data, temp_data = train_test_split(data, test_size=(test_size + eval_size), random_state=42)
    test_data, eval_data = train_test_split(temp_data, test_size=eval_size / (test_size + eval_size), random_state=42)
    return train_data, test_data, eval_data


def prepare_data(data_preparation, data):
    data = data_preparation.lower_case(data)
    data = data_preparation.remove_stopwords(data)
    data = data_preparation.remove_special_characters(data)
    data = data_preparation.lemmatize_text(data)
    data = data_preparation.handle_negations(data)
    return data


def preprocess_single(data_preparation, text):
    preprocessed_data = [{
        'text': text,
        'label': -1,
        'text_in_lower': '',
        'text_without_stopwords': '',
        'text_without_special_characters': '',
        'lemmatized_text': '',
        'negation_handled_text': '',
    }]
    prepare_data(data_preparation, preprocessed_data)
    return preprocessed_data[0]['negation_handled_text']


def evaluate_model(classifier, data_preparation, vectorizer, data, model_type):
    if model_type == 'NaiveBayes':
        classifier.calculate_prior_probability(data)
        classifier.calculate_likelihood_probabilities(data)
    elif model_type == 'SVM':
        texts = [item['lemmatized_text'] for item in data]
        labels = [str(item['label']) for item in data]
        vectors = vectorizer.fit_transform(texts).toarray()
        classifier.fit(vectors, labels)

    y_true = []
    y_pred = []

    for item in data:
        text = item['text']
        true_label = str(item['label'])
        negation_handled_text = preprocess_single(data_preparation, text)

        if model_type == 'NaiveBayes':
            predicted_emotion = classifier.predict(negation_handled_text)
        elif model_type == 'SVM':
            text_vector = vectorizer.transform([negation_handled_text]).toarray()
            predicted_emotion = classifier.predict(text_vector)[0]

        y_true.append(classifier.emotion_classes[int(true_label)])
        y_pred.append(predicted_emotion)

    unique_classes = sorted(set(y_true).union(set(y_pred)))
    target_names = unique_classes

    return classification_report(y_true, y_pred, target_names=target_names, output_dict=True)


def main():
    emotion_detection = EmotionDetection()
    data_preparation = DataPreparation()
    data = emotion_detection.read_data('data/training.json')

    split_ratios = [
        (0.5, 0.25, 0.25), (0.5, 0.20, 0.30), (0.6, 0.2, 0.2),
        (0.6, 0.3, 0.1), (0.7, 0.15, 0.15), (0.8, 0.1, 0.1),
        (0.8, 0.2, 0.1), (0.9, 0.05, 0.05)
    ]
    results = {}

    for train_size, test_size, eval_size in split_ratios:
        split_key = f"train_{int(train_size * 100)}_test_{int(test_size * 100)}_eval_{int(eval_size * 100)}"
        results[split_key] = {}

        train_data, test_data, eval_data = split_data(data, train_size, test_size, eval_size)
        prepared_eval_data = prepare_data(data_preparation, eval_data)

        nb_classifier = NaiveBayesClassifier()
        results[split_key]['NaiveBayes'] = evaluate_model(nb_classifier, data_preparation, None, prepared_eval_data,
                                                          'NaiveBayes')

        svm_classifier = SupportVectorMachine()
        vectorizer = TfidfVectorizer()
        results[split_key]['SVM'] = evaluate_model(svm_classifier, data_preparation, vectorizer, prepared_eval_data,
                                                   'SVM')

    with open('data/evaluation_results.json', 'w') as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()

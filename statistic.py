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
    training_data = emotion_detection.read_data('data/prepared_training_data.json')
    classifier.calculate_prior_probability(training_data)
    classifier.calculate_likelihood_probabilities(training_data)
    correct_count = 0
    
    test_data = emotion_detection.read_data('data/test.json')
    
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
        'stemmed_data': '',
        'text_pos_tagged': []
        }]
        preprocessed_data = data_preparation.lower_case(preprocessed_data)
        preprocessed_data = data_preparation.remove_stopwords(preprocessed_data)
        preprocessed_data = data_preparation.remove_special_characters(preprocessed_data)
        preprocessed_data = data_preparation.lemmatize_text(preprocessed_data)
        lemmatized_text = preprocessed_data[0]['lemmatized_text']
        predicted_emotion = classifier.predict(lemmatized_text)
        
        if get_key_from_value(classifier.emotion_classes, predicted_emotion) == true_label:
            correct_count += 1
        
    print("result: ", correct_count/len(test_data))
    print("correct count: ", correct_count )

if __name__ == "__main__":
    main()
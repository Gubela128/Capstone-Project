import json

from data_preparation import DataPreparation


class EmotionDetection:

    @classmethod
    def read_data(cls, file_name):
        with open(file_name, 'r') as file:
            data = json.load(file)
        return data

    @classmethod
    def save_prepared_data(cls):
        training_data = cls.read_data('data/training.json')
        data_preparation = DataPreparation()
        training_data = data_preparation.lower_case(training_data)
        training_data = data_preparation.remove_stopwords(training_data)
        training_data = data_preparation.remove_special_characters(training_data)
        training_data = data_preparation.lemmatize_text(training_data)
        training_data = data_preparation.stem_text(training_data)
        training_data = data_preparation.pos_tagging(training_data)
        with open('data/prepared_training_data.json', 'w') as file:
            json.dump(training_data, file, indent=4)

    @classmethod
    def main(cls):
        cls.save_prepared_data()


if __name__ == '__main__':
    EmotionDetection.main()

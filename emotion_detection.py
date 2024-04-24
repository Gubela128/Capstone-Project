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
        print(training_data[0])

    @classmethod
    def main(cls):
        cls.save_prepared_data()


if __name__ == '__main__':
    EmotionDetection.main()

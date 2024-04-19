from data_preparation import DataPreparation
import pandas as pd


class EmotionDetection:

    @classmethod
    def read_data(cls, file_name):
        data = pd.read_csv(f"data/{file_name}")
        text = data["text"]
        return text

    @classmethod
    def save_prepared_data(cls):
        training_data = cls.read_data('training.csv')
        data_preparation = DataPreparation()
        training_data = data_preparation.lower_case(training_data)
        training_data = data_preparation.stop_words(training_data)
        training_data = data_preparation.lemmatizer(training_data)
        training_data.to_csv(f'data/processed_training_data.csv', index=False)

    @classmethod
    def main(cls):
        cls.save_prepared_data()


if __name__ == '__main__':
    EmotionDetection.main()

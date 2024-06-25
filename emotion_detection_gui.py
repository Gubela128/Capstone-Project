import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Style
from tkinter import ttk
from data_preparation import DataPreparation
from emotion_detection import EmotionDetection
from naive_bayes_classifier import NaiveBayesClassifier
from support_vector_machine import SupportVectorMachine
from ttkthemes import ThemedStyle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.feature_extraction.text import CountVectorizer


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection")
        self.root.geometry("800x650")
        self.root.configure(bg='white')

        self.initialize_classifiers()

        self.initialize_components()

    def initialize_components(self):
        style = ThemedStyle(self.root)
        style.set_theme("arc")

        buttion_style = Style()
        buttion_style.configure('Green.TButton', background='green', foreground='black')

        self.setup_labels()
        self.setup_classifier_dropdown()
        self.setup_entry()
        self.setup_buttons()
        self.setup_result_label()

    def setup_labels(self):
        self.label = ttk.Label(self.root, text="Select classifier:", font=("Helvetica", 14))
        self.label.grid(row=0, column=0, columnspan=2, pady=(12, 0), padx=20, sticky="w")

        self.text_label = ttk.Label(self.root, text="Enter text to detect emotion:", font=("Helvetica", 14))
        self.text_label.grid(row=2, column=0, columnspan=2, pady=(12, 0), padx=20, sticky="w")

    def setup_classifier_dropdown(self):
        self.classifier_var = tk.StringVar()
        self.classifier_dropdown = ttk.Combobox(self.root, textvariable=self.classifier_var,
                                                values=["Naive Bayes", "Support Vector Machine"],
                                                state="readonly", font=("Helvetica", 12))
        self.classifier_dropdown.current(0)
        self.classifier_dropdown.grid(row=1, column=0, columnspan=2, pady=(0, 10), padx=20, sticky="ew")

    def setup_entry(self):
        self.text_entry = ttk.Entry(self.root, width=50, font=("Helvetica", 12))
        self.text_entry.grid(row=3, column=0, columnspan=2, pady=10, padx=20, sticky="ew")
        self.text_entry.bind("<Return>", self.predict_emotion)  # Bind Enter key to predict_emotion

    def setup_buttons(self):
        self.predict_button = ttk.Button(self.root, text="Predict Emotion", command=self.predict_emotion,
                                         style="Green.TButton")
        self.predict_button.grid(row=4, column=0, columnspan=2, pady=10, padx=20)

    def setup_result_label(self):
        self.result_label = ttk.Label(self.root, text="", font=("Helvetica", 16))
        self.result_label.grid(row=5, column=0, columnspan=2, pady=10, padx=20)

        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.graph = FigureCanvasTkAgg(self.figure, self.root)
        self.graph.get_tk_widget().grid(row=6, column=0, columnspan=2, pady=10, padx=20)

    def initialize_classifiers(self):
        self.data_preparation = DataPreparation()
        self.naive_bayes = NaiveBayesClassifier()
        self.svm = SupportVectorMachine()

        emotion_detection = EmotionDetection()
        self.training_data = emotion_detection.read_data('data/prepared_training_data.json')

        # Initialize Naive Bayes
        self.naive_bayes.calculate_prior_probability(self.training_data)
        self.naive_bayes.calculate_likelihood_probabilities(self.training_data)

        # Initialize SVM
        X = [item['negation_handled_text'] for item in self.training_data]
        y = [item['label'] for item in self.training_data]
        self.vectorizer = CountVectorizer()
        X_vectorized = self.vectorizer.fit_transform(X)
        self.svm.fit(X_vectorized, y)

    def preprocess_text(self, text):
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
        preprocessed_data = self.data_preparation.lower_case(preprocessed_data)
        preprocessed_data = self.data_preparation.remove_stopwords(preprocessed_data)
        preprocessed_data = self.data_preparation.remove_special_characters(preprocessed_data)
        preprocessed_data = self.data_preparation.lemmatize_text(preprocessed_data)
        preprocessed_data = self.data_preparation.handle_negations(preprocessed_data)
        return preprocessed_data[0]['negation_handled_text']

    def predict_emotion(self, event=None):
        text = self.text_entry.get()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return

        try:
            negation_handled_text = self.preprocess_text(text)
            selected_classifier = self.classifier_var.get()

            if selected_classifier == "Naive Bayes":
                predicted_emotion, probabilities = self.naive_bayes.predict_with_probabilities(negation_handled_text)
            elif selected_classifier == "Support Vector Machine":
                X_vectorized = self.vectorizer.transform([negation_handled_text])
                predicted_emotion, probabilities = self.svm.predict_with_probabilities(X_vectorized)
                probabilities = probabilities[0]  # SVM returns a list of dicts, we need only the first one
            else:
                raise ValueError("Invalid classifier selected")

            self.display_result(predicted_emotion, probabilities)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def display_result(self, predicted_emotion, probabilities):
        self.result_label.config(text=f"Predicted Emotion: {predicted_emotion}", foreground="blue")

        emotions = list(probabilities.keys())
        probs = [probabilities.get(emotion, 0.0) * 100 for emotion in emotions]

        self.ax.clear()
        bars = self.ax.bar(emotions, probs, color='blue')
        self.ax.set_ylabel('Probability (%)')
        self.ax.set_title('Emotion Probabilities')

        for bar, prob in zip(bars, probs):
            self.ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{prob:.1f}%',
                         ha='center', va='bottom')

        self.graph.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
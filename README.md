
# Emotion Detection Project

This project is aimed at detecting emotions from text using machine learning techniques. It includes data preparation, training classifiers (Naive Bayes and Support Vector Machine), and a graphical user interface for interactive emotion detection.

## Capstone Project

Capstone Project by Kutaisi International University Students - Nikoloz Gubeladze, Ana Chkhikvadze, Natia Pruidze

## Overview

This repository contains Python scripts and files organized as follows:

- `data_preparation.py`: This script handles text preprocessing tasks such as tokenization, stop word removal, and lemmatization using the NLTK library.
- `training_data_preparation.py`: This script prepares the training data, handles data splits for training and testing, and ensures the data is ready for the classifiers.
- `naive_bayes_classifier.py`: Implementation of the Naive Bayes classifier for emotion detection.
- `support_vector_machine.py`: Implementation of the Support Vector Machine (SVM) classifier for emotion detection.
- `emotion_detection_gui.py`: A graphical user interface (GUI) application for real-time emotion detection from text input.
- `splited_data_statistics.py`: Script for splitting data and evaluating the performance of the classifiers.

## Usage

To use the GUI application for emotion detection, follow these steps:

1. Ensure you have all the required dependencies installed.
2. Run the `emotion_detection_gui.py` script:
   ```bash
   python emotion_detection_gui.py
   ```

This will launch the GUI application where you can input text and get the detected emotion.

## Dependencies

Make sure you have the following Python libraries installed:

- `nltk`: For natural language processing tasks.
- `tkinter`: For creating the graphical user interface.
- `matplotlib`: For plotting graphs and visualizations.
- `scikit-learn`: For implementing machine learning algorithms.
- `ttkthemes`: For adding themes to the tkinter GUI.

You can install these dependencies using pip:
```bash
pip install nltk tkinter matplotlib scikit-learn ttkthemes
```

## Installation

Follow these steps to set up the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gubela128/Capstone-Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Capstone-Project
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

The `data_preparation.py` script is responsible for preprocessing the raw text data. This includes:

- Tokenizing the text into words.
- Removing stop words.
- Lemmatizing the words to their base forms.


## Training the Classifiers

The `training_data_preparation.py` script prepares the data for training. It splits the data into training and testing sets and ensures the data is formatted correctly for the classifiers.


## Data Splitting and Evaluation

The `splited_data_statistics.py` script handles splitting the data into training and testing sets and evaluates the performance of the classifiers. This script provides insights into the accuracy and efficiency of the models.

Run the script using:
```bash
python splited_data_statistics.py
```

## Contact

For any questions or inquiries, please contact:

- Nikoloz Gubeladze
- Ana Chkhikvadze
- Natia Pruidze

Thank you for using our Emotion Detection Project!

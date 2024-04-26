class NaiveBayesClassifier:
    def __init__(self):
        self.emotion_classes = {
            0: 'sadness',
            1: 'joy',
            2: 'love',
            3: 'anger',
            4: 'fear'
        }
        self.class_probs = {emotion: 0.0 for emotion in self.emotion_classes.values()}
        self.word_probs = {emotion: {} for emotion in self.emotion_classes.values()}
        self.class_vocabulary_size = {emotion: set() for emotion in self.emotion_classes.values()}

    def calculate_prior_probability(self, data):
        total_samples = len(data)
        for emotion_label, emotion_name in self.emotion_classes.items():
            class_samples = [sample for sample in data if sample['label'] == emotion_label]
            self.class_probs[emotion_name] = len(class_samples) / total_samples

    def calculate_likelihood_probabilities(self, data):
        word_count = {emotion: {} for emotion in self.emotion_classes.values()}

        for sample in data:
            emotion_label = sample['label']
            if emotion_label not in self.emotion_classes:
                continue
            emotion_name = self.emotion_classes[emotion_label]
            for word in sample['lemmatized_text'].split():
                self.class_vocabulary_size[emotion_name].add(word)
                if word not in word_count[emotion_name]:
                    word_count[emotion_name][word] = 0
                word_count[emotion_name][word] += 1

        for emotion_name, counts in word_count.items():
            total_words = sum(counts.values())
            for word, count in counts.items():
                self.word_probs[emotion_name][word] = ((count + 1) /
                                                       (total_words + len(self.class_vocabulary_size[emotion_name])))

    def predict(self, lemmatized_text):
        posterior_probs = {emotion: self.class_probs[emotion] for emotion in self.emotion_classes.values()}

        for word in lemmatized_text.split():
            for emotion_name in self.emotion_classes.values():
                if word in self.word_probs[emotion_name]:
                    posterior_probs[emotion_name] *= self.word_probs[emotion_name][word]

        predicted_emotion = max(posterior_probs, key=posterior_probs.get)
        return predicted_emotion

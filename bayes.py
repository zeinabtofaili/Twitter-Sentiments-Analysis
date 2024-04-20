from utils import *
import copy
from collections import defaultdict

class Bayes:
    def __init__(self, testing_filepath, training_filepath, bayes_values):
        self.testing_filepath = testing_filepath
        self.training_filepath = training_filepath
        self.tweets_to_annotate = load_tweets_to_annotate(self.testing_filepath)
        self.annotated_tweets = copy.deepcopy(self.tweets_to_annotate)
        self.name = "bayes"
        self.bayes_method = bayes_values["bayes_method"]
        self.word_frequencies = defaultdict(lambda: defaultdict(int))
        self.num_tweets = 0
        self.num_tweets_per_class = defaultdict(int)
        self.vocab = set()
        self.training_data = load_training_data(self.training_filepath)

    def generate_bigrams(self, tweet_words):
        tweet_size = len(tweet_words) - 1
        return [tweet_words[i]+ ' ' +tweet_words[i + 1] for i in range(tweet_size)]

    def train_naive_bayes(self):
        for tweet, label in self.training_data:
            tweet = clean_tweet(tweet)
            self.num_tweets_per_class[label] += 1
            self.num_tweets += 1
            tweet_words = tweet.split()

            if self.bayes_method in ["Use Bigrams Only", "Use Bigrams and Unigrams"]:
                bigrams = self.generate_bigrams(tweet_words)
                for bigram in bigrams:
                    self.word_frequencies[label][bigram] += 1
                    self.vocab.add(bigram)

            if self.bayes_method !="Use Bigrams Only":
                for word in tweet_words:
                    self.word_frequencies[label][word] += 1
                    self.vocab.add(word)

    def proba_calc(self, numerator, denominator):
        return numerator/ denominator
    
    def calc_proba_word_given_class(self, tweet_word, label): #P(w|c)
        denominator = len(self.vocab) + sum(self.word_frequencies[label].values())
        numerator = self.word_frequencies.get(label, {}).get(tweet_word, 0) + 1

        return self.proba_calc(numerator, denominator)

    def classify_with_naive_bayes(self, tweet_to_annotate):

        max_class = None
        max_log_prob = float('-inf')
        tweet = clean_tweet(tweet_to_annotate)
        
        for label in [0, 2, 4]:
            
            #P(c)
            label_proba = self.proba_calc(self.num_tweets_per_class[label], self.num_tweets)
            
            mul_word_given_class_prob = 1
            prob_class_given_tweet = 1
            if self.bayes_method == "Default":
                for tweet_word in set(tweet.split()):
                    mul_word_given_class_prob *= self.calc_proba_word_given_class(tweet_word, label)

            elif self.bayes_method == "Consider Repeated Words":
                for tweet_word in tweet.split():
                    mul_word_given_class_prob *= self.calc_proba_word_given_class(tweet_word, label)

            elif self.bayes_method == "Remove stop words":
                stopwords = load_stopwords('./helper_files/stopwords.txt')
                tweet_words = tweet.strip("\"").split()
                tweet_words = [word for word in tweet_words if word.lower() not in stopwords]
                new_tweet = ' '.join(tweet_words)
                for tweet_word in set(new_tweet.split()):
                    mul_word_given_class_prob *= self.calc_proba_word_given_class(tweet_word, label)
            
            if self.bayes_method == "Use Bigrams":
                tweet_words = tweet.split()
                tweet_bigrams = self.generate_bigrams(tweet_words)
                for bigram in tweet_bigrams:
                    mul_word_given_class_prob *= self.calc_proba_word_given_class(bigram, label)

            elif self.bayes_method == "Use Unigrams and Bigrams":
                tweet_words = tweet.split()
                tweet_bigrams = self.generate_bigrams(tweet_words)
                all_features = set(tweet_words + tweet_bigrams)
                for feature in all_features:
                    mul_word_given_class_prob *= self.calc_proba_word_given_class(feature, label)
            
            prob_class_given_tweet = mul_word_given_class_prob * label_proba          

            if prob_class_given_tweet > max_log_prob:
                max_class = label
                max_log_prob = prob_class_given_tweet

        return max_class

    def get_bayes_annotated_tweets(self):
        self.train_naive_bayes()
        for tweet_data in self.annotated_tweets:
            tweet = tweet_data[-1].strip("\"")
            predicted_label = self.classify_with_naive_bayes(tweet)
            tweet_data[0] = predicted_label
        return self.annotated_tweets   

    def error_rate(self, k):
        folds = stratified_split(self.training_data, k)
        error_rates = [] 

        for i in range(k):
            self.reset_values()
            self.annotated_tweets = folds[i]
            self.training_data = [entry for j, fold in enumerate(folds) if j != i for entry in fold]
            
            self.train_naive_bayes()

            true_labels = [label for tweet, label in self.annotated_tweets]
            predicted_labels = []
            for tweet_data, label in self.annotated_tweets:
                predicted_label = self.classify_with_naive_bayes(tweet_data)
                predicted_labels.append(predicted_label)

            error_rate = calc_error(true_labels, predicted_labels)
            error_rates.append(error_rate)
            

        return sum(error_rates) / len(error_rates) if error_rates else 0

    def reset_values(self):
        self.word_frequencies = defaultdict(lambda: defaultdict(int))
        self.num_tweets = 0
        self.num_tweets_per_class = defaultdict(int)
        self.vocab = set()

    def get_original_labels(self):
        return [tweet[0] for tweet in self.tweets_to_annotate]
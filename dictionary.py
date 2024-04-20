from utils import *
import tkinter as tk
import copy

class Dictionary:
    def __init__(self, testing_filepath, pos_and_neg_filepaths, training_filepath):
        self.testing_filepath = testing_filepath
        self.tweets_to_annotate = load_tweets_to_annotate(self.testing_filepath)
        self.annotated_tweets = copy.deepcopy(self.tweets_to_annotate)
        self.positive_words = self.get_words(pos_and_neg_filepaths["positive_words_path"])
        self.negative_words = self.get_words(pos_and_neg_filepaths["negative_words_path"])
        self.name = "dictionary"
        self.training_data = load_training_data(training_filepath)

    
    def get_words(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read().split(', ')

    def annotate_tweet(self, tweet):
        label = 0
        cleaned_tweet = clean_tweet(tweet)
        cleaned_tweet_lowered =  [word.lower() for word in cleaned_tweet.split()]

        pos_words_count = sum(word in self.positive_words for word in cleaned_tweet_lowered)
        neg_words_count = sum(word in self.negative_words for word in cleaned_tweet_lowered)
        if pos_words_count < neg_words_count:
            label = '0'
        elif pos_words_count > neg_words_count:
            label = '4'
        else:
            label = '2'
        return label

    def get_dictionary_annotated_tweets(self):
        if len(self.positive_words) <= 0 or len(self.negative_words) <= 0:
            tk.messagebox.showinfo("Info", "Please add both positive and negative word files.")
        else: 
            for tweet_data in self.annotated_tweets:
                tweet = tweet_data[-1].strip("\"")
                predicted_label = self.annotate_tweet(tweet)
                tweet_data[0] = predicted_label
        return self.annotated_tweets
    
    def error_rate(self, k):
        folds = stratified_split(self.training_data, k)
        error_rates = [] 

        for i in range(k):
            self.annotated_tweets = folds[i]
            self.training_data = [entry for j, fold in enumerate(folds) if j != i for entry in fold]

            true_labels = [label for tweet, label in self.annotated_tweets]
            predicted_labels = []
            for tweet_data, label in self.annotated_tweets:
                predicted_label = self.annotate_tweet(tweet_data)
                predicted_labels.append(int(predicted_label))

            error_rate = calc_error(true_labels, predicted_labels)
            error_rates.append(error_rate)
            

        return sum(error_rates) / len(error_rates) if error_rates else 0
    
    def get_original_labels(self):
        return [tweet[0] for tweet in self.tweets_to_annotate]
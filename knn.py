from utils import *
import tkinter as tk
import copy
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class KNN:
    def __init__(self, testing_filepath, training_filepath, knn_values):
        self.testing_filepath = testing_filepath
        self.training_filepath = training_filepath
        self.tweets_to_annotate = load_tweets_to_annotate(self.testing_filepath)
        self.annotated_tweets = copy.deepcopy(self.tweets_to_annotate)
        self.name = "knn"
        self.k = knn_values["k_value"]
        self.distance_type = knn_values["distance_type"]
        self.voting_type = knn_values["voting_type"]

    def jaccard_distance(self, set1, set2):
        total_num_of_words = len(set1.union(set2))
        num_common_words = len(set1.intersection(set2))
        return 1 - (num_common_words / total_num_of_words)

    def knn_classify(self, tweet_to_annotate, training_tweets):

        tweet_to_annotate_words = set(clean_tweet(tweet_to_annotate).lower().split())
        distances = []

        for train_tweet_data in training_tweets:
            train_tweet, label = train_tweet_data
            train_tokens = set(clean_tweet(train_tweet).lower().split())
            if self.distance_type == "jaccard":
                distance = self.jaccard_distance(tweet_to_annotate_words, train_tokens)
            elif self.distance_type == "euclidean":
                vectorizer = CountVectorizer()
                vectors = vectorizer.fit_transform([clean_tweet(tweet_to_annotate).lower(), clean_tweet(train_tweet).lower()]).toarray()
                distance = euclidean_distances(vectors)[0, 1]
            elif self.distance_type == "cosine":
                vectorizer = CountVectorizer()
                vectors = vectorizer.fit_transform([clean_tweet(tweet_to_annotate).lower(), clean_tweet(train_tweet).lower()]).toarray()
                cos_sim = cosine_similarity(vectors)[0, 1]
                distance = 1 - cos_sim
            else:
                tk.messagebox.showwarning("Warning", "Unsupported distance type: {}".format(self.distance_type))

            distances.append((label, distance))
        
        distances.sort(key=lambda element: element[1])
        tweet_neighbors = distances[:self.k]

        if self.voting_type == 'weighted':
            vote_counts = Counter()
            for label, distance in tweet_neighbors:
                vote_counts[label] += 1/(distance +1)**2
            predicted_label, _ = vote_counts.most_common(1)[0]
        elif self.voting_type == 'majority':
            votes = [label for label, distance in tweet_neighbors]
            vote_counts = Counter(votes)
            predicted_label, _ = vote_counts.most_common(1)[0]
        else:
            tk.messagebox.showwarning("Warning", "Unsupported vote type: {}".format(self.voting_type))
            return None
        return predicted_label

    def get_knn_annotated_tweets(self):
        training_data = load_training_data(self.training_filepath)
        for tweet_data in self.annotated_tweets:
            tweet = tweet_data[-1]
            predicted_label = self.knn_classify(tweet, training_data)
            tweet_data[0] = predicted_label
        return self.annotated_tweets
    
    def error_rate(self, k):
        training_data = load_training_data(self.training_filepath)
        folds = stratified_split(training_data, k)
        error_rates = [] 

        for i in range(k):
            self.annotated_tweets = folds[i]
            training_data = [entry for j, fold in enumerate(folds) if j != i for entry in fold]

            true_labels = [label for tweet, label in self.annotated_tweets]
            predicted_labels = []
            for tweet_data, label in self.annotated_tweets:
                predicted_label = self.knn_classify(tweet_data, training_data)
                predicted_labels.append(predicted_label)

            error_rate = calc_error(true_labels, predicted_labels)
            error_rates.append(error_rate)
            

        return sum(error_rates) / len(error_rates) if error_rates else 0
    
    def get_original_labels(self):
        return [tweet[0] for tweet in self.tweets_to_annotate]
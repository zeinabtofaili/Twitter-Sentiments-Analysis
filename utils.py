import csv
import random
from collections import defaultdict
import re

def calc_error(true_labels, predicted_labels):
    num_false_labels = sum(t != p for p, t in zip(predicted_labels, true_labels))
    error_rate = num_false_labels / len(true_labels)
    return error_rate

def load_tweets_to_annotate(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        tweets = [row for row in reader if row]
        return tweets
    
def load_training_data(file_path):
    training_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            label = int(line[0])
            tweet = line[-1].strip("\"")
            training_data.append((tweet, label))
    # we remove the duplicates
    training_data = list(set(training_data))
    return training_data

def stratified_split(tweets, k):
    folds = []
    for num in range(k):
        folds.append([])

    grouped_tweets_by_label = defaultdict(list)
    for entry in tweets:
        grouped_tweets_by_label[entry[1]].append(entry)

    for label, tweets_of_this_label in grouped_tweets_by_label.items():
        random.shuffle(tweets_of_this_label)
        for i, entry in enumerate(tweets_of_this_label):
            folds[i % k].append(entry)

    index = 0
    while index < len(folds):
        random.shuffle(folds[index])
        index += 1

    return folds

def clean_tweet(tweet):
    tweet = re.sub(r'@\w+?:', '', tweet.strip("\""))
    tweet = re.sub(r'https?://\S+', '', tweet.strip("\""))
    tweet = re.sub(r'#\w+', '', tweet.strip("\""))
    tweet = re.sub(r'RT ', '', tweet.strip("\""))
    tweet = re.sub(r'([?!".,;:])', r' \1 ', tweet.strip("\"")) 
    tweet = re.sub(r'\€[0-9]+', '€XX', tweet.strip("\""))
    tweet = re.sub(r'\$[0-9]+', '$XX', tweet.strip("\""))
    tweet = re.sub(r'[0-9]+n%', 'XXn%', tweet.strip("\""))
    tweet_words = tweet.strip("\"").split()
    return ' '.join(tweet_words)        

def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = file.read()
        return set(stopwords.strip().split(','))
from utils import *
import tkinter as tk
import copy
class Manual:
    def __init__(self, testing_filepath):
        self.testing_filepath = testing_filepath
        self.tweets_to_annotate = load_tweets_to_annotate(self.testing_filepath)
        self.annotated_tweets = copy.deepcopy(self.tweets_to_annotate)
        self.index = 0

    def get_annotated_tweets(self):
        return self.annotated_tweets

    def show_tweet(self):
        try:
            current_tweet = self.tweets_to_annotate[self.index]
            return current_tweet[-1].strip("\"")
        except Exception as e:
            tk.messagebox.showinfo("Info", "You have reached the end of the file.")
            return None
        
    def annotate_tweet(self, sentiment):
        if self.index < len(self.tweets_to_annotate):
            if sentiment:
                self.annotated_tweets[self.index][0] = sentiment
                self.index += 1
            else:
                tk.messagebox.showwarning("Warning", "Please select a sentiment before proceeding.")
        else:
            messagebox.showinfo("Info", "You have reached the end of the file.")
            # reset_manual_annotation()
            # if general_reset_function:  # Check if the function was set
            #     general_reset_function()
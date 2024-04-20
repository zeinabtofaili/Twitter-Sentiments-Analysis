import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image
from manual import Manual
from dictionary import Dictionary
from knn import KNN
import csv
from bayes import Bayes

fileUploadImage = ctk.CTkImage(Image.open("./images/fileUpload.png"), size=(30, 30))
annotateTweetsImage = ctk.CTkImage(Image.open("./images/annotateTweets.png"), size=(30, 30))
saveTweetsImage = ctk.CTkImage(Image.open("./images/saveTweets.png"), size=(30, 30))
nextTweetImage = ctk.CTkImage(Image.open("./images/nextTweet.png"), size=(30, 30))
positiveImage = ctk.CTkImage(Image.open("./images/positiveImage.png"), size=(30, 30))
negativeImage = ctk.CTkImage(Image.open("./images/negativeImage.png"), size=(30, 30))
annotateImage = ctk.CTkImage(Image.open("./images/annotateTweets.png"), size=(30, 30))

class Annotator:
    def __init__(self):
        self.values = {
            "general": {
                "training_filepath": "./helper_files/training_data.csv",
                "testing_filepath": "",
                "annotated_tweets": [],
                "original_labels": []
            },
            "dictionary": {
                "positive_words_path": "./helper_files/positive.txt",
                "negative_words_path": "./helper_files/negative.txt",
            },
            "knn": {
                "k_value": 7,
                "distance_type": "jaccard",
                "voting_type": "majority",
            },
            "bayes": {
                "bayes_method": "Default",
            },
            "scoring": {
                "num_folds": 10
            }
        }
        self.annotation_methods = ["Manual annotation", "Dictionary Approach", "KNN Approach", "Naive Bayes Approach"]
        self.annotation_method = None
    def set_general_value(self, key, value):
        if key in self.values['general']:
            self.values['general'][key] = value

    def set_dictionary_value(self, key, value):
        if key in self.values['dictionary']:
            self.values['dictionary'][key] = value

    def set_knn_value(self, key, value):
        if key in self.values['knn']:
            self.values['knn'][key] = value

    def set_bayes_value(self, key, value):
        if key in self.values['bayes']:
            self.values['bayes'][key] = value

annotator = Annotator()

def select_annotation_method(event):
    method = event.widget.get()

    for widget in main_frame.winfo_children():
        widget.destroy()
    if not annotator.values["general"]["testing_filepath"]:
        messagebox.showwarning("Warning", "Please upload a tweet file first.")
    else:
        if(method == "Manual annotation"):
            annotator.annotation_method = Manual(annotator.values["general"]["testing_filepath"])

            sentiments = [("Negative", "0"), ("Neutral", "2"), ("Positive", "4")]
            current_tweet_sentiment = tk.StringVar(root)
            current_tweet_sentiment.set("0")
            tweet_content = tk.StringVar(root)
            
            tweet_label = ctk.CTkLabel(main_frame, textvariable=tweet_content, font=("Arial", 16), text_color="black")
            tweet_label.pack(pady=10)
            
            for text, value in sentiments:
                radiobutton = ctk.CTkRadioButton(main_frame, text=text, variable=current_tweet_sentiment, value=value, text_color="black", font=("Arial", 12, "bold"))
                radiobutton.pack(anchor="w", pady=10)

            def next_tweet():
                annotator.values["general"]["annotated_tweets"] = annotator.annotation_method.get_annotated_tweets()
                annotator.annotation_method.annotate_tweet(current_tweet_sentiment.get())
                tweet = annotator.annotation_method.show_tweet()
                if not tweet:
                    for widget in main_frame.winfo_children():
                        widget.destroy()
                else:
                    tweet_content.set(tweet)
                   
            next_button = ctk.CTkButton(main_frame, text="Next", image = nextTweetImage, command = next_tweet, compound = "right", anchor ="e", font=("Arial", 16, "bold"))
            next_button.pack(pady=10)
            tweet_content.set(annotator.annotation_method.show_tweet())
            
        elif method == "Dictionary Approach":

            pos_btn = ctk.CTkButton(main_frame, text="Upload Positive Words", image = positiveImage, command= lambda: annotator.set_dictionary_value("positive_words_path", tk.filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])), compound = "right", anchor ="e", font=("Arial", 12, "bold")).pack(pady=10)
            neg_btn = ctk.CTkButton(main_frame, text="Upload Negative Words", image = negativeImage, command=lambda: annotator.set_dictionary_value("positive_words_path", tk.filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])), compound = "right", anchor ="e", font=("Arial", 12, "bold")).pack(pady=10)
            do_labeling_btn_dic = ctk.CTkButton(main_frame, text="Label the Tweets", image = annotateImage, command= lambda: doLabeling(method), compound = "right", anchor ="e", font=("Arial", 12, "bold")).pack(pady=10)
            
            error_label = ctk.CTkLabel(main_frame, text="If you wish to calculate the average error rate using cross validation, please upload a labeled file (a default one is already provided)", font=("Arial", 16), text_color="black").pack(pady=10)
            error_file_btn = ctk.CTkButton(main_frame, text="Upload Labeled File", image = fileUploadImage, command= lambda: annotator.set_general_value("training_filepath", tk.filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])), compound = "right", anchor ="e", font=("Arial", 12, "bold")).pack(pady=10)
            calc_error_btn_dic = ctk.CTkButton(main_frame, text="Calculate Error", command = lambda: calc_error_cross_validation("Dictionary", None)).pack(pady=10)
        
        elif method == "KNN Approach":

            def knn_on_combobox_select(event=None):
                annotator.set_knn_value("voting_type", voting_combobox.get())
                annotator.set_knn_value("distance_type", distance_combobox.get())

            k_label = ctk.CTkLabel(main_frame, text="Enter the number of neighbors to consider (default is 7):", font=("Arial", 12), text_color="black").pack(pady = 5)
            k_entry = tk.Entry(main_frame)
            k_entry.pack(pady= 5)
            k_entry.insert(0, str(annotator.values["knn"]["k_value"]))

            distance_label = ctk.CTkLabel(main_frame, text="Distance Type (default is 'jaccard'):", font=("Arial", 12), text_color="black").pack(pady =5)
            distance_combobox = tk.ttk.Combobox(main_frame, values=["jaccard", "euclidean", "cosine"])
            distance_combobox.set(annotator.values["knn"]["distance_type"])
            distance_combobox.bind("<<ComboboxSelected>>", knn_on_combobox_select)
            distance_combobox.pack(pady =5)

            method_label = ctk.CTkLabel(main_frame, text="Method (default is 'majority'):", font=("Arial", 12), text_color="black").pack(pady = 5)
            voting_combobox = tk.ttk.Combobox(main_frame, values=["majority", "weighted"])
            voting_combobox.set(annotator.values["knn"]["voting_type"])
            voting_combobox.bind("<<ComboboxSelected>>", knn_on_combobox_select)
            voting_combobox.pack(pady = 5)

            training_file_btn = ctk.CTkButton(main_frame, text="Upload Training File (default provided)", image = fileUploadImage, command= lambda: annotator.set_general_value("training_filepath", tk.filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])), compound = "right", anchor ="e", font=("Arial", 12, "bold")).pack(pady=5)           
            do_labeling_btn_knn = ctk.CTkButton(main_frame, text="Label the Tweets", image = annotateImage, command= lambda: doLabeling(method, k_entry), compound = "right", anchor ="e", font=("Arial", 12, "bold")).pack(pady=5)

            calc_error_btn_knn = ctk.CTkButton(main_frame, text="Calculate Error", command = lambda: calc_error_cross_validation("KNN", k_entry)).pack(pady=5)
        elif method == "Naive Bayes Approach":
            def bayes_on_combobox_select(event=None):
                annotator.set_bayes_value("bayes_method", bayes_combobox.get())
                
            training_file_btn = ctk.CTkButton(main_frame, text="Upload Training File (default provided)", image = fileUploadImage, command= lambda: annotator.set_general_value("training_filepath", tk.filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])), compound = "right", anchor ="e", font=("Arial", 12, "bold")).pack(pady=5)           
            method_label = ctk.CTkLabel(main_frame, text="Method (a default option is provided):", font=("Arial", 12), text_color="black").pack(pady = 10)
            bayes_combobox = tk.ttk.Combobox(main_frame, values=["Default", "Consider Repeated Words", "Remove stop words", "Use Bigrams", "Use Unigrams and Bigrams"])
            bayes_combobox.set(annotator.values["bayes"]["bayes_method"])
            bayes_combobox.bind("<<ComboboxSelected>>", bayes_on_combobox_select)
            bayes_combobox.pack(pady=10)

            do_labeling_btn_bayes = ctk.CTkButton(main_frame, text="Label the Tweets", image = annotateImage, command= lambda: doLabeling(method), compound = "right", anchor ="e", font=("Arial", 12, "bold")).pack(pady=10)

            calc_error_btn_dic = ctk.CTkButton(main_frame, text="Calculate Error", command = lambda: calc_error_cross_validation("Bayes", None)).pack(pady=10)

def doLabeling(method, k_entry = None):
    if method == "Dictionary Approach":
        annotator.annotation_method = Dictionary(annotator.values["general"]["testing_filepath"], annotator.values["dictionary"], annotator.values["general"]["training_filepath"])
        annotator.set_general_value("annotated_tweets", annotator.annotation_method.get_dictionary_annotated_tweets())
        annotator.set_general_value("original_labels", annotator.annotation_method.get_original_labels())
    elif method == "KNN Approach":
        try:
            annotator.values["knn"]["k_value"] = int(k_entry.get())
        except ValueError:
            annotator.values["knn"]["k_value"] = 7
        annotator.annotation_method = KNN(annotator.values["general"]["testing_filepath"], annotator.values["general"]["training_filepath"], annotator.values["knn"])
        annotator.set_general_value("annotated_tweets", annotator.annotation_method.get_knn_annotated_tweets())
        annotator.set_general_value("original_labels", annotator.annotation_method.get_original_labels())

    elif method == "Naive Bayes Approach":
        annotator.annotation_method = Bayes(annotator.values["general"]["testing_filepath"], annotator.values["general"]["training_filepath"], annotator.values["bayes"])
        annotator.set_general_value("annotated_tweets", annotator.annotation_method.get_bayes_annotated_tweets())
        annotator.set_general_value("original_labels", annotator.annotation_method.get_original_labels())

    open_labeled_frame()

def open_labeled_frame():

    new_labels = [i[0] for i in annotator.values["general"]["annotated_tweets"]]
    tweet_texts = [i[-1] for i in annotator.values["general"]["annotated_tweets"]]
    headers = ["Old Label", "New Label", "Tweet Text"]

    changed_labels_frame = tk.Toplevel(root, width=520)
    changed_labels_frame.title("Old Vs New Labels")
    frame_canvas = tk.Canvas(changed_labels_frame, width=520)
    scrollbar = tk.Scrollbar(changed_labels_frame, orient="vertical", command=frame_canvas.yview)

    table_frame = tk.Frame(frame_canvas, width=520)
    table_frame.bind(
        "<Configure>",
        lambda e: frame_canvas.configure(scrollregion=frame_canvas.bbox("all"))
    )

    frame_canvas.create_window((0, 0), window=table_frame, anchor="nw")
    frame_canvas.configure(yscrollcommand=scrollbar.set)

    frame_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for col, header in enumerate(headers):
        label = ctk.CTkLabel(table_frame, text=header, font=("Arial", 12, "bold"), text_color="black")
        label.grid(row=0, column=col, padx=10, pady=5)

    n = len(annotator.values["general"]["original_labels"])
    for row in range(1, n + 1):
        label1 = ctk.CTkLabel(table_frame, text=annotator.values["general"]["original_labels"][row-1], text_color="black", bg_color="white")
        label1.grid(row=row, column=0, sticky="nsew", padx=10, pady=10)

        label2 = ctk.CTkLabel(table_frame, text=new_labels[row-1], text_color="black", bg_color="white")
        label2.grid(row=row, column=1, sticky="nsew", padx=10, pady=10)

        label3 = ctk.CTkLabel(table_frame, text=tweet_texts[row-1], text_color="black", bg_color="white",  wraplength=350)
        label3.grid(row=row, column=2, sticky="nsew", padx=10, pady=10)

def save_to_file():
    annotated_tweets = annotator.values["general"]["annotated_tweets"]
    
    if annotated_tweets == []:
        messagebox.showwarning("Warning", "Please press the label button first.")

    if len(annotated_tweets) > 0:
        user_choice = tk.messagebox.askyesnocancel("Save Tweets Annotations",
                                            "Would you like to overwrite the original file?\n"
                                            "Yes: Overwrite the original file\n"
                                            "No: Save tweets to a new file\n"
                                            "Cancel: Cancel the operation")

        if user_choice is True:
            with open(annotator.values["general"]["testing_filepath"], "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                for tweet in annotated_tweets:
                    writer.writerow(tweet)
            # reset values
            annotator.set_general_value("annotated_tweets",[])


        elif user_choice is False:
            new_filepath = tk.filedialog.asksaveasfilename(filetypes=[("CSV files", "*.csv")], defaultextension=".csv")
            if new_filepath: 
                with open(new_filepath, "w", newline='', encoding="utf-8") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    for tweet in annotated_tweets:
                        writer.writerow(tweet)
                # reset values
                annotator.set_general_value("annotated_tweets",[])

def calc_error_cross_validation(method_name, k_entry= None):
    if method_name == "Dictionary":
        annotator.annotation_method = Dictionary(annotator.values["general"]["testing_filepath"], annotator.values["dictionary"], annotator.values["general"]["training_filepath"])
    elif method_name == "KNN":
        try:
            annotator.values["knn"]["k_value"] = int(k_entry.get())
        except ValueError:
            annotator.values["knn"]["k_value"] = 7

        annotator.annotation_method = KNN(annotator.values["general"]["testing_filepath"], annotator.values["general"]["training_filepath"], annotator.values["knn"])
    elif method_name == "Bayes":
        annotator.annotation_method = Bayes(annotator.values["general"]["testing_filepath"], annotator.values["general"]["training_filepath"], annotator.values["bayes"])
    average_error_rate = annotator.annotation_method.error_rate(annotator.values["scoring"]["num_folds"])
    tk.messagebox.showinfo("Average Error Rate", "The average error rate is: "+ str(round(average_error_rate, 2)))

root = tk.Tk()
root.state('zoomed')
root.title("Tweet Sentiment Analysis")

upload_test_tweets_btn = ctk.CTkButton(root, image = fileUploadImage, text="Upload Tweet File", command= lambda: annotator.set_general_value("testing_filepath", tk.filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])), compound = "right", anchor ="e", font=("Arial", 16, "bold")).pack(pady=20)
choose_method_label = ctk.CTkLabel(root, text="Choose Annotation Method:", font=("Arial", 12, "bold"), text_color="black").pack(pady=10)

annotation_method_combobox = tk.ttk.Combobox(root, values=annotator.annotation_methods)
annotation_method_combobox.bind("<<ComboboxSelected>>", select_annotation_method)
annotation_method_combobox.pack(pady=10)

main_frame = tk.Frame(root)
main_frame.pack(pady=20, padx=20, fill='both', expand=True)

save_tweets_btn = ctk.CTkButton(root, text="Save Annotations", image = saveTweetsImage, command=save_to_file, compound = "right", anchor ="e", font=("Arial", 16, "bold")).pack(pady=20)
root.mainloop()
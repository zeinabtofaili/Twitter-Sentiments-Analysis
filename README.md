# Twitter Sentiment Classifier

## Overview
This Python application was developed as part of a university project course. It uses a graphical interface built with Tkinter to classify the sentiment (positive, negative, neutral) of tweets using supervised learning algorithms. It is designed to process, analyze, and classify large sets of tweet data effectively.

## CSV File Format
The application expects tweet data in a specific CSV format where each record includes the following fields:

### Column Descriptions
- **target**: The polarity of the tweet, where "0" represents negative, "2" represents neutral, and "4" represents positive sentiments.
- **ids**: The unique identifier for the tweet.
- **date**: The date and time when the tweet was posted.
- **flag**: A placeholder for the query tag. 
- **user**: The username of the tweet's author.
- **text**: The actual text content of the tweet.

### Example
Here is an example of how your CSV file should look:
```
"0","1467810369","Mon Apr 06 22:19:45 PDT 2023","NO_QUERY","_TheSpecialOne_","@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer. You shoulda got David Carr of Third Day to do it ;D"
"0","1467810672","Mon Apr 06 22:19:49 PDT 2023","NO_QUERY","scotthamilton","is upset that he can't update his Facebook by texting it... and might cry as a result School today too."
```

## Getting Started

### Prerequisites
Make sure Python is installed on your machine.

### Setting up the Environment
Install the necessary dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Running the Application
Start the application with the following command:
```bash
python mainGUI.py
```

## Usage
- **Upload and Clean Data**: Load your CSV files through the GUI and preprocess them as necessary.
- **Classify Tweets**: Select the desired classification algorithm and classify the sentiments of the tweets.
- **Review and Save Results**: Evaluate the output, make manual adjustments if needed, and save the results.

Please note that an example training data file is provided in `helper_files` folder along with sentiment dictionaries files uselful for the dictionary mode in the application. I do not own those training data. 

from tweepy import OAuthHandler, API, Cursor
from joblib import load
import preprocessor as p
import numpy as np
import joblib
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from autocorrect import Speller
import nltk
from flask import Flask, render_template, request, send_file
from nltk.stem import WordNetLemmatizer


# Create Flask app
app = Flask(__name__)

# Load the trained spam detector model
model = load('tweet.joblib')
vectorizer = joblib.load("vectorizer.joblib")

# Set up Twitter API credentials
consumer_key = "22oAmCTccX7Tc419UvzO6T0XS"
consumer_secret = "ZE7DUrAcSgGYo0Kfw9ECsutpr5FPGK464x7frUxmsVBa1FK4Pt"
access_token = "733711577334075393-iRx9B3PzyQ5A4Lcso9EAxQRZfsuqGRQ"
access_token_secret = "01sEGY7Z9Dx6AlMdRzotN9vNuidCNvj7I252uaszeEguQ"

# Authenticate with Twitter API
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
spell = Speller(lang='en')


def preprocess(text):
    # Convert to lower case
    text = text.lower()
    # Remove stop words
    words = word_tokenize(text)
    words = [word for word in words if not word in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [spell(word) for word in words]
    return ' '.join(words)


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to collect and classify tweets

@app.route('/classify_tweets')
def classify_tweets():
    tweets = Cursor(api.search_tweets, q='Congratulations!', lang='en').items(
        10)
    classified_tweets = []
    for tweet in tweets:
        tweet_text = tweet.text
        tweet_text = preprocess(tweet_text)
        message_vector = vectorizer.transform([tweet_text])
        prediction = model.predict(message_vector)[0]
        if prediction == 1:
            label = 'Spam'
        else:
            label = 'Non-Spam'
        classified_tweets.append({'text': tweet_text, 'label': label})

    return render_template('tweets.html', tweets=classified_tweets)


if __name__ == '__main__':
    app.run(debug=True)

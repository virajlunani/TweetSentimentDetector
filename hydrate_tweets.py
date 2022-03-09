import datetime
from lib2to3.pgen2 import token
import os
import string
import re

from twarc.client2 import Twarc2
from twarc.expansions import ensure_flattened
from dotenv import load_dotenv

from extract_feats import *
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import emoji

load_dotenv()
BEARER_TOKEN = os.getenv('BEARER_TOKEN')

# Twitter v2 API Client
t = Twarc2(bearer_token=BEARER_TOKEN)
tk = TweetTokenizer()

ps = PorterStemmer()
nltk.download('stopwords')

punctuation = [ c for c in string.punctuation ] + [u'\u201c',u'\u201d',u'\u2018',u'\u2019']

def isNumber(token):
    try:
        val = int(token)
    except ValueError:
        return False
    return True

def removeURL(text):
    return re.sub(r"http\S+", "", text)

def stemTweets(tweets):
    return [ps.stem(word) for word in tweets]

def tokenizeTweets(tweets):
    return [tk.tokenize(word) for word in tweets]

def removePunctuation(tweets_tokens):
    noPuncTweets = []
    for tweet_tokens in tweets_tokens:
        for tweet_token in tweet_tokens:
            new_tweet_token = emoji.demojize(tweet_token)
            if new_tweet_token not in punctuation and not isNumber(tweet_token) and not tweet_token.startswith(('@', '#', ':')):
                noPuncTweets.append(new_tweet_token)
    return noPuncTweets
#remove emojis
def cleanRetweets(tweet):
    workingTweet = tweet
    if "referenced_tweets" in tweet:
        workingTweet = tweet["referenced_tweets"][0]["text"]
    else:
        workingTweet = tweet["text"]
    return removeURL(workingTweet)

# https://developer.twitter.com/en/docs/twitter-api/tweets/lookup/api-reference/get-tweets
def hydrateTweets(df, tweet_fields, user_fields):
    # take first column of tweet ids
    tweet_ids = list(df[0])

    lookup_results = t.tweet_lookup(tweet_ids=tweet_ids, tweet_fields=tweet_fields, media_fields=None, poll_fields=None, place_fields=None, user_fields=user_fields, expansions=None)

    tweets = []
    users = []
    # Get all results page by page: all tweets
    for page in lookup_results:
        for tweet in ensure_flattened(page):
            tweets.append(cleanRetweets(tweet))
            users.append(removeURL(tweet["author"]["description"]))
        tweets = stemTweets(tweets)
        tweets_tokens = tokenizeTweets(tweets)
        tweets_tokens = removePunctuation(tweets_tokens)

        users = stemTweets(users)
        users_tokens = tokenizeTweets(users)
        users_tokens = removePunctuation(users_tokens)

        stopeng = set(stopwords.words('english'))
        cleaned_tweets = []
        cleaned_users = []
        for tokens in tweets_tokens:
            cleaned_tweets.append([token for token in tokens if token not in stopeng])
        for tokens in users_tokens:
            cleaned_users.append([token for token in tokens if token not in stopeng])
        # Stop iteration prematurely, to only get 1 page of results.
        break
    
    final_tweets = [' '.join(cleaned_tweet) for cleaned_tweet in cleaned_tweets]
    final_users = [' '.join(cleaned_user) for cleaned_user in cleaned_users]
    return final_tweets, final_users

def createBoWFeatureVecTweets(tweets, labels, vectorizer):
    X = vectorizer.fit_transform(tweets).toarray()
    y = labels
    return X, y

def createTfIdfFeatureVecTweets(tweets, labels, vectorizer):
	X = vectorizer.fit_transform(tweets).toarray()
	y = labels
	return X, y

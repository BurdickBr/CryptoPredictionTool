#|===================|
#|      Imports      |
#|===================|
import pandas as pd                 # Pandas dataframe library
import pandas_datareader as pdr     # Pandas datareader that allows me to lookup & store live crypto prices from yahoo finance.
import numpy as np                  # Numpy
from alpha_vantage.timeseries import TimeSeries     # Library used for pulling live price data from alphavantage api

import secrets               # This import is referring to the secrets.py file supplied with the submission of the project.
                                    # Secrets contains the API keys that are used for AlphaVantage and the Twitter API.

from datetime import datetime, timedelta, timezone             # Datetime library.
import warnings
warnings.simplefilter(action='ignore', category=ResourceWarning)
warnings.filterwarnings('ignore')

import json
import os                           # For changing/finding proper directory (when opening files)
import requests
import nest_asyncio                 # Import required for twint usage.
nest_asyncio.apply()                

import re                           # Regex for string cleaning (used for Textblob Sentiment Analysis)
from textblob import TextBlob       # Textblob used for sentiment analysis of cleaned data.

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer    # Sentiment analysis tool that works great on determining social media sentiment.
import requests                     # Used for sending get requests to the NewsAPI client.

from sklearn.preprocessing import MinMaxScaler                          # Scaler used for scaling data (LSTMRNN Implementation)
from sklearn.metrics import accuracy_score, classification_report       
from sklearn.model_selection import train_test_split                    # Used for splitting data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    # Used for implementing SVM
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Run app with command: python -m uvicorn CryptoPredictionAPI:app --reload
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#|===========================|
#| Additional files and keys:|
#|===========================|

# Read in stopwords file to list for sifting tweets later on.
os.chdir(r'C:\Users\WaKaBurd\Documents\GitHub\CryptoPredictionTool\archive')
stopwords_file = open("stopwords.txt", "r+")
stopwords = list(stopwords_file.read().split('\n'))
av_api_key = secrets.av_api_key

#|===========|
#| Functions |
#|===========|

# Need to create function for cleaning the tweets so we can derive the subjectivity and polarity using textblob.
def sift_tweet(tweet, stop_words):
    cleaned_tweet = tweet
    cleaned_tweet = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet) # regex to remove all @userame, emojis, and links from tweets.
    for word in cleaned_tweet:
        if word in stop_words: cleaned_tweet.replace(word, '')
    return cleaned_tweet

def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

def send_request(url, headers, params, next_token=None):
    params['next_token'] = next_token
    response = requests.request('GET', url, headers=headers, params=params)
    print('Endpoint response code:' + str(response.status_code))
    if (response.status_code != 200):
        raise Exception(response.status_code, response.text)
    return response.json()

def pull_live_tweets(coin):

    # Pull tweets from the last hour
    path = r'c:\Users\WaKaBurd\Documents\GitHub\CryptoPredictionTool\predicted_trends'
    os.chdir(path)

    print('performing twitter search for coin:', coin)

    # 1 hour ago
    from_date = datetime.now(timezone.utc) - timedelta(hours = 1)
    to_date = datetime.now(timezone.utc) + timedelta(seconds=-30)
    
    iso_from_date = from_date.isoformat()
    iso_to_date = to_date.isoformat()

    from_date = from_date.strftime('%Y-%m-%d %H:%M:%S')
    to_date = to_date.strftime('%Y-%m-%d %H:%M:%S')

    print(f'searching {from_date} to {to_date}')
    
    bearer_token = secrets.bearer_token

    headers = {
        "Authorization": "Bearer {}".format(bearer_token)
    }

    url = 'https://api.twitter.com/2/tweets/search/recent'

    params = {
        'query': coin,
        'start_time': iso_from_date,
        'end_time': iso_to_date,
        'max_results': 100,
        'next_token':{}
    }

    json_response = send_request(url, headers, params)
    return json_response

# Pull financial data from yahoo finance for the current hour
# Uses AlphaVantage API with their CRYPTO_INTRADAY endpoint.
def get_prices(coin):
    path = r'c:\Users\WaKaBurd\Documents\GitHub\CryptoPredictionTool\prices\LivePrices'
    os.chdir(path)
    url = f'https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol={coin}&market=USD&interval=1min&apikey={av_api_key}&datatype=csv'
    req = requests.get(url)
    data = req.content
    csv_file = open(f'{coin}_prices.csv','wb')
    csv_file.write(data)
    csv_file.close()
    return

def gen_model():
    # These are all the columns we actually want to keep for the purposes of training & using the model.
    model_cols = ['open', 'high', 'low', 'Volume USD', 'compound', 'positive', 'negative', 'neutral', 'polarity', 'subjectivity', 'price_change']
    os.chdir(r'C:\Users\WaKaBurd\Documents\GitHub\CryptoPredictionTool\hourly_coin_data')

    model_df = pd.read_csv('model_df_2.csv')
    model_df = model_df.iloc[: , 1:]                # Drops first column in the dataframe as we don't want/need it.
    # Feature Dataset
    x = model_df
    # Target Dataset
    y = np.array(model_df['price_change'])
    x.drop(['price_change'], axis=1, inplace=True)
    np.asarray(x)

    # split into test & train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Create LDA model
    model = LinearDiscriminantAnalysis().fit(x_train, y_train)
    predictions = model.predict(x_test)
    return model

def make_live_prediction(coin, fetched_tweets_df, model):
    path = r'c:\Users\WaKaBurd\Documents\GitHub\CryptoPredictionTool\prices\LivePrices'
    os.chdir(path)    

    live_prices = pd.read_csv(f'{coin}_prices.csv')        # read in live prices csv
    kept_prices = live_prices.head(60)                  # keep only the last 60 minutes.
    high = kept_prices['high'].max(axis=0)              # Find the max value in the last 60 minutes
    low = kept_prices['low'].min(axis=0)                # find the lowesst value in the last 60 minutes
    open = kept_prices['open'].values[59]                 # Price from 60 minutes ago. (opening price of the last hour)
    volume = kept_prices['volume'].sum(axis=0)          # summate the total volume traded from the last hour

    live_coin_data = pd.DataFrame([[open, high, low, volume]], columns =['open', 'high', 'low', 'volume'])

    # Run textblob on tweets for polarity & subjectivity
    combined_tweets = ' '.join(fetched_tweets_df['text'])

    # Clean tweet so we can use textblob on it.
    fetched_tweets_df['cleaned_tweet'] = fetched_tweets_df['text'].apply(lambda x: sift_tweet(str(x).lower(), stopwords))
    combined_cleaned_tweets = ' '.join(fetched_tweets_df['cleaned_tweet'])

                
    # Get sentiment values on tweets using VADER sentiment analyzer
    sia = get_sentiment(combined_tweets)
    compound = sia['compound']                    # Score representing sum(lexicon ratings)
    pos = sia['pos']
    neg = sia['neg']
    neu = sia['neu']

    live_coin_data.loc[live_coin_data.index[0],'compound'] = compound
    live_coin_data.loc[live_coin_data.index[0],'pos'] = pos
    live_coin_data.loc[live_coin_data.index[0],'neg'] = neg
    live_coin_data.loc[live_coin_data.index[0],'neu'] = neu
    live_coin_data.loc[live_coin_data.index[0],'polarity'] = TextBlob(combined_cleaned_tweets).sentiment[0]            
    live_coin_data.loc[live_coin_data.index[0],'subjectivity'] = TextBlob(combined_cleaned_tweets).sentiment[1]            


    # make the prediction
    return model.predict_proba(live_coin_data)
    

#|=======|
#| Main: |
#|=======|

@app.get("/prediction_generator/{coin}")
def generate_prediction(coin: Optional[str] = None):
    
    # use old model/generate model for that coin
    model = gen_model()
    
    fetched_tweets = pull_live_tweets(f'{coin} lang:en')
    fetched_tweets_df = pd.DataFrame(fetched_tweets['data'])
    fetched_tweets_df.to_csv('recently_fetched_tweets.csv')

    # look up prices for that coin from the last hour
    get_prices(coin)   

    # make prediction
    prediction = make_live_prediction(coin, fetched_tweets_df, model)
    d = dict(enumerate(prediction.flatten(), 1))
    resp = json.dumps(d)
    
    # return prediction
    return resp

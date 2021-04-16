from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import tweepy
from tweepy import OAuthHandler

import re, string, random, nltk
nltk.download('twitter_samples')
nltk.download('stopwords')

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                      '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


class TwitterClient(object):
    def __init__(self):
        consumer_key = 'kHZAFhMkr63lwmosmhkXT5gyI'
        consumer_secret = 'NHX3OGe1EDN1PXfwlY0dQEBCIIUQNYi3uJiFa7XZ5cMUSf8LJY'
        access_token = '624014297-wmSPstrVSEEC0LQzJfs1elVknZbXmhgNEL6aOOs0'
        access_token_secret = 'Ssrv95mgS1WnZ1Ae9U0rRw2u2gO3KbZ1S5Var9aNvtHCo'
        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def get_tweets(self, query, count=10):
            tweets = []
            try:
                fetched_tweets = self.api.search(q=query, count=count)
                for tweet in fetched_tweets:
                    parsed_tweet = {'text': tweet.text, 'author': tweet.author}
                    if tweet.retweet_count > 0:
                        if parsed_tweet not in tweets:
                            tweets.append(parsed_tweet)
                    else:
                        tweets.append(parsed_tweet)
                return tweets

            except tweepy.TweepError as e:
                print("Error : " + str(e))

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)



if __name__ == "__main__":
    api = TwitterClient()
    tweet_array = []

    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)
    still_searching = False
    while not still_searching:
        search_input = input("What would you like to search? ")
        is_number = False
        query = 0
        while not is_number:
            test = input("How many tweets from this topic would you like to look at?  ")
            try:
                num = int(test)
                print("Beginning query search...")
                query = num
                is_number = True
            except ValueError:
                print(
                    "This is not a valid number.")
        tweet_array = api.get_tweets(search_input, query)
        increment = 1
        for newTweet in tweet_array:
            custom_tokens = remove_noise(word_tokenize(newTweet['text']))
            print('TWEET #' + str(increment) + ': ')
            print(newTweet['text'], classifier.classify(dict([token, True] for token in custom_tokens)))
            print('\n')
            increment += 1
        search_input = input("Would you like to continue searching (Y/N)? ")
        if search_input == "N":
            still_searching = True
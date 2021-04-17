from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier
from nltk import classify
import tweepy
from tweepy import OAuthHandler

import re, string, random, nltk, operator
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('movie_reviews')

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):

        # First, work on the regex side. Make sure that no links, hashtags, or
        # @usernames count as tokens
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                      '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub('(@[A-Za-z0-9_]+)','', token)
        token = re.sub('(#[A-Za-z0-9]+)','', token)
        token = re.sub('&amp', 'and', token)
        token = re.sub('\\\'[A-Za-z0-9_]+', '', token)

        # Classify the tag as a noun, verb, or adjective/adverb that will fit with
        # the WordNetLemmatizer
        if tag.startswith("NN"):
            typeOfWord = 'n'
        elif tag.startswith('VB'):
            typeOfWord = 'v'
        else:
            typeOfWord = 'a'
        wordNet = WordNetLemmatizer()
        token = wordNet.lemmatize(token, typeOfWord)

        # Check to see if token is valid and impactful towards judging sentiment
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

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
                    parsed_tweet = {'text': tweet.text, 'author': tweet.author.id}
                    if tweet.retweet_count > 0:
                        if parsed_tweet not in tweets:
                            tweets.append(parsed_tweet)
                    else:
                        tweets.append(parsed_tweet)
                return tweets

            except tweepy.TweepError as e:
                print("Error : " + str(e))

    def get_author_tweets(self, user):
        try:
            currentUser = self.api.get_user(user)
            username = currentUser.screen_name
            tweets = self.api.user_timeline(screen_name=username, count=50, include_rts=False)
            return_text = []
            for tweet in tweets:
                parsed_tweet = {'text': tweet.text}
                return_text.append(parsed_tweet)
            return return_text
        except tweepy.TweepError as e:
            print("Error : " + str(e))

    def get_username_tweets(self, user, search_count):
        try:
            tweets = self.api.user_timeline(screen_name=user, count=search_count, include_rts=True)
            return_text = []
            for tweet in tweets:
                parsed_tweet = {'text': tweet.text}
                return_text.append(parsed_tweet)
            return return_text
        except tweepy.TweepError as e:
            print("Error : " + str(e))


if __name__ == "__main__":
    api = TwitterClient()
    tweet_array = []
    positive_cleaned = []
    negative_cleaned = []
    stop_words = stopwords.words('english')
    positive_reviews = nltk.corpus.movie_reviews.fileids(categories=["pos"])
    negative_reviews = nltk.corpus.movie_reviews.fileids(categories=["neg"])
    print("Tokenizing and removing noise from untokenized data sets")
    for movie in positive_reviews:
        text = nltk.corpus.movie_reviews.raw(movie)
        positive_cleaned.append(remove_noise(word_tokenize(text)))
    for movie in negative_reviews:
        text = nltk.corpus.movie_reviews.raw(movie)
        negative_cleaned.append(remove_noise(word_tokenize(text)))
    positive_original = twitter_samples.tokenized('positive_tweets.json')
    negative_original = twitter_samples.tokenized('negative_tweets.json')
    print("Gathering and cleaning tokens from already tokenized data sets")
    for tokens in positive_original:
        positive_cleaned.append(remove_noise(tokens, stop_words))

    for tokens in negative_original:
        negative_cleaned.append(remove_noise(tokens, stop_words))
    positive_final = []

    print("Classifying tokens as positive or negative")
    for tweets in positive_cleaned:
        for token in tweets:
            addition = ({token: True}, "Positive")
            positive_final.append(addition)
    negative_final = []
    for tweets in negative_cleaned:
        for token in tweets:
            addition = ({token: True}, "Negative")
            negative_final.append(addition)

    dataset = positive_final + negative_final

    random.shuffle(dataset)

    # train_data = dataset[:7000]
    # test_data = dataset[:7000]
    train_data = dataset[:1200000]
    test_data = dataset[1200000:]
    print("Training the data\n")
    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy of the Naive Bayes Classifier is:", classify.accuracy(classifier, test_data))
    print("All set to use!\n")

    still_searching = False
    while not still_searching:
        author_or_topic = input("Would you like to look at a user or topic? ")
        if author_or_topic.lower() == "topic":
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
            positive_count = 0
            negative_count = 0
            score = 0
            most_surprising = {}
            for newTweet in tweet_array:
                author_score = 0
                custom_tokens = remove_noise(word_tokenize(newTweet['text']))
                print('TWEET #' + str(increment) + ': ')
                currentTweetSatisfaction = classifier.classify(dict([token, True] for token in custom_tokens))
                if currentTweetSatisfaction == "Positive":
                    positive_count += 1
                    score = 0
                else:
                    negative_count += 1
                    score = 1
                print(newTweet['text'], '\nTHIS TWEET IS CLASSIFIED AS: ' + currentTweetSatisfaction)
                author = api.get_author_tweets(newTweet['author'])
                positive_increment = 0
                negative_increment = 0
                for tweet in author:
                    author_token = remove_noise(word_tokenize(tweet['text']))
                    if classifier.classify(dict([a_token, True] for a_token in author_token)) == "Positive":
                        positive_increment += 1
                    else:
                        negative_increment += 1
                print('Author has ' + str(positive_increment) + ' recent positive posts and '
                      + str(negative_increment) + ' recent negative posts.')
                if negative_increment == 0:
                    author_score = 1
                else:
                    author_score = positive_increment / (positive_increment + negative_increment)
                most_surprising[increment] = abs(score - author_score)
                print('\n')
                increment += 1
            print("Number of positive posts: " + str(positive_count))
            print("Number of negative posts: " + str(negative_count))
            sorted_most_surprising = dict(sorted(most_surprising.items(), key=operator.itemgetter(1)))
            print("Most surprising tweets to least surprising tweets: ")
            print(list(sorted_most_surprising.keys()))
            search_input = input("Would you like to continue searching (Y/N)? ")
            if search_input == "N" or search_input == "n":
                still_searching = True
        elif author_or_topic.lower() == "user":
            search_input = input("What is their username? ")
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
            tweet_array = api.get_username_tweets(search_input, query)
            increment = 1
            posi_increment = 0
            negative_increment = 0
            for newTweet in tweet_array:
                custom_tokens = remove_noise(word_tokenize(newTweet['text']))
                print('TWEET #' + str(increment) + ': ')
                currentTweetSatisfaction = classifier.classify(dict([token, True] for token in custom_tokens))
                print(newTweet['text'], '\nTHIS TWEET IS CLASSIFIED AS: ' + currentTweetSatisfaction + '\n')
                if currentTweetSatisfaction == "Positive":
                    posi_increment += 1
                else:
                    negative_increment += 1
                increment += 1
            print("User has " + str(posi_increment) + " positive posts and "
                  + str(negative_increment) + " negative posts.")
            search_input = input("Would you like to continue searching (Y/N)? ")
            if search_input == "N" or search_input == "n":
                still_searching = True

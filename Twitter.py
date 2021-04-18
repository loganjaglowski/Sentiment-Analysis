from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier
from nltk import classify
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

import re, string, random, nltk, operator

# Download data sets from nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('movie_reviews')

def remove_noise(tweet_tokens, stop_words = ()):
    """
    This function removes the noise from the tokens created.
    Also ensures that the tokens are classified correctly and are
    relevant towards the search terms

    Inspiration for this taken from: https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
    This site really helped us with the regex side of things, we did edit it to make the tokens more relevant

    :param tweet_tokens: Original tweet tokens sent into the function
    :param stop_words: Common english words that shouldn't be considered in evaluation
    :return: a cleaned list of tokens for future use
    """
    cleaned = []
    for token, tag in pos_tag(tweet_tokens):

        # First, work on the regex side. Make sure that no links, hashtags, or
        # @usernames count as tokens
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                      '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub('(@[A-Za-z0-9_]+)','', token)
        token = re.sub('(#[A-Za-z0-9]+)','', token)
        token = re.sub('.*&amp.*', 'and', token)
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
            cleaned.append(token.lower())
    return cleaned

class TwitterClient(object):
    """
    This Class is used to access tweet batches, rather than the streamed requests
    This class utilizes Twitter's API to connect to Twitter, as well as easily fetch
    and extract data from tweets
    """
    def __init__(self):
        """
        The init is necessary so we can use the proper credentials
        and connect to Twitter through Tweepy
        """
        consumer_key = 'kHZAFhMkr63lwmosmhkXT5gyI'
        consumer_secret = 'NHX3OGe1EDN1PXfwlY0dQEBCIIUQNYi3uJiFa7XZ5cMUSf8LJY'
        access_token = '624014297-wmSPstrVSEEC0LQzJfs1elVknZbXmhgNEL6aOOs0'
        access_token_secret = 'Ssrv95mgS1WnZ1Ae9U0rRw2u2gO3KbZ1S5Var9aNvtHCo'
        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth)
        except:
            # If anything goes wrong, return an authentication error
            print("Error: Authentication Failed")

    def get_tweets(self, query, count=10):
        """
        Get a batch of tweets based on the topic and the count
        the user submits

        :param query: The topic that is being searched
        :param count: The number of  queries to look up
        :return: 
        """
        tweets = []
        try:
            # Make sure no retweets are added
            query += ' -filter:retweets'
            fetched_tweets = self.api.search(q=query, count=count)
            for tweet in fetched_tweets:
                # Save the text of the tweet and the author of the tweet
                parsed_tweet = {'text': tweet.text, 'author': tweet.author.screen_name}
                tweets.append(parsed_tweet)
            return tweets

        except tweepy.TweepError as e:
            print("Error : " + str(e))

    def get_author_tweets(self, user):
        """
        Get the most recent 50 tweets from a specific author
        :param user: username of the author
        :return: array of tweets from the author
        """
        try:
            username = user
            # Get the most recent 50 tweets from the user's timeline
            tweets = self.api.user_timeline(screen_name=username, count=50, include_rts=False)
            return_text = []
            for tweet in tweets:
                # Take the text of the tweet and store it into the return array
                parsed_tweet = {'text': tweet.text}
                return_text.append(parsed_tweet)
            return return_text
        except tweepy.TweepError as e:
            print("Error : " + str(e))

    def get_username_tweets(self, user, search_count):
        """
        Get a user-specified number of tweets from the user-submitted author
        :param user: username of the author being searched
        :param search_count: number of items to search from the author
        :return: final array of tweets from the author
        """
        try:
            tweets = self.api.user_timeline(screen_name=user, count=search_count, include_rts=True)
            return_text = []
            for tweet in tweets:
                parsed_tweet = {'text': tweet.text}
                return_text.append(parsed_tweet)
            return return_text
        except tweepy.TweepError as e:
            print("Error : " + str(e))

    def get_auth(self):
        return self.auth

class myStreamListener(StreamListener):
    """
    Class is used to show the user the current stream of tweets for a specific topic.
    After each tweet, sentiment statistics are shown about the tweet and the user.
    """
    # Unfortunately need to save this information to access Tweepy API in this class
    consumer_key = 'kHZAFhMkr63lwmosmhkXT5gyI'
    consumer_secret = 'NHX3OGe1EDN1PXfwlY0dQEBCIIUQNYi3uJiFa7XZ5cMUSf8LJY'
    access_token = '624014297-wmSPstrVSEEC0LQzJfs1elVknZbXmhgNEL6aOOs0'
    access_token_secret = 'Ssrv95mgS1WnZ1Ae9U0rRw2u2gO3KbZ1S5Var9aNvtHCo'
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)


    def on_status(self, status):
        """
        This definition is needed for when new tweets found in the stream.
        From there, sentiment information is recorded about the user and the tweet
        :param status: The 'status' brought from the stream, contains information about new tweets
        :return: prints tweet sentiment information about the new tweet in the stream
        """

        # if detected as a retweet, don't recognize
        if status.retweeted:
            return
        if 'RT @' in status.text:
            return

        #Print the tweet and the author of it
        print("NEW TWEET")
        print("By: @" + str(status.user.screen_name))
        tweet = status.text
        print(status.text)

        # Remove noise and tokenize the tweet
        custom_tokens = remove_noise(word_tokenize(tweet))

        # Detect the satisfaction of the tweet and print it
        # Inspiration for this classifier line (used several times in the program) from:
        # https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
        currentTweetSatisfaction = classifier.classify(dict([token, True] for token in custom_tokens))

        print("THIS TWEET IS DETERMINED TO BE: " + currentTweetSatisfaction)
        author = api.get_author_tweets(status.user.screen_name)

        # Save tweet in integer form
        score = 0
        if currentTweetSatisfaction == "Positive":
            score = 1
        positive_increment = 0
        negative_increment = 0
        surprise_score = 0

        # check author's 50 recent posts and record information
        for tweets in author:
            author_token = remove_noise(word_tokenize(tweets['text']))

            # Inspiration for this classifier line (used several times in the program) from:
            # https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
            if classifier.classify(dict([a_token, True] for a_token in author_token)) == "Positive":
                positive_increment += 1
            else:
                negative_increment += 1

        # print the count of the negative and positive posts
        print('Author has ' + str(positive_increment) + ' recent positive posts and '
              + str(negative_increment) + ' recent negative posts.')

        # calculate if this tweet seems surprising based off whether not this user usually posts positive/negative posts
        if negative_increment == 0:
            surprise_score = abs(score - (positive_increment))
        else:
            surprise_score = abs(score - (positive_increment/ (positive_increment + negative_increment)))
        if surprise_score >= 0.7:
            print("This tweet is deemed to be surprising\n")
        elif surprise_score >= 0.4:
            print("This tweet is deemed to be somewhat surprising\n")
        else:
            print("This tweet is deemed to not be surprising\n")



    def on_error(self, status_code):
        """
        If an error occurs, inform the user of the error
        :param status_code: status code of the error
        :return: return False
        """
        if status_code == 420:
            print("The Rate is Being Limited at the current moment")
            return False
        else:
            print("Error occurred: ")
            print(str(status_code))
            return False

if __name__ == "__main__":
    """
    Main Function of the program. Handles the input from the user,
    calling of functions, and the looping of events for user input.
    """

    # get the API running and create initial arrays
    api = TwitterClient()
    tweet_array = []
    positive_cleaned = []
    negative_cleaned = []

    # Assign variables for the stopwords and the review ID's
    stop_words = stopwords.words('english')
    positive_reviews = nltk.corpus.movie_reviews.fileids(categories=["pos"])
    negative_reviews = nltk.corpus.movie_reviews.fileids(categories=["neg"])
    print("Tokenizing and removing noise from untokenized data sets")

    # For the positive and negative reviews, get the raw data of each ID
    # Then, tokenize, remove the noise, and append to their respective cleaned array
    for movie in positive_reviews:
        text = nltk.corpus.movie_reviews.raw(movie)
        positive_cleaned.append(remove_noise(word_tokenize(text)))
    for movie in negative_reviews:
        text = nltk.corpus.movie_reviews.raw(movie)
        negative_cleaned.append(remove_noise(word_tokenize(text)))

    # Gather tokenized positive and negative tweet samples, then remove noise
    # and append them to their respective arrays
    positive_original = twitter_samples.tokenized('positive_tweets.json')
    negative_original = twitter_samples.tokenized('negative_tweets.json')
    print("Gathering and cleaning tokens from already tokenized data sets")
    for tokens in positive_original:
        positive_cleaned.append(remove_noise(tokens, stop_words))
    for tokens in negative_original:
        negative_cleaned.append(remove_noise(tokens, stop_words))
    positive_final = []

    # Classify their tokens as positive or negative
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

    # Combine positive and negative, then train and test the data set.
    dataset = positive_final + negative_final
    random.shuffle(dataset)

    # train_data = dataset[:7000]
    # test_data = dataset[:7000]
    train_data = dataset[:1000000]
    test_data = dataset[1000000:]
    print("Training the data\n")
    classifier = NaiveBayesClassifier.train(train_data)

    # Record accuracy of the Classifier
    print("Accuracy of the Naive Bayes Classifier is:", classify.accuracy(classifier, test_data))
    print("All set to use!\n")

    # A loop asking what the user would like to look at in terms of tweets
    still_searching = False
    while not still_searching:
        author_or_topic = input("Would you like to look at a user, topic, or stream? ")
        if author_or_topic.lower() == "topic":

            # If the user searches for a topic, prompt for a topic name and number of tweets
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

            # Get the tweets for the topic and instantiate necessary variables
            tweet_array = api.get_tweets(search_input, query)
            increment = 1
            positive_count = 0
            negative_count = 0
            score = 0
            most_surprising = {}

            for newTweet in tweet_array:

                # For each tweet: Print out the current tweet, then tokenize it. Then, increment
                # proper variables based on if it's a positive or negative tweet
                author_score = 0
                custom_tokens = remove_noise(word_tokenize(newTweet['text']))
                print('TWEET #' + str(increment) + ': ')
                print("By: @" + newTweet['author'])

                # Classify the tweet and increment the positive and negative counter
                # Inspiration for this classifier line (used several times in the program) from:
                # https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
                currentTweetSatisfaction = classifier.classify(dict([token, True] for token in custom_tokens))
                if currentTweetSatisfaction == "Positive":
                    positive_count += 1
                    score = 0
                else:
                    negative_count += 1
                    score = 1

                # Give user information about classification and get author's tweets
                print(newTweet['text'], '\nTHIS TWEET IS CLASSIFIED AS: ' + currentTweetSatisfaction)
                author = api.get_author_tweets(newTweet['author'])
                positive_increment = 0
                negative_increment = 0

                # Look at author's most recent 50 tweets, then tokenize them and classify them as positive and negative
                for tweet in author:
                    author_token = remove_noise(word_tokenize(tweet['text']))

                    # Inspiration for this classifier line (used several times in the program) from:
                    # https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
                    if classifier.classify(dict([a_token, True] for a_token in author_token)) == "Positive":
                        positive_increment += 1
                    else:
                        negative_increment += 1
                print('Author has ' + str(positive_increment) + ' recent positive posts and '
                      + str(negative_increment) + ' recent negative posts.')

                # Give a surprising score. If a user posts a negative tweet but is generally positive,
                # their score will be higher
                if negative_increment == 0:
                    author_score = 1
                else:
                    author_score = positive_increment / (positive_increment + negative_increment)
                most_surprising[increment] = abs(score - author_score)
                print('\n')
                increment += 1

            # Print off statistics to the user
            print("Number of positive posts: " + str(positive_count))
            print("Number of negative posts: " + str(negative_count))
            sorted_most_surprising = dict(sorted(most_surprising.items(), key=operator.itemgetter(1)))
            print("Most surprising tweets to least surprising tweets: ")
            print(list(sorted_most_surprising.keys()))

            # Ask if user wants to continue
            search_input = input("Would you like to continue searching (Y/N)? ")
            if search_input == "N" or search_input == "n":
                still_searching = True
        elif author_or_topic.lower() == "user":

            # Ask for the username the user would like to
            # research more as well as number of queries
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

            # Get the tweets from the author and instantiate necessary variables
            tweet_array = api.get_username_tweets(search_input, query)
            increment = 1
            posi_increment = 0
            negative_increment = 0
            for newTweet in tweet_array:

                # For each tweet: Print out the current tweet, then tokenize it. Then, increment
                # proper variables based on if it's a positive or negative tweet
                custom_tokens = remove_noise(word_tokenize(newTweet['text']))
                print('TWEET #' + str(increment) + ': ')

                # Inspiration for this classifier line (used several times in the program) from:
                # https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
                currentTweetSatisfaction = classifier.classify(dict([token, True] for token in custom_tokens))

                print(newTweet['text'], '\nTHIS TWEET IS CLASSIFIED AS: ' + currentTweetSatisfaction + '\n')
                if currentTweetSatisfaction == "Positive":
                    posi_increment += 1
                else:
                    negative_increment += 1
                increment += 1

            # Print author's statistic to the user
            print("User has " + str(posi_increment) + " positive posts and "
                  + str(negative_increment) + " negative posts.")
            search_input = input("Would you like to continue searching (Y/N)? ")

            # Ask if the user would like to continue searching
            if search_input == "N" or search_input == "n":
                still_searching = True
        elif author_or_topic.lower() == "stream":

            # Ask the user what topic they'd like to view the stream of
            search_input = input("What topic would you like to constantly look at? ")
            auth = api.get_auth()
            listener = myStreamListener()

            # Begin the stream, most of the activity will exist in its specific class
            stream = Stream(auth, listener)
            stream.filter(languages=['en'], track=[search_input])

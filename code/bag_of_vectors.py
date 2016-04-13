import re
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import indicoio
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


def review_to_sentiment( review):
    '''
    Function to split a review into parsed sentences and get sentiment
    This uses one function call for each review... good for testing, not
    good for production, since you have a limited number of API calls
    '''
    # 1. Use Indico to split the review into sentences, with sentiment
    results = indicoio.sentiment(review, split='sentence')
    # 2. Loop over each sentence
    sums = 0
    for item in results:
        sums += item['results']
    avg_sentiment = sums / len(results)
    return avg_sentiment


def result_to_sentiment(result):
    '''
    The result is a list of dictionaries, one per sentence in the review.
    The sentiment value of that sentence is the value for the 'results' key.
    The weight of the sentiment is the proportion of review given by that sentence.
    '''
    sums = 0
    # total_len = result[-1]['position'][1]
    for sentence in result:
        #sent_len = sentence['position'][1] - sentence['position'][0]
        sums += sentence['results']
    avg_sentiment = sums / len(result)
    return avg_sentiment


def load_data():
    '''
    Import the labeled training data into the train dataframe. Import the unlabeled
    test and training data into the test and extra dataframes. The train data will
    be subset into training and testing data, and the unlabeled test and extra data
    will be used only for building vocabulary.
    '''
    train = pd.read_csv('./labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('./testData.tsv', header=0, delimiter="\t", quoting=3 )
    extra = pd.read_csv('./unlabeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    return train, test, extra


def test_train(train):
    '''
    Given the train dataframe with target sentiment values, divide it into
    training and testing dataframes for features, with training and testing targets.
    This is used instead of the supplied test data, because that had no targets.
    '''
    print "Dividing the data..."
    target = train['sentiment']
    train = train.drop('sentiment', axis=1)
    Xtr, Xt, ytr, yt = train_test_split(train, target, test_size=.3, random_state=42)
    Xtr.reset_index(inplace=True)
    Xt.reset_index(inplace=True)
    ytr = pd.Series(ytr.values)
    yt = pd.Series(yt.values)
    return Xtr, Xt, ytr, yt


def remove_html(df):
    '''
    Given a dataframe (df) with a column 'review', for each review, remove the
    html using BeautifulSoup, make sure all sentence delimiters are followed by
    spaces, and return a list of the reviews.
    '''
    cleaned_reviews=[]
    for review in df['review']:
        # get rid of any html in the review
        review_text = BeautifulSoup(review, "lxml").get_text()
        # remove the opening and closing quotes around the review.
        review_text = review_text[1:-1]
        # add a space after every period, question mark, and exclamation point
        # then collapse ellipses
        review_text = review_text.replace('.','. ')
        review_text = review_text.replace('?','? ')
        review_text = review_text.replace('!','! ')
        review_text = review_text.replace('. . . ','...')
        cleaned_reviews.append(review_text)
    return cleaned_reviews


def use_indico(train):
    '''
    Batch reviews in groups of 1000 and send them to the Indico API to get
    sentiment results. Return the list of results, as well as the test features
    and test targets, to be used in testing the results.
    Isolate the fetching of the sentiment results from Indico from the use of
    those results, so that if something goes wrong, we don't need to fetch again.
    No need to clean and vectorize training reviews, or train a random forest
    on them, because Indico has done all of that already. Just strip out html.
    '''
    Xtrain, Xtest, ytrain, ytest = test_train(train)
    print "Cleaning html from the test set of movie reviews..."
    clean_test_reviews = remove_html(Xtest)
    print "Running Indico queries..."
    print "This will take a while..."
    # process the reviews in batches of 1000, then finish with the leftovers, if any
    sentiment_lists = []
    for i in range(1000,len(Xtest),1000):
        print "Processing reviews {0} to {1}...".format(i-1000, i-1)
        batch = clean_test_reviews[i-1000:i]
        results = indicoio.sentiment(batch, split='sentence')
        sentiment_lists += results
    if len(sentiment_lists)<len(Xtest):
        print "Processing final reviews {0} to {1}...".format(len(sentiment_lists),len(Xtest))
        batch = clean_test_reviews[len(sentiment_lists):]
        results = indicoio.sentiment(batch, split='sentence')
        sentiment_lists += results
    print "{0} Indico sentiments returned".format(len(sentiment_lists))
    return sentiment_lists, Xtest, ytest


def calculate_sentiment(sentiment_lists, Xtest, ytest):
    '''
    Convert the returned sentiment_lists into individual document results and
    compute the overall sentiment for each document result. Copy the results to
    a DataFrame and write it as a CSV file for analysis.
    '''
    Xpred = []
    i=0
    for result in sentiment_lists:
        # If the index is evenly divisible by 1000, print a message
        if( (i+1)%1000 == 0 ):
            print "Review %d of %d" % (i+1, len(sentiment_lists))
        Xpred.append(result_to_sentiment(result))
        i += 1
    output = pd.DataFrame( data={"id":Xtest["id"], "percent_prediction":Xpred, "actual":ytest} )
    output['prediction']=output['percent_prediction'].round(0)
    # Use pandas to write the comma-separated output file
    output.to_csv('./Indico_api_with_spaces.csv', index=False, quoting=3)
    print "Wrote results to Indico_api_with_spaces.csv"

if __name__ == '__main__':
    #Hide API before pushing to GitHub!!!
    indicoio.config.api_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    train, test, extra = load_data()
    # run indico sentiment
    sentiment_lists, Xtest, ytest = use_indico(train)
    print "Calculating sentiment..."
    calculate_sentiment(sentiment_lists, Xtest, ytest)

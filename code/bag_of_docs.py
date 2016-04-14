import re
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import indicoio
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk.data
import pattern
import pattern.en
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing
from random import shuffle
import datetime
from contextlib import contextmanager
from timeit import default_timer
import time
from gensim.models.doc2vec import TaggedDocument

@contextmanager
def elapsed_timer():
    '''
    Timing function to calculate how long is spent during epochs when training.
    '''
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

# global variables for all non alphanumeric characters to be used in clean function
contractions = re.compile(r"'|-")
symbols = re.compile(r'(\W+)', re.U)
numeric = re.compile(r'(?<=\s)(\d+|\w\d+|\d+\w)(?=\s)', re.I)
seps = re.compile(r'\s+')

def clean(text):
    '''
    Given a review (text), use the above defined global variables to add spaces
    around words, lower case words, replace apostrophes with hyphens for contractions,
    and convert all numbers into a single designated number, 000. These choices
    for cleaning came from Matt Taddy's Doc2Vec parser.
    '''
    # clean uses the above defined substitutions (order matters)
    text = u' ' +  text.lower() + u' '
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = numeric.sub('000', text)
    text = seps.sub(' ', text)
    return text

def remove_html(df):
    '''
    Rather than doing a full text cleaning, as the clean function does, this
    takes all reviews in the 'review' column of a dataframe (df) and removes html,
    adds spaces after sentence delimiters, and returns a list of cleaned_reviews.
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


def review_to_words(raw_review, stem=False, stop=False):
    '''
    Converts a raw review to a string of words. The input is a single string
    (a raw movie review), and the output is a single string (a preprocessed
    movie review).
    '''
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    # 2. Remove non-letters, convert letters to lowercase
    letters_only = clean(review_text)
    # 3. Split into individual words
    words = letters_only.split()
    # 4. Optionally remove stop words and stem using Porter Stemming
    if stem and stop:
        meaningful_words = [stemmer.stem(w) for w in words if not w in stops]
    elif stem and not stop:
        meaningful_words = [stemmer.stem(w) for w in words]
    elif stop and not stem:
        meaningful_words = [w for w in words if not w in stops]
    else:
        meaningful_words = words
    # 5. Return the result.
    return meaningful_words

def load_data():
    '''
    Import the labeled training data into the train dataframe. Import the unlabeled
    test and training data into the test and extra dataframes. The train data will
    be subset into training and testing data, and the unlabeled test and extra data
    will be used only for building vocabulary.
    '''
    print "Loading the data..."
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
    #train = train.drop('sentiment', axis=1)
    Xtr, Xt, ytr, yt = train_test_split(train, target, test_size=.3, random_state=42)
    Xtr.reset_index(inplace=True)
    Xt.reset_index(inplace=True)
    ytr = pd.Series(ytr.values)
    yt = pd.Series(yt.values)
    return Xtr, Xt, ytr, yt


def clean_reviews(data, method='bagofwords'):
    '''
    Given a dataframe (data) with a column 'review', clean the reviews and
    create a list of cleaned reviews depending on the method that needs to
    use them. For bag of words, create a string of words separated by spaces for
    each document. For word2vec, create a list of words. For doc2vec, create a
    document in the gensim TaggedDocument format. Return a list of the chosen format.
    '''
    cleaned_reviews = []
    num_reviews = len(data['review'])
    print 'Cleaning {0} reviews using {1} method...'.format(num_reviews, method)
    for i in xrange( 0, num_reviews):
        # If the index is evenly divisible by 1000, print a message
        if( (i+1)%1000 == 0 ):
            print "Review %d of %d" % (i+1, num_reviews)
        # bagofwords needs a single string of words, separated by spaces
        # word2vec needs the words provided in a list
        # doc2vec needs the words provided as TaggedDocument format
        if method=='bagofwords':
            clean_review = ( " ".join(review_to_words(data["review"][i])))
        elif method=='word2vec':
            clean_review = review_to_words(data["review"][i])
        elif method=='doc2vec':
            clean_review = TaggedDocument(review_to_words(data["review"][i], stem=False, stop=True),[str(data["id"][i][1:-1])])
        else:
            print "only 'bagofwords', 'word2vec', and 'doc2vec' accepted as method"
        cleaned_reviews.append(clean_review)
    return cleaned_reviews


def build_doc2vec_models(train):
    '''
    Given a dataframe (train), get a list of gensim TaggedDocuments that contain
    a list of the words in the document and the document id. Create three different
    doc2vec models (distributed bag of words, distributed model with averaging of
    vectors, and distributed model with concatenation of vectors). To get the
    300-feature vector representing each document, pass the document id to the model.
    Models are trained in multiple epochs, with the data sorted prior to each one.
    Save the models so they can be used later without needing to retrain.
    '''
    # Pass the entire training set, because this is how word2vec will
    # know what the 300-feature vector for that document is.
    # The sentiment values are not used at this time, so no data leakage
    cleaned_tagged_docs = clean_reviews(train, method='doc2vec')
    # The three models of doc2vec being tested share:
    # 300 feature size (to match word2vec and indico models)
    # window=5 (both sides) approximates a 10-word total window size
    # min_count=2 gets rid of unique words
    simple_models = [
    # PV-DM w/concatenation
    Doc2Vec(dm=1, dm_concat=1, size=300, window=5, negative=5, hs=0, min_count=1, workers=cores),
    # PV-DBOW
    Doc2Vec(dm=0, size=300, negative=5, hs=0, min_count=1, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=300, window=10, negative=5, hs=0, min_count=1, workers=cores),
    ]
    models_by_name = OrderedDict((str(model), model) for model in simple_models)
    # speed setup by sharing results of 1st model's vocabulary scan
    simple_models[0].build_vocab(cleaned_tagged_docs)
    # PV-DM/concat requires one special NULL word so it serves as template
    for model in simple_models[1:]:
        model.reset_from(simple_models[0])
    # start each model with basic values
    alpha, min_alpha, passes = (0.025, 0.001, 20)
    alpha_delta = (alpha - min_alpha) / passes
    print("START %s" % datetime.datetime.now())
    # run through multiple epochs, shuffling between, for best training
    for epoch in range(passes):
        shuffle(cleaned_tagged_docs)  # shuffling gets best results
        for name, train_model in models_by_name.items():
            # train
            duration = 'na'
            train_model.alpha, train_model.min_alpha = alpha, alpha
            with elapsed_timer() as elapsed:
                train_model.train(cleaned_tagged_docs)
                duration = '%.1f' % elapsed()
                if ((epoch + 1) % 5) == 0 or epoch == 0:
                    print("%i passes : %s %ss " % (epoch + 1, name, duration))
        print('completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta
    print("END %s" % str(datetime.datetime.now()))
    # save the final models so we don't need to do this again
    for name, train_model in models_by_name.items():
        train_model.init_sims(replace=True)
        name = name.replace('(','_')
        name = name.replace(')','')
        name = name.replace('/','-')
        name = name.replace(',','_')
        train_model.save(name)
    print "models saved"


def build_forest(model, Xtrain, ytrain):
    '''
    Given a particular doc2vec model (model), use the 'id' column in the dataframe
    Xtrain to obtain a 300-feature array for each review in the Xtrain dataframe.
    Pass that feature array along with the targets (ytrain) to the fit_forest
    function to create and train a random forest model. Return the model.
    '''
    print "Creating list of features for documents..."
    features=[]
    for id in Xtrain['id']:
        # the ids are double-quoted strings, this removes the extra quotes
        features.append(model.docvecs[id[1:-1]])
    print "Training the random forest (this may take a while)..."
    forest = fit_forest(features, ytrain)
    return forest


def fit_forest(features, target):
    '''
    Initialize a Random Forest classifier with 500 trees. Fit the forest to the
    training set, using the bag of words as features and the sentiment labels as
    the target. Return the fitted forest.
    '''
    forest = RandomForestClassifier(n_estimators = 500, n_jobs=-1, random_state=42)
    # This may take a few minutes to run
    forest = forest.fit(features, target)
    return forest


def use_forest(forest, features, target, filename):
    '''
    Use the Random Forest model that was created and fitted by the fit_forest
    function to predict sentiment labels for the testing set of features.
    Copy the results to a DataFrame and write it as a CSV file for analysis.
    '''
    print "Predicting test labels..."
    result = forest.predict(features)
    # Copy the results to a pandas dataframe
    output = pd.DataFrame( data={"id":Xtest["id"], "prediction":result, "actual":target} )
    # Use pandas to write the comma-separated output file
    output.to_csv('./'+filename, index=False, quoting=3)
    print "Wrote results to {}".format(filename)


def test_build(Xtrain, ytrain, Xtest, ytest):
    '''
    Load the three varieties of Doc2Vec models that were previously saved.
    Build a random forest model for each Doc2Vec model. Test each random
    forest model with the same test data, and write the results to a CSV
    file for each Doc2Vec model.
    '''
    print "Loading the model..."
    models = [Doc2Vec.load("Doc2Vec_dbow_d300_n5_t4"), \
    Doc2Vec.load("Doc2Vec_dm-c_d300_n5_w5_t4"),  \
    Doc2Vec.load("Doc2Vec_dm-m_d300_n5_w10_t4")]
    filenames = ['Doc2Vec_dbow.csv', 'Doc2Vec_dm-c.csv', 'Doc2Vec_dm-m.csv']
    forests = []
    for model in models:
        forests.append(build_forest(model, Xtrain, ytrain))
    for i in xrange(3):
        model = models[i]
        forest = forests[i]
        filename = filenames[i]
        features = []
        print "Creating feature list for test data..."
        for id in Xtest['id']:
            # remove the extra quotes around the id
            features.append(model.docvecs[id[1:-1]])
        print "Predicting test sentiment..."
        use_forest(forest, features, ytest, filename)


def use_indico(Xtest, ytest):
    '''
    Batch reviews in groups of 100 and send them to the Indico high quality API to get
    sentiment results. Return the list of results, as well as the test features
    and test targets, to be used in testing the results. Process each document as a
    whole, rather than processing each sentence individually.
    Isolate the fetching of the sentiment results from Indico from the use of
    those results, so that if something goes wrong, we don't need to fetch again.
    No need to clean and vectorize training reviews, or train a random forest
    on them, because Indico has done all of that already. Just strip out html.
    '''
    print "Cleaning html from the test set of movie reviews..."
    clean_test_reviews = remove_html(Xtest)
    print "Running Indico queries..."
    print "This will take a while..."
    # process the reviews in batches of 1000, then finish with the leftovers, if any
    # Indico is not splitting on sentences, returns one sentiment per review
    sentiment_lists = []
    for i in range(100,len(Xtest),100):
        print "Processing reviews {0} to {1}...".format(i-100, i-1)
        batch = clean_test_reviews[i-100:i]
        results = indicoio.sentiment_hq(batch)
        sentiment_lists += results
    if len(sentiment_lists)<len(Xtest):
        print "Processing final reviews {0} to {1}...".format(len(sentiment_lists),len(Xtest))
        batch = clean_test_reviews[len(sentiment_lists):]
        results = indicoio.sentiment_hq(batch)
        sentiment_lists += results
    print "{0} Indico sentiments returned".format(len(sentiment_lists))
    return sentiment_lists


def calculate_sentiment(sentiment_lists, Xtest, ytest):
    '''
    Convert the returned sentiment_lists into individual document results and compute
    the overall sentiment for each document result. Write the results to a CSV
    file for future analysis.
    '''
    output = pd.DataFrame( data={"id":Xtest["id"], "percent_prediction":sentiment_lists, "actual":ytest} )
    output['prediction']=output['percent_prediction'].round(0)
    # Use pandas to write the comma-separated output file
    output.to_csv('./Indico_hq_api_by_doc.csv', index=False, quoting=3)
    print "Wrote results to Indico_hq_api_by_doc.csv"


def use_pattern(Xtest, ytest):
    '''
    Use the built-in pattern sentiment function to create sentiment predictions for the
    reviews in the Xtest dataframe. Return the lists of sentiment and subjectivity.
    '''
    print "Cleaning html from the test set of movie reviews..."
    clean_test_reviews = remove_html(Xtest)
    print "Running Pattern queries..."
    print "This will take a while..."
    # process the reviews in batches of 1000, then finish with the leftovers, if any
    # Pattern returns one sentiment as a polarity (-1 to 1), subjectivity (0 to 1) per review
    sent_lists = []
    subj_lists = []
    num_reviews = len(Xtest)
    for i in range(num_reviews):
        if( (i)%1000 == 0 ):
            print "Processing review {0} of {1}".format(i, num_reviews)
        results = pattern.en.sentiment(Xtest['review'][i])
        sent_lists.append(results[0])
        subj_lists.append(results[1])
    print "{0} Pattern sentiments returned".format(len(sent_lists))
    return sent_lists, subj_lists


def calculate_pattern_sentiment(sent_lists, subj_lists, Xtest, ytest):
    '''
    Convert the returned sentiment_lists into individual document results and
    compute the overall sentiment for each document result. Write the results to a CSV
    file for future analysis.
    '''
    # convert
    #
    output = pd.DataFrame( data={"id":Xtest["id"], "percent_prediction":sent_lists, "subjectivity":subj_lists, "actual":ytest} )
    output['prediction']=output['percent_prediction'].round(0)
    # Use pandas to write the comma-separated output file
    output.to_csv('./Pattern_api_by_doc.csv', index=False, quoting=3)
    print "Wrote results to Pattern_api_by_doc.csv"


if __name__ == '__main__':
    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1

    #Hide API before pushing to GitHub!!!
    indicoio.config.api_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

    # performance gain - move this to a global variable
    stops = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # Load the data from TSV files
    train, test, extra = load_data()
    # Divide the provided training data into test/train split for statistics
    # we can apply accross all variations of processing.
    # to make shuffling between epochs easier, alldocs contains both train and target data
    # it can be shuffled and split each epoch, rather than having to shuffle divided data
    # the same way each time
    alldocs, Xtest, ytrain, ytest = test_train(train)
    Xtrain = alldocs.drop('sentiment', axis=1)
    #build_doc2vec_models(train)
    # no need to build the model again after it's been built
    #test_build(Xtrain, ytrain, Xtest, ytest)
    sentiment_lists = use_indico(Xtest, ytest)
    calculate_sentiment(sentiment_lists, Xtest, ytest)
    #sent_lists, subj_lists = use_pattern(Xtest, ytest)
    #calculate_pattern_sentiment(sent_lists, subj_lists, Xtest, ytest)

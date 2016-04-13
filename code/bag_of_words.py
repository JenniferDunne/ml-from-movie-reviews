import re
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk.data
from gensim.models import word2vec
import pattern

# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)


def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    '''
    Function to split a review into parsed sentences. Returns a
    list of sentences, where each sentence is a list of words
    '''
    # 0. Get rid of non-tokenizable characters
    review = BeautifulSoup(review, "lxml").get_text()
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, get a list of words
            letters_only = re.sub("[^a-zA-Z]", " ", raw_sentence)
            words = letters_only.lower().split()
            sentences.append(words)
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists)
    return sentences

def review_to_words(raw_review, stem=False, stop=False):
    '''
    Function to convert a raw review to a string of words.
    The input is a single string (a raw movie review), and
    the output is a single string (a preprocessed movie review).
    '''
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
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
    train = train.drop('sentiment', axis=1)
    Xtr, Xt, ytr, yt = train_test_split(train, target, test_size=.3, random_state=42)
    Xtr.reset_index(inplace=True)
    Xt.reset_index(inplace=True)
    ytr = pd.Series(ytr.values)
    yt = pd.Series(yt.values)
    return Xtr, Xt, ytr, yt

def clean_reviews(data, bagofwords=True):
    '''
    Given a data dataframe containing a 'review' column, clean the reviews.
    Reviews are returned as either a single string of words separated by
    spaces (if bagofwords=True) or a list of words (if bagofwords=False).
    '''
    cleaned_reviews = []
    num_reviews = len(data['review'])
    for i in xrange( 0, num_reviews):
        # If the index is evenly divisible by 1000, print a message
        if( (i+1)%1000 == 0 ):
            print "Review %d of %d\n" % (i+1, num_reviews)
        # bagofwords needs a single string of words, separated by spaces
        # word vector needs the words provided in a list
        if bagofwords:
            clean_review = ( " ".join(review_to_words(data["review"][i])))
        else:
            clean_review = review_to_words(data["review"][i])
        cleaned_reviews.append(clean_review)
    return cleaned_reviews

def build_bag_of_words(reviews, test_reviews):
    '''
    Initialize CountVectorizer, then fit it with our vocabulary, supplied as a
    list of reviews, and transform the reviews into feature vectors (max 6000).
    Use the fitted model to transform test_reviews into feature vectors, and
    return both arrays of feature vectors.
    '''
    # Initialize the "CountVectorizer" object
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 6000)
    # The input to fit_transform should be a list of strings
    features = vectorizer.fit_transform(reviews)
    # Numpy arrays are easy to work with, so convert the result
    features = features.toarray()
    # calculate the test arrays for this bag of words vector
    test_features = vectorizer.transform(test_reviews)
    test_features = test_features.toarray()
    return features, test_features

def fit_forest(features, target):
    '''
    Initialize a Random Forest classifier with 500 trees. Fit the forest to the
    training set of features provided by the build_bag_of_words function. Use
    the sentiment labels as the target. Return the fitted model.
    '''
    forest = RandomForestClassifier(n_estimators = 500, n_jobs=-1)
    # This may take a few minutes to run
    forest = forest.fit(features, target)
    return forest

def use_forest(forest, features, target):
    '''
    Use the Random Forest model that was created and fitted by the fit_forest
    function to predict sentiment labels for the testing set of features
    provided by the build_bag_of_words function. Copy the results to a DataFrame
    and write it as a CSV file for analysis.
    '''
    print "Predicting test labels...\n"
    result = forest.predict(features)
    # Copy the results to a pandas dataframe
    output = pd.DataFrame( data={"id":Xtest["id"], "prediction":result, "actual":target} )
    # Use pandas to write the comma-separated output file
    output.to_csv('./Bag_of_Words_model.csv', index=False, quoting=3)
    print "Wrote results to Bag_of_Words_model.csv"

def compute_bag_of_words(Xtrain, Xtest, ytrain, ytest):
    '''
    This is the main I/O of the bag of words processing. Given the test and train
    features and targets, it describes the size of the training dataframe and
    outputs a sample review for visual inspection. It then cleans the reviews,
    builds a bag of words model, builds a random forest, fits the random forest,
    and tests the predictive abilities of the model using the test features and
    test targets.
    '''
    print "The size of the training file is " + str(Xtrain.shape)
    print 'The first review is:'
    print Xtrain["review"][0]
    print "Cleaning and parsing the training set movie reviews...\n"
    clean_train_reviews = clean_reviews(Xtrain)
    print "Cleaning and parsing the test set of movie reviews...\n"
    clean_test_reviews = clean_reviews(Xtest)
    print "Creating the bag of words...\n"
    train_data_features, test_data_features = build_bag_of_words(clean_train_reviews, clean_test_reviews)
    print "Training the random forest (this may take a while)..."
    forest = fit_forest(train_data_features, ytrain)
    use_forest(forest, test_data_features, ytest)

def compute_word_vector(train, unlabeled_train, tokenizer):
    '''
    This creates a word2vec model of the training reviews. The tokenizer is
    from the NLTK package. Because the word2vec model requires an input of
    lists of words, first the reviews are processed through the review_to_sentences
    function to return a list of lists of words, one list per sentence. The model
    is saved for future use.
    '''
    sentences = []  # Initialize an empty list of sentences
    print "Parsing sentences from training set"
    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)
    print "Parsing sentences from unlabeled set"
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)
    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    # Initialize and train the model (this will take some time)
    print "Training model... please stand by...."
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
    # Call init_sims to freeze the model. Much more memory-efficient.
    model.init_sims(replace=True)
    # Save the model for later use. Load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    return model

def all_sentences(df,tokenizer):
    '''
    Given a dataframe (df) containing a 'review' column, returns a list of
    all sentences in each review.
    '''
    print "Making lists of {0} documents".format(len(df))
    sentence_list = []
    for review in df['review']:
        sentence_list.append(review_to_sentences(review, tokenizer, remove_stopwords=False ))
    return sentence_list

def use_word_vector(forest, features, target, ids):
    '''
    Given a random forest model (forest), test features (features), and test
    targets (target), run the features through the forest to predict the sentiment
    labels and write the predicted and true results to a CSV file. Include the ids
    for each review so that manual analysis can be performed later.
    '''
    print "Predicting results for {0} feature arrays".format(len(target))
    Xpred = []
    i = 0
    for vector in features:
        # If the index is evenly divisible by 1000, print a message
        if( (i+1)%1000 == 0 ):
            print "Review %d of %d\n" % (i+1, len(target))
        # reshape required by numpy not allowing passing a 1-dimensional array
        vector = vector.reshape(1,-1)
        Xpred.append(forest.predict(vector)[0])
        i += 1
    print "Creating dataframe of results"
    output = pd.DataFrame( data={"id":ids, "prediction":Xpred, "actual":target} )
    # Use pandas to write the comma-separated output file
    print "Writing dataframe of results"
    output.to_csv('./Word2Vec_model.csv', index=False, quoting=3)

def words_to_vectors(model, sentences):
    '''
    Given a word2vec model, for every word in sentences that is in the vocabulary
    of that model, add the model's vector for that word to the existing vector
    for that sentence. (Sentence vectors are initialized as zero arrays.) Return
    a list of all sentence vectors.
    '''
    print "Creating vectors for {} documents".format(len(sentences))
    result = []
    for document in sentences:
        doc_arr = np.zeros(300)
        for sentence in document:
            for word in sentence:
                if word in model.vocab:
                    doc_arr += model[word]
        result.append(doc_arr)
    return result

if __name__ == '__main__':
    # performance gain - move this to a global variable
    stops = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # Load the data from TSV files
    train, test, extra = load_data()
    # Divide the provided training data into test/train split for statistics
    # we can apply accross all variations of processing.
    Xtrain, Xtest, ytrain, ytest = test_train(train)
    # run doc2vec on the training data
    # model = compute_doc_vector(Xtrain, ytrain, tokenizer)
    # no need to build the model again after it's been built
    print "Loading the model..."
    model = word2vec.Word2Vec.load("300features_40minwords_10context")
    # fit a random forest to the training data
    sentences = all_sentences(Xtrain,tokenizer)
    train_data_features = words_to_vectors(model, sentences)
    forest = fit_forest(train_data_features, ytrain)
    # see how well word2vec does on the test data
    sentences = all_sentences(Xtest, tokenizer)
    test_data_features = words_to_vectors(model, sentences)
    use_word_vector(forest, test_data_features, ytest, Xtest['id'])
